from __future__ import annotations

from su_modfjsp_ha_pc.rules import job_pickup_of, move_leg_usage

from pathlib import Path

from su_modfjsp_ha_pc.generator import generate_instance
from su_modfjsp_ha_pc.neural_operation_rollout import (
    _RolloutState,
    _candidate_completion,
    _detail_row,
    _pairwise_rows,
    _summary_rows,
    decode_rule_operation_rollout,
)
from su_modfjsp_ha_pc.offline_logs import write_csv
from su_modfjsp_ha_pc.path_conflict_neural_dataset import (
    _explicit_preference_id,
    _explicit_teacher_preferences,
    _preference_proxy_score,
    _preference_transport_candidates,
    decode_preference_conditioned_teacher_schedule,
)
from su_modfjsp_ha_pc.rules import _available_operations, _first_resource_event_feasible_start
from su_modfjsp_ha_pc.schedule import ScheduleResult, ScheduledOperation, TransportMove
from su_modfjsp_ha_pc.state_action_attention_value_policy import (
    _lookahead_row,
    _ordered_trace,
    _state_token_rows_for_candidates,
    _trace_rows_for_decision,
    build_state_action_attention_dataset,
    decode_state_action_attention_rollout,
    score_state_action_attention_rows,
    train_state_action_attention_artifact,
)
from su_modfjsp_ha_pc.joint_operation_machine_value_policy import (
    _joint_group_rows,
    _resolve_joint_candidate,
)
from su_modfjsp_ha_pc.structural import realize_structural_events


def build_hierarchical_transport_attention_policy_rows(
    *,
    levels: tuple[str, ...],
    modes: tuple[str, ...],
    size: str,
    train_start_seed: int,
    train_end_seed: int,
    eval_start_seed: int,
    eval_end_seed: int,
    max_preferences: int,
    epochs: int,
    hidden_size: int,
    attention_heads: int,
    layers: int,
    random_state: int,
    max_train_groups_per_context: int = 0,
    completion_delta_cap: float = 15.0,
) -> dict[str, tuple[dict[str, object], ...]]:
    dataset = build_state_action_attention_dataset(
        levels=levels,
        modes=modes,
        size=size,
        start_seed=train_start_seed,
        end_seed=train_end_seed,
        max_preferences=max_preferences,
        max_groups_per_context=max_train_groups_per_context,
    )
    artifact = train_state_action_attention_artifact(
        dataset.rows,
        dataset.token_rows,
        epochs=epochs,
        hidden_size=hidden_size,
        attention_heads=attention_heads,
        layers=layers,
        random_state=random_state,
    )

    detail_rows = []
    trace_rows = []
    for level in levels:
        for seed in range(eval_start_seed, eval_end_seed + 1):
            raw_instance = generate_instance(level, size, seed)
            for mode in modes:
                instance = realize_structural_events(raw_instance).instance if mode == "realized" else raw_instance
                preferences = _explicit_teacher_preferences(len(instance.objectives), max_preferences=max_preferences)
                for preference_index, preference in enumerate(preferences):
                    preference_id = _explicit_preference_id(preference_index, preference)
                    teacher = decode_preference_conditioned_teacher_schedule(
                        instance,
                        preference,
                        algorithm="e119-teacher-rollout",
                    )
                    teacher_ops = tuple(operation.operation_id for operation in teacher.operations)
                    schedules = {
                        "teacher": teacher,
                        "state-action-attention-safe": decode_state_action_attention_rollout(
                            instance=instance,
                            preference=preference,
                            artifact=artifact,
                            level=level,
                            seed=seed,
                            mode=mode,
                            preference_id=preference_id,
                            method="state-action-attention-safe",
                            completion_delta_cap=completion_delta_cap,
                        ),
                        "hier-safe-pref-transport": decode_hierarchical_safe_transport_rollout(
                            instance=instance,
                            preference=preference,
                            artifact=artifact,
                            level=level,
                            seed=seed,
                            mode=mode,
                            preference_id=preference_id,
                            completion_delta_cap=completion_delta_cap,
                            trace_rows=trace_rows,
                        ),
                        "lookahead-rule": decode_rule_operation_rollout(
                            instance=instance,
                            preference=preference,
                            rule="lookahead",
                            random_seed=random_state + seed + preference_index,
                        ),
                        "random-rule": decode_rule_operation_rollout(
                            instance=instance,
                            preference=preference,
                            rule="random",
                            random_seed=random_state + seed * 1000 + preference_index,
                        ),
                    }
                    for method, schedule in schedules.items():
                        detail_rows.append(
                            _detail_row(
                                instance=instance,
                                schedule=schedule,
                                teacher_operations=teacher_ops,
                                level=level,
                                seed=seed,
                                mode=mode,
                                preference_id=preference_id,
                                preference=preference,
                                method=method,
                            )
                        )
    pairwise = tuple(_pairwise_rows(tuple(detail_rows), candidate_method="state-action-attention-safe")) + tuple(
        _pairwise_rows(tuple(detail_rows), candidate_method="hier-safe-pref-transport")
    )
    return {
        "train": (
            {
                "method": "hier-safe-pref-transport",
                "train_groups": artifact.train_groups,
                "train_rows": artifact.train_rows,
                "token_rows": artifact.token_rows,
                "label_rows": artifact.label_rows,
                "epochs": epochs,
                "hidden_size": hidden_size,
                "attention_heads": attention_heads,
                "layers": layers,
                "final_loss": round(artifact.final_loss, 10),
            },
        ),
        "detail": tuple(detail_rows),
        "summary": _summary_rows(tuple(detail_rows)),
        "pairwise": pairwise,
        "trace": tuple(_ordered_trace(row) for row in trace_rows),
    }


def decode_hierarchical_safe_transport_rollout(
    *,
    instance: object,
    preference: tuple[float, ...],
    artifact: object,
    level: str,
    seed: int,
    mode: str,
    preference_id: str,
    completion_delta_cap: float = 15.0,
    trace_rows: list[dict[str, object]] | None = None,
) -> ScheduleResult:
    state = _RolloutState(
        machine_available={machine.machine_id: 0 for machine in instance.machines},
        job_ready={job.job_id: job.release_time for job in instance.jobs},
        next_op_index={job.job_id: 0 for job in instance.jobs},
        agv_available={agv.agv_id: 0 for agv in instance.agvs},
        agv_node={agv.agv_id: agv.start_node_id for agv in instance.agvs},
        edge_reservations={edge.edge_id: [] for edge in instance.path_edges},
        total_operations=sum(len(job.operations) for job in instance.jobs),
    )
    scheduled_operations: list[ScheduledOperation] = []
    transport_moves: list[TransportMove] = []
    step_index = 0
    while len(scheduled_operations) < state.total_operations:
        available = _available_operations(instance.jobs, state.next_op_index)
        if not available:
            raise ValueError("no schedulable operations remain")
        group_id = f"e119-rollout-{level}-{seed:03d}-{mode}-{preference_id}-{step_index:03d}-joint"
        rows = _joint_group_rows(
            instance=instance,
            available=available,
            preference=preference,
            preference_id=preference_id,
            level=level,
            seed=seed,
            mode=mode,
            group_id=group_id,
            candidate_uid_start=step_index * 10000,
            step_index=step_index,
            selected_operation_id="",
            selected_machine_id="",
            total_operations=state.total_operations,
            machine_available=state.machine_available,
            job_ready=state.job_ready,
            agv_available=state.agv_available,
            agv_node=state.agv_node,
            edge_reservations=state.edge_reservations,
        )
        tokens = _state_token_rows_for_candidates(
            instance=instance,
            candidate_rows=rows,
            step_index=step_index,
            total_operations=state.total_operations,
            machine_available=state.machine_available,
            job_ready=state.job_ready,
            next_op_index=state.next_op_index,
            agv_available=state.agv_available,
            agv_node=state.agv_node,
            edge_reservations=state.edge_reservations,
        )
        scored = score_state_action_attention_rows(rows, tokens, artifact=artifact, preference=preference)
        lookahead = _lookahead_row(scored)
        lookahead_completion = _candidate_completion(lookahead)
        eligible = tuple(
            row
            for row in scored
            if _candidate_completion(row) <= lookahead_completion + completion_delta_cap
        ) or scored
        chosen = min(
            eligible,
            key=lambda row: (
                float(row.get("predicted_weighted_value", 0.0)),
                _candidate_completion(row),
                str(row.get("candidate_operation_id", "")),
                str(row.get("candidate_machine_id", "")),
            ),
        )
        if trace_rows is not None:
            trace_rows.extend(
                _trace_rows_for_decision(
                    scored,
                    method="hier-safe-pref-transport",
                    level=level,
                    seed=seed,
                    mode=mode,
                    preference_id=preference_id,
                    step_index=step_index,
                    chosen=chosen,
                    lookahead=lookahead,
                    completion_delta_cap=completion_delta_cap,
                )
            )
        job, operation, machine = _resolve_joint_candidate(instance, available, chosen)
        _append_fixed_joint_preference_transport_action(
            instance=instance,
            preference=preference,
            job=job,
            operation=operation,
            machine=machine,
            machine_available=state.machine_available,
            job_ready=state.job_ready,
            next_op_index=state.next_op_index,
            agv_available=state.agv_available,
            agv_node=state.agv_node,
            edge_reservations=state.edge_reservations,
            scheduled_operations=scheduled_operations,
            transport_moves=transport_moves,
        )
        step_index += 1
    return ScheduleResult(
        algorithm="e119-hier-safe-pref-transport",
        path_aware=bool(instance.path_edges),
        operations=tuple(scheduled_operations),
        transports=tuple(transport_moves),
        event_aware=bool(instance.events),
    )


def write_hierarchical_transport_attention_policy_outputs(
    out_dir: Path,
    rows: dict[str, tuple[dict[str, object], ...]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "e119_hier_transport_train.csv", rows["train"])
    write_csv(out_dir / "e119_hier_transport_detail.csv", rows["detail"])
    write_csv(out_dir / "e119_hier_transport_summary.csv", rows["summary"])
    write_csv(out_dir / "e119_hier_transport_pairwise.csv", rows["pairwise"])
    write_csv(out_dir / "e119_hier_transport_trace.csv", rows["trace"])


def _append_fixed_joint_preference_transport_action(
    *,
    instance: object,
    preference: tuple[float, ...],
    job: object,
    operation: object,
    machine: object,
    machine_available: dict[str, int],
    job_ready: dict[str, int],
    next_op_index: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
    scheduled_operations: list[ScheduledOperation],
    transport_moves: list[TransportMove],
) -> None:
    selected = _best_fixed_machine_preference_transport(
        instance=instance,
        preference=preference,
        job=job,
        operation=operation,
        machine=machine,
        machine_available=machine_available,
        job_ready=job_ready,
        agv_available=agv_available,
        agv_node=agv_node,
        edge_reservations=edge_reservations,
    )
    transport_end = job_ready[job.job_id]
    move = selected["transport"]
    if move is not None:
        transport = TransportMove(
            move_id=f"T{len(transport_moves)}",
            job_id=job.job_id,
            operation_id=operation.operation_id,
            agv_id=move["agv_id"],
            from_node=move["from_node"],
            to_node=machine.node_id,
            edge_ids=move["route"],
            start=move["start"],
            end=move["end"],
            transport_energy=move["transport_energy"],
        )
        transport_moves.append(transport)
        transport_end = transport.end
        for _legs, _ls, _le in move_leg_usage(instance, transport, job_pickup_of(instance, agv_node, transport.job_id)):
            for edge_id in dict.fromkeys(_legs):
                edge_reservations.setdefault(edge_id, []).append((_ls, _le))
        agv_available[transport.agv_id] = transport.end
        agv_node[transport.agv_id] = transport.to_node
        agv_node["__job__:" + str(transport.job_id)] = transport.to_node

    start = max(job_ready[job.job_id], machine_available[machine.machine_id], transport_end)
    start = _first_resource_event_feasible_start(
        start,
        operation.processing_time[machine.machine_id],
        machine.machine_id,
        "machine_breakdown",
        instance,
    )
    end = start + operation.processing_time[machine.machine_id]
    scheduled_operations.append(
        ScheduledOperation(
            operation_id=operation.operation_id,
            job_id=operation.job_id,
            op_index=operation.op_index,
            machine_id=machine.machine_id,
            start=start,
            end=end,
            processing_energy=operation.processing_energy[machine.machine_id],
        )
    )
    machine_available[machine.machine_id] = end
    job_ready[job.job_id] = end
    next_op_index[job.job_id] += 1


def _best_fixed_machine_preference_transport(
    *,
    instance: object,
    preference: tuple[float, ...],
    job: object,
    operation: object,
    machine: object,
    machine_available: dict[str, int],
    job_ready: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
) -> dict[str, object]:
    transports = _preference_transport_candidates(
        instance=instance,
        job=job,
        machine=machine,
        agv_available=agv_available,
        agv_node=agv_node,
        edge_reservations=edge_reservations,
        earliest_job_ready=job_ready[job.job_id],
    )
    candidates = []
    for transport in transports:
        transport_end = transport["end"] if transport is not None else job_ready[job.job_id]
        start = max(job_ready[job.job_id], machine_available[machine.machine_id], int(transport_end))
        start = _first_resource_event_feasible_start(
            start,
            operation.processing_time[machine.machine_id],
            machine.machine_id,
            "machine_breakdown",
            instance,
        )
        end = start + operation.processing_time[machine.machine_id]
        route = transport["route"] if transport is not None else ()
        score = _preference_proxy_score(
            instance=instance,
            preference=preference,
            job=job,
            operation=operation,
            machine=machine,
            route=route,
            operation_start=start,
            operation_end=end,
            transport=transport,
        )
        candidates.append(
            {
                "score": score,
                "transport": transport,
                "start": start,
                "end": end,
            }
        )
    if not candidates:
        raise ValueError("no fixed-machine transport candidates")
    return min(
        candidates,
        key=lambda item: (
            item["score"],
            "" if item["transport"] is None else item["transport"]["agv_id"],
            () if item["transport"] is None else tuple(item["transport"]["route"]),  # v3 section 3: same tie-break as _best_preference_action
        ),
    )
