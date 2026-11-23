from __future__ import annotations

from su_modfjsp_ha_pc.rules import job_pickup_of, move_leg_usage

from pathlib import Path

from su_modfjsp_ha_pc.generator import generate_instance
from su_modfjsp_ha_pc.metrics import evaluate_schedule
from su_modfjsp_ha_pc.neural_operation_rollout import (
    _RolloutState,
    _candidate_completion,
    _detail_row,
    _pairwise_rows,
    _summary_rows,
    decode_rule_operation_rollout,
)
from su_modfjsp_ha_pc.offline_logs import write_csv
from su_modfjsp_ha_pc.oracle_value_policy import (
    DATASET_FIELDS,
    OBJECTIVE_NAMES,
    _append_scheduled_teacher_step,
    _future_regret_field,
    _future_value_field,
    _ordered_dataset,
    _selected_label_steps,
    score_oracle_value_rows,
    train_oracle_value_artifact,
)
from su_modfjsp_ha_pc.path_conflict_neural_dataset import (
    _attach_group_critical_path_regrets,
    _best_preference_action,
    _best_transport_projection,
    _candidate_row,
    _explicit_preference_id,
    _explicit_teacher_preferences,
    decode_preference_conditioned_teacher_schedule,
    preference_teacher_surrogate_metrics,
)
from su_modfjsp_ha_pc.rules import _available_operations, _first_resource_event_feasible_start
from su_modfjsp_ha_pc.schedule import ScheduleResult, ScheduledOperation, TransportMove
from su_modfjsp_ha_pc.structural import realize_structural_events


TRACE_FIELDS = (
    "level",
    "seed",
    "mode",
    "target_preference_id",
    "step_index",
    "available_count",
    "candidate_operation_id",
    "candidate_machine_id",
    "selected_by_joint_model",
    "selected_by_lookahead",
    "predicted_weighted_value",
    "predicted_makespan",
    "predicted_tardiness",
    "predicted_energy",
    "predicted_path_conflict_delay",
    "predicted_stability",
    "lookahead_completion",
)


def build_joint_operation_machine_value_policy_rows(
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
) -> dict[str, tuple[dict[str, object], ...]]:
    dataset_rows = build_joint_operation_machine_dataset_rows(
        levels=levels,
        modes=modes,
        size=size,
        start_seed=train_start_seed,
        end_seed=train_end_seed,
        max_preferences=max_preferences,
        max_groups_per_context=max_train_groups_per_context,
    )
    artifact = train_oracle_value_artifact(
        dataset_rows,
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
                        algorithm="e117-teacher-rollout",
                    )
                    teacher_ops = tuple(operation.operation_id for operation in teacher.operations)
                    schedules = {
                        "teacher": teacher,
                        "joint-op-machine-value-neural": decode_joint_operation_machine_value_rollout(
                            instance=instance,
                            preference=preference,
                            artifact=artifact,
                            level=level,
                            seed=seed,
                            mode=mode,
                            preference_id=preference_id,
                            trace_rows=trace_rows,
                        ),
                        "lookahead-rule": decode_rule_operation_rollout(
                            instance=instance,
                            preference=preference,
                            rule="lookahead",
                            random_seed=random_state + seed + preference_index,
                        ),
                        "spt-rule": decode_rule_operation_rollout(
                            instance=instance,
                            preference=preference,
                            rule="spt",
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

    return {
        "dataset": tuple(_ordered_dataset(row) for row in dataset_rows),
        "train": (
            {
                "method": "joint-op-machine-value-neural",
                "train_groups": artifact.train_groups,
                "train_rows": artifact.train_rows,
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
        "pairwise": _pairwise_rows(tuple(detail_rows), candidate_method="joint-op-machine-value-neural"),
        "trace": tuple(_ordered_trace(row) for row in trace_rows),
    }


def build_joint_operation_machine_dataset_rows(
    *,
    levels: tuple[str, ...],
    modes: tuple[str, ...],
    size: str,
    start_seed: int,
    end_seed: int,
    max_preferences: int,
    max_groups_per_context: int = 0,
) -> tuple[dict[str, object], ...]:
    rows = []
    candidate_uid = 0
    for level in levels:
        for seed in range(start_seed, end_seed + 1):
            raw_instance = generate_instance(level, size, seed)
            for mode in modes:
                instance = realize_structural_events(raw_instance).instance if mode == "realized" else raw_instance
                preferences = _explicit_teacher_preferences(len(instance.objectives), max_preferences=max_preferences)
                for preference_index, preference in enumerate(preferences):
                    preference_id = _explicit_preference_id(preference_index, preference)
                    schedule = decode_preference_conditioned_teacher_schedule(
                        instance,
                        preference,
                        algorithm="e117-joint-label-teacher",
                    )
                    context_rows = _labeled_joint_rows_for_schedule(
                        instance=instance,
                        schedule=schedule,
                        preference=preference,
                        preference_id=preference_id,
                        level=level,
                        seed=seed,
                        mode=mode,
                        candidate_uid_start=candidate_uid,
                        max_groups_per_context=max_groups_per_context,
                    )
                    rows.extend(context_rows)
                    candidate_uid += len(context_rows)
    return tuple(_ordered_dataset(row) for row in rows)


def decode_joint_operation_machine_value_rollout(
    *,
    instance: object,
    preference: tuple[float, ...],
    artifact: object,
    level: str,
    seed: int,
    mode: str,
    preference_id: str,
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
        rows = _joint_group_rows(
            instance=instance,
            available=available,
            preference=preference,
            preference_id=preference_id,
            level=level,
            seed=seed,
            mode=mode,
            group_id=f"e117-rollout-{level}-{seed:03d}-{mode}-{preference_id}-{step_index:03d}-joint",
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
        scored = score_oracle_value_rows(rows, artifact=artifact, preference=preference)
        chosen = min(
            scored,
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
                    level=level,
                    seed=seed,
                    mode=mode,
                    preference_id=preference_id,
                    step_index=step_index,
                    chosen_operation_id=str(chosen.get("candidate_operation_id", "")),
                    chosen_machine_id=str(chosen.get("candidate_machine_id", "")),
                )
            )
        job, operation, machine = _resolve_joint_candidate(instance, available, chosen)
        _append_fixed_joint_action(
            instance=instance,
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
        algorithm="e117-joint-op-machine-value-neural-rollout",
        path_aware=bool(instance.path_edges),
        operations=tuple(scheduled_operations),
        transports=tuple(transport_moves),
        event_aware=bool(instance.events),
    )


def write_joint_operation_machine_value_policy_outputs(
    out_dir: Path,
    rows: dict[str, tuple[dict[str, object], ...]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "e117_joint_op_machine_dataset.csv", rows["dataset"])
    write_csv(out_dir / "e117_joint_op_machine_train.csv", rows["train"])
    write_csv(out_dir / "e117_joint_op_machine_detail.csv", rows["detail"])
    write_csv(out_dir / "e117_joint_op_machine_summary.csv", rows["summary"])
    write_csv(out_dir / "e117_joint_op_machine_pairwise.csv", rows["pairwise"])
    write_csv(out_dir / "e117_joint_op_machine_trace.csv", rows["trace"])


def _labeled_joint_rows_for_schedule(
    *,
    instance: object,
    schedule: object,
    preference: tuple[float, ...],
    preference_id: str,
    level: str,
    seed: int,
    mode: str,
    candidate_uid_start: int,
    max_groups_per_context: int,
) -> tuple[dict[str, object], ...]:
    operation_by_id = {operation.operation_id: operation for job in instance.jobs for operation in job.operations}
    job_by_id = {job.job_id: job for job in instance.jobs}
    machine_available = {machine.machine_id: 0 for machine in instance.machines}
    job_ready = {job.job_id: job.release_time for job in instance.jobs}
    next_op_index = {job.job_id: 0 for job in instance.jobs}
    agv_available = {agv.agv_id: 0 for agv in instance.agvs}
    agv_node = {agv.agv_id: agv.start_node_id for agv in instance.agvs}
    edge_reservations: dict[str, list[tuple[int, int]]] = {edge.edge_id: [] for edge in instance.path_edges}
    total_operations = max(1, len(schedule.operations))
    selected_steps = _selected_label_steps(total_operations, max_groups_per_context)
    rows = []
    transport_by_operation = {move.operation_id: move for move in schedule.transports}

    for step_index, scheduled in enumerate(schedule.operations):
        operation = operation_by_id[scheduled.operation_id]
        job = job_by_id[operation.job_id]
        available = _available_operations(instance.jobs, next_op_index)
        available_ids = {item[1].operation_id for item in available}
        if operation.operation_id not in available_ids:
            available = available + ((job, operation),)
        group_id = f"e117-{level}-{seed:03d}-{mode}-{preference_id}-{step_index:03d}-joint"
        group_rows = _joint_group_rows(
            instance=instance,
            available=available,
            preference=preference,
            preference_id=preference_id,
            level=level,
            seed=seed,
            mode=mode,
            group_id=group_id,
            candidate_uid_start=candidate_uid_start + len(rows),
            step_index=step_index,
            selected_operation_id=operation.operation_id,
            selected_machine_id=scheduled.machine_id,
            total_operations=total_operations,
            machine_available=machine_available,
            job_ready=job_ready,
            agv_available=agv_available,
            agv_node=agv_node,
            edge_reservations=edge_reservations,
        )
        if step_index in selected_steps:
            prefix_operation_ids = {item.operation_id for item in schedule.operations[:step_index]}
            group_rows = _attach_future_joint_value_labels(
                instance=instance,
                rows=group_rows,
                available=available,
                preference=preference,
                machine_available=machine_available,
                job_ready=job_ready,
                next_op_index=next_op_index,
                agv_available=agv_available,
                agv_node=agv_node,
                edge_reservations=edge_reservations,
                scheduled_prefix=tuple(schedule.operations[:step_index]),
                transport_prefix=tuple(move for move in schedule.transports if move.operation_id in prefix_operation_ids),
            )
        rows.extend(group_rows)
        _append_scheduled_teacher_step(
            instance=instance,
            scheduled=scheduled,
            transport=transport_by_operation.get(operation.operation_id),
            machine_available=machine_available,
            job_ready=job_ready,
            next_op_index=next_op_index,
            agv_available=agv_available,
            agv_node=agv_node,
            edge_reservations=edge_reservations,
        )
    return tuple(_ordered_dataset(row) for row in rows)


def _joint_group_rows(
    *,
    instance: object,
    available: tuple[tuple[object, object], ...],
    preference: tuple[float, ...],
    preference_id: str,
    level: str,
    seed: int,
    mode: str,
    group_id: str,
    candidate_uid_start: int,
    step_index: int,
    selected_operation_id: str,
    selected_machine_id: str,
    total_operations: int,
    machine_available: dict[str, int],
    job_ready: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
) -> tuple[dict[str, object], ...]:
    machine_by_id = {machine.machine_id: machine for machine in instance.machines}
    rows = []
    candidate_index = 0
    group_size = sum(len(operation.eligible_machines) for _job, operation in available)
    for job, operation in available:
        for machine_id in operation.eligible_machines:
            machine = machine_by_id[machine_id]
            projection = _best_transport_projection(
                instance=instance,
                job=job,
                machine=machine,
                agv_available=agv_available,
                agv_node=agv_node,
                edge_reservations=edge_reservations,
                earliest_job_ready=job_ready[job.job_id],
            )
            rows.append(
                _candidate_row(
                    instance=instance,
                    candidate_uid=candidate_uid_start + candidate_index,
                    split="train",
                    level=level,
                    seed=seed,
                    mode=mode,
                    algorithm="e117-joint-op-machine-value-policy",
                    teacher_family="e117-preference-teacher",
                    source_solution_id=preference_id,
                    teacher_schedule_id=f"e117-{level}-{seed:03d}-{mode}-{preference_id}",
                    target_preference_id=preference_id,
                    decision_group_id=group_id,
                    stage="operation",
                    target_preference=preference,
                    step_index=step_index,
                    group_size=group_size,
                    selected_action=(
                        bool(selected_operation_id)
                        and operation.operation_id == selected_operation_id
                        and machine.machine_id == selected_machine_id
                    ),
                    available_operation_count=len(available),
                    total_operations=total_operations,
                    job=job,
                    operation=operation,
                    job_ready_time=job_ready[job.job_id],
                    machine=machine,
                    machine_available_time=machine_available[machine.machine_id],
                    agv=projection.get("agv"),
                    agv_available_time=projection.get("agv_available_time", 0),
                    route=projection.get("route", ()),
                    route_start=projection.get("route_start", job_ready[job.job_id]),
                    route_end=projection.get("route_end", job_ready[job.job_id]),
                    route_wait_time=projection.get("route_wait_time", 0.0),
                    candidate_has_transport=bool(projection.get("has_transport", False)),
                    route_legs=projection.get("route_legs"),
                )
            )
            candidate_index += 1
    return tuple(_zero_joint_extra_labels(row) for row in _attach_group_critical_path_regrets(tuple(rows)))


def _attach_future_joint_value_labels(
    *,
    instance: object,
    rows: tuple[dict[str, object], ...],
    available: tuple[tuple[object, object], ...],
    preference: tuple[float, ...],
    machine_available: dict[str, int],
    job_ready: dict[str, int],
    next_op_index: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
    scheduled_prefix: tuple[object, ...],
    transport_prefix: tuple[object, ...],
) -> tuple[dict[str, object], ...]:
    operation_by_id = {operation.operation_id: (job, operation) for job, operation in available}
    machine_by_id = {machine.machine_id: machine for machine in instance.machines}
    outcomes: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        operation_id = str(row.get("candidate_operation_id", ""))
        machine_id = str(row.get("candidate_machine_id", ""))
        job, operation = operation_by_id[operation_id]
        schedule = _complete_schedule_after_forced_joint_action(
            instance=instance,
            preference=preference,
            forced_job=job,
            forced_operation=operation,
            forced_machine=machine_by_id[machine_id],
            machine_available=machine_available,
            job_ready=job_ready,
            next_op_index=next_op_index,
            agv_available=agv_available,
            agv_node=agv_node,
            edge_reservations=edge_reservations,
            scheduled_prefix=scheduled_prefix,
            transport_prefix=transport_prefix,
        )
        metrics = evaluate_schedule(instance, schedule)
        surrogates = preference_teacher_surrogate_metrics(instance, schedule)
        outcomes[(operation_id, machine_id)] = _objective_values(instance, metrics, surrogates)
    best_by_objective = {
        objective: min(float(values.get(objective, 0.0)) for values in outcomes.values())
        for objective in OBJECTIVE_NAMES
    }
    patched = []
    for row in rows:
        key = (str(row.get("candidate_operation_id", "")), str(row.get("candidate_machine_id", "")))
        values = outcomes[key]
        copy = dict(row)
        copy["future_value_label_available"] = 1.0
        weighted_value = 0.0
        weighted_regret = 0.0
        for index, objective in enumerate(OBJECTIVE_NAMES):
            value = float(values.get(objective, 0.0))
            regret = max(0.0, value - best_by_objective[objective])
            copy[_future_value_field(objective)] = value
            copy[_future_regret_field(objective)] = regret
            if index < len(preference):
                weighted_value += float(preference[index]) * value
                weighted_regret += float(preference[index]) * regret
        copy["future_weighted_value_after_action"] = weighted_value
        copy["future_weighted_completion_regret"] = weighted_regret
        patched.append(copy)
    return tuple(_ordered_dataset(row) for row in patched)


def _complete_schedule_after_forced_joint_action(
    *,
    instance: object,
    preference: tuple[float, ...],
    forced_job: object,
    forced_operation: object,
    forced_machine: object,
    machine_available: dict[str, int],
    job_ready: dict[str, int],
    next_op_index: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
    scheduled_prefix: tuple[object, ...],
    transport_prefix: tuple[object, ...],
) -> ScheduleResult:
    local_machine_available = dict(machine_available)
    local_job_ready = dict(job_ready)
    local_next_op_index = dict(next_op_index)
    local_agv_available = dict(agv_available)
    local_agv_node = dict(agv_node)
    local_edge_reservations = {edge_id: list(items) for edge_id, items in edge_reservations.items()}
    scheduled_operations = list(scheduled_prefix)
    transport_moves = list(transport_prefix)
    total_operations = sum(len(job.operations) for job in instance.jobs)
    _append_fixed_joint_action(
        instance=instance,
        job=forced_job,
        operation=forced_operation,
        machine=forced_machine,
        machine_available=local_machine_available,
        job_ready=local_job_ready,
        next_op_index=local_next_op_index,
        agv_available=local_agv_available,
        agv_node=local_agv_node,
        edge_reservations=local_edge_reservations,
        scheduled_operations=scheduled_operations,
        transport_moves=transport_moves,
    )
    while len(scheduled_operations) < total_operations:
        available = _available_operations(instance.jobs, local_next_op_index)
        if not available:
            raise ValueError("no schedulable operations remain during joint-value completion")
        selected = _best_preference_action(
            instance=instance,
            available=available,
            preference=preference,
            machine_available=local_machine_available,
            job_ready=local_job_ready,
            agv_available=local_agv_available,
            agv_node=local_agv_node,
            edge_reservations=local_edge_reservations,
        )
        _append_selected_preference_action(
            instance=instance,
            selected=selected,
            machine_available=local_machine_available,
            job_ready=local_job_ready,
            next_op_index=local_next_op_index,
            agv_available=local_agv_available,
            agv_node=local_agv_node,
            edge_reservations=local_edge_reservations,
            scheduled_operations=scheduled_operations,
            transport_moves=transport_moves,
        )
    return ScheduleResult(
        algorithm="e117-joint-op-machine-counterfactual",
        path_aware=bool(instance.path_edges),
        operations=tuple(scheduled_operations),
        transports=tuple(transport_moves),
        event_aware=bool(instance.events),
    )


def _append_fixed_joint_action(
    *,
    instance: object,
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
    projection = _best_transport_projection(
        instance=instance,
        job=job,
        machine=machine,
        agv_available=agv_available,
        agv_node=agv_node,
        edge_reservations=edge_reservations,
        earliest_job_ready=job_ready[job.job_id],
    )
    transport_end = job_ready[job.job_id]
    if projection.get("has_transport", False):
        agv = projection["agv"]
        transport = TransportMove(
            move_id=f"T{len(transport_moves)}",
            job_id=job.job_id,
            operation_id=operation.operation_id,
            agv_id=agv.agv_id,
            from_node=agv_node[agv.agv_id],
            to_node=machine.node_id,
            edge_ids=tuple(projection.get("route", ())),
            start=int(projection.get("route_start", job_ready[job.job_id])),
            end=int(projection.get("route_end", job_ready[job.job_id])),
            transport_energy=float(projection.get("transport_energy", 0.0)),
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


def _append_selected_preference_action(
    *,
    instance: object,
    selected: dict[str, object],
    machine_available: dict[str, int],
    job_ready: dict[str, int],
    next_op_index: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
    scheduled_operations: list[ScheduledOperation],
    transport_moves: list[TransportMove],
) -> None:
    job = selected["job"]
    operation = selected["operation"]
    machine = selected["machine"]
    transport_end = job_ready[job.job_id]
    move = selected.get("transport")
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


def _resolve_joint_candidate(
    instance: object,
    available: tuple[tuple[object, object], ...],
    row: dict[str, object],
) -> tuple[object, object, object]:
    operation_id = str(row.get("candidate_operation_id", ""))
    machine_id = str(row.get("candidate_machine_id", ""))
    machine_by_id = {machine.machine_id: machine for machine in instance.machines}
    for job, operation in available:
        if operation.operation_id == operation_id:
            if machine_id not in operation.eligible_machines:
                raise ValueError(f"machine {machine_id} is not eligible for operation {operation_id}")
            return job, operation, machine_by_id[machine_id]
    raise ValueError(f"joint policy selected unavailable operation {operation_id}")


def _trace_rows_for_decision(
    scored_rows: tuple[dict[str, object], ...],
    *,
    level: str,
    seed: int,
    mode: str,
    preference_id: str,
    step_index: int,
    chosen_operation_id: str,
    chosen_machine_id: str,
) -> tuple[dict[str, object], ...]:
    lookahead = min(
        scored_rows,
        key=lambda row: (
            _candidate_completion(row),
            float(row.get("candidate_processing_time", 0.0)),
            float(row.get("candidate_processing_energy", 0.0)),
            str(row.get("candidate_operation_id", "")),
            str(row.get("candidate_machine_id", "")),
        ),
    )
    lookahead_key = (str(lookahead.get("candidate_operation_id", "")), str(lookahead.get("candidate_machine_id", "")))
    chosen_key = (chosen_operation_id, chosen_machine_id)
    return tuple(
        {
            "level": level,
            "seed": seed,
            "mode": mode,
            "target_preference_id": preference_id,
            "step_index": step_index,
            "available_count": len(scored_rows),
            "candidate_operation_id": row.get("candidate_operation_id", ""),
            "candidate_machine_id": row.get("candidate_machine_id", ""),
            "selected_by_joint_model": 1
            if (str(row.get("candidate_operation_id", "")), str(row.get("candidate_machine_id", ""))) == chosen_key
            else 0,
            "selected_by_lookahead": 1
            if (str(row.get("candidate_operation_id", "")), str(row.get("candidate_machine_id", ""))) == lookahead_key
            else 0,
            "predicted_weighted_value": round(float(row.get("predicted_weighted_value", 0.0)), 10),
            "predicted_makespan": round(float(row.get("predicted_makespan", 0.0)), 10),
            "predicted_tardiness": round(float(row.get("predicted_tardiness", 0.0)), 10),
            "predicted_energy": round(float(row.get("predicted_energy", 0.0)), 10),
            "predicted_path_conflict_delay": round(float(row.get("predicted_path_conflict_delay", 0.0)), 10),
            "predicted_stability": round(float(row.get("predicted_stability", 0.0)), 10),
            "lookahead_completion": round(_candidate_completion(row), 10),
        }
        for row in scored_rows
    )


def _objective_values(instance: object, metrics: dict[str, object], surrogates: dict[str, object]) -> dict[str, float]:
    values = {}
    for objective in OBJECTIVE_NAMES:
        if objective in metrics:
            values[objective] = float(metrics.get(objective, 0.0))
        else:
            values[objective] = float(surrogates.get(objective, 0.0))
    if "path_conflict_delay" not in values:
        values["path_conflict_delay"] = float(surrogates.get("path_conflict_delay", 0.0))
    if "stability" not in values:
        values["stability"] = float(surrogates.get("stability", 0.0))
    del instance
    return values


def _zero_joint_extra_labels(row: dict[str, object]) -> dict[str, object]:
    copy = dict(row)
    for field in DATASET_FIELDS:
        copy.setdefault(field, "")
    copy["future_value_label_available"] = float(copy.get("future_value_label_available") or 0.0)
    for objective in OBJECTIVE_NAMES:
        copy.setdefault(_future_value_field(objective), 0.0)
        copy.setdefault(_future_regret_field(objective), 0.0)
    copy.setdefault("future_weighted_value_after_action", 0.0)
    return copy


def _ordered_trace(row: dict[str, object]) -> dict[str, object]:
    return {field: row.get(field, "") for field in TRACE_FIELDS}

