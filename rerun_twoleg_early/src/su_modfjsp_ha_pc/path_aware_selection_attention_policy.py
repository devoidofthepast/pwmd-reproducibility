from __future__ import annotations

from pathlib import Path

from su_modfjsp_ha_pc.deterministic_dispatch_rules import (
    DETERMINISTIC_RULES,
    select_operation_by_rule,
)
from su_modfjsp_ha_pc.generator import generate_instance
from su_modfjsp_ha_pc.metrics import evaluate_schedule
from su_modfjsp_ha_pc.objective_contract import (
    OBJECTIVE_NAMES as CONTRACT_OBJECTIVE_NAMES,
    ObjectiveScales,
    evaluate_objective_contract,
    objective_breakdown,
)
from su_modfjsp_ha_pc.neural_operation_rollout import (
    _RolloutState,
    _candidate_completion,
    _detail_row,
    _objective_values as _rollout_objective_values,
    _pairwise_rows,
    _summary_rows,
    decode_rule_operation_rollout,
)
from su_modfjsp_ha_pc.offline_logs import write_csv
from su_modfjsp_ha_pc.oracle_value_policy import (
    OBJECTIVE_NAMES,
    _append_scheduled_teacher_step,
    _future_regret_field,
    _future_value_field,
    _ordered_dataset,
    _selected_label_steps,
)
from su_modfjsp_ha_pc.path_conflict_neural_dataset import (
    _attach_group_critical_path_regrets,
    _best_preference_action,
    _candidate_row,
    _explicit_preference_id,
    _explicit_teacher_preferences,
    _operation_stage_lookahead,
    decode_preference_conditioned_teacher_schedule,
    preference_teacher_surrogate_metrics,
)
from su_modfjsp_ha_pc.rules import _available_operations
from su_modfjsp_ha_pc.schedule import ScheduleResult, ScheduledOperation, TransportMove
from su_modfjsp_ha_pc.hierarchical_transport_attention_policy import (
    _append_fixed_joint_preference_transport_action,
    _best_fixed_machine_preference_transport,
)
from su_modfjsp_ha_pc.joint_operation_machine_value_policy import (
    _append_selected_preference_action,
    _objective_values,
    _resolve_joint_candidate,
)
from su_modfjsp_ha_pc.state_action_attention_value_policy import (
    StateActionAttentionDataset,
    _lookahead_row,
    _ordered_token,
    _ordered_trace,
    _state_token_rows_for_candidates,
    _trace_rows_for_decision,
    score_state_action_attention_rows,
    train_state_action_attention_artifact,
)
from su_modfjsp_ha_pc.structural import realize_structural_events


def build_path_aware_selection_attention_policy_rows(
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
    dataset = build_path_aware_selection_dataset(
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
                        algorithm="e120-teacher-rollout",
                    )
                    teacher_ops = tuple(operation.operation_id for operation in teacher.operations)
                    schedules = {
                        "teacher": teacher,
                        "path-aware-safe-attention": decode_path_aware_safe_attention_rollout(
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
    return {
        "dataset": tuple(_ordered_dataset(row) for row in dataset.rows),
        "tokens": tuple(_ordered_token(row) for row in dataset.token_rows),
        "train": (
            {
                "method": "path-aware-safe-attention",
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
        "pairwise": _pairwise_rows(tuple(detail_rows), candidate_method="path-aware-safe-attention"),
        "trace": tuple(_ordered_trace(row) for row in trace_rows),
    }


def build_path_aware_selection_dataset(
    *,
    levels: tuple[str, ...],
    modes: tuple[str, ...],
    size: str,
    start_seed: int,
    end_seed: int,
    max_preferences: int,
    max_groups_per_context: int = 0,
    trajectory_policy: str = "teacher",
    label_continuation_rule: str = "teacher",
    label_objective_value_mode: str = "legacy",
    preference_indexes: tuple[int, ...] | None = None,
    objective_scales: ObjectiveScales | None = None,
) -> StateActionAttentionDataset:
    rows = []
    token_rows = []
    candidate_uid = 0
    for level in levels:
        for seed in range(start_seed, end_seed + 1):
            raw_instance = generate_instance(level, size, seed)
            for mode in modes:
                instance = realize_structural_events(raw_instance).instance if mode == "realized" else raw_instance
                preferences = _explicit_teacher_preferences(len(instance.objectives), max_preferences=max_preferences)
                for preference_index, preference in enumerate(preferences):
                    if preference_indexes is not None and preference_index not in preference_indexes:
                        continue
                    preference_id = _explicit_preference_id(preference_index, preference)
                    if trajectory_policy == "teacher":
                        schedule = decode_preference_conditioned_teacher_schedule(
                            instance,
                            preference,
                            algorithm="e120-path-aware-label-teacher",
                        )
                    elif trajectory_policy == "lookahead":
                        schedule = decode_rule_operation_rollout(
                            instance=instance,
                            preference=preference,
                            rule="lookahead",
                            random_seed=seed + preference_index,
                        )
                    else:
                        raise ValueError("trajectory_policy must be teacher or lookahead")
                    context = _labeled_path_aware_rows_for_schedule(
                        instance=instance,
                        schedule=schedule,
                        preference=preference,
                        preference_id=preference_id,
                        level=level,
                        seed=seed,
                        mode=mode,
                        candidate_uid_start=candidate_uid,
                        max_groups_per_context=max_groups_per_context,
                        label_continuation_rule=label_continuation_rule,
                        label_objective_value_mode=label_objective_value_mode,
                        objective_scales=objective_scales,
                    )
                    rows.extend(context.rows)
                    token_rows.extend(context.token_rows)
                    candidate_uid += len(context.rows)
    return StateActionAttentionDataset(
        rows=tuple(_ordered_dataset(row) for row in rows),
        token_rows=tuple(_ordered_token(row) for row in token_rows),
    )


def decode_path_aware_safe_attention_rollout(
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
        group_id = f"e120-rollout-{level}-{seed:03d}-{mode}-{preference_id}-{step_index:03d}-path-aware"
        rows = _path_aware_joint_group_rows(
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
        eligible = tuple(row for row in scored if _candidate_completion(row) <= lookahead_completion + completion_delta_cap) or scored
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
                    method="path-aware-safe-attention",
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
        algorithm="e120-path-aware-safe-attention",
        path_aware=bool(instance.path_edges),
        operations=tuple(scheduled_operations),
        transports=tuple(transport_moves),
        event_aware=bool(instance.events),
    )


def write_path_aware_selection_attention_policy_outputs(
    out_dir: Path,
    rows: dict[str, tuple[dict[str, object], ...]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "e120_path_aware_selection_dataset.csv", rows["dataset"])
    write_csv(out_dir / "e120_path_aware_selection_tokens.csv", rows["tokens"])
    write_csv(out_dir / "e120_path_aware_selection_train.csv", rows["train"])
    write_csv(out_dir / "e120_path_aware_selection_detail.csv", rows["detail"])
    write_csv(out_dir / "e120_path_aware_selection_summary.csv", rows["summary"])
    write_csv(out_dir / "e120_path_aware_selection_pairwise.csv", rows["pairwise"])
    write_csv(out_dir / "e120_path_aware_selection_trace.csv", rows["trace"])


def _labeled_path_aware_rows_for_schedule(
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
    label_continuation_rule: str = "teacher",
    label_objective_value_mode: str = "legacy",
    group_namespace: str = "e120",
    objective_scales: ObjectiveScales | None = None,
) -> StateActionAttentionDataset:
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
    token_rows = []
    transport_by_operation = {move.operation_id: move for move in schedule.transports}

    for step_index, scheduled in enumerate(schedule.operations):
        operation = operation_by_id[scheduled.operation_id]
        job = job_by_id[operation.job_id]
        available = _available_operations(instance.jobs, next_op_index)
        available_ids = {item[1].operation_id for item in available}
        if operation.operation_id not in available_ids:
            available = available + ((job, operation),)
        group_id = f"{group_namespace}-{level}-{seed:03d}-{mode}-{preference_id}-{step_index:03d}-path-aware"
        group_rows = _path_aware_joint_group_rows(
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
            token_rows.extend(
                _state_token_rows_for_candidates(
                    instance=instance,
                    candidate_rows=group_rows,
                    step_index=step_index,
                    total_operations=total_operations,
                    machine_available=machine_available,
                    job_ready=job_ready,
                    next_op_index=next_op_index,
                    agv_available=agv_available,
                    agv_node=agv_node,
                    edge_reservations=edge_reservations,
                )
            )
            prefix_operation_ids = {item.operation_id for item in schedule.operations[:step_index]}
            group_rows = _attach_future_path_aware_value_labels(
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
                continuation_rule=label_continuation_rule,
                continuation_level=level,
                continuation_seed=seed,
                continuation_mode=mode,
                continuation_preference_id=preference_id,
                objective_value_mode=label_objective_value_mode,
                objective_scales=objective_scales,
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
    return StateActionAttentionDataset(
        rows=tuple(_ordered_dataset(row) for row in rows),
        token_rows=tuple(_ordered_token(row) for row in token_rows),
    )


def _path_aware_joint_group_rows(
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
    agv_by_id = {agv.agv_id: agv for agv in instance.agvs}
    rows = []
    candidate_index = 0
    group_size = sum(len(operation.eligible_machines) for _job, operation in available)
    for job, operation in available:
        for machine_id in operation.eligible_machines:
            machine = machine_by_id[machine_id]
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
            transport = selected["transport"]
            agv = agv_by_id[str(transport["agv_id"])] if transport is not None else None
            rows.append(
                _candidate_row(
                    instance=instance,
                    candidate_uid=candidate_uid_start + candidate_index,
                    split="train",
                    level=level,
                    seed=seed,
                    mode=mode,
                    algorithm="e120-path-aware-selection",
                    teacher_family="e120-preference-teacher",
                    source_solution_id=preference_id,
                    teacher_schedule_id=f"e120-{level}-{seed:03d}-{mode}-{preference_id}",
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
                    agv=agv,
                    agv_available_time=agv_available[agv.agv_id] if agv is not None else 0,
                    route=transport["route"] if transport is not None else (),
                    route_start=transport["start"] if transport is not None else job_ready[job.job_id],
                    route_end=transport["end"] if transport is not None else job_ready[job.job_id],
                    route_wait_time=transport["wait"] if transport is not None else 0.0,
                    candidate_has_transport=transport is not None,
                    route_legs=transport.get("route_legs") if transport is not None else None,
                )
            )
            candidate_index += 1
    return tuple(_zero_path_aware_extra_labels(row) for row in _attach_group_critical_path_regrets(tuple(rows)))


def _attach_future_path_aware_value_labels(
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
    continuation_rule: str = "teacher",
    continuation_level: str = "L4",
    continuation_seed: int = 0,
    continuation_mode: str = "raw",
    continuation_preference_id: str = "",
    objective_value_mode: str = "legacy",
    objective_scales: ObjectiveScales | None = None,
) -> tuple[dict[str, object], ...]:
    operation_by_id = {operation.operation_id: (job, operation) for job, operation in available}
    machine_by_id = {machine.machine_id: machine for machine in instance.machines}
    outcomes: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        operation_id = str(row.get("candidate_operation_id", ""))
        machine_id = str(row.get("candidate_machine_id", ""))
        job, operation = operation_by_id[operation_id]
        schedule = _complete_schedule_after_forced_path_aware_joint_action(
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
            continuation_rule=continuation_rule,
            continuation_level=continuation_level,
            continuation_seed=continuation_seed,
            continuation_mode=continuation_mode,
            continuation_preference_id=continuation_preference_id,
        )
        metrics = evaluate_schedule(instance, schedule)
        surrogates = preference_teacher_surrogate_metrics(instance, schedule)
        if objective_value_mode == "contract":
            if objective_scales is None:
                raise ValueError("objective_scales is required for contract objective values")
            evaluation = evaluate_objective_contract(instance, schedule)
            outcomes[(operation_id, machine_id)] = objective_breakdown(
                evaluation.raw,
                preference,
                objective_scales,
            ).normalized
        elif objective_value_mode == "rollout":
            outcomes[(operation_id, machine_id)] = _rollout_objective_values(instance, metrics, surrogates)
        elif objective_value_mode == "legacy":
            outcomes[(operation_id, machine_id)] = _objective_values(instance, metrics, surrogates)
        else:
            raise ValueError("objective_value_mode must be legacy, rollout, or contract")
    objective_names = CONTRACT_OBJECTIVE_NAMES if objective_value_mode == "contract" else OBJECTIVE_NAMES
    best_by_objective = {
        objective: min(float(values.get(objective, 0.0)) for values in outcomes.values())
        for objective in objective_names
    }
    patched = []
    for row in rows:
        key = (str(row.get("candidate_operation_id", "")), str(row.get("candidate_machine_id", "")))
        values = outcomes[key]
        copy = dict(row)
        copy["future_value_label_available"] = 1.0
        weighted_value = 0.0
        weighted_regret = 0.0
        for index, objective in enumerate(objective_names):
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
    if objective_value_mode == "contract":
        return tuple(patched)
    return tuple(_ordered_dataset(row) for row in patched)


def _complete_schedule_after_forced_path_aware_joint_action(
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
    continuation_rule: str = "teacher",
    continuation_level: str = "L4",
    continuation_seed: int = 0,
    continuation_mode: str = "raw",
    continuation_preference_id: str = "",
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
    _append_fixed_joint_preference_transport_action(
        instance=instance,
        preference=preference,
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
            raise ValueError("no schedulable operations remain during path-aware completion")
        if continuation_rule == "lookahead":
            job, operation = min(
                available,
                key=lambda item: (
                    _operation_stage_lookahead(
                        instance=instance,
                        job=item[0],
                        operation=item[1],
                        machine_available=local_machine_available,
                        job_ready_time=local_job_ready[item[0].job_id],
                        agv_available=local_agv_available,
                        agv_node=local_agv_node,
                        edge_reservations=local_edge_reservations,
                    )["score"],
                    item[1].operation_id,
                ),
            )
            selected = _best_preference_action(
                instance=instance,
                available=((job, operation),),
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
        elif continuation_rule in DETERMINISTIC_RULES:
            job, operation = select_operation_by_rule(
                rule=continuation_rule,
                instance=instance,
                available=available,
                machine_available=local_machine_available,
                job_ready=local_job_ready,
                agv_available=local_agv_available,
                agv_node=local_agv_node,
                edge_reservations=local_edge_reservations,
            )
            selected = _best_preference_action(
                instance=instance,
                available=((job, operation),),
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
        elif continuation_rule == "teacher":
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
        else:
            raise ValueError(f"unsupported continuation rule: {continuation_rule}")
    return ScheduleResult(
        algorithm="e120-path-aware-counterfactual",
        path_aware=bool(instance.path_edges),
        operations=tuple(scheduled_operations),
        transports=tuple(transport_moves),
        event_aware=bool(instance.events),
    )


def _zero_path_aware_extra_labels(row: dict[str, object]) -> dict[str, object]:
    copy = dict(row)
    copy.setdefault("future_value_label_available", 0.0)
    for objective in OBJECTIVE_NAMES:
        copy.setdefault(_future_value_field(objective), 0.0)
        copy.setdefault(_future_regret_field(objective), 0.0)
    copy.setdefault("future_weighted_value_after_action", 0.0)
    return copy
