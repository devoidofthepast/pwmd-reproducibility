from __future__ import annotations

from su_modfjsp_ha_pc.rules import job_pickup_of, move_leg_usage

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from su_modfjsp_ha_pc.generator import generate_instance
from su_modfjsp_ha_pc.metrics import evaluate_schedule
from su_modfjsp_ha_pc.objective_contract import (
    OBJECTIVE_NAMES as CANONICAL_OBJECTIVE_NAMES,
    ObjectiveScales,
    evaluate_objective_contract,
    objective_breakdown,
)
from su_modfjsp_ha_pc.offline_logs import write_csv
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
from su_modfjsp_ha_pc.path_conflict_operation_sequence_policy import (
    OperationGroup,
    _operation_sequence_features,
    _score_groups,
    _train_sequence_ranker,
    build_operation_groups,
    fit_standardizer,
    read_candidate_rows,
)
from su_modfjsp_ha_pc.rules import _available_operations, _first_resource_event_feasible_start
from su_modfjsp_ha_pc.schedule import ScheduleResult, ScheduledOperation, TransportMove
from su_modfjsp_ha_pc.structural import realize_structural_events


DETAIL_FIELDS = (
    "level",
    "seed",
    "mode",
    "target_preference_id",
    "preference",
    "method",
    "algorithm",
    "operation_match_rate_to_teacher",
    "operation_prefix_match_rate_to_teacher",
    "target_weighted_value",
    "dominant_objective",
    "dominant_objective_value",
    "makespan",
    "tardiness",
    "energy",
    "path_conflict_delay",
    "stability",
    "teacher_path_pressure",
    "teacher_stability_pressure",
    "teacher_projected_tardiness",
    "feasible",
    "precedence_violations",
    "machine_overlap_violations",
    "agv_overlap_violations",
    "edge_capacity_violations",
    "machine_unavailability_violations",
    "agv_unavailability_violations",
    "path_event_violations",
) + tuple(
    field
    for objective in CANONICAL_OBJECTIVE_NAMES
    for field in (
        f"raw_{objective}",
        f"normalized_{objective}",
        f"weighted_component_{objective}",
        f"contribution_rate_{objective}",
    )
) + (
    "route_pressure",
    "event_exposure_pressure",
    "objective_scale_source_sha256",
) + (
    "route_bottleneck_exposure",
    "route_length_exposure",
    "route_event_overlap_exposure",
    "route_blockage_exposure",
    "route_capacity_exposure",
)

SUMMARY_FIELDS = (
    "method",
    "group",
    "cases",
    "mean_operation_match_rate_to_teacher",
    "mean_target_weighted_value",
    "mean_dominant_objective_value",
    "mean_makespan",
    "mean_tardiness",
    "mean_energy",
    "mean_teacher_path_pressure",
    "mean_teacher_stability_pressure",
    "feasible_rate",
)

PAIRWISE_FIELDS = (
    "candidate_method",
    "baseline_method",
    "group",
    "cases",
    "wins",
    "equals",
    "losses",
    "mean_target_delta",
)

TRACE_FIELDS = (
    "level",
    "seed",
    "mode",
    "target_preference_id",
    "dominant_objective",
    "step_index",
    "available_count",
    "candidate_operation_id",
    "selected_by_neural",
    "selected_by_lookahead",
    "selected_by_teacher_step",
    "neural_rank",
    "lookahead_rank",
    "predicted_score",
    "predicted_future_regret",
    "regret_aware_score",
    "lookahead_completion",
    "candidate_processing_time",
    "candidate_processing_energy",
    "candidate_machine_start_lower_bound",
    "candidate_route_wait_time",
    "candidate_route_length",
    "candidate_route_bottleneck_count",
    "candidate_route_event_overlap_duration",
    "candidate_active_path_event_count",
    "candidate_active_machine_event_count",
    "selected_completion_delta_to_lookahead",
    "selected_energy_delta_to_lookahead",
    "selected_path_pressure_delta_to_lookahead",
)


GATE_TRACE_FIELDS = (
    "level",
    "seed",
    "mode",
    "target_preference_id",
    "dominant_objective",
    "step_index",
    "available_count",
    "chosen_operation_id",
    "neural_operation_id",
    "lookahead_operation_id",
    "teacher_operation_id",
    "allow_override",
    "gate_reason",
    "regret_cap",
    "completion_regret",
    "energy_gain",
    "path_gain",
    "weighted_gain",
    "same_as_lookahead",
    "same_as_teacher",
)

GATE_SUMMARY_FIELDS = (
    "dominant_objective",
    "decisions",
    "neural_override_rate",
    "different_action_rate",
    "effective_neural_override_rate",
    "blocked_override_rate",
    "mean_completion_regret_allowed",
    "mean_completion_regret_blocked",
    "mean_weighted_gain_allowed",
    "mean_weighted_gain_blocked",
    "same_as_lookahead_rate",
    "same_as_teacher_rate",
)


@dataclass(frozen=True)
class OperationRankerArtifact:
    model: object
    standardizer: object
    train_groups: int
    train_rows: int
    final_loss: float


@dataclass(frozen=True)
class RegretGateDecision:
    allow_override: bool
    reason: str
    regret_cap: float
    weighted_gain: float


def train_operation_ranker_artifact(
    dataset_path: Path,
    *,
    epochs: int,
    hidden_size: int,
    attention_heads: int,
    layers: int,
    random_state: int,
) -> OperationRankerArtifact:
    rows = read_candidate_rows(dataset_path)
    operation_train_rows = tuple(
        row for row in rows if row.get("stage") == "operation" and row.get("split") == "train"
    )
    if not operation_train_rows:
        raise ValueError("dataset must contain train operation rows")
    train_groups = build_operation_groups(operation_train_rows)
    standardizer = fit_standardizer(tuple(feature for group in train_groups for feature in group.feature_rows))
    model, final_loss = _train_sequence_ranker(
        train_groups,
        standardizer=standardizer,
        epochs=epochs,
        hidden_size=hidden_size,
        attention_heads=attention_heads,
        layers=layers,
        random_state=random_state,
    )
    return OperationRankerArtifact(
        model=model,
        standardizer=standardizer,
        train_groups=len(train_groups),
        train_rows=len(operation_train_rows),
        final_loss=final_loss,
    )


def build_neural_operation_rollout_rows(
    *,
    dataset_path: Path,
    levels: tuple[str, ...],
    modes: tuple[str, ...],
    size: str,
    start_seed: int,
    end_seed: int,
    max_preferences: int,
    epochs: int,
    hidden_size: int,
    attention_heads: int,
    layers: int,
    random_state: int,
) -> dict[str, tuple[dict[str, object], ...]]:
    artifact = train_operation_ranker_artifact(
        dataset_path,
        epochs=epochs,
        hidden_size=hidden_size,
        attention_heads=attention_heads,
        layers=layers,
        random_state=random_state,
    )
    detail_rows = []
    trace_rows = []
    gate_trace_rows = []
    for level in levels:
        for seed in range(start_seed, end_seed + 1):
            raw_instance = generate_instance(level, size, seed)
            for mode in modes:
                instance = realize_structural_events(raw_instance).instance if mode == "realized" else raw_instance
                preferences = _explicit_teacher_preferences(len(instance.objectives), max_preferences=max_preferences)
                for preference_index, preference in enumerate(preferences):
                    preference_id = _explicit_preference_id(preference_index, preference)
                    teacher = decode_preference_conditioned_teacher_schedule(
                        instance,
                        preference,
                        algorithm="e110-teacher-rollout",
                    )
                    schedules = {
                        "teacher": teacher,
                        "neural-operation": decode_neural_operation_rollout(
                            instance=instance,
                            preference=preference,
                            artifact=artifact,
                            level=level,
                            seed=seed,
                            mode=mode,
                            preference_id=preference_id,
                            teacher_operations=tuple(operation.operation_id for operation in teacher.operations),
                            trace_rows=trace_rows,
                        ),
                        "regret-controlled-neural": decode_regret_controlled_neural_rollout(
                            instance=instance,
                            preference=preference,
                            artifact=artifact,
                            level=level,
                            seed=seed,
                            mode=mode,
                            preference_id=preference_id,
                            teacher_operations=tuple(operation.operation_id for operation in teacher.operations),
                            gate_trace_rows=gate_trace_rows,
                        ),
                        "regret-aware-neural": decode_regret_aware_neural_rollout(
                            instance=instance,
                            preference=preference,
                            artifact=artifact,
                            level=level,
                            seed=seed,
                            mode=mode,
                            preference_id=preference_id,
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
                    teacher_ops = tuple(operation.operation_id for operation in teacher.operations)
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
        "detail": tuple(_ordered_detail(row) for row in detail_rows),
        "summary": _summary_rows(tuple(detail_rows)),
        "pairwise": (
            _pairwise_rows(tuple(detail_rows), candidate_method="neural-operation")
            + _pairwise_rows(tuple(detail_rows), candidate_method="regret-controlled-neural")
            + _pairwise_rows(tuple(detail_rows), candidate_method="regret-aware-neural")
        ),
        "trace": tuple(_ordered_trace(row) for row in trace_rows),
        "gate_trace": tuple(_ordered_gate_trace(row) for row in gate_trace_rows),
        "gate_summary": _gate_summary_rows(tuple(gate_trace_rows)),
        "train": (
            {
                "method": "neural-operation",
                "train_groups": artifact.train_groups,
                "train_rows": artifact.train_rows,
                "epochs": epochs,
                "hidden_size": hidden_size,
                "attention_heads": attention_heads,
                "layers": layers,
                "final_loss": round(artifact.final_loss, 10),
            },
        ),
    }


def decode_neural_operation_rollout(
    *,
    instance: object,
    preference: tuple[float, ...],
    artifact: OperationRankerArtifact,
    level: str,
    seed: int,
    mode: str,
    preference_id: str = "e110-rollout",
    teacher_operations: tuple[str, ...] = (),
    trace_rows: list[dict[str, object]] | None = None,
) -> ScheduleResult:
    def pick_operation(state: _RolloutState, available: tuple[tuple[object, object], ...], step_index: int):
        rows = _operation_candidate_rows_for_state(
            instance=instance,
            state=state,
            available=available,
            preference=preference,
            level=level,
            seed=seed,
            mode=mode,
            step_index=step_index,
            total_operations=state.total_operations,
        )
        scored = _score_operation_rows(rows, artifact)
        if trace_rows is not None:
            trace_rows.extend(
                _decision_trace_rows(
                    scored,
                    level=level,
                    seed=seed,
                    mode=mode,
                    preference_id=preference_id,
                    preference=preference,
                    objectives=tuple(instance.objectives),
                    step_index=step_index,
                    teacher_operation_id=teacher_operations[step_index] if step_index < len(teacher_operations) else "",
                )
            )
        best = max(
            scored,
            key=lambda row: (
                float(row["predicted_score"]),
                -float(row.get("candidate_machine_start_lower_bound", 0.0)),
                str(row.get("candidate_operation_id", "")),
            ),
        )
        operation_id = str(best["candidate_operation_id"])
        for job, operation in available:
            if operation.operation_id == operation_id:
                return job, operation
        raise ValueError(f"neural policy selected unavailable operation {operation_id}")

    return _decode_with_operation_picker(
        instance=instance,
        preference=preference,
        algorithm="e110-neural-operation-rollout",
        picker=pick_operation,
    )


def decode_regret_aware_neural_rollout(
    *,
    instance: object,
    preference: tuple[float, ...],
    artifact: OperationRankerArtifact,
    level: str,
    seed: int,
    mode: str,
    preference_id: str = "e113-rollout",
) -> ScheduleResult:
    del preference_id

    def pick_operation(state: _RolloutState, available: tuple[tuple[object, object], ...], step_index: int):
        rows = _operation_candidate_rows_for_state(
            instance=instance,
            state=state,
            available=available,
            preference=preference,
            level=level,
            seed=seed,
            mode=mode,
            step_index=step_index,
            total_operations=state.total_operations,
        )
        scored = _score_operation_rows(rows, artifact)
        best = _regret_aware_choice(scored)
        operation_id = str(best["candidate_operation_id"])
        for job, operation in available:
            if operation.operation_id == operation_id:
                return job, operation
        raise ValueError(f"regret-aware neural policy selected unavailable operation {operation_id}")

    return _decode_with_operation_picker(
        instance=instance,
        preference=preference,
        algorithm="e113-regret-aware-neural-rollout",
        picker=pick_operation,
    )


def decode_regret_controlled_neural_rollout(
    *,
    instance: object,
    preference: tuple[float, ...],
    artifact: OperationRankerArtifact,
    level: str,
    seed: int,
    mode: str,
    preference_id: str = "e111-rollout",
    teacher_operations: tuple[str, ...] = (),
    gate_trace_rows: list[dict[str, object]] | None = None,
) -> ScheduleResult:
    def pick_operation(state: _RolloutState, available: tuple[tuple[object, object], ...], step_index: int):
        rows = _operation_candidate_rows_for_state(
            instance=instance,
            state=state,
            available=available,
            preference=preference,
            level=level,
            seed=seed,
            mode=mode,
            step_index=step_index,
            total_operations=state.total_operations,
        )
        scored = _score_operation_rows(rows, artifact)
        neural_choice = _neural_choice(scored)
        lookahead_choice = _lookahead_choice(scored)
        dominant_objective = _gate_dominant_objective(tuple(instance.objectives), preference)
        completion_regret = _candidate_completion(neural_choice) - _candidate_completion(lookahead_choice)
        energy_gain = float(lookahead_choice.get("candidate_processing_energy", 0.0)) - float(
            neural_choice.get("candidate_processing_energy", 0.0)
        )
        path_gain = _candidate_path_pressure(lookahead_choice) - _candidate_path_pressure(neural_choice)
        decision = _regret_gate_decision(
            dominant_objective=dominant_objective,
            preference=preference,
            completion_regret=completion_regret,
            energy_gain=energy_gain,
            path_gain=path_gain,
            objectives=tuple(instance.objectives),
        )
        chosen = neural_choice if decision.allow_override else lookahead_choice
        teacher_operation_id = teacher_operations[step_index] if step_index < len(teacher_operations) else ""
        if gate_trace_rows is not None:
            gate_trace_rows.append(
                _gate_trace_row(
                    level=level,
                    seed=seed,
                    mode=mode,
                    preference_id=preference_id,
                    dominant_objective=dominant_objective,
                    step_index=step_index,
                    available_count=len(scored),
                    chosen=chosen,
                    neural_choice=neural_choice,
                    lookahead_choice=lookahead_choice,
                    teacher_operation_id=teacher_operation_id,
                    decision=decision,
                    completion_regret=completion_regret,
                    energy_gain=energy_gain,
                    path_gain=path_gain,
                )
            )
        operation_id = str(chosen["candidate_operation_id"])
        for job, operation in available:
            if operation.operation_id == operation_id:
                return job, operation
        raise ValueError(f"regret-controlled policy selected unavailable operation {operation_id}")

    return _decode_with_operation_picker(
        instance=instance,
        preference=preference,
        algorithm="e111-regret-controlled-neural-rollout",
        picker=pick_operation,
    )


def decode_rule_operation_rollout(
    *,
    instance: object,
    preference: tuple[float, ...],
    rule: str,
    random_seed: int,
) -> ScheduleResult:
    rng = random.Random(random_seed)

    def pick_operation(state: _RolloutState, available: tuple[tuple[object, object], ...], step_index: int):
        del step_index
        if rule == "random":
            return available[rng.randrange(len(available))]
        if rule == "spt":
            return min(
                available,
                key=lambda item: (
                    min(item[1].processing_time.values()),
                    item[1].operation_id,
                ),
            )
        if rule == "edd":
            return min(
                available,
                key=lambda item: (
                    float(getattr(item[0], "due_date", 0)),
                    item[1].operation_id,
                ),
            )
        if rule == "fifo":
            return min(
                available,
                key=lambda item: (
                    float(getattr(item[0], "release_time", 0)),
                    item[0].job_id,
                    item[1].operation_id,
                ),
            )
        if rule == "mwkr":
            def _remaining_work(item):
                job, operation = item
                ops = job.operations
                index = ops.index(operation)
                return sum(min(op.processing_time.values()) for op in ops[index:])
            return min(
                available,
                key=lambda item: (
                    -_remaining_work(item),
                    item[1].operation_id,
                ),
            )
        if rule == "lookahead":
            return min(
                available,
                key=lambda item: (
                    _operation_stage_lookahead(
                        instance=instance,
                        job=item[0],
                        operation=item[1],
                        machine_available=state.machine_available,
                        job_ready_time=state.job_ready[item[0].job_id],
                        agv_available=state.agv_available,
                        agv_node=state.agv_node,
                        edge_reservations=state.edge_reservations,
                    )["score"],
                    item[1].operation_id,
                ),
            )
        raise ValueError(f"unsupported rule {rule}")

    return _decode_with_operation_picker(
        instance=instance,
        preference=preference,
        algorithm=f"e110-{rule}-operation-rollout",
        picker=pick_operation,
    )


@dataclass
class _RolloutState:
    machine_available: dict[str, int]
    job_ready: dict[str, int]
    next_op_index: dict[str, int]
    agv_available: dict[str, int]
    agv_node: dict[str, str]
    edge_reservations: dict[str, list[tuple[int, int]]]
    total_operations: int


OperationPicker = Callable[[_RolloutState, tuple[tuple[object, object], ...], int], tuple[object, object]]


def _decode_with_operation_picker(
    *,
    instance: object,
    preference: tuple[float, ...],
    algorithm: str,
    picker: OperationPicker,
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
        job, operation = picker(state, available, step_index)
        selected = _best_preference_action(
            instance=instance,
            available=((job, operation),),
            preference=preference,
            machine_available=state.machine_available,
            job_ready=state.job_ready,
            agv_available=state.agv_available,
            agv_node=state.agv_node,
            edge_reservations=state.edge_reservations,
        )
        machine = selected["machine"]
        transport_end = state.job_ready[job.job_id]
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
            for _legs, _ls, _le in move_leg_usage(instance, transport, job_pickup_of(instance, state.agv_node, transport.job_id)):
                for edge_id in dict.fromkeys(_legs):
                    state.edge_reservations.setdefault(edge_id, []).append((_ls, _le))
            state.agv_available[transport.agv_id] = transport.end
            state.agv_node[transport.agv_id] = transport.to_node
            state.agv_node["__job__:" + str(transport.job_id)] = transport.to_node

        start = max(state.job_ready[job.job_id], state.machine_available[machine.machine_id], transport_end)
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
        state.machine_available[machine.machine_id] = end
        state.job_ready[job.job_id] = end
        state.next_op_index[job.job_id] += 1
        step_index += 1
    return ScheduleResult(
        algorithm=algorithm,
        path_aware=bool(instance.path_edges),
        operations=tuple(scheduled_operations),
        transports=tuple(transport_moves),
        event_aware=bool(instance.events),
    )


def _operation_candidate_rows_for_state(
    *,
    instance: object,
    state: _RolloutState,
    available: tuple[tuple[object, object], ...],
    preference: tuple[float, ...],
    level: str,
    seed: int,
    mode: str,
    step_index: int,
    total_operations: int,
) -> tuple[dict[str, object], ...]:
    rows = []
    group_id = f"e110-{level}-{seed:03d}-{mode}-{step_index:03d}-operation"
    for index, (job, operation) in enumerate(available):
        best = _operation_stage_lookahead(
            instance=instance,
            job=job,
            operation=operation,
            machine_available=state.machine_available,
            job_ready_time=state.job_ready[job.job_id],
            agv_available=state.agv_available,
            agv_node=state.agv_node,
            edge_reservations=state.edge_reservations,
        )
        rows.append(
            _candidate_row(
                instance=instance,
                candidate_uid=step_index * 1000 + index,
                split="rollout",
                level=level,
                seed=seed,
                mode=mode,
                algorithm="e110-neural-operation-rollout",
                teacher_family="e110-rollout",
                source_solution_id="e110-rollout",
                teacher_schedule_id="e110-rollout",
                target_preference_id="e110-rollout",
                decision_group_id=group_id,
                stage="operation",
                target_preference=preference,
                step_index=step_index,
                group_size=len(available),
                selected_action=index == 0,
                available_operation_count=len(available),
                total_operations=total_operations,
                job=job,
                operation=operation,
                job_ready_time=state.job_ready[job.job_id],
                machine=best["machine"],
                machine_available_time=state.machine_available[best["machine"].machine_id],
                agv=best.get("agv"),
                agv_available_time=best.get("agv_available_time", 0),
                route=best.get("route", ()),
                route_start=best.get("route_start", state.job_ready[job.job_id]),
                route_end=best.get("route_end", state.job_ready[job.job_id]),
                route_wait_time=best.get("route_wait_time", 0.0),
                candidate_has_transport=bool(best.get("has_transport", False)),
                route_legs=best.get("route_legs"),
            )
        )
    return _attach_group_critical_path_regrets(tuple(rows))


def _score_operation_rows(
    rows: tuple[dict[str, object], ...],
    artifact: OperationRankerArtifact,
) -> tuple[dict[str, object], ...]:
    sorted_rows = tuple(sorted(rows, key=lambda row: str(row.get("candidate_operation_id", ""))))
    feature_rows = tuple(
        _operation_sequence_features(row, index, sorted_rows)
        for index, row in enumerate(sorted_rows)
    )
    group = OperationGroup(
        group_id=str(sorted_rows[0].get("decision_group_id", "e110-rollout")),
        rows=sorted_rows,
        feature_rows=feature_rows,
        selected_index=0,
        regret_targets=tuple(0.0 for _ in sorted_rows),
        regret_label_available=False,
    )
    return _score_groups(artifact.model, (group,), standardizer=artifact.standardizer)[0]


def _neural_choice(scored_rows: tuple[dict[str, object], ...]) -> dict[str, object]:
    return max(
        scored_rows,
        key=lambda row: (
            float(row["predicted_score"]),
            -float(row.get("candidate_machine_start_lower_bound", 0.0)),
            str(row.get("candidate_operation_id", "")),
        ),
    )


def _regret_aware_choice(scored_rows: tuple[dict[str, object], ...]) -> dict[str, object]:
    return max(
        scored_rows,
        key=lambda row: (
            float(row.get("regret_aware_score", row.get("predicted_score", 0.0))),
            -float(row.get("candidate_machine_start_lower_bound", 0.0)),
            str(row.get("candidate_operation_id", "")),
        ),
    )


def _lookahead_choice(scored_rows: tuple[dict[str, object], ...]) -> dict[str, object]:
    return min(
        scored_rows,
        key=lambda row: (
            _candidate_completion(row),
            float(row.get("candidate_processing_time", 0.0)),
            float(row.get("candidate_processing_energy", 0.0)),
            str(row.get("candidate_operation_id", "")),
        ),
    )


def _gate_dominant_objective(objectives: tuple[str, ...], preference: tuple[float, ...]) -> str:
    if preference and max(preference) - min(preference) <= 1.0e-9:
        return "balanced"
    if not preference or not objectives:
        return "balanced"
    index = max(range(len(preference)), key=lambda item: (preference[item], -item))
    return objectives[index] if index < len(objectives) else "balanced"


def _regret_gate_decision(
    *,
    dominant_objective: str,
    preference: tuple[float, ...],
    completion_regret: float,
    energy_gain: float,
    path_gain: float,
    objectives: tuple[str, ...] = ("makespan", "tardiness", "energy", "path_conflict_delay", "stability"),
) -> RegretGateDecision:
    regret_caps = {
        "makespan": 0.0,
        "tardiness": 5.0,
        "energy": 30.0,
        "path_conflict_delay": 40.0,
        "stability": 50.0,
        "balanced": 15.0,
    }
    regret_cap = regret_caps.get(dominant_objective, regret_caps["balanced"])
    energy_weight = _preference_weight(objectives, preference, "energy")
    path_weight = _preference_weight(objectives, preference, "path_conflict_delay") + 0.5 * _preference_weight(
        objectives,
        preference,
        "stability",
    )
    weighted_gain = energy_weight * max(0.0, energy_gain) + path_weight * max(0.0, path_gain)
    if completion_regret <= regret_cap:
        return RegretGateDecision(
            allow_override=True,
            reason="under-cap",
            regret_cap=regret_cap,
            weighted_gain=weighted_gain,
        )
    if weighted_gain >= completion_regret:
        return RegretGateDecision(
            allow_override=True,
            reason="gain-covers-regret",
            regret_cap=regret_cap,
            weighted_gain=weighted_gain,
        )
    return RegretGateDecision(
        allow_override=False,
        reason="blocked-regret",
        regret_cap=regret_cap,
        weighted_gain=weighted_gain,
    )


def _preference_weight(objectives: tuple[str, ...], preference: tuple[float, ...], objective: str) -> float:
    try:
        index = objectives.index(objective)
    except ValueError:
        return 0.0
    return float(preference[index]) if index < len(preference) else 0.0


def _gate_trace_row(
    *,
    level: str,
    seed: int,
    mode: str,
    preference_id: str,
    dominant_objective: str,
    step_index: int,
    available_count: int,
    chosen: dict[str, object],
    neural_choice: dict[str, object],
    lookahead_choice: dict[str, object],
    teacher_operation_id: str,
    decision: RegretGateDecision,
    completion_regret: float,
    energy_gain: float,
    path_gain: float,
) -> dict[str, object]:
    chosen_id = str(chosen.get("candidate_operation_id", ""))
    lookahead_id = str(lookahead_choice.get("candidate_operation_id", ""))
    teacher_id = str(teacher_operation_id)
    return {
        "level": level,
        "seed": seed,
        "mode": mode,
        "target_preference_id": preference_id,
        "dominant_objective": dominant_objective,
        "step_index": step_index,
        "available_count": available_count,
        "chosen_operation_id": chosen_id,
        "neural_operation_id": neural_choice.get("candidate_operation_id", ""),
        "lookahead_operation_id": lookahead_id,
        "teacher_operation_id": teacher_id,
        "allow_override": 1 if decision.allow_override else 0,
        "gate_reason": decision.reason,
        "regret_cap": round(decision.regret_cap, 10),
        "completion_regret": round(float(completion_regret), 10),
        "energy_gain": round(float(energy_gain), 10),
        "path_gain": round(float(path_gain), 10),
        "weighted_gain": round(decision.weighted_gain, 10),
        "same_as_lookahead": 1 if chosen_id == lookahead_id else 0,
        "same_as_teacher": 1 if chosen_id == teacher_id else 0,
    }


def _decision_trace_rows(
    scored_rows: tuple[dict[str, object], ...],
    *,
    level: str,
    seed: int,
    mode: str,
    preference_id: str,
    preference: tuple[float, ...],
    objectives: tuple[str, ...],
    step_index: int,
    teacher_operation_id: str,
) -> tuple[dict[str, object], ...]:
    dominant_index = max(range(len(preference)), key=lambda index: (preference[index], -index)) if preference else 0
    dominant_objective = objectives[dominant_index] if dominant_index < len(objectives) else ""
    by_neural = sorted(
        scored_rows,
        key=lambda row: (
            float(row["predicted_score"]),
            -float(row.get("candidate_machine_start_lower_bound", 0.0)),
            str(row.get("candidate_operation_id", "")),
        ),
        reverse=True,
    )
    by_lookahead = sorted(
        scored_rows,
        key=lambda row: (
            _candidate_completion(row),
            float(row.get("candidate_processing_time", 0.0)),
            float(row.get("candidate_processing_energy", 0.0)),
            str(row.get("candidate_operation_id", "")),
        ),
    )
    neural_rank = {str(row["candidate_operation_id"]): rank for rank, row in enumerate(by_neural, start=1)}
    lookahead_rank = {str(row["candidate_operation_id"]): rank for rank, row in enumerate(by_lookahead, start=1)}
    neural_choice = by_neural[0]
    lookahead_choice = by_lookahead[0]
    completion_delta = _candidate_completion(neural_choice) - _candidate_completion(lookahead_choice)
    energy_delta = float(neural_choice.get("candidate_processing_energy", 0.0)) - float(
        lookahead_choice.get("candidate_processing_energy", 0.0)
    )
    path_delta = _candidate_path_pressure(neural_choice) - _candidate_path_pressure(lookahead_choice)
    rows = []
    for row in scored_rows:
        operation_id = str(row.get("candidate_operation_id", ""))
        selected_by_neural = operation_id == str(neural_choice.get("candidate_operation_id", ""))
        rows.append(
            {
                "level": level,
                "seed": seed,
                "mode": mode,
                "target_preference_id": preference_id,
                "dominant_objective": dominant_objective,
                "step_index": step_index,
                "available_count": len(scored_rows),
                "candidate_operation_id": operation_id,
                "selected_by_neural": 1 if selected_by_neural else 0,
                "selected_by_lookahead": 1 if operation_id == str(lookahead_choice.get("candidate_operation_id", "")) else 0,
                "selected_by_teacher_step": 1 if operation_id == teacher_operation_id else 0,
                "neural_rank": neural_rank[operation_id],
                "lookahead_rank": lookahead_rank[operation_id],
                "predicted_score": round(float(row.get("predicted_score", 0.0)), 10),
                "predicted_future_regret": round(float(row.get("predicted_future_regret", 0.0)), 10),
                "regret_aware_score": round(float(row.get("regret_aware_score", row.get("predicted_score", 0.0))), 10),
                "lookahead_completion": round(_candidate_completion(row), 10),
                "candidate_processing_time": row.get("candidate_processing_time", 0.0),
                "candidate_processing_energy": row.get("candidate_processing_energy", 0.0),
                "candidate_machine_start_lower_bound": row.get("candidate_machine_start_lower_bound", 0.0),
                "candidate_route_wait_time": row.get("candidate_route_wait_time", 0.0),
                "candidate_route_length": row.get("candidate_route_length", 0.0),
                "candidate_route_bottleneck_count": row.get("candidate_route_bottleneck_count", 0.0),
                "candidate_route_event_overlap_duration": row.get("candidate_route_event_overlap_duration", 0.0),
                "candidate_active_path_event_count": row.get("candidate_active_path_event_count", 0.0),
                "candidate_active_machine_event_count": row.get("candidate_active_machine_event_count", 0.0),
                "selected_completion_delta_to_lookahead": round(completion_delta, 10) if selected_by_neural else "",
                "selected_energy_delta_to_lookahead": round(energy_delta, 10) if selected_by_neural else "",
                "selected_path_pressure_delta_to_lookahead": round(path_delta, 10) if selected_by_neural else "",
            }
        )
    return tuple(rows)


def _candidate_completion(row: dict[str, object]) -> float:
    return float(row.get("candidate_machine_start_lower_bound", 0.0)) + float(row.get("candidate_processing_time", 0.0))


def _candidate_path_pressure(row: dict[str, object]) -> float:
    return (
        float(row.get("candidate_route_wait_time", 0.0))
        + 25.0 * float(row.get("candidate_route_bottleneck_count", 0.0))
        + 12.0 * float(row.get("candidate_route_event_overlap_duration", 0.0))
        + 2.0 * float(row.get("candidate_active_path_event_count", 0.0))
        + 0.05 * float(row.get("candidate_route_length", 0.0))
    )


def _detail_row(
    *,
    instance: object,
    schedule: ScheduleResult,
    teacher_operations: tuple[str, ...],
    level: str,
    seed: int,
    mode: str,
    preference_id: str,
    preference: tuple[float, ...],
    method: str,
    objective_scales: ObjectiveScales | None = None,
) -> dict[str, object]:
    metrics = evaluate_schedule(instance, schedule)
    surrogates = preference_teacher_surrogate_metrics(instance, schedule)
    contract = None
    if objective_scales is None:
        objective_values = _objective_values(instance, metrics, surrogates)
        objective_names = tuple(instance.objectives)
        target_weighted_value = _weighted_value(instance, preference, objective_values)
    else:
        evaluation = evaluate_objective_contract(instance, schedule)
        contract = objective_breakdown(evaluation.raw, preference, objective_scales)
        objective_values = contract.raw
        objective_names = CANONICAL_OBJECTIVE_NAMES
        target_weighted_value = contract.weighted_value
    dominant_index = max(range(len(preference)), key=lambda index: (preference[index], -index)) if preference else 0
    dominant_objective = objective_names[dominant_index] if dominant_index < len(objective_names) else ""
    schedule_operations = tuple(operation.operation_id for operation in schedule.operations)
    match_count = sum(1 for left, right in zip(schedule_operations, teacher_operations) if left == right)
    prefix_match = 0
    for left, right in zip(schedule_operations, teacher_operations):
        if left != right:
            break
        prefix_match += 1
    total = max(1, len(teacher_operations))
    row = {
        "level": level,
        "seed": seed,
        "mode": mode,
        "target_preference_id": preference_id,
        "preference": ";".join(f"{value:.4f}" for value in preference),
        "method": method,
        "algorithm": schedule.algorithm,
        "operation_match_rate_to_teacher": round(match_count / total, 10),
        "operation_prefix_match_rate_to_teacher": round(prefix_match / total, 10),
        "target_weighted_value": round(target_weighted_value, 10),
        "dominant_objective": dominant_objective,
        "dominant_objective_value": round(objective_values.get(dominant_objective, 0.0), 10),
        "makespan": round(metrics.get("makespan", 0.0), 10),
        "tardiness": round(metrics.get("tardiness", 0.0), 10),
        "energy": round(metrics.get("energy", 0.0), 10),
        "path_conflict_delay": round(metrics.get("path_conflict_delay", 0.0), 10),
        "stability": round(metrics.get("stability", 0.0), 10),
        "teacher_path_pressure": round(surrogates.get("teacher_path_pressure", 0.0), 10),
        "teacher_stability_pressure": round(surrogates.get("teacher_stability_pressure", 0.0), 10),
        "teacher_projected_tardiness": round(surrogates.get("teacher_projected_tardiness", 0.0), 10),
        "feasible": round(metrics.get("feasible", 0.0), 10),
        "precedence_violations": round(metrics.get("precedence_violations", 0.0), 10),
        "machine_overlap_violations": round(metrics.get("machine_overlap_violations", 0.0), 10),
        "agv_overlap_violations": round(metrics.get("agv_overlap_violations", 0.0), 10),
        "edge_capacity_violations": round(metrics.get("edge_capacity_violations", 0.0), 10),
        "machine_unavailability_violations": round(metrics.get("machine_unavailability_violations", 0.0), 10),
        "agv_unavailability_violations": round(metrics.get("agv_unavailability_violations", 0.0), 10),
        "path_event_violations": round(metrics.get("path_event_violations", 0.0), 10),
    }
    if contract is not None:
        row["route_pressure"] = contract.raw["route_pressure"]
        row["event_exposure_pressure"] = contract.raw["event_exposure_pressure"]
        row["objective_scale_source_sha256"] = objective_scales.source_sha256
        row.update(evaluation.soft_components)
        for name in CANONICAL_OBJECTIVE_NAMES:
            row[f"raw_{name}"] = contract.raw[name]
            row[f"normalized_{name}"] = contract.normalized[name]
            row[f"weighted_component_{name}"] = contract.contributions[name]
            row[f"contribution_rate_{name}"] = contract.contribution_rates[name]
    return row


def _objective_values(
    instance: object,
    metrics: dict[str, float],
    surrogates: dict[str, float],
) -> dict[str, float]:
    values = dict(metrics)
    if "path_conflict_delay" in instance.objectives:
        values["path_conflict_delay"] = float(surrogates.get("teacher_path_pressure", metrics.get("path_conflict_delay", 0.0)))
    if "stability" in instance.objectives:
        values["stability"] = float(surrogates.get("teacher_stability_pressure", metrics.get("stability", 0.0)))
    if "tardiness" in instance.objectives:
        values.setdefault("teacher_projected_tardiness", float(surrogates.get("teacher_projected_tardiness", 0.0)))
    return values


def _weighted_value(instance: object, preference: tuple[float, ...], values: dict[str, float]) -> float:
    total = 0.0
    for index, objective in enumerate(instance.objectives):
        if index < len(preference):
            total += float(preference[index]) * float(values.get(objective, 0.0))
    return total


def _summary_rows(detail_rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in detail_rows:
        grouped.setdefault((str(row["method"]), "ALL"), []).append(row)
        grouped.setdefault((str(row["method"]), str(row["dominant_objective"])), []).append(row)
    summary = []
    for (method, group), rows in sorted(grouped.items()):
        summary.append(
            {
                "method": method,
                "group": group,
                "cases": len(rows),
                "mean_operation_match_rate_to_teacher": round(_mean(rows, "operation_match_rate_to_teacher"), 10),
                "mean_target_weighted_value": round(_mean(rows, "target_weighted_value"), 10),
                "mean_dominant_objective_value": round(_mean(rows, "dominant_objective_value"), 10),
                "mean_makespan": round(_mean(rows, "makespan"), 10),
                "mean_tardiness": round(_mean(rows, "tardiness"), 10),
                "mean_energy": round(_mean(rows, "energy"), 10),
                "mean_teacher_path_pressure": round(_mean(rows, "teacher_path_pressure"), 10),
                "mean_teacher_stability_pressure": round(_mean(rows, "teacher_stability_pressure"), 10),
                "feasible_rate": round(_mean(rows, "feasible"), 10),
            }
        )
    return tuple(_ordered_summary(row) for row in summary)


def _pairwise_rows(
    detail_rows: tuple[dict[str, object], ...],
    *,
    candidate_method: str,
) -> tuple[dict[str, object], ...]:
    key_fields = ("level", "seed", "mode", "target_preference_id")
    by_case: dict[tuple[object, ...], dict[str, dict[str, object]]] = {}
    for row in detail_rows:
        case_key = tuple(row[field] for field in key_fields)
        by_case.setdefault(case_key, {})[str(row["method"])] = row
    baseline_methods = sorted(
        {
            str(row["method"])
            for row in detail_rows
            if str(row["method"]) != candidate_method
        }
    )
    rows = []
    for baseline_method in baseline_methods:
        for group in ("ALL", "energy", "makespan", "path_conflict_delay", "stability", "tardiness"):
            deltas = []
            wins = equals = losses = 0
            for methods in by_case.values():
                candidate = methods.get(candidate_method)
                baseline = methods.get(baseline_method)
                if candidate is None or baseline is None:
                    continue
                if group != "ALL" and str(candidate["dominant_objective"]) != group:
                    continue
                delta = float(candidate["target_weighted_value"]) - float(baseline["target_weighted_value"])
                deltas.append(delta)
                if delta < -1.0e-9:
                    wins += 1
                elif delta > 1.0e-9:
                    losses += 1
                else:
                    equals += 1
            if not deltas:
                continue
            rows.append(
                {
                    "candidate_method": candidate_method,
                    "baseline_method": baseline_method,
                    "group": group,
                    "cases": len(deltas),
                    "wins": wins,
                    "equals": equals,
                    "losses": losses,
                    "mean_target_delta": round(sum(deltas) / len(deltas), 10),
                }
            )
    return tuple(_ordered_pairwise(row) for row in rows)


def _gate_summary_rows(gate_trace_rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in gate_trace_rows:
        grouped.setdefault("ALL", []).append(row)
        grouped.setdefault(str(row["dominant_objective"]), []).append(row)
    rows = []
    for dominant_objective, group_rows in sorted(grouped.items()):
        allowed = [row for row in group_rows if int(row.get("allow_override", 0)) == 1]
        blocked = [row for row in group_rows if int(row.get("allow_override", 0)) == 0]
        different_actions = [row for row in group_rows if int(row.get("same_as_lookahead", 0)) == 0]
        effective_overrides = [
            row
            for row in group_rows
            if int(row.get("allow_override", 0)) == 1 and int(row.get("same_as_lookahead", 0)) == 0
        ]
        decisions = len(group_rows)
        rows.append(
            {
                "dominant_objective": dominant_objective,
                "decisions": decisions,
                "neural_override_rate": round(len(allowed) / decisions, 10) if decisions else 0.0,
                "different_action_rate": round(len(different_actions) / decisions, 10) if decisions else 0.0,
                "effective_neural_override_rate": round(len(effective_overrides) / decisions, 10) if decisions else 0.0,
                "blocked_override_rate": round(len(blocked) / decisions, 10) if decisions else 0.0,
                "mean_completion_regret_allowed": round(_mean(allowed, "completion_regret"), 10),
                "mean_completion_regret_blocked": round(_mean(blocked, "completion_regret"), 10),
                "mean_weighted_gain_allowed": round(_mean(allowed, "weighted_gain"), 10),
                "mean_weighted_gain_blocked": round(_mean(blocked, "weighted_gain"), 10),
                "same_as_lookahead_rate": round(_mean(group_rows, "same_as_lookahead"), 10),
                "same_as_teacher_rate": round(_mean(group_rows, "same_as_teacher"), 10),
            }
        )
    return tuple(_ordered_gate_summary(row) for row in rows)


def _mean(rows: list[dict[str, object]], field: str) -> float:
    return sum(float(row.get(field, 0.0)) for row in rows) / len(rows) if rows else 0.0


def _ordered_detail(row: dict[str, object]) -> dict[str, object]:
    return {field: row.get(field, "") for field in DETAIL_FIELDS}


def _ordered_summary(row: dict[str, object]) -> dict[str, object]:
    return {field: row.get(field, "") for field in SUMMARY_FIELDS}


def _ordered_pairwise(row: dict[str, object]) -> dict[str, object]:
    return {field: row.get(field, "") for field in PAIRWISE_FIELDS}


def _ordered_trace(row: dict[str, object]) -> dict[str, object]:
    return {field: row.get(field, "") for field in TRACE_FIELDS}


def _ordered_gate_trace(row: dict[str, object]) -> dict[str, object]:
    return {field: row.get(field, "") for field in GATE_TRACE_FIELDS}


def _ordered_gate_summary(row: dict[str, object]) -> dict[str, object]:
    return {field: row.get(field, "") for field in GATE_SUMMARY_FIELDS}


def write_neural_operation_rollout_outputs(out_dir: Path, rows: dict[str, tuple[dict[str, object], ...]]) -> None:
    write_csv(out_dir / "e110_neural_operation_rollout_detail.csv", rows["detail"])
    write_csv(out_dir / "e110_neural_operation_rollout_summary.csv", rows["summary"])
    write_csv(out_dir / "e110_neural_operation_rollout_pairwise.csv", rows["pairwise"])
    write_csv(out_dir / "e110_neural_operation_rollout_trace.csv", rows["trace"])
    write_csv(out_dir / "e110_neural_operation_rollout_train.csv", rows["train"])
    write_csv(out_dir / "e111_regret_controlled_neural_rollout_detail.csv", rows["detail"])
    write_csv(out_dir / "e111_regret_controlled_neural_rollout_summary.csv", rows["summary"])
    write_csv(out_dir / "e111_regret_controlled_neural_rollout_pairwise.csv", rows["pairwise"])
    write_csv(out_dir / "e111_regret_controlled_neural_rollout_trace.csv", rows["gate_trace"])
    write_csv(out_dir / "e111_regret_controlled_neural_rollout_gate_summary.csv", rows["gate_summary"])
