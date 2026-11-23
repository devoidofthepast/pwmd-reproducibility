from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from su_modfjsp_ha_pc.generator import generate_instance
from su_modfjsp_ha_pc.metrics import evaluate_schedule
from su_modfjsp_ha_pc.moea import MoeaConfig, _ordered_operations, run_moea_baseline
from su_modfjsp_ha_pc.offline_logs import write_csv
from su_modfjsp_ha_pc.path_conflict_transport_decoder import (
    _safe_shortest_path,
    JOB_KEY,
    job_pickup_node,
    two_leg_routes,
    two_leg_safe,
    candidate_routes_for_transport,
)
from su_modfjsp_ha_pc.rules import (
    _available_operations,
    _first_resource_event_feasible_start,
    _first_transport_feasible_start,
    job_pickup_of,
    move_leg_usage,
    two_leg_early_schedule,
)
from su_modfjsp_ha_pc.schedule import ScheduleResult, ScheduledOperation, TransportMove
from su_modfjsp_ha_pc.seeded_moea import (
    _apply_seed_teacher_mask,
    _seed_initial_quota_for_mode,
    _seed_preferences_by_context,
    _trained_seed_hints_by_context,
    read_seed_rows,
)
from su_modfjsp_ha_pc.structural import realize_structural_events


ROUTE_FAMILIES = ("no-transport", "shortest", "bottleneck-avoid", "event-avoid", "anchored-other")

FEATURE_FIELDS = (
    "level_l5",
    "mode_realized",
    "seed_scaled",
    "step_fraction",
    "available_operation_count",
    "job_release_time",
    "job_due_date",
    "job_due_slack_at_op_start",
    "operation_index",
    "job_operation_count",
    "operation_progress",
    "eligible_machine_count",
    "processing_time_min",
    "processing_time_mean",
    "processing_time_max",
    "chosen_processing_time",
    "chosen_processing_energy",
    "chosen_machine_energy_rate",
    "chosen_machine_nominal_capacity",
    "machine_start",
    "machine_wait_since_transport",
    "has_transport",
    "chosen_agv_capacity",
    "chosen_agv_speed",
    "transport_start",
    "transport_wait_since_release",
    "transport_travel_time",
    "route_edge_count",
    "route_length",
    "route_bottleneck_count",
    "route_capacity_min",
    "route_blockage_probability_max",
    "route_event_overlap_duration",
    "route_uses_bottleneck",
    "active_path_event_count",
    "active_agv_event_count",
    "active_machine_event_count",
)

OBJECTIVE_AUDIT_FIELDS = (
    "objective_makespan",
    "objective_tardiness",
    "objective_energy",
    "objective_path_conflict_delay",
    "objective_stability",
)

LABEL_FIELDS = (
    "selected_available_rank",
    "selected_machine_index",
    "selected_eligible_machine_index",
    "selected_agv_index",
    "route_family_index",
    "route_uses_bottleneck_label",
)

IDENTITY_FIELDS = (
    "sample_id",
    "split",
    "level",
    "seed",
    "mode",
    "algorithm",
    "teacher_family",
    "source_solution_id",
    "teacher_schedule_id",
    "step_index",
    "operation_id",
    "job_id",
    "machine_id",
    "agv_id",
    "route_family",
    "route_edges",
    "objective_vector",
)

DATASET_FIELDS = IDENTITY_FIELDS + OBJECTIVE_AUDIT_FIELDS + FEATURE_FIELDS + LABEL_FIELDS

CANDIDATE_FEATURE_FIELDS = (
    "level_l5",
    "mode_realized",
    "seed_scaled",
    "target_preference_0",
    "target_preference_1",
    "target_preference_2",
    "target_preference_3",
    "target_preference_4",
    "stage_operation",
    "stage_machine",
    "stage_transport",
    "step_fraction",
    "available_operation_count",
    "candidate_operation_index",
    "candidate_job_release_time",
    "candidate_job_due_date",
    "candidate_job_ready_time",
    "candidate_job_due_slack_at_ready",
    "candidate_job_remaining_ops",
    "candidate_eligible_machine_count",
    "candidate_processing_time_min",
    "candidate_processing_time_mean",
    "candidate_processing_time_max",
    "candidate_machine_energy_rate",
    "candidate_machine_available_time",
    "candidate_processing_time",
    "candidate_processing_energy",
    "candidate_machine_start_lower_bound",
    "candidate_has_transport",
    "candidate_agv_speed",
    "candidate_agv_load_capacity",
    "candidate_agv_available_time",
    "candidate_route_edge_count",
    "candidate_route_length",
    "candidate_route_bottleneck_count",
    "candidate_route_capacity_min",
    "candidate_route_blockage_probability_max",
    "candidate_route_event_overlap_duration",
    "candidate_route_wait_time",
    "candidate_route_travel_time",
    "candidate_active_path_event_count",
    "candidate_active_agv_event_count",
    "candidate_active_machine_event_count",
    "candidate_operation_completion_lower_bound",
    "candidate_job_remaining_processing_after",
    "candidate_job_tail_completion_lower_bound",
    "candidate_job_due_slack_after_tail",
    "candidate_job_projected_tardiness_lower_bound",
    "candidate_job_remaining_work_ratio",
    "candidate_machine_queue_delay",
    "candidate_transport_queue_delay",
    "candidate_completion_regret_to_group_best",
    "candidate_tail_regret_to_group_best",
    "candidate_machine_event_proximity_pressure",
    "candidate_path_event_proximity_pressure",
    "candidate_agv_event_proximity_pressure",
    "candidate_future_risk_pressure",
)

FUTURE_REGRET_LABEL_FIELDS = (
    "future_regret_label_available",
    "future_makespan_after_action",
    "future_tardiness_after_action",
    "future_makespan_regret_to_group_best",
    "future_tardiness_regret_to_group_best",
    "future_weighted_completion_regret",
)

CANDIDATE_FIELDS = (
    "candidate_id",
    "split",
    "level",
    "seed",
    "mode",
    "algorithm",
    "teacher_family",
    "source_solution_id",
    "teacher_schedule_id",
    "target_preference_id",
    "decision_group_id",
    "stage",
    "step_index",
    "candidate_operation_id",
    "candidate_job_id",
    "candidate_machine_id",
    "candidate_agv_id",
    "candidate_route_edges",
    "selected_action",
    *FUTURE_REGRET_LABEL_FIELDS,
    *CANDIDATE_FEATURE_FIELDS,
)


@dataclass(frozen=True)
class PathConflictNeuralDatasetConfig:
    size: str = "small"
    levels: tuple[str, ...] = ("L4", "L5")
    modes: tuple[str, ...] = ("raw", "realized")
    start_seed: int | None = None
    end_seed: int | None = None
    test_seed: int | None = None
    algorithm: str = "nsga3"
    population_size: int = 24
    max_evaluations: int = 96
    moea_seed: int = 104
    include_realized: bool = True
    max_seed_preferences: int = 5
    max_seed_hints: int = 5
    seed_initial_quota_raw: int = 12
    seed_initial_quota_realized: int = 4
    seed_neighbor_count: int = 1
    seed_teacher_mask: str = "full"
    max_schedules_per_context: int = 4
    teacher_family: str = "e104-path-conflict-decoder-teacher"
    max_teacher_preferences: int = 6
    include_future_regret_labels: bool = False
    future_regret_max_groups_per_context: int = 0


def build_path_conflict_neural_decision_rows(
    seed_rows_path: Path,
    config: PathConflictNeuralDatasetConfig | None = None,
) -> tuple[dict[str, object], ...]:
    resolved = config or PathConflictNeuralDatasetConfig()
    rows: list[dict[str, object]] = []
    sample_uid = 0
    for context in _teacher_schedules_from_seed_rows(seed_rows_path, resolved):
        context_rows = _decision_rows_for_schedule(
            instance=context["instance"],
            schedule=context["schedule"],
            operations=_ordered_operations(context["instance"]),
            level=str(context["level"]),
            seed=int(context["seed"]),
            mode=str(context["mode"]),
            algorithm=resolved.algorithm,
            split=_split(int(context["seed"]), resolved),
            teacher_family=str(context["teacher_family"]),
            source_solution_id=str(context["source_solution_id"]),
            teacher_schedule_id=str(context["teacher_schedule_id"]),
            sample_uid_start=sample_uid,
        )
        rows.extend(context_rows)
        sample_uid += len(context_rows)
    return tuple(_ordered_row(row) for row in rows)


def build_path_conflict_neural_candidate_rows(
    seed_rows_path: Path,
    config: PathConflictNeuralDatasetConfig | None = None,
) -> tuple[dict[str, object], ...]:
    resolved = config or PathConflictNeuralDatasetConfig()
    rows: list[dict[str, object]] = []
    candidate_uid = 0
    for context in _teacher_schedules_from_seed_rows(seed_rows_path, resolved):
        schedule = context["schedule"]
        instance = context["instance"]
        metrics = evaluate_schedule(instance, schedule)
        objective_vector = tuple(metrics.get(objective, 0.0) for objective in instance.objectives)
        context_rows = _candidate_rows_for_schedule(
            instance=instance,
            schedule=schedule,
            target_preference_id=f"proxy-{int(context['schedule_index']):03d}",
            target_preference=_target_preference_proxy(objective_vector, (objective_vector,)),
            level=str(context["level"]),
            seed=int(context["seed"]),
            mode=str(context["mode"]),
            algorithm=resolved.algorithm,
            split=_split(int(context["seed"]), resolved),
            teacher_family=str(context["teacher_family"]),
            source_solution_id=str(context["source_solution_id"]),
            teacher_schedule_id=str(context["teacher_schedule_id"]),
            candidate_uid_start=candidate_uid,
            include_future_regret_labels=resolved.include_future_regret_labels,
            future_regret_max_groups_per_context=resolved.future_regret_max_groups_per_context,
        )
        rows.extend(context_rows)
        candidate_uid += len(context_rows)
    return tuple(_ordered_candidate_row(row) for row in rows)


def build_preference_conditioned_neural_candidate_rows(
    config: PathConflictNeuralDatasetConfig | None = None,
) -> tuple[dict[str, object], ...]:
    resolved = config or PathConflictNeuralDatasetConfig()
    rows: list[dict[str, object]] = []
    candidate_uid = 0
    start_seed = resolved.start_seed if resolved.start_seed is not None else 1
    end_seed = resolved.end_seed if resolved.end_seed is not None else start_seed
    for level in resolved.levels:
        for seed in range(start_seed, end_seed + 1):
            raw_instance = generate_instance(level, resolved.size, seed)
            for mode in resolved.modes:
                if not _include_context(level, seed, mode, resolved):
                    continue
                instance = realize_structural_events(raw_instance).instance if mode == "realized" else raw_instance
                preferences = _explicit_teacher_preferences(
                    len(instance.objectives),
                    max_preferences=resolved.max_teacher_preferences,
                )
                for preference_index, preference in enumerate(preferences):
                    preference_id = _explicit_preference_id(preference_index, preference)
                    schedule = decode_preference_conditioned_teacher_schedule(
                        instance,
                        preference,
                        algorithm=f"{resolved.algorithm}-e107-teacher",
                    )
                    context_rows = _candidate_rows_for_schedule(
                        instance=instance,
                        schedule=schedule,
                        target_preference_id=preference_id,
                        target_preference=preference,
                        level=level,
                        seed=seed,
                        mode=mode,
                        algorithm=resolved.algorithm,
                        split=_split(seed, resolved),
                        teacher_family="e107-preference-conditioned-rollout-teacher",
                        source_solution_id=preference_id,
                        teacher_schedule_id=_teacher_schedule_id(level, seed, mode, preference_id, preference_index),
                        candidate_uid_start=candidate_uid,
                        include_future_regret_labels=resolved.include_future_regret_labels,
                        future_regret_max_groups_per_context=resolved.future_regret_max_groups_per_context,
                    )
                    rows.extend(context_rows)
                    candidate_uid += len(context_rows)
    return tuple(_ordered_candidate_row(row) for row in rows)


def write_path_conflict_neural_dataset(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    write_csv(path, tuple(_ordered_row(row) for row in rows))


def write_path_conflict_neural_candidate_dataset(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    write_csv(path, tuple(_ordered_candidate_row(row) for row in rows))


def preference_teacher_surrogate_metrics(instance: object, schedule: ScheduleResult) -> dict[str, float]:
    metrics = dict(evaluate_schedule(instance, schedule))
    job_by_id = {job.job_id: job for job in instance.jobs}
    operation_by_id = {operation.operation_id: operation for job in instance.jobs for operation in job.operations}

    path_pressure = 0.0
    stability_pressure = 0.0
    move_legs = _schedule_move_legs(instance, schedule)
    for move in schedule.transports:
        legs = move_legs[move.move_id]
        route_stats = _route_stats(instance, move.edge_ids, move.start, move.end)
        path_pressure += (
            25.0 * route_stats["route_bottleneck_count"]
            + 0.05 * route_stats["route_length"]
            + 12.0 * float(_legs_overlap_duration(instance, legs, ""))
            + 20.0 * _route_blockage_risk(instance, move.edge_ids)
            + 8.0 * _route_capacity_risk(instance, move.edge_ids)
        )
        stability_pressure += _legs_proximity_pressure(
            instance,
            event_types=("agv_failure",),
            legs=tuple(((move.agv_id,), s, e) for _r, s, e in legs),
        )
        stability_pressure += _legs_proximity_pressure(
            instance,
            event_types=("path_blockage", "congestion_spike"),
            legs=legs,
        )

    for scheduled in schedule.operations:
        operation = operation_by_id.get(scheduled.operation_id)
        job = job_by_id.get(scheduled.job_id)
        remaining = _remaining_min_processing(job, int(getattr(operation, "op_index", scheduled.op_index)) + 1) if job and operation else 0.0
        projected_completion = scheduled.end + remaining
        metrics.setdefault("teacher_projected_tardiness", 0.0)
        metrics["teacher_projected_tardiness"] += max(0.0, projected_completion - float(getattr(job, "due_date", 0)))
        stability_pressure += _event_proximity_pressure(
            instance,
            event_types=("machine_breakdown",),
            entity_ids=(scheduled.machine_id,),
            start=scheduled.start,
            end=scheduled.end,
        )

    metrics["teacher_path_pressure"] = round(path_pressure, 10)
    metrics["teacher_stability_pressure"] = round(stability_pressure, 10)
    return metrics


def decode_preference_conditioned_teacher_schedule(
    instance: object,
    preference: tuple[float, ...],
    *,
    algorithm: str,
) -> ScheduleResult:
    machine_by_id = {machine.machine_id: machine for machine in instance.machines}
    machine_available = {machine.machine_id: 0 for machine in instance.machines}
    job_ready = {job.job_id: job.release_time for job in instance.jobs}
    next_op_index = {job.job_id: 0 for job in instance.jobs}
    agv_available = {agv.agv_id: 0 for agv in instance.agvs}
    agv_node = {agv.agv_id: agv.start_node_id for agv in instance.agvs}
    edge_reservations: dict[str, list[tuple[int, int]]] = {edge.edge_id: [] for edge in instance.path_edges}

    scheduled_operations: list[ScheduledOperation] = []
    transport_moves: list[TransportMove] = []
    total_operations = sum(len(job.operations) for job in instance.jobs)
    while len(scheduled_operations) < total_operations:
        available = _available_operations(instance.jobs, next_op_index)
        if not available:
            raise ValueError("no schedulable operations remain")
        selected = _best_preference_action(
            instance=instance,
            available=available,
            preference=preference,
            machine_available=machine_available,
            job_ready=job_ready,
            agv_available=agv_available,
            agv_node=agv_node,
            edge_reservations=edge_reservations,
        )
        job = selected["job"]
        operation = selected["operation"]
        machine = selected["machine"]
        transport_end = job_ready[job.job_id]
        move = selected.get("transport")
        if move is not None:
            move = TransportMove(
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
            transport_moves.append(move)
            transport_end = move.end
            for _legs, _ls, _le in move_leg_usage(instance, move, job_pickup_of(instance, agv_node, move.job_id)):
                for edge_id in dict.fromkeys(_legs):
                    edge_reservations.setdefault(edge_id, []).append((_ls, _le))
            agv_available[move.agv_id] = move.end
            agv_node[move.agv_id] = move.to_node
            agv_node[JOB_KEY + str(move.job_id)] = move.to_node

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

    return ScheduleResult(
        algorithm=algorithm,
        path_aware=bool(instance.path_edges),
        operations=tuple(scheduled_operations),
        transports=tuple(transport_moves),
        event_aware=bool(instance.events),
    )


def _teacher_schedules_from_seed_rows(
    seed_rows_path: Path,
    config: PathConflictNeuralDatasetConfig,
) -> tuple[dict[str, object], ...]:
    seed_rows = read_seed_rows(seed_rows_path)
    seed_hint_rows = _apply_seed_teacher_mask(seed_rows, config.seed_teacher_mask)
    start_seed = config.start_seed if config.start_seed is not None else _min_seed(seed_rows)
    end_seed = config.end_seed if config.end_seed is not None else _max_seed(seed_rows)
    seed_preferences = _seed_preferences_by_context(
        seed_rows,
        levels=config.levels,
        start_seed=start_seed,
        end_seed=end_seed,
        include_realized=config.include_realized,
        max_seed_preferences=config.max_seed_preferences,
    )
    seed_hints = _trained_seed_hints_by_context(
        seed_hint_rows,
        context_preferences=seed_preferences,
        size=config.size,
        max_seed_hints=config.max_seed_hints,
        seed_neighbor_count=config.seed_neighbor_count,
    )
    contexts = []
    for level in config.levels:
        for seed in range(start_seed, end_seed + 1):
            raw_instance = generate_instance(level, config.size, seed)
            for mode in config.modes:
                if not _include_context(level, seed, mode, config):
                    continue
                instance = realize_structural_events(raw_instance).instance if mode == "realized" else raw_instance
                preferences = seed_preferences.get((level, seed, mode), ())
                hints = seed_hints.get((level, seed, mode), ())
                if not preferences:
                    preferences = instance.preferences[: config.max_seed_preferences]
                resolved_quota = _seed_initial_quota_for_mode(
                    mode,
                    seed_initial_quota=None,
                    seed_initial_quota_raw=config.seed_initial_quota_raw,
                    seed_initial_quota_realized=config.seed_initial_quota_realized,
                )
                archive = run_moea_baseline(
                    instance,
                    config.algorithm,
                    MoeaConfig(
                        population_size=config.population_size,
                        max_evaluations=config.max_evaluations,
                        seed=config.moea_seed,
                        path_aware=True,
                        event_aware=True,
                        seed_preferences=preferences,
                        seed_hints=hints,
                        seed_initial_quota=resolved_quota,
                    ),
                )
                for schedule_index, solution in enumerate(archive[: config.max_schedules_per_context]):
                    contexts.append(
                        {
                            "instance": instance,
                            "schedule": solution.schedule,
                            "level": level,
                            "seed": seed,
                            "mode": mode,
                            "teacher_family": config.teacher_family,
                            "source_solution_id": solution.solution_id,
                            "teacher_schedule_id": _teacher_schedule_id(level, seed, mode, solution.solution_id, schedule_index),
                            "schedule_index": schedule_index,
                        }
                    )
                if archive:
                    continue
                for preference_index, preference in enumerate(preferences[: config.max_schedules_per_context]):
                    schedule = decode_preference_conditioned_teacher_schedule(
                        instance,
                        preference,
                        algorithm=f"{config.algorithm}-fallback-teacher",
                    )
                    preference_id = _explicit_preference_id(preference_index, preference)
                    contexts.append(
                        {
                            "instance": instance,
                            "schedule": schedule,
                            "level": level,
                            "seed": seed,
                            "mode": mode,
                            "teacher_family": f"{config.teacher_family}-fallback",
                            "source_solution_id": preference_id,
                            "teacher_schedule_id": _teacher_schedule_id(level, seed, mode, preference_id, preference_index),
                            "schedule_index": preference_index,
                        }
                    )
    return tuple(contexts)


def _decision_rows_for_schedule(
    *,
    instance: object,
    schedule: ScheduleResult,
    operations: tuple[object, ...],
    level: str,
    seed: int,
    mode: str,
    algorithm: str,
    split: str,
    teacher_family: str,
    source_solution_id: str,
    teacher_schedule_id: str,
    sample_uid_start: int,
) -> tuple[dict[str, object], ...]:
    operation_by_id = {operation.operation_id: operation for job in instance.jobs for operation in job.operations}
    job_by_id = {job.job_id: job for job in instance.jobs}
    machine_by_id = {machine.machine_id: machine for machine in instance.machines}
    transport_by_operation = {move.operation_id: move for move in schedule.transports}
    metrics = evaluate_schedule(instance, schedule)
    objective_vector = tuple(metrics.get(objective, 0.0) for objective in instance.objectives)

    machine_available = {machine.machine_id: 0 for machine in instance.machines}
    job_ready = {job.job_id: job.release_time for job in instance.jobs}
    next_op_index = {job.job_id: 0 for job in instance.jobs}
    rows = []
    total_operations = max(1, len(schedule.operations))
    for step_index, scheduled in enumerate(schedule.operations):
        operation = operation_by_id[scheduled.operation_id]
        job = job_by_id[operation.job_id]
        available = _available_operations(instance.jobs, next_op_index)
        available_ids = [item[1].operation_id for item in available]
        selected_available_rank = available_ids.index(operation.operation_id) if operation.operation_id in available_ids else 0
        machine = machine_by_id[scheduled.machine_id]
        transport = transport_by_operation.get(operation.operation_id)
        route_stats = _route_stats(
            instance,
            transport.edge_ids if transport else (),
            transport.start if transport else scheduled.start,
            transport.end if transport else scheduled.start,
        )
        row = {
            "sample_id": f"n{sample_uid_start + len(rows):07d}",
            "split": split,
            "level": level,
            "seed": f"{seed:03d}",
            "mode": mode,
            "algorithm": algorithm,
            "teacher_family": teacher_family,
            "source_solution_id": source_solution_id,
            "teacher_schedule_id": teacher_schedule_id,
            "step_index": step_index,
            "operation_id": operation.operation_id,
            "job_id": job.job_id,
            "machine_id": scheduled.machine_id,
            "agv_id": transport.agv_id if transport else "",
            "route_family": _route_family(instance, transport),
            "route_edges": ";".join(transport.edge_ids) if transport else "",
            "objective_vector": _format_vector(objective_vector),
            "selected_available_rank": selected_available_rank,
            "selected_machine_index": _machine_index(instance, scheduled.machine_id),
            "selected_eligible_machine_index": operation.eligible_machines.index(scheduled.machine_id),
            "selected_agv_index": _agv_index(instance, transport.agv_id) if transport else -1,
            "route_family_index": ROUTE_FAMILIES.index(_route_family(instance, transport)),
            "route_uses_bottleneck_label": 1 if route_stats["route_bottleneck_count"] > 0 else 0,
            **_objective_audit(metrics),
            **_decision_feature_values(
                instance=instance,
                level=level,
                seed=seed,
                mode=mode,
                step_index=step_index,
                total_operations=total_operations,
                available_count=len(available),
                job=job,
                operation=operation,
                scheduled=scheduled,
                machine=machine,
                transport=transport,
                route_stats=route_stats,
            ),
        }
        rows.append(row)
        machine_available[scheduled.machine_id] = scheduled.end
        job_ready[job.job_id] = scheduled.end
        next_op_index[job.job_id] += 1
    return tuple(rows)


def _candidate_rows_for_schedule(
    *,
    instance: object,
    schedule: ScheduleResult,
    target_preference_id: str,
    target_preference: tuple[float, ...],
    level: str,
    seed: int,
    mode: str,
    algorithm: str,
    split: str,
    teacher_family: str,
    source_solution_id: str,
    teacher_schedule_id: str,
    candidate_uid_start: int,
    include_future_regret_labels: bool = False,
    future_regret_max_groups_per_context: int = 0,
) -> tuple[dict[str, object], ...]:
    job_by_id = {job.job_id: job for job in instance.jobs}
    operation_by_id = {operation.operation_id: operation for job in instance.jobs for operation in job.operations}
    machine_by_id = {machine.machine_id: machine for machine in instance.machines}
    transport_by_operation = {move.operation_id: move for move in schedule.transports}

    machine_available = {machine.machine_id: 0 for machine in instance.machines}
    job_ready = {job.job_id: job.release_time for job in instance.jobs}
    next_op_index = {job.job_id: 0 for job in instance.jobs}
    agv_available = {agv.agv_id: 0 for agv in instance.agvs}
    agv_node = {agv.agv_id: agv.start_node_id for agv in instance.agvs}
    edge_reservations: dict[str, list[tuple[int, int]]] = {edge.edge_id: [] for edge in instance.path_edges}

    rows: list[dict[str, object]] = []
    total_operations = max(1, len(schedule.operations))
    for step_index, scheduled in enumerate(schedule.operations):
        operation = operation_by_id[scheduled.operation_id]
        job = job_by_id[operation.job_id]
        available = _available_operations(instance.jobs, next_op_index)
        available_ids = {item[1].operation_id for item in available}
        if operation.operation_id not in available_ids:
            available = available + ((job, operation),)
        operation_group_id = f"{teacher_schedule_id}|{target_preference_id}|{step_index:03d}|operation"
        operation_group_rows = []
        for candidate_job, candidate_operation in available:
            best = _operation_stage_lookahead(
                instance=instance,
                job=candidate_job,
                operation=candidate_operation,
                machine_available=machine_available,
                job_ready_time=job_ready[candidate_job.job_id],
                agv_available=agv_available,
                agv_node=agv_node,
                edge_reservations=edge_reservations,
            )
            operation_group_rows.append(
                _candidate_row(
                    instance=instance,
                    candidate_uid=candidate_uid_start + len(rows) + len(operation_group_rows),
                    split=split,
                    level=level,
                    seed=seed,
                    mode=mode,
                    algorithm=algorithm,
                    teacher_family=teacher_family,
                    source_solution_id=source_solution_id,
                    teacher_schedule_id=teacher_schedule_id,
                    target_preference_id=target_preference_id,
                    decision_group_id=operation_group_id,
                    stage="operation",
                    target_preference=target_preference,
                    step_index=step_index,
                    group_size=len(available),
                    selected_action=candidate_operation.operation_id == operation.operation_id,
                    available_operation_count=len(available),
                    total_operations=total_operations,
                    job=candidate_job,
                    operation=candidate_operation,
                    job_ready_time=job_ready[candidate_job.job_id],
                    machine=best["machine"],
                    machine_available_time=machine_available[best["machine"].machine_id],
                    agv=best.get("agv"),
                    agv_available_time=best.get("agv_available_time", 0),
                    route=best.get("route", ()),
                    route_start=best.get("route_start", job_ready[candidate_job.job_id]),
                    route_end=best.get("route_end", job_ready[candidate_job.job_id]),
                    route_wait_time=best.get("route_wait_time", 0.0),
                    candidate_has_transport=bool(best.get("has_transport", False)),
                    route_legs=best.get("route_legs"),
                )
            )
        operation_group_rows = list(_attach_group_critical_path_regrets(tuple(operation_group_rows)))
        if include_future_regret_labels and _future_regret_step_selected(
            step_index,
            total_operations,
            future_regret_max_groups_per_context,
        ):
            prefix_operation_ids = {item.operation_id for item in schedule.operations[:step_index]}
            operation_group_rows = list(
                _attach_future_regret_labels(
                    instance=instance,
                    rows=tuple(operation_group_rows),
                    available=available,
                    preference=target_preference,
                    machine_available=machine_available,
                    job_ready=job_ready,
                    next_op_index=next_op_index,
                    agv_available=agv_available,
                    agv_node=agv_node,
                    edge_reservations=edge_reservations,
                    scheduled_prefix=tuple(schedule.operations[:step_index]),
                    transport_prefix=tuple(
                        move for move in schedule.transports if move.operation_id in prefix_operation_ids
                    ),
                )
            )
        rows.extend(operation_group_rows)

        machine_group_id = f"{teacher_schedule_id}|{target_preference_id}|{step_index:03d}|machine"
        for machine_id in operation.eligible_machines:
            machine = machine_by_id[machine_id]
            transport_projection = _best_transport_projection(
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
                    candidate_uid=candidate_uid_start + len(rows),
                    split=split,
                    level=level,
                    seed=seed,
                    mode=mode,
                    algorithm=algorithm,
                    teacher_family=teacher_family,
                    source_solution_id=source_solution_id,
                    teacher_schedule_id=teacher_schedule_id,
                    target_preference_id=target_preference_id,
                    decision_group_id=machine_group_id,
                    stage="machine",
                    target_preference=target_preference,
                    step_index=step_index,
                    group_size=len(operation.eligible_machines),
                    selected_action=machine_id == scheduled.machine_id,
                    available_operation_count=len(available),
                    total_operations=total_operations,
                    job=job,
                    operation=operation,
                    job_ready_time=job_ready[job.job_id],
                    machine=machine,
                    machine_available_time=machine_available[machine_id],
                    agv=transport_projection.get("agv"),
                    agv_available_time=transport_projection.get("agv_available_time", 0),
                    route=transport_projection.get("route", ()),
                    route_start=transport_projection.get("route_start", job_ready[job.job_id]),
                    route_end=transport_projection.get("route_end", job_ready[job.job_id]),
                    route_wait_time=transport_projection.get("route_wait_time", 0.0),
                    candidate_has_transport=bool(transport_projection.get("has_transport", False)),
                    route_legs=transport_projection.get("route_legs"),
                )
            )

        transport = transport_by_operation.get(operation.operation_id)
        if instance.agvs and instance.path_edges and transport is not None:
            transport_group_id = f"{teacher_schedule_id}|{target_preference_id}|{step_index:03d}|transport"
            transport_candidates = _transport_candidate_rows(
                instance=instance,
                job=job,
                operation=operation,
                machine=machine_by_id[scheduled.machine_id],
                teacher_transport=transport,
                agv_available=agv_available,
                agv_node=agv_node,
                edge_reservations=edge_reservations,
                earliest_job_ready=job_ready[job.job_id],
            )
            for candidate in transport_candidates:
                rows.append(
                    _candidate_row(
                        instance=instance,
                        candidate_uid=candidate_uid_start + len(rows),
                        split=split,
                        level=level,
                        seed=seed,
                        mode=mode,
                        algorithm=algorithm,
                        teacher_family=teacher_family,
                        source_solution_id=source_solution_id,
                        teacher_schedule_id=teacher_schedule_id,
                        target_preference_id=target_preference_id,
                        decision_group_id=transport_group_id,
                        stage="transport",
                        target_preference=target_preference,
                        step_index=step_index,
                        group_size=len(transport_candidates),
                        selected_action=bool(candidate["selected"]),
                        available_operation_count=len(available),
                        total_operations=total_operations,
                        job=job,
                        operation=operation,
                        job_ready_time=job_ready[job.job_id],
                        machine=machine_by_id[scheduled.machine_id],
                        machine_available_time=machine_available[scheduled.machine_id],
                        agv=candidate["agv"],
                        agv_available_time=agv_available[candidate["agv"].agv_id],
                        route=candidate["route"],
                        route_start=candidate["start"],
                        route_end=candidate["end"],
                        route_wait_time=candidate["wait_time"],
                        candidate_has_transport=True,
                        route_legs=candidate.get("route_legs") or _teacher_legs(instance, candidate),
                    )
                )

        machine_available[scheduled.machine_id] = scheduled.end
        job_ready[job.job_id] = scheduled.end
        next_op_index[job.job_id] += 1
        if transport is not None:
            for _legs, _ls, _le in move_leg_usage(instance, transport, job_pickup_of(instance, agv_node, transport.job_id)):
                for edge_id in dict.fromkeys(_legs):
                    edge_reservations.setdefault(edge_id, []).append((_ls, _le))
            agv_available[transport.agv_id] = transport.end
            agv_node[transport.agv_id] = transport.to_node
            agv_node[JOB_KEY + str(transport.job_id)] = transport.to_node
    return tuple(rows)


def _operation_stage_lookahead(
    *,
    instance: object,
    job: object,
    operation: object,
    machine_available: dict[str, int],
    job_ready_time: int,
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
) -> dict[str, object]:
    machine_by_id = {machine.machine_id: machine for machine in instance.machines}
    best = None
    for machine_id in operation.eligible_machines:
        machine = machine_by_id[machine_id]
        projection = _best_transport_projection(
            instance=instance,
            job=job,
            machine=machine,
            agv_available=agv_available,
            agv_node=agv_node,
            edge_reservations=edge_reservations,
            earliest_job_ready=job_ready_time,
        )
        transport_end = projection.get("route_end", job_ready_time)
        start = max(job_ready_time, machine_available[machine_id], int(transport_end))
        completion = start + operation.processing_time[machine_id]
        score = (
            completion,
            operation.processing_time[machine_id],
            operation.processing_energy[machine_id],
            machine_id,
        )
        candidate = {
            "score": score,
            "machine": machine,
            **projection,
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate
    if best is None:
        raise ValueError(f"operation {operation.operation_id} has no eligible machines")
    return best


def _best_transport_projection(
    *,
    instance: object,
    job: object,
    machine: object,
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
    earliest_job_ready: int,
) -> dict[str, object]:
    if not instance.agvs or not instance.path_edges:
        return {
            "has_transport": False,
            "agv": None,
            "agv_available_time": 0,
            "route": (),
            "route_start": earliest_job_ready,
            "route_end": earliest_job_ready,
            "route_wait_time": 0.0,
        }
    candidates = []
    edge_by_id = {edge.edge_id: edge for edge in instance.path_edges}
    for agv in instance.agvs:
        origin = agv_node[agv.agv_id]
        pickup = job_pickup_node(instance, agv_node, job.job_id)
        routes = two_leg_routes(
            instance.path_edges,
            source=origin,
            pickup=pickup,
            target=machine.node_id,
            avoid_edges=_active_event_edges_after(instance, earliest=max(earliest_job_ready, agv_available[agv.agv_id])),
        )
        if not routes:
            safe = two_leg_safe(instance.path_edges, origin, pickup, machine.node_id)
            routes = (safe,) if safe is not None else ((),)
        for route in routes:
            route_length = _route_length(instance, route)
            _tl = two_leg_early_schedule(instance, agv, origin, pickup, route, agv_available[agv.agv_id],
                                         earliest_job_ready, edge_reservations)
            start = _tl["start"]
            travel_time = _tl["end"] - _tl["start"]
            earliest = _tl["start"] - _tl["delay"]
            end = start + travel_time
            stats = _route_stats(instance, route, start, end)
            _pj = ((tuple(route[:_tl["n_empty"]]), _tl["start"], _tl["empty_end"]), (tuple(route[_tl["n_empty"]:]), _tl["loaded_start"], _tl["end"]))
            stats = {**stats, "route_event_overlap_duration": float(_legs_overlap_duration(instance, _pj, ""))}
            score = (
                end,
                start - earliest,
                stats["route_event_overlap_duration"],
                stats["route_bottleneck_count"],
                route_length,
                agv.agv_id,
                ";".join(route),
            )
            candidates.append(
                {
                    "score": score,
                    "has_transport": True,
                    "agv": agv,
                    "agv_available_time": agv_available[agv.agv_id],
                    "route": route,
                    "route_start": start,
                    "route_end": end,
                    "route_wait_time": float(start - earliest),
                    "route_length": route_length,
                    "travel_time": travel_time,
                    "transport_energy": round(route_length * agv.consumption_rate, 6),
                    "route_legs": _legs_of(route, _tl),
                }
            )
    if not candidates:
        raise ValueError(f"no transport candidate to {machine.machine_id}")
    return min(candidates, key=lambda row: row["score"])


def _candidate_row(
    *,
    instance: object,
    candidate_uid: int,
    split: str,
    level: str,
    seed: int,
    mode: str,
    algorithm: str,
    teacher_family: str,
    source_solution_id: str,
    teacher_schedule_id: str,
    target_preference_id: str,
    decision_group_id: str,
    stage: str,
    target_preference: tuple[float, ...],
    step_index: int,
    group_size: int,
    selected_action: bool,
    available_operation_count: int,
    total_operations: int,
    job: object,
    operation: object,
    job_ready_time: int,
    machine: object | None,
    machine_available_time: int,
    agv: object | None,
    agv_available_time: int,
    route: tuple[str, ...],
    route_start: int,
    route_end: int,
    route_wait_time: float,
    candidate_has_transport: bool,
    route_legs: tuple | None = None,
) -> dict[str, object]:
    del group_size
    if candidate_has_transport and route and route_legs is None:
        raise RuntimeError("rerun A1: transport candidate without route legs")
    _cl = route_legs if route_legs is not None else ((tuple(route), int(route_start), int(route_end)),)
    machine_id = machine.machine_id if machine is not None else ""
    agv_id = agv.agv_id if agv is not None else ""
    processing_times = tuple(operation.processing_time[machine_id] for machine_id in operation.eligible_machines)
    processing_energy = operation.processing_energy.get(machine_id, min(operation.processing_energy.values()))
    processing_time = operation.processing_time.get(machine_id, min(processing_times))
    machine_start = max(job_ready_time, machine_available_time, route_end)
    route_stats = _route_stats(instance, route, route_start, route_end)
    route_stats = {**route_stats, "route_event_overlap_duration": float(_legs_overlap_duration(instance, _cl, ""))}
    operation_completion = machine_start + processing_time
    remaining_after = _remaining_min_processing(job, int(getattr(operation, "op_index", 0)) + 1)
    remaining_from_current = processing_time + remaining_after
    total_job_min_processing = _remaining_min_processing(job, 0)
    tail_completion = operation_completion + remaining_after
    due_date = float(getattr(job, "due_date", 0.0))
    due_slack_after_tail = due_date - tail_completion
    projected_tardiness = max(0.0, tail_completion - due_date)
    machine_queue_delay = max(0.0, float(machine_available_time) - max(float(job_ready_time), float(route_end)))
    transport_queue_delay = max(0.0, float(route_start) - max(float(job_ready_time), float(agv_available_time)))
    machine_event_pressure = _event_proximity_pressure(
        instance,
        event_types=("machine_breakdown",),
        entity_ids=(machine_id,),
        start=int(machine_start),
        end=int(operation_completion),
    )
    path_event_pressure = _legs_proximity_pressure(
        instance,
        event_types=("path_blockage", "congestion_spike"),
        legs=_cl,
    )
    agv_event_pressure = _legs_proximity_pressure(
        instance,
        event_types=("agv_failure",),
        legs=tuple(((agv_id,), s, e) for _r, s, e in _cl),
    )
    future_risk_pressure = (
        projected_tardiness
        + machine_queue_delay
        + transport_queue_delay
        + machine_event_pressure
        + path_event_pressure
        + agv_event_pressure
    )
    row = {
        "candidate_id": f"c{candidate_uid:09d}",
        "split": split,
        "level": level,
        "seed": f"{seed:03d}",
        "mode": mode,
        "algorithm": algorithm,
        "teacher_family": teacher_family,
        "source_solution_id": source_solution_id,
        "teacher_schedule_id": teacher_schedule_id,
        "target_preference_id": target_preference_id,
        "decision_group_id": decision_group_id,
        "stage": stage,
        "step_index": step_index,
        "candidate_operation_id": operation.operation_id,
        "candidate_job_id": job.job_id,
        "candidate_machine_id": machine_id,
        "candidate_agv_id": agv_id,
        "candidate_route_edges": ";".join(route),
        "selected_action": 1 if selected_action else 0,
        "future_regret_label_available": 0.0,
        "future_makespan_after_action": 0.0,
        "future_tardiness_after_action": 0.0,
        "future_makespan_regret_to_group_best": 0.0,
        "future_tardiness_regret_to_group_best": 0.0,
        "future_weighted_completion_regret": 0.0,
        "level_l5": 1.0 if level == "L5" else 0.0,
        "mode_realized": 1.0 if mode == "realized" else 0.0,
        "seed_scaled": seed / 100.0,
        "stage_operation": 1.0 if stage == "operation" else 0.0,
        "stage_machine": 1.0 if stage == "machine" else 0.0,
        "stage_transport": 1.0 if stage == "transport" else 0.0,
        "step_fraction": step_index / max(1, total_operations - 1),
        "available_operation_count": float(available_operation_count),
        "candidate_operation_index": float(operation.op_index),
        "candidate_job_release_time": float(job.release_time),
        "candidate_job_due_date": float(job.due_date),
        "candidate_job_ready_time": float(job_ready_time),
        "candidate_job_due_slack_at_ready": float(job.due_date - job_ready_time),
        "candidate_job_remaining_ops": float(len(job.operations) - operation.op_index),
        "candidate_eligible_machine_count": float(len(operation.eligible_machines)),
        "candidate_processing_time_min": float(min(processing_times)),
        "candidate_processing_time_mean": float(_mean(tuple(float(value) for value in processing_times))),
        "candidate_processing_time_max": float(max(processing_times)),
        "candidate_machine_energy_rate": float(getattr(machine, "energy_rate", 0.0)),
        "candidate_machine_available_time": float(machine_available_time),
        "candidate_processing_time": float(processing_time),
        "candidate_processing_energy": float(processing_energy),
        "candidate_machine_start_lower_bound": float(machine_start),
        "candidate_has_transport": 1.0 if candidate_has_transport else 0.0,
        "candidate_agv_speed": float(getattr(agv, "speed", 0.0)),
        "candidate_agv_load_capacity": float(getattr(agv, "load_capacity", 0.0)),
        "candidate_agv_available_time": float(agv_available_time),
        "candidate_route_edge_count": float(route_stats["route_edge_count"]),
        "candidate_route_length": float(route_stats["route_length"]),
        "candidate_route_bottleneck_count": float(route_stats["route_bottleneck_count"]),
        "candidate_route_capacity_min": float(route_stats["route_capacity_min"]),
        "candidate_route_blockage_probability_max": float(route_stats["route_blockage_probability_max"]),
        "candidate_route_event_overlap_duration": float(route_stats["route_event_overlap_duration"]),
        "candidate_route_wait_time": float(route_wait_time),
        "candidate_route_travel_time": float(max(0, route_end - route_start)),
        "candidate_active_path_event_count": float(_active_path_event_count(instance, route, route_start, route_end)),
        "candidate_active_agv_event_count": float(_active_agv_event_count(instance, agv_id, route_start, route_end)),
        "candidate_active_machine_event_count": float(_active_machine_event_count(instance, machine_id, machine_start, machine_start + processing_time)),
        "candidate_operation_completion_lower_bound": float(operation_completion),
        "candidate_job_remaining_processing_after": float(remaining_after),
        "candidate_job_tail_completion_lower_bound": float(tail_completion),
        "candidate_job_due_slack_after_tail": float(due_slack_after_tail),
        "candidate_job_projected_tardiness_lower_bound": float(projected_tardiness),
        "candidate_job_remaining_work_ratio": float(remaining_from_current / max(1.0, total_job_min_processing)),
        "candidate_machine_queue_delay": float(machine_queue_delay),
        "candidate_transport_queue_delay": float(transport_queue_delay),
        "candidate_completion_regret_to_group_best": 0.0,
        "candidate_tail_regret_to_group_best": 0.0,
        "candidate_machine_event_proximity_pressure": float(machine_event_pressure),
        "candidate_path_event_proximity_pressure": float(path_event_pressure),
        "candidate_agv_event_proximity_pressure": float(agv_event_pressure),
        "candidate_future_risk_pressure": float(future_risk_pressure),
    }
    for index in range(5):
        row[f"target_preference_{index}"] = _preference_at(target_preference, index)
    return row


def _attach_group_critical_path_regrets(rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    operation_rows = tuple(row for row in rows if row.get("stage") == "operation")
    if not operation_rows:
        return rows
    best_completion = min(float(row.get("candidate_operation_completion_lower_bound", 0.0)) for row in operation_rows)
    best_tail = min(float(row.get("candidate_job_tail_completion_lower_bound", 0.0)) for row in operation_rows)
    patched = []
    for row in rows:
        if row.get("stage") != "operation":
            patched.append(row)
            continue
        copy = dict(row)
        copy["candidate_completion_regret_to_group_best"] = max(
            0.0,
            float(copy.get("candidate_operation_completion_lower_bound", 0.0)) - best_completion,
        )
        copy["candidate_tail_regret_to_group_best"] = max(
            0.0,
            float(copy.get("candidate_job_tail_completion_lower_bound", 0.0)) - best_tail,
        )
        patched.append(copy)
    return tuple(patched)


def _attach_future_regret_labels(
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
    scheduled_prefix: tuple[ScheduledOperation, ...],
    transport_prefix: tuple[TransportMove, ...],
) -> tuple[dict[str, object], ...]:
    operation_by_id = {operation.operation_id: (job, operation) for job, operation in available}
    outcomes = []
    for row in rows:
        if row.get("stage") != "operation":
            continue
        operation_id = str(row.get("candidate_operation_id", ""))
        if operation_id not in operation_by_id:
            continue
        job, operation = operation_by_id[operation_id]
        schedule = _complete_schedule_after_forced_operation(
            instance=instance,
            preference=preference,
            forced_job=job,
            forced_operation=operation,
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
        outcomes.append(
            {
                "operation_id": operation_id,
                "makespan": float(metrics.get("makespan", 0.0)),
                "tardiness": float(metrics.get("tardiness", 0.0)),
            }
        )
    if not outcomes:
        return rows
    outcome_by_operation = {row["operation_id"]: row for row in outcomes}
    best_makespan = min(row["makespan"] for row in outcomes)
    best_tardiness = min(row["tardiness"] for row in outcomes)
    makespan_weight = _objective_weight(instance, preference, "makespan")
    tardiness_weight = _objective_weight(instance, preference, "tardiness")
    if makespan_weight <= 0.0 and tardiness_weight <= 0.0:
        makespan_weight = 1.0
    patched = []
    for row in rows:
        operation_id = str(row.get("candidate_operation_id", ""))
        outcome = outcome_by_operation.get(operation_id)
        if outcome is None:
            patched.append(row)
            continue
        makespan_regret = max(0.0, outcome["makespan"] - best_makespan)
        tardiness_regret = max(0.0, outcome["tardiness"] - best_tardiness)
        copy = dict(row)
        copy["future_regret_label_available"] = 1.0
        copy["future_makespan_after_action"] = outcome["makespan"]
        copy["future_tardiness_after_action"] = outcome["tardiness"]
        copy["future_makespan_regret_to_group_best"] = makespan_regret
        copy["future_tardiness_regret_to_group_best"] = tardiness_regret
        copy["future_weighted_completion_regret"] = makespan_weight * makespan_regret + tardiness_weight * tardiness_regret
        patched.append(copy)
    return tuple(patched)


def _complete_schedule_after_forced_operation(
    *,
    instance: object,
    preference: tuple[float, ...],
    forced_job: object,
    forced_operation: object,
    machine_available: dict[str, int],
    job_ready: dict[str, int],
    next_op_index: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
    scheduled_prefix: tuple[ScheduledOperation, ...],
    transport_prefix: tuple[TransportMove, ...],
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

    def append_selected(selected: dict[str, object]) -> None:
        job = selected["job"]
        operation = selected["operation"]
        machine = selected["machine"]
        transport_end = local_job_ready[job.job_id]
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
            for _legs, _ls, _le in move_leg_usage(instance, transport, job_pickup_of(instance, local_agv_node, transport.job_id)):
                for edge_id in dict.fromkeys(_legs):
                    local_edge_reservations.setdefault(edge_id, []).append((_ls, _le))
            local_agv_available[transport.agv_id] = transport.end
            local_agv_node[transport.agv_id] = transport.to_node
            local_agv_node[JOB_KEY + str(transport.job_id)] = transport.to_node

        start = max(local_job_ready[job.job_id], local_machine_available[machine.machine_id], transport_end)
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
        local_machine_available[machine.machine_id] = end
        local_job_ready[job.job_id] = end
        local_next_op_index[job.job_id] += 1

    forced = _best_preference_action(
        instance=instance,
        available=((forced_job, forced_operation),),
        preference=preference,
        machine_available=local_machine_available,
        job_ready=local_job_ready,
        agv_available=local_agv_available,
        agv_node=local_agv_node,
        edge_reservations=local_edge_reservations,
    )
    append_selected(forced)
    while len(scheduled_operations) < total_operations:
        available = _available_operations(instance.jobs, local_next_op_index)
        if not available:
            raise ValueError("no schedulable operations remain during future-regret completion")
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
        append_selected(selected)
    return ScheduleResult(
        algorithm="e113-future-regret-counterfactual",
        path_aware=bool(instance.path_edges),
        operations=tuple(scheduled_operations),
        transports=tuple(transport_moves),
        event_aware=bool(instance.events),
    )


def _objective_weight(instance: object, preference: tuple[float, ...], objective: str) -> float:
    try:
        index = tuple(instance.objectives).index(objective)
    except ValueError:
        return 0.0
    return _preference_at(preference, index)


def _future_regret_step_selected(step_index: int, total_operations: int, max_groups: int) -> bool:
    if max_groups <= 0 or max_groups >= total_operations:
        return True
    if max_groups == 1:
        return step_index == 0
    selected = {
        round(index * (total_operations - 1) / (max_groups - 1))
        for index in range(max_groups)
    }
    return step_index in selected


def _transport_candidate_rows(
    *,
    instance: object,
    job: object,
    operation: object,
    machine: object,
    teacher_transport: TransportMove,
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
    earliest_job_ready: int,
) -> tuple[dict[str, object], ...]:
    candidates = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for agv in instance.agvs:
        origin = agv_node[agv.agv_id]
        pickup = job_pickup_node(instance, agv_node, job.job_id)
        routes = two_leg_routes(
            instance.path_edges,
            source=origin,
            pickup=pickup,
            target=machine.node_id,
            avoid_edges=_active_event_edges_after(instance, earliest=max(earliest_job_ready, agv_available[agv.agv_id])),
        )
        if agv.agv_id == teacher_transport.agv_id:
            routes = (teacher_transport.edge_ids,) + tuple(route for route in routes if route != teacher_transport.edge_ids)
        for route in routes:
            key = (agv.agv_id, route)
            if key in seen:
                continue
            seen.add(key)
            route_length = _route_length(instance, route)
            _tl = two_leg_early_schedule(instance, agv, origin, pickup, route, agv_available[agv.agv_id],
                                         earliest_job_ready, edge_reservations)
            start = _tl["start"]
            travel_time = _tl["end"] - _tl["start"]
            earliest = _tl["start"] - _tl["delay"]
            candidates.append(
                {
                    "agv": agv,
                    "route": route,
                    "start": start,
                    "end": start + travel_time,
                    "wait_time": float(start - earliest),
                    "route_legs": _legs_of(route, _tl),
                    "selected": agv.agv_id == teacher_transport.agv_id and route == teacher_transport.edge_ids,
                }
            )
    if not any(bool(row["selected"]) for row in candidates):
        agv = _agv_by_id(instance, teacher_transport.agv_id)
        candidates.append(
            {
                "agv": agv,
                "route": teacher_transport.edge_ids,
                "start": teacher_transport.start,
                "end": teacher_transport.end,
                "wait_time": float(max(0, teacher_transport.start - earliest_job_ready)),
                "selected": True,
            }
        )
    selected_seen = False
    fixed = []
    for candidate in candidates:
        selected = bool(candidate["selected"]) and not selected_seen
        selected_seen = selected_seen or selected
        fixed.append({**candidate, "selected": selected})
    return tuple(fixed)


def _candidate_route_feature_values(
    instance: object,
    route: tuple[str, ...],
    agv_id: str,
    start: int,
    end: int,
    wait_time: float,
) -> dict[str, float]:
    stats = _route_stats(instance, route, start, end)
    return {
        "candidate_route_edge_count": float(stats["route_edge_count"]),
        "candidate_route_length": float(stats["route_length"]),
        "candidate_route_bottleneck_count": float(stats["route_bottleneck_count"]),
        "candidate_route_capacity_min": float(stats["route_capacity_min"]),
        "candidate_route_blockage_probability_max": float(stats["route_blockage_probability_max"]),
        "candidate_route_event_overlap_duration": float(stats["route_event_overlap_duration"]),
        "candidate_route_wait_time": float(wait_time),
        "candidate_route_travel_time": float(max(0, end - start)),
        "candidate_active_path_event_count": float(_active_path_event_count(instance, route, start, end)),
        "candidate_active_agv_event_count": float(_active_agv_event_count(instance, agv_id, start, end)),
    }


def _best_preference_action(
    *,
    instance: object,
    available: tuple[tuple[object, object], ...],
    preference: tuple[float, ...],
    machine_available: dict[str, int],
    job_ready: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
) -> dict[str, object]:
    machine_by_id = {machine.machine_id: machine for machine in instance.machines}
    candidates = []
    for job, operation in available:
        for machine_id in operation.eligible_machines:
            machine = machine_by_id[machine_id]
            transports = _preference_transport_candidates(
                instance=instance,
                job=job,
                machine=machine,
                agv_available=agv_available,
                agv_node=agv_node,
                edge_reservations=edge_reservations,
                earliest_job_ready=job_ready[job.job_id],
            )
            for transport in transports:
                transport_end = transport["end"] if transport is not None else job_ready[job.job_id]
                start = max(job_ready[job.job_id], machine_available[machine_id], int(transport_end))
                start = _first_resource_event_feasible_start(
                    start,
                    operation.processing_time[machine_id],
                    machine_id,
                    "machine_breakdown",
                    instance,
                )
                end = start + operation.processing_time[machine_id]
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
                        "job": job,
                        "operation": operation,
                        "machine": machine,
                        "transport": transport,
                    }
                )
    if not candidates:
        raise ValueError("no preference action candidate")
    return min(
        candidates,
        key=lambda item: (
            item["score"],
            item["operation"].operation_id,
            item["machine"].machine_id,
            "" if item["transport"] is None else item["transport"]["agv_id"],
            () if item["transport"] is None else tuple(item["transport"]["route"]),
        ),
    )


def _preference_transport_candidates(
    *,
    instance: object,
    job: object,
    machine: object,
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
    earliest_job_ready: int,
) -> tuple[dict[str, object] | None, ...]:
    if not instance.agvs or not instance.path_edges:
        return (None,)
    candidates = []
    for agv in instance.agvs:
        origin = agv_node[agv.agv_id]
        pickup = job_pickup_node(instance, agv_node, job.job_id)
        routes = two_leg_routes(
            instance.path_edges,
            source=origin,
            pickup=pickup,
            target=machine.node_id,
            avoid_edges=_active_event_edges_after(instance, earliest=max(earliest_job_ready, agv_available[agv.agv_id])),
        )
        if not routes:
            safe = two_leg_safe(instance.path_edges, origin, pickup, machine.node_id)
            routes = (safe,) if safe is not None else ((),)
        for route in routes:
            route_length = _route_length(instance, route)
            _tl = two_leg_early_schedule(instance, agv, origin, pickup, route, agv_available[agv.agv_id],
                                         earliest_job_ready, edge_reservations)
            start = _tl["start"]
            travel_time = _tl["end"] - _tl["start"]
            earliest = _tl["start"] - _tl["delay"]
            candidates.append(
                {
                    "agv_id": agv.agv_id,
                    "from_node": origin,
                    "route": route,
                    "start": start,
                    "end": start + travel_time,
                    "wait": start - earliest,
                    "travel_time": travel_time,
                    "route_length": route_length,
                    "transport_energy": round(route_length * agv.consumption_rate, 6),
                    "empty_end": _tl["empty_end"], "loaded_start": _tl["loaded_start"], "n_empty": _tl["n_empty"], "route_legs": _legs_of(route, _tl),
                }
            )
    return tuple(candidates) if candidates else (None,)


def _preference_proxy_score(
    *,
    instance: object,
    preference: tuple[float, ...],
    job: object,
    operation: object,
    machine: object,
    route: tuple[str, ...],
    operation_start: int,
    operation_end: int,
    transport: dict[str, object] | None,
) -> float:
    weights = {
        objective: _preference_at(preference, index)
        for index, objective in enumerate(instance.objectives)
    }
    processing = operation.processing_time[machine.machine_id]
    energy = operation.processing_energy[machine.machine_id]
    transport_energy = float(transport["transport_energy"]) if transport is not None else 0.0
    route_length = _route_length(instance, route)
    transport_start = int(transport["start"]) if transport is not None else operation_start
    transport_end = int(transport["end"]) if transport is not None else operation_start
    agv_id = str(transport["agv_id"]) if transport is not None else ""
    import os
    _leg = transport is not None and "loaded_start" in transport
    if _leg:  # diagnosis only (default off): event terms over the two legs, not the wait at the pickup node
        _ne = int(transport["n_empty"]); _legs = ((tuple(route[:_ne]), transport_start, int(transport["empty_end"])),
                                                  (tuple(route[_ne:]), int(transport["loaded_start"]), transport_end))
        route_event = _legs_overlap_duration(instance, _legs, agv_id)
    else:
        route_event = _route_event_overlap_duration(
            instance,
            route,
            agv_id,
            transport_start,
            transport_end,
        )
    remaining_processing = _remaining_min_processing(job, operation.op_index + 1)
    projected_job_completion = operation_end + remaining_processing
    projected_tardiness = max(0.0, projected_job_completion - float(job.due_date))
    tardiness_pressure = (
        operation_end
        + 0.20 * remaining_processing
        + 0.001 * projected_tardiness
    )
    path_pressure = (
        25.0 * _route_bottleneck_count(instance, route)
        + 12.0 * route_event
        + 20.0 * _route_blockage_risk(instance, route)
        + 8.0 * _route_capacity_risk(instance, route)
        + 0.05 * route_length
    )
    stability_pressure = _event_proximity_pressure(
        instance,
        event_types=("machine_breakdown",),
        entity_ids=(machine.machine_id,),
        start=operation_start,
        end=operation_end,
    )
    if transport is not None:
        _pl = _legs if _leg else ((route, transport_start, transport_end),)
        stability_pressure += _legs_proximity_pressure(
            instance,
            event_types=("agv_failure",),
            legs=tuple(((agv_id,), s, e) for _r, s, e in _pl),
        )
        stability_pressure += _legs_proximity_pressure(
            instance,
            event_types=("path_blockage", "congestion_spike"),
            legs=_pl,
        )
    score = (
        weights.get("makespan", 0.0) * (operation_end + 0.20 * remaining_processing)
        + weights.get("tardiness", 0.0) * tardiness_pressure
        + weights.get("energy", 0.0) * (energy + transport_energy)
        + weights.get("machine_idle_time", 0.0) * (operation_start + processing)
        + weights.get("path_conflict_delay", 0.0) * (path_pressure + route_length * 0.01)
        + weights.get("stability", 0.0) * stability_pressure
    )
    return score


def _decision_feature_values(
    *,
    instance: object,
    level: str,
    seed: int,
    mode: str,
    step_index: int,
    total_operations: int,
    available_count: int,
    job: object,
    operation: object,
    scheduled: ScheduledOperation,
    machine: object,
    transport: TransportMove | None,
    route_stats: dict[str, float],
) -> dict[str, object]:
    processing_values = tuple(operation.processing_time[machine_id] for machine_id in operation.eligible_machines)
    transport_start = transport.start if transport else scheduled.start
    transport_end = transport.end if transport else scheduled.start
    agv = _agv_by_id(instance, transport.agv_id) if transport else None
    return {
        "level_l5": 1.0 if level == "L5" else 0.0,
        "mode_realized": 1.0 if mode == "realized" else 0.0,
        "seed_scaled": seed / 100.0,
        "step_fraction": step_index / max(1, total_operations - 1),
        "available_operation_count": float(available_count),
        "job_release_time": float(job.release_time),
        "job_due_date": float(job.due_date),
        "job_due_slack_at_op_start": float(job.due_date - scheduled.start),
        "operation_index": float(operation.op_index),
        "job_operation_count": float(len(job.operations)),
        "operation_progress": operation.op_index / max(1, len(job.operations) - 1),
        "eligible_machine_count": float(len(operation.eligible_machines)),
        "processing_time_min": float(min(processing_values)),
        "processing_time_mean": float(_mean(tuple(float(value) for value in processing_values))),
        "processing_time_max": float(max(processing_values)),
        "chosen_processing_time": float(operation.processing_time[scheduled.machine_id]),
        "chosen_processing_energy": float(operation.processing_energy[scheduled.machine_id]),
        "chosen_machine_energy_rate": float(machine.energy_rate),
        "chosen_machine_nominal_capacity": 1.0,
        "machine_start": float(scheduled.start),
        "machine_wait_since_transport": float(max(0, scheduled.start - transport_end)),
        "has_transport": 1.0 if transport else 0.0,
        "chosen_agv_capacity": float(getattr(agv, "load_capacity", 0.0)),
        "chosen_agv_speed": float(getattr(agv, "speed", 0.0)),
        "transport_start": float(transport_start),
        "transport_wait_since_release": float(max(0, transport_start - job.release_time)) if transport else 0.0,
        "transport_travel_time": float(max(0, transport_end - transport_start)) if transport else 0.0,
        "route_edge_count": float(route_stats["route_edge_count"]),
        "route_length": float(route_stats["route_length"]),
        "route_bottleneck_count": float(route_stats["route_bottleneck_count"]),
        "route_capacity_min": float(route_stats["route_capacity_min"]),
        "route_blockage_probability_max": float(route_stats["route_blockage_probability_max"]),
        "route_event_overlap_duration": float(route_stats["route_event_overlap_duration"]),
        "route_uses_bottleneck": 1.0 if route_stats["route_bottleneck_count"] > 0 else 0.0,
        "active_path_event_count": float(_active_path_event_count(instance, transport.edge_ids if transport else (), transport_start, transport_end)),
        "active_agv_event_count": float(_active_agv_event_count(instance, transport.agv_id if transport else "", transport_start, transport_end)),
        "active_machine_event_count": float(_active_machine_event_count(instance, scheduled.machine_id, scheduled.start, scheduled.end)),
    }


def _route_stats(instance: object, route: tuple[str, ...], start: int, end: int) -> dict[str, float]:
    edge_by_id = {edge.edge_id: edge for edge in instance.path_edges}
    edges = tuple(edge_by_id[edge_id] for edge_id in route if edge_id in edge_by_id)
    return {
        "route_edge_count": float(len(route)),
        "route_length": float(sum(edge.length for edge in edges)),
        "route_bottleneck_count": float(sum(1 for edge in edges if bool(edge.is_bottleneck))),
        "route_capacity_min": float(min((edge.capacity for edge in edges), default=0)),
        "route_blockage_probability_max": float(max((edge.blockage_probability for edge in edges), default=0.0)),
        "route_event_overlap_duration": float(_route_event_overlap_duration(instance, route, "", start, end)),
    }


def _route_length(instance: object, route: tuple[str, ...]) -> float:
    edge_by_id = {edge.edge_id: edge for edge in instance.path_edges}
    return float(sum(edge_by_id[edge_id].length for edge_id in route if edge_id in edge_by_id))


def _route_bottleneck_count(instance: object, route: tuple[str, ...]) -> int:
    edge_by_id = {edge.edge_id: edge for edge in instance.path_edges}
    return sum(1 for edge_id in route if edge_id in edge_by_id and bool(edge_by_id[edge_id].is_bottleneck))


def _route_blockage_risk(instance: object, route: tuple[str, ...]) -> float:
    edge_by_id = {edge.edge_id: edge for edge in instance.path_edges}
    return sum(
        float(edge_by_id[edge_id].blockage_probability) * max(1.0, float(edge_by_id[edge_id].length))
        for edge_id in route
        if edge_id in edge_by_id
    )


def _route_capacity_risk(instance: object, route: tuple[str, ...]) -> float:
    edge_by_id = {edge.edge_id: edge for edge in instance.path_edges}
    return sum(
        1.0 / max(1, int(edge_by_id[edge_id].capacity))
        for edge_id in route
        if edge_id in edge_by_id
    )


def _active_event_edges_after(instance: object, earliest: int) -> frozenset[str]:
    edges = set()
    for event in instance.events:
        if event.event_type not in {"path_blockage", "congestion_spike"} or event.duration <= 0:
            continue
        if event.event_time + event.duration >= earliest:
            edges.add(event.affected_entity)
    return frozenset(edges)


def _route_family(instance: object, transport: TransportMove | None) -> str:
    if transport is None:
        return "no-transport"
    shortest = _safe_shortest_path(instance.path_edges, transport.from_node, transport.to_node)
    if transport.edge_ids == shortest:
        return "shortest"
    stats = _route_stats(instance, transport.edge_ids, transport.start, transport.end)
    if stats["route_bottleneck_count"] <= 0:
        return "bottleneck-avoid"
    if stats["route_event_overlap_duration"] <= 0:
        return "event-avoid"
    return "anchored-other"


def _route_event_overlap_duration(instance: object, route: tuple[str, ...], agv_id: str, start: int, end: int) -> int:
    overlap = 0
    for event in instance.events:
        if event.duration <= 0:
            continue
        event_start = int(event.event_time)
        event_end = int(event.event_time + event.duration)
        if not _overlaps(start, end, event_start, event_end):
            continue
        if event.event_type == "agv_failure" and event.affected_entity == agv_id:
            overlap += event_end - event_start
        if event.event_type in {"path_blockage", "congestion_spike"} and event.affected_entity in route:
            overlap += event_end - event_start
    return overlap


def _active_path_event_count(instance: object, route: tuple[str, ...], start: int, end: int) -> int:
    return sum(
        1
        for event in instance.events
        if event.event_type in {"path_blockage", "congestion_spike"}
        and event.affected_entity in route
        and _overlaps(start, end, event.event_time, event.event_time + event.duration)
    )


def _active_agv_event_count(instance: object, agv_id: str, start: int, end: int) -> int:
    return sum(
        1
        for event in instance.events
        if event.event_type == "agv_failure"
        and event.affected_entity == agv_id
        and _overlaps(start, end, event.event_time, event.event_time + event.duration)
    )


def _active_machine_event_count(instance: object, machine_id: str, start: int, end: int) -> int:
    return sum(
        1
        for event in instance.events
        if event.event_type == "machine_breakdown"
        and event.affected_entity == machine_id
        and _overlaps(start, end, event.event_time, event.event_time + event.duration)
    )


def _event_proximity_pressure(
    instance: object,
    *,
    event_types: tuple[str, ...],
    entity_ids: tuple[str, ...],
    start: int,
    end: int,
    horizon: int = 40,
) -> float:
    entities = set(entity_ids)
    pressure = 0.0
    for event in instance.events:
        if event.event_type not in event_types or event.affected_entity not in entities or event.duration <= 0:
            continue
        event_start = int(event.event_time)
        event_end = int(event.event_time + event.duration)
        if _overlaps(start, end, event_start, event_end):
            distance = 0
        elif end <= event_start:
            distance = event_start - end
        else:
            distance = start - event_end
        if distance <= horizon:
            proximity = (horizon - max(0, distance)) / max(1, horizon)
            pressure += proximity * max(1.0, float(event.duration)) * max(0.1, float(event.severity))
    return pressure


def _remaining_min_processing(job: object, start_index: int) -> float:
    if job is None:
        return 0.0
    return float(
        sum(
            min(operation.processing_time.values())
            for operation in job.operations[start_index:]
            if operation.processing_time
        )
    )


def _machine_by_id(instance: object, machine_id: str) -> object:
    return next(machine for machine in instance.machines if machine.machine_id == machine_id)


def _agv_by_id(instance: object, agv_id: str) -> object:
    return next(agv for agv in instance.agvs if agv.agv_id == agv_id)


def _overlaps(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    if right_start >= right_end:
        return False
    return left_start < right_end and right_start < left_end


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def _include_context(level: str, seed: int, mode: str, config: PathConflictNeuralDatasetConfig) -> bool:
    if level not in config.levels:
        return False
    if mode not in config.modes:
        return False
    if config.start_seed is not None and seed < config.start_seed:
        return False
    if config.end_seed is not None and seed > config.end_seed:
        return False
    if mode == "realized" and not config.include_realized:
        return False
    return True


def _target_preference_proxy(
    objective_vector: tuple[float, ...],
    context_vectors: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
    if not objective_vector:
        return ()
    if not context_vectors:
        return tuple(1.0 / len(objective_vector) for _ in objective_vector)
    columns = tuple(zip(*context_vectors))
    scores = []
    for index, value in enumerate(objective_vector):
        column = columns[index] if index < len(columns) else (value,)
        low = min(column)
        high = max(column)
        normalized = 0.5 if high <= low else (value - low) / (high - low)
        scores.append(1.0 - normalized + 1.0e-6)
    total = sum(scores)
    return tuple(score / total for score in scores)


def _explicit_teacher_preferences(n_objectives: int, *, max_preferences: int) -> tuple[tuple[float, ...], ...]:
    if n_objectives <= 0 or max_preferences <= 0:
        return ()
    preferences: list[tuple[float, ...]] = [tuple(1.0 / n_objectives for _ in range(n_objectives))]
    for index in range(n_objectives):
        preferences.append(tuple(1.0 if item == index else 0.0 for item in range(n_objectives)))
    for left in range(n_objectives):
        for right in range(left + 1, n_objectives):
            preferences.append(
                tuple(
                    0.5 if item in {left, right} else 0.0
                    for item in range(n_objectives)
                )
            )
    near_anchor_order = tuple(
        index
        for index in (3, 4, *range(n_objectives))
        if 0 <= index < n_objectives
    )
    seen_anchors = set()
    for anchor in near_anchor_order:
        if anchor in seen_anchors:
            continue
        seen_anchors.add(anchor)
        for other in range(n_objectives):
            if other == anchor:
                continue
            preferences.append(
                tuple(
                    0.8 if item == anchor else 0.2 if item == other else 0.0
                    for item in range(n_objectives)
                )
            )
    if n_objectives >= 5:
        preferences.extend(
            (
                (0.25, 0.25, 0.0, 0.5, 0.0),
                (0.20, 0.20, 0.0, 0.30, 0.30),
                (0.0, 0.35, 0.0, 0.35, 0.30),
            )
        )
    deduped = []
    seen = set()
    for preference in preferences:
        normalized = _normalize_preference(preference[:n_objectives])
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
        if len(deduped) >= max_preferences:
            break
    return tuple(deduped)


def _normalize_preference(preference: tuple[float, ...]) -> tuple[float, ...]:
    total = sum(max(0.0, float(value)) for value in preference)
    if total <= 1.0e-9:
        return tuple(1.0 / len(preference) for _ in preference)
    return tuple(max(0.0, float(value)) / total for value in preference)


def _explicit_preference_id(index: int, preference: tuple[float, ...]) -> str:
    values = "-".join(f"{value:.2f}" for value in preference)
    return f"pref{index:02d}-{values}"


def _preference_at(preference: tuple[float, ...], index: int) -> float:
    return float(preference[index]) if index < len(preference) else 0.0


def _teacher_schedule_id(level: str, seed: int, mode: str, source_solution_id: str, schedule_index: int) -> str:
    return f"{level}-{seed:03d}-{mode}-{source_solution_id}-{schedule_index:03d}"


def _int(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _split(seed: int, config: PathConflictNeuralDatasetConfig) -> str:
    return "test" if config.test_seed is not None and seed >= config.test_seed else "train"


def _min_seed(rows: tuple[dict[str, str], ...]) -> int:
    seeds = [_int(row.get("seed", "")) for row in rows if row.get("seed", "")]
    return min(seeds) if seeds else 1


def _max_seed(rows: tuple[dict[str, str], ...]) -> int:
    seeds = [_int(row.get("seed", "")) for row in rows if row.get("seed", "")]
    return max(seeds) if seeds else 1


def _machine_index(instance: object, machine_id: str) -> int:
    for index, machine in enumerate(instance.machines):
        if machine.machine_id == machine_id:
            return index
    return 0


def _agv_index(instance: object, agv_id: str) -> int:
    for index, agv in enumerate(instance.agvs):
        if agv.agv_id == agv_id:
            return index
    return -1


def _objective_audit(metrics: dict[str, float]) -> dict[str, object]:
    return {
        "objective_makespan": metrics.get("makespan", 0.0),
        "objective_tardiness": metrics.get("tardiness", 0.0),
        "objective_energy": metrics.get("energy", 0.0),
        "objective_path_conflict_delay": metrics.get("path_conflict_delay", 0.0),
        "objective_stability": metrics.get("stability", 0.0),
    }


def _format_vector(vector: tuple[float, ...]) -> str:
    return ";".join(f"{value:.10f}" for value in vector)


def _ordered_row(row: dict[str, object]) -> dict[str, object]:
    return {field: row.get(field, "") for field in DATASET_FIELDS}


def _ordered_candidate_row(row: dict[str, object]) -> dict[str, object]:
    return {field: row.get(field, "") for field in CANDIDATE_FIELDS}


# ---- leg-aware event terms (2026-11-09, 改正后复核) ----
def _legs_overlap_duration(instance, legs, agv_id):
    """Like _route_event_overlap_duration, over legs ((route, start, end), ...); an event counts once."""
    overlap = 0
    for event in instance.events:
        if event.duration <= 0:
            continue
        event_start = int(event.event_time)
        event_end = int(event.event_time + event.duration)
        hit = False
        for route, start, end in legs:
            if end <= start or not _overlaps(start, end, event_start, event_end):
                continue
            if event.event_type == "agv_failure" and event.affected_entity == agv_id:
                hit = True
            if event.event_type in {"path_blockage", "congestion_spike"} and event.affected_entity in route:
                hit = True
        if hit:
            overlap += event_end - event_start
    return overlap


def _legs_proximity_pressure(instance, *, event_types, legs, horizon=40):
    """Like _event_proximity_pressure, over legs ((entity_ids, start, end), ...); distance to the nearest leg."""
    pressure = 0.0
    for event in instance.events:
        if event.event_type not in event_types or event.duration <= 0:
            continue
        event_start = int(event.event_time)
        event_end = int(event.event_time + event.duration)
        best = None
        for entity_ids, start, end in legs:
            if end <= start or event.affected_entity not in set(entity_ids):
                continue
            if _overlaps(start, end, event_start, event_end):
                distance = 0
            elif end <= event_start:
                distance = event_start - end
            else:
                distance = start - event_end
            best = distance if best is None else min(best, distance)
        if best is not None and best <= horizon:
            proximity = (horizon - max(0, best)) / max(1, horizon)
            pressure += proximity * max(1.0, float(event.duration)) * max(0.1, float(event.severity))
    return pressure


def _schedule_move_legs(instance, schedule):
    from su_modfjsp_ha_pc.metrics import _early_legs

    return {move.move_id: legs for move, legs in _early_legs(instance, schedule)}



# ---- rerun A1 helpers ----
def _legs_of(route, tl):
    n = int(tl["n_empty"])
    return ((tuple(route[:n]), int(tl["start"]), int(tl["empty_end"])), (tuple(route[n:]), int(tl["loaded_start"]), int(tl["end"])))


def _teacher_legs(instance, candidate):
    """The teacher's own move appended without re-timing: recover its legs from the committed timing."""
    raise RuntimeError("rerun A1: teacher-transport candidate without legs (not expected on the label path)")


# ---- exact speed-ups (2026-11-12, 全部重跑规则_冻结v3 6b: implementation only; parity in v3_step0/fast_parity.py) ----
# V3_FAST_PROJ=0 restores the reference functions above. The fast versions return identical results:
# (1) the projection score uses only route_bottleneck_count from _route_stats (its overlap term is overwritten), so only that
#     count is computed; (2) a route whose end cannot be below the best end so far is skipped before scheduling it, because
#     conflicts only delay a leg, so end >= max(agv_free + te, job_ready) + tl; (3) the tie-break terms are computed only when
#     (end, wait) does not already lose; (4) in the stage lookahead the candidates are compared by the transport score (the
#     projection's "score" key overrides the machine score in the merged dict), so the best transport score so far is passed
#     to the next machine's projection as a bound and routes that cannot beat it strictly are skipped. Ties keep the first
#     candidate, as min and the strict < do. (5) two_leg_routes is cached per path graph and arguments.
import os as _os_fast  # noqa: E402

from su_modfjsp_ha_pc.rules import early_leg_times as _early_leg_times, _memo_table as _memo_table_fast  # noqa: E402

_FAST_PROJ = _os_fast.environ.get("V3_FAST_PROJ", "1") != "0"
_best_transport_projection_ref = _best_transport_projection
_operation_stage_lookahead_ref = _operation_stage_lookahead


def _bottleneck_count_fast(instance, route):
    path_edges = instance.path_edges
    if not isinstance(path_edges, tuple):
        return _route_stats(instance, route, 0, 0)["route_bottleneck_count"]
    table = _memo_table_fast(path_edges, "bnk")
    key = tuple(route)
    hit = table.get(key)
    if hit is None:
        edge_by_id = {edge.edge_id: edge for edge in path_edges}
        hit = table[key] = float(sum(1 for e in route if e in edge_by_id and bool(edge_by_id[e].is_bottleneck)))
    return hit


def _two_leg_routes_memo(instance, origin, pickup, target, avoid):
    """two_leg_routes with the two_leg_safe fallback, as in the reference; a pure function of the path graph and its arguments."""
    path_edges = instance.path_edges
    table = _memo_table_fast(path_edges, "tlr") if isinstance(path_edges, tuple) else {}
    key = (origin, pickup, target, avoid)
    hit = table.get(key)
    if hit is None:
        routes = two_leg_routes(path_edges, source=origin, pickup=pickup, target=target, avoid_edges=avoid)
        if not routes:
            safe = two_leg_safe(path_edges, origin, pickup, target)
            routes = (safe,) if safe is not None else ((),)
        hit = table[key] = routes
    return hit


def _best_transport_projection_fast(*, instance, job, machine, agv_available, agv_node, edge_reservations, earliest_job_ready, bound=None):
    """bound given: the best candidate whose score is strictly below bound, or None."""
    if not instance.agvs or not instance.path_edges:
        return _best_transport_projection_ref(instance=instance, job=job, machine=machine, agv_available=agv_available,
                                              agv_node=agv_node, edge_reservations=edge_reservations, earliest_job_ready=earliest_job_ready)
    best_score = bound
    best = None
    for agv in instance.agvs:
        origin = agv_node[agv.agv_id]
        pickup = job_pickup_node(instance, agv_node, job.job_id)
        free = agv_available[agv.agv_id]
        routes = _two_leg_routes_memo(instance, origin, pickup, machine.node_id,
                                      _active_event_edges_after(instance, earliest=max(earliest_job_ready, free)))
        for route in routes:
            _n, te, tl = _early_leg_times(instance, agv, origin, pickup, tuple(route))
            if best_score is not None and max(free + te, earliest_job_ready) + tl > best_score[0]:
                continue
            _tl = two_leg_early_schedule(instance, agv, origin, pickup, route, free, earliest_job_ready, edge_reservations)
            start = _tl["start"]
            travel_time = _tl["end"] - _tl["start"]
            earliest = _tl["start"] - _tl["delay"]
            end = start + travel_time
            if best_score is not None and (end, start - earliest) > best_score[:2]:
                continue
            route_length = _route_length(instance, route)
            _pj = ((tuple(route[:_tl["n_empty"]]), _tl["start"], _tl["empty_end"]), (tuple(route[_tl["n_empty"]:]), _tl["loaded_start"], _tl["end"]))
            score = (
                end,
                start - earliest,
                float(_legs_overlap_duration(instance, _pj, "")),
                _bottleneck_count_fast(instance, route),
                route_length,
                agv.agv_id,
                ";".join(route),
            )
            if best_score is None or score < best_score:
                best_score = score
                best = {
                    "score": score,
                    "has_transport": True,
                    "agv": agv,
                    "agv_available_time": free,
                    "route": route,
                    "route_start": start,
                    "route_end": end,
                    "route_wait_time": float(start - earliest),
                    "route_length": route_length,
                    "travel_time": travel_time,
                    "transport_energy": round(route_length * agv.consumption_rate, 6),
                    "route_legs": _legs_of(route, _tl),
                }
    if best is None and bound is None:
        raise ValueError(f"no transport candidate to {machine.machine_id}")
    return best


def _operation_stage_lookahead_fast(*, instance, job, operation, machine_available, job_ready_time, agv_available, agv_node, edge_reservations):
    if not instance.agvs or not instance.path_edges:
        return _operation_stage_lookahead_ref(instance=instance, job=job, operation=operation, machine_available=machine_available,
                                              job_ready_time=job_ready_time, agv_available=agv_available, agv_node=agv_node,
                                              edge_reservations=edge_reservations)
    machine_by_id = {machine.machine_id: machine for machine in instance.machines}
    best = None
    for machine_id in operation.eligible_machines:
        machine = machine_by_id[machine_id]
        p = operation.processing_time[machine_id]
        projection = _best_transport_projection_fast(
            instance=instance,
            job=job,
            machine=machine,
            agv_available=agv_available,
            agv_node=agv_node,
            edge_reservations=edge_reservations,
            earliest_job_ready=job_ready_time,
            bound=None if best is None else best["score"],
        )
        if projection is None:
            continue
        transport_end = projection.get("route_end", job_ready_time)
        start = max(job_ready_time, machine_available[machine_id], int(transport_end))
        completion = start + p
        score = (completion, p, operation.processing_energy[machine_id], machine_id)
        candidate = {"score": score, "machine": machine, **projection}
        if best is None or candidate["score"] < best["score"]:
            best = candidate
    if best is None:
        raise ValueError(f"operation {operation.operation_id} has no eligible machines")
    return best


def _best_transport_projection(**kw):
    return (_best_transport_projection_fast if _FAST_PROJ else _best_transport_projection_ref)(**kw)


def _operation_stage_lookahead(**kw):
    return (_operation_stage_lookahead_fast if _FAST_PROJ else _operation_stage_lookahead_ref)(**kw)
