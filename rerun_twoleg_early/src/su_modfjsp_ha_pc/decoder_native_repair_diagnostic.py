from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from su_modfjsp_ha_pc.generator import generate_instance
from su_modfjsp_ha_pc.metrics import evaluate_schedule
from su_modfjsp_ha_pc.moea import MoeaConfig, _Genome, _evaluate_genome, _ordered_operations, run_moea_baseline
from su_modfjsp_ha_pc.repair_inheritance_diagnostic import (
    _format_vector,
    _relation,
)
from su_modfjsp_ha_pc.rules import _available_operations, _first_resource_event_feasible_start, _schedule_transport
from su_modfjsp_ha_pc.schedule import ScheduleResult, ScheduledOperation
from su_modfjsp_ha_pc.seeded_moea import (
    _apply_seed_teacher_mask,
    _schedule_to_seed_genes,
    _seed_initial_quota_for_mode,
    _seed_preferences_by_context,
    _trained_seed_hints_by_context,
    read_seed_rows,
)
from su_modfjsp_ha_pc.structural import realize_structural_events
from su_modfjsp_ha_pc.trace_lns import (
    TraceLnsConfig,
    apply_trace_lns_repair_with_neighborhood,
    select_trace_lns_neighborhood,
)


DIAGNOSTIC_FIELDS = (
    "algorithm",
    "level",
    "seed",
    "mode",
    "base_solution_id",
    "method",
    "base_objective_vector",
    "candidate_objective_vector",
    "current_projection_objective_vector",
    "native_projection_objective_vector",
    "candidate_relation_to_base",
    "current_projection_relation_to_candidate",
    "native_projection_relation_to_candidate",
    "native_projection_relation_to_current",
    "current_projection_preserved_candidate",
    "native_projection_preserved_candidate",
    "current_projection_lost_candidate",
    "native_projection_lost_candidate",
    "native_gain_over_current_projection",
    "base_total_delay",
    "candidate_total_delay",
    "current_projection_total_delay",
    "native_projection_total_delay",
    "candidate_delay_delta",
    "current_projection_delay_delta_to_candidate",
    "native_projection_delay_delta_to_candidate",
    "same_machine_assignment_rate",
    "same_global_order_rate",
    "same_machine_sequence_rate",
    "mean_start_abs_delta",
    "trace_neighborhood_size",
    "candidate_neighborhood_size",
)

SUMMARY_FIELDS = (
    "algorithm",
    "level",
    "mode",
    "method",
    "trials",
    "phenotype_success_trials",
    "phenotype_success_rate",
    "current_projection_preserve_rate",
    "native_projection_preserve_rate",
    "current_projection_loss_rate",
    "native_projection_loss_rate",
    "success_current_projection_preserve_rate",
    "success_native_projection_preserve_rate",
    "success_current_projection_loss_rate",
    "success_native_projection_loss_rate",
    "native_gain_over_current_projection_rate",
    "success_native_gain_over_current_projection_rate",
    "mean_current_projection_delay_delta_to_candidate",
    "mean_native_projection_delay_delta_to_candidate",
    "mean_same_machine_assignment_rate",
    "mean_same_global_order_rate",
    "mean_same_machine_sequence_rate",
    "mean_start_abs_delta",
)


@dataclass(frozen=True)
class DecoderNativeGenome:
    operation_priority: tuple[float, ...]
    operation_machine_choice: tuple[float, ...]
    operation_start_anchor: tuple[float, ...] = ()


@dataclass(frozen=True)
class DecoderNativeConfig:
    size: str = "small"
    levels: tuple[str, ...] = ("L4", "L5")
    modes: tuple[str, ...] = ("raw", "realized")
    start_seed: int | None = None
    end_seed: int | None = None
    algorithms: tuple[str, ...] = ("nsga3",)
    methods: tuple[str, ...] = ("delay-trace-lns", "delay-topk-lns")
    max_base_rows_per_context: int = 48
    population_size: int = 48
    max_evaluations: int = 384
    moea_seed: int = 96
    random_seed: int = 9600
    operator_trials: int = 0
    include_realized: bool = True
    max_seed_preferences: int = 5
    max_seed_hints: int = 5
    seed_initial_quota_raw: int = 24
    seed_initial_quota_realized: int = 8
    seed_neighbor_count: int = 1
    seed_teacher_mask: str = "full"
    top_records: int = 8
    neighbor_radius: int = 1
    priority_boost: float = 3.0
    process_weight: float = 0.5


def project_decoder_native_genome(instance: object, operations: tuple, schedule: ScheduleResult) -> DecoderNativeGenome:
    operation_count = len(operations)
    operation_rank = {
        operation.operation_id: index
        for index, operation in enumerate(sorted(schedule.operations, key=lambda item: (item.start, item.end, item.operation_id)))
    }
    scheduled_machine = {operation.operation_id: operation.machine_id for operation in schedule.operations}
    priority = tuple(
        operation_rank.get(operation.operation_id, index) / max(1, operation_count - 1)
        for index, operation in enumerate(operations)
    )
    machine_choice = []
    for operation in operations:
        machine_id = scheduled_machine.get(operation.operation_id, operation.eligible_machines[0])
        try:
            machine_index = operation.eligible_machines.index(machine_id)
        except ValueError:
            machine_index = 0
        machine_choice.append(machine_index / max(1, len(operation.eligible_machines) - 1))
    start_anchor = tuple(
        float(next((item.start for item in schedule.operations if item.operation_id == operation.operation_id), 0.0))
        for operation in operations
    )
    return DecoderNativeGenome(priority, tuple(machine_choice), start_anchor)


def decode_decoder_native_genome(
    instance: object,
    operations: tuple,
    genome: DecoderNativeGenome,
    *,
    algorithm: str,
    path_aware: bool,
    event_aware: bool,
    repair_process_log: list[dict[str, object]] | None = None,
    repair_process_context: dict[str, object] | None = None,
) -> ScheduleResult:
    operation_index = {operation.operation_id: index for index, operation in enumerate(operations)}

    def select_operation(available):
        return min(
            available,
            key=lambda item: (
                _operation_priority(genome, operation_index[item[1].operation_id]),
                item[1].operation_id,
            ),
        )

    def select_machine(machine_by_id, job, operation):
        del machine_by_id, job
        index = operation_index[operation.operation_id]
        target = _machine_choice_index(genome, index, len(operation.eligible_machines))
        return min(
            operation.eligible_machines,
            key=lambda machine_id: (
                abs(operation.eligible_machines.index(machine_id) - target),
                machine_id,
            ),
        )

    return _schedule_decoder_native(
        instance=instance,
        algorithm=algorithm,
        path_aware=path_aware,
        event_aware=event_aware,
        select_operation=select_operation,
        select_machine=select_machine,
        operation_index=operation_index,
        genome=genome,
        repair_process_log=repair_process_log,
        repair_process_context=repair_process_context,
    )


def _schedule_decoder_native(
    *,
    instance: object,
    algorithm: str,
    path_aware: bool,
    event_aware: bool,
    select_operation,
    select_machine,
    operation_index: dict[str, int],
    genome: DecoderNativeGenome,
    repair_process_log: list[dict[str, object]] | None = None,
    repair_process_context: dict[str, object] | None = None,
) -> ScheduleResult:
    machine_by_id = {machine.machine_id: machine for machine in instance.machines}
    machine_available = {machine.machine_id: 0 for machine in instance.machines}
    job_ready = {job.job_id: job.release_time for job in instance.jobs}
    next_op_index = {job.job_id: 0 for job in instance.jobs}
    agv_available = {agv.agv_id: 0 for agv in instance.agvs}
    agv_node = {agv.agv_id: agv.start_node_id for agv in instance.agvs}
    edge_reservations: dict[str, list[tuple[int, int]]] = {edge.edge_id: [] for edge in instance.path_edges}

    scheduled_operations: list[ScheduledOperation] = []
    transport_moves = []
    total_operations = sum(len(job.operations) for job in instance.jobs)

    while len(scheduled_operations) < total_operations:
        available = _available_operations(instance.jobs, next_op_index)
        if not available:
            raise ValueError("no schedulable operations remain")
        job, operation = select_operation(available)
        machine_id = select_machine(machine_by_id, job, operation)
        machine = machine_by_id[machine_id]
        transport_end = job_ready[job.job_id]
        if instance.agvs and instance.path_edges:
            move = _schedule_transport(
                instance=instance,
                job=job,
                operation=operation,
                destination=machine,
                agv_available=agv_available,
                agv_node=agv_node,
                edge_reservations=edge_reservations,
                path_aware=path_aware,
                event_aware=event_aware,
                move_index=len(transport_moves),
                repair_process_log=repair_process_log,
                repair_process_context=repair_process_context,
            )
            transport_moves.append(move)
            transport_end = move.end

        target_start = _operation_start_anchor(genome, operation_index[operation.operation_id])
        start = max(job_ready[job.job_id], machine_available[machine_id], transport_end, target_start)
        if event_aware:
            start = _first_resource_event_feasible_start(
                start,
                operation.processing_time[machine_id],
                machine_id,
                "machine_breakdown",
                instance,
                repair_process_log=repair_process_log,
                repair_process_context=repair_process_context,
                subject={
                    "job_id": job.job_id,
                    "operation_id": operation.operation_id,
                    "machine_id": machine_id,
                },
            )
        end = start + operation.processing_time[machine_id]
        scheduled_operations.append(
            ScheduledOperation(
                operation_id=operation.operation_id,
                job_id=operation.job_id,
                op_index=operation.op_index,
                machine_id=machine_id,
                start=start,
                end=end,
                processing_energy=operation.processing_energy[machine_id],
            )
        )
        machine_available[machine_id] = end
        job_ready[job.job_id] = end
        next_op_index[job.job_id] += 1

    return ScheduleResult(
        algorithm=algorithm,
        path_aware=path_aware,
        operations=tuple(scheduled_operations),
        transports=tuple(transport_moves),
        event_aware=event_aware,
    )


def run_decoder_native_repair_diagnostic_from_seed_rows(
    seed_rows_path: Path,
    *,
    config: DecoderNativeConfig | None = None,
) -> dict[str, tuple[dict[str, object], ...]]:
    resolved = config or DecoderNativeConfig()
    seed_rows = read_seed_rows(seed_rows_path)
    seed_hint_rows = _apply_seed_teacher_mask(seed_rows, resolved.seed_teacher_mask)
    seed_values = [_int(row.get("seed", 0)) for row in seed_rows if row.get("seed", "")]
    start_seed = resolved.start_seed or (min(seed_values) if seed_values else 1)
    end_seed = resolved.end_seed or (max(seed_values) if seed_values else 1)
    seed_preferences = _seed_preferences_by_context(
        seed_rows,
        levels=resolved.levels,
        start_seed=start_seed,
        end_seed=end_seed,
        include_realized=resolved.include_realized and "realized" in resolved.modes,
        max_seed_preferences=resolved.max_seed_preferences,
    )
    seed_hints = _trained_seed_hints_by_context(
        seed_hint_rows,
        context_preferences=seed_preferences,
        size=resolved.size,
        max_seed_hints=resolved.max_seed_hints,
        seed_neighbor_count=resolved.seed_neighbor_count,
    )

    diagnostic_rows: list[dict[str, object]] = []
    for level, seed, mode in sorted(set(seed_preferences) | set(seed_hints)):
        if level not in resolved.levels or mode not in resolved.modes:
            continue
        if seed < start_seed or seed > end_seed:
            continue
        raw_instance = generate_instance(level, resolved.size, seed)
        instance = realize_structural_events(raw_instance).instance if mode == "realized" else raw_instance
        seed_initial_quota = _seed_initial_quota_for_mode(
            mode,
            seed_initial_quota=None,
            seed_initial_quota_raw=resolved.seed_initial_quota_raw,
            seed_initial_quota_realized=resolved.seed_initial_quota_realized,
        )
        for algorithm in resolved.algorithms:
            archive = run_moea_baseline(
                instance,
                algorithm=algorithm,
                config=MoeaConfig(
                    population_size=resolved.population_size,
                    max_evaluations=resolved.max_evaluations,
                    seed=resolved.moea_seed,
                    path_aware=True,
                    event_aware=True,
                    seed_preferences=seed_preferences.get((level, seed, mode), ()),
                    seed_hints=seed_hints.get((level, seed, mode), ()),
                    seed_initial_quota=seed_initial_quota,
                ),
            )
            for solution in archive[: resolved.max_base_rows_per_context]:
                diagnostic_rows.extend(
                    _diagnose_solution(
                        instance=instance,
                        solution=solution,
                        algorithm=algorithm,
                        level=level,
                        seed=seed,
                        mode=mode,
                        config=resolved,
                    )
                )
    diagnostic = tuple(diagnostic_rows)
    return {
        "diagnostic": diagnostic,
        "summary": summarize_decoder_native_repair(diagnostic),
    }


def summarize_decoder_native_repair(rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["algorithm"]), str(row["level"]), str(row["mode"]), str(row["method"]))].append(row)
        grouped[(str(row["algorithm"]), "ALL", "ALL", str(row["method"]))].append(row)
    summaries = []
    for (algorithm, level, mode, method), group in sorted(grouped.items()):
        success = [row for row in group if _candidate_success(str(row["candidate_relation_to_base"]))]
        trials = len(group)
        summaries.append(
            {
                "algorithm": algorithm,
                "level": level,
                "mode": mode,
                "method": method,
                "trials": trials,
                "phenotype_success_trials": len(success),
                "phenotype_success_rate": _ratio(len(success), trials),
                "current_projection_preserve_rate": _ratio(
                    sum(1 for row in group if _bool(row["current_projection_preserved_candidate"])),
                    trials,
                ),
                "native_projection_preserve_rate": _ratio(
                    sum(1 for row in group if _bool(row["native_projection_preserved_candidate"])),
                    trials,
                ),
                "current_projection_loss_rate": _ratio(
                    sum(1 for row in group if _bool(row["current_projection_lost_candidate"])),
                    trials,
                ),
                "native_projection_loss_rate": _ratio(
                    sum(1 for row in group if _bool(row["native_projection_lost_candidate"])),
                    trials,
                ),
                "success_current_projection_preserve_rate": _ratio(
                    sum(1 for row in success if _bool(row["current_projection_preserved_candidate"])),
                    len(success),
                ),
                "success_native_projection_preserve_rate": _ratio(
                    sum(1 for row in success if _bool(row["native_projection_preserved_candidate"])),
                    len(success),
                ),
                "success_current_projection_loss_rate": _ratio(
                    sum(1 for row in success if _bool(row["current_projection_lost_candidate"])),
                    len(success),
                ),
                "success_native_projection_loss_rate": _ratio(
                    sum(1 for row in success if _bool(row["native_projection_lost_candidate"])),
                    len(success),
                ),
                "native_gain_over_current_projection_rate": _ratio(
                    sum(1 for row in group if _bool(row["native_gain_over_current_projection"])),
                    trials,
                ),
                "success_native_gain_over_current_projection_rate": _ratio(
                    sum(1 for row in success if _bool(row["native_gain_over_current_projection"])),
                    len(success),
                ),
                "mean_current_projection_delay_delta_to_candidate": round(
                    _avg(float(row["current_projection_delay_delta_to_candidate"]) for row in group),
                    10,
                ),
                "mean_native_projection_delay_delta_to_candidate": round(
                    _avg(float(row["native_projection_delay_delta_to_candidate"]) for row in group),
                    10,
                ),
                "mean_same_machine_assignment_rate": round(
                    _avg(float(row["same_machine_assignment_rate"]) for row in group),
                    10,
                ),
                "mean_same_global_order_rate": round(
                    _avg(float(row["same_global_order_rate"]) for row in group),
                    10,
                ),
                "mean_same_machine_sequence_rate": round(
                    _avg(float(row["same_machine_sequence_rate"]) for row in group),
                    10,
                ),
                "mean_start_abs_delta": round(
                    _avg(float(row["mean_start_abs_delta"]) for row in group),
                    10,
                ),
            }
        )
    return tuple(summaries)


def write_decoder_native_outputs(
    *,
    diagnostic_out: Path,
    summary_out: Path,
    rows: dict[str, tuple[dict[str, object], ...]],
) -> None:
    _write_rows(diagnostic_out, rows["diagnostic"], DIAGNOSTIC_FIELDS)
    _write_rows(summary_out, rows["summary"], SUMMARY_FIELDS)


def _diagnose_solution(
    *,
    instance: object,
    solution: object,
    algorithm: str,
    level: str,
    seed: int,
    mode: str,
    config: DecoderNativeConfig,
) -> tuple[dict[str, object], ...]:
    operations = _ordered_operations(instance)
    base_trace: list[dict[str, object]] = []
    base = _evaluate_genome(
        _stable_seed(config.random_seed, level, seed, mode, solution.solution_id, "base"),
        solution.generation,
        _Genome(solution.genome_genes),
        instance,
        operations,
        algorithm,
        MoeaConfig(
            population_size=2,
            max_evaluations=2,
            seed=config.moea_seed,
            path_aware=True,
            event_aware=True,
            repair_process_log=base_trace,
            repair_process_context=_repair_context(level, seed, mode, solution.solution_id, algorithm, "base"),
        ),
    )
    if not base_trace:
        return ()
    trace_neighborhood = select_trace_lns_neighborhood(
        instance,
        base.schedule,
        tuple(base_trace),
        top_records=config.top_records,
        neighbor_radius=config.neighbor_radius,
    )
    if not trace_neighborhood:
        return ()
    rows = []
    for method in config.methods:
        neighborhood = _neighborhood_for_method(
            method,
            tuple(operation.operation_id for operation in base.schedule.operations),
            tuple(base_trace),
            trace_neighborhood,
            _stable_seed(config.random_seed, level, seed, mode, solution.solution_id, method),
        )
        repaired = apply_trace_lns_repair_with_neighborhood(
            instance,
            base.schedule,
            _pressure_rows_for_method(method, tuple(base_trace), neighborhood),
            neighborhood_operation_ids=neighborhood,
            config=TraceLnsConfig(
                top_records=config.top_records,
                neighbor_radius=config.neighbor_radius,
                priority_boost=config.priority_boost,
                process_weight=config.process_weight,
                acceptance_policy="raw",
            ),
            repair_process_context=_repair_context(level, seed, mode, solution.solution_id, method, "candidate"),
        )
        candidate_metrics = evaluate_schedule(instance, repaired.schedule)
        candidate_vector = tuple(float(candidate_metrics[objective]) for objective in instance.objectives)

        current_projection_genes = _schedule_to_seed_genes(instance, operations, repaired.schedule, _Genome(solution.genome_genes))
        current_trace: list[dict[str, object]] = []
        current_projection = _evaluate_genome(
            _stable_seed(config.random_seed, level, seed, mode, solution.solution_id, f"current:{method}"),
            solution.generation + 1,
            _Genome(current_projection_genes),
            instance,
            operations,
            algorithm,
            MoeaConfig(
                population_size=2,
                max_evaluations=2,
                seed=config.moea_seed,
                path_aware=True,
                event_aware=True,
                repair_process_log=current_trace,
                repair_process_context=_repair_context(level, seed, mode, solution.solution_id, algorithm, "current-projection"),
            ),
        )

        native_genome = project_decoder_native_genome(instance, operations, repaired.schedule)
        native_trace: list[dict[str, object]] = []
        native_schedule = decode_decoder_native_genome(
            instance,
            operations,
            native_genome,
            algorithm=f"decoder-native-{method}",
            path_aware=True,
            event_aware=True,
            repair_process_log=native_trace,
            repair_process_context=_repair_context(level, seed, mode, solution.solution_id, method, "native-projection"),
        )
        native_metrics = evaluate_schedule(instance, native_schedule)
        native_vector = tuple(float(native_metrics[objective]) for objective in instance.objectives)

        rows.append(
            _diagnostic_row(
                algorithm=algorithm,
                level=level,
                seed=seed,
                mode=mode,
                solution_id=solution.solution_id,
                method=_canonical_method(method),
                base_vector=base.objective_vector,
                candidate_vector=candidate_vector,
                current_vector=current_projection.objective_vector,
                native_vector=native_vector,
                base_delay=_total_delay(tuple(base_trace)),
                candidate_delay=_total_delay(repaired.trace_rows),
                current_delay=_total_delay(tuple(current_trace)),
                native_delay=_total_delay(tuple(native_trace)),
                candidate_schedule=repaired.schedule,
                native_schedule=native_schedule,
                trace_neighborhood_size=len(trace_neighborhood),
                candidate_neighborhood_size=len(repaired.neighborhood_operation_ids),
            )
        )
    return tuple(rows)


def _diagnostic_row(
    *,
    algorithm: str,
    level: str,
    seed: int,
    mode: str,
    solution_id: str,
    method: str,
    base_vector: tuple[float, ...],
    candidate_vector: tuple[float, ...],
    current_vector: tuple[float, ...],
    native_vector: tuple[float, ...],
    base_delay: int,
    candidate_delay: int,
    current_delay: int,
    native_delay: int,
    candidate_schedule: ScheduleResult,
    native_schedule: ScheduleResult,
    trace_neighborhood_size: int,
    candidate_neighborhood_size: int,
) -> dict[str, object]:
    candidate_relation = _relation(base_vector, candidate_vector, "base", "candidate")
    current_relation = _relation(candidate_vector, current_vector, "candidate", "current")
    native_relation = _relation(candidate_vector, native_vector, "candidate", "native")
    native_to_current = _relation(current_vector, native_vector, "current", "native")
    return {
        "algorithm": algorithm,
        "level": level,
        "seed": f"{seed:03d}",
        "mode": mode,
        "base_solution_id": solution_id,
        "method": method,
        "base_objective_vector": _format_vector(base_vector),
        "candidate_objective_vector": _format_vector(candidate_vector),
        "current_projection_objective_vector": _format_vector(current_vector),
        "native_projection_objective_vector": _format_vector(native_vector),
        "candidate_relation_to_base": candidate_relation,
        "current_projection_relation_to_candidate": current_relation,
        "native_projection_relation_to_candidate": native_relation,
        "native_projection_relation_to_current": native_to_current,
        "current_projection_preserved_candidate": str(_projection_preserved(current_relation, "current")),
        "native_projection_preserved_candidate": str(_projection_preserved(native_relation, "native")),
        "current_projection_lost_candidate": str(current_relation == "candidate-dominates-current"),
        "native_projection_lost_candidate": str(native_relation == "candidate-dominates-native"),
        "native_gain_over_current_projection": str(native_to_current == "native-dominates-current"),
        "base_total_delay": base_delay,
        "candidate_total_delay": candidate_delay,
        "current_projection_total_delay": current_delay,
        "native_projection_total_delay": native_delay,
        "candidate_delay_delta": candidate_delay - base_delay,
        "current_projection_delay_delta_to_candidate": current_delay - candidate_delay,
        "native_projection_delay_delta_to_candidate": native_delay - candidate_delay,
        "same_machine_assignment_rate": round(_same_machine_assignment_rate(candidate_schedule, native_schedule), 10),
        "same_global_order_rate": round(_same_global_order_rate(candidate_schedule, native_schedule), 10),
        "same_machine_sequence_rate": round(_same_machine_sequence_rate(candidate_schedule, native_schedule), 10),
        "mean_start_abs_delta": round(_mean_start_abs_delta(candidate_schedule, native_schedule), 10),
        "trace_neighborhood_size": trace_neighborhood_size,
        "candidate_neighborhood_size": candidate_neighborhood_size,
    }


def _neighborhood_for_method(
    method: str,
    operation_ids: tuple[str, ...],
    trace_rows: tuple[dict[str, object], ...],
    trace_neighborhood: tuple[str, ...],
    seed: int,
) -> tuple[str, ...]:
    canonical = _canonical_method(method)
    if canonical == "delay-trace-lns":
        return trace_neighborhood
    if canonical == "delay-topk-lns":
        return _delay_topk_neighborhood(operation_ids, trace_rows, len(trace_neighborhood))
    if canonical == "random-lns":
        return _random_neighborhood(operation_ids, len(trace_neighborhood), seed)
    raise ValueError(f"unsupported decoder-native method: {method}")


def _pressure_rows_for_method(
    method: str,
    trace_rows: tuple[dict[str, object], ...],
    neighborhood: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    canonical = _canonical_method(method)
    if canonical == "delay-trace-lns":
        return trace_rows
    if canonical == "delay-topk-lns":
        pressure = _trace_delay_by_operation(trace_rows)
        return tuple({"operation_id": operation_id, "repair_delay": pressure.get(operation_id, 0.0)} for operation_id in neighborhood)
    if canonical == "random-lns":
        return tuple({"operation_id": operation_id, "repair_delay": 1.0} for operation_id in neighborhood)
    raise ValueError(f"unsupported decoder-native method: {method}")


def _delay_topk_neighborhood(
    operation_ids: tuple[str, ...],
    trace_rows: tuple[dict[str, object], ...],
    target_size: int,
) -> tuple[str, ...]:
    pressure = _trace_delay_by_operation(trace_rows)
    selected = {
        operation_id
        for operation_id, _delay in sorted(pressure.items(), key=lambda item: (-item[1], item[0]))[:target_size]
    }
    if len(selected) < min(target_size, len(operation_ids)):
        for operation_id in operation_ids:
            selected.add(operation_id)
            if len(selected) >= min(target_size, len(operation_ids)):
                break
    return tuple(operation_id for operation_id in operation_ids if operation_id in selected)


def _random_neighborhood(operation_ids: tuple[str, ...], target_size: int, seed: int) -> tuple[str, ...]:
    if target_size <= 0:
        return ()
    rng = random.Random(seed)
    selected = set(rng.sample(list(operation_ids), k=min(target_size, len(operation_ids))))
    return tuple(operation_id for operation_id in operation_ids if operation_id in selected)


def _trace_delay_by_operation(rows: tuple[dict[str, object], ...]) -> dict[str, float]:
    pressure: dict[str, float] = defaultdict(float)
    for row in rows:
        operation_id = str(row.get("operation_id", ""))
        if operation_id:
            pressure[operation_id] += _float(row.get("repair_delay", 0.0))
    return dict(pressure)


def _same_machine_assignment_rate(left: ScheduleResult, right: ScheduleResult) -> float:
    left_machine = {operation.operation_id: operation.machine_id for operation in left.operations}
    right_machine = {operation.operation_id: operation.machine_id for operation in right.operations}
    operation_ids = set(left_machine) & set(right_machine)
    if not operation_ids:
        return 0.0
    return sum(1 for operation_id in operation_ids if left_machine[operation_id] == right_machine[operation_id]) / len(operation_ids)


def _same_global_order_rate(left: ScheduleResult, right: ScheduleResult) -> float:
    left_order = _operation_order(left)
    right_order = _operation_order(right)
    count = min(len(left_order), len(right_order))
    if count == 0:
        return 0.0
    return sum(1 for index in range(count) if left_order[index] == right_order[index]) / count


def _same_machine_sequence_rate(left: ScheduleResult, right: ScheduleResult) -> float:
    left_by_machine = _machine_sequences(left)
    right_by_machine = _machine_sequences(right)
    machines = set(left_by_machine) | set(right_by_machine)
    if not machines:
        return 0.0
    return sum(1 for machine in machines if left_by_machine.get(machine, ()) == right_by_machine.get(machine, ())) / len(machines)


def _mean_start_abs_delta(left: ScheduleResult, right: ScheduleResult) -> float:
    left_start = {operation.operation_id: operation.start for operation in left.operations}
    right_start = {operation.operation_id: operation.start for operation in right.operations}
    operation_ids = set(left_start) & set(right_start)
    if not operation_ids:
        return 0.0
    return sum(abs(left_start[operation_id] - right_start[operation_id]) for operation_id in operation_ids) / len(operation_ids)


def _operation_order(schedule: ScheduleResult) -> tuple[str, ...]:
    return tuple(
        operation.operation_id
        for operation in sorted(schedule.operations, key=lambda item: (item.start, item.end, item.operation_id))
    )


def _machine_sequences(schedule: ScheduleResult) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[object]] = defaultdict(list)
    for operation in schedule.operations:
        grouped[operation.machine_id].append(operation)
    return {
        machine_id: tuple(operation.operation_id for operation in sorted(items, key=lambda item: (item.start, item.end, item.operation_id)))
        for machine_id, items in grouped.items()
    }


def _operation_priority(genome: DecoderNativeGenome, index: int) -> float:
    if index >= len(genome.operation_priority):
        return 1.0
    return genome.operation_priority[index]


def _machine_choice_index(genome: DecoderNativeGenome, index: int, machine_count: int) -> int:
    if machine_count <= 1:
        return 0
    value = genome.operation_machine_choice[index] if index < len(genome.operation_machine_choice) else 0.0
    return min(machine_count - 1, max(0, round(value * (machine_count - 1))))


def _operation_start_anchor(genome: DecoderNativeGenome, index: int) -> int:
    if index >= len(genome.operation_start_anchor):
        return 0
    return max(0, int(round(genome.operation_start_anchor[index])))


def _projection_preserved(relation: str, contender: str) -> bool:
    return relation in {f"{contender}-dominates-candidate", "equal"}


def _candidate_success(relation: str) -> bool:
    return relation in {"candidate-dominates-base", "equal"}


def _canonical_method(method: str) -> str:
    if method in {"trace-lns", "delay-trace-lns"}:
        return "delay-trace-lns"
    if method in {"delay-topk-lns", "delay-top-k-lns"}:
        return "delay-topk-lns"
    if method == "random-lns":
        return "random-lns"
    raise ValueError(f"unsupported decoder-native method: {method}")


def _repair_context(level: str, seed: int, mode: str, solution_id: str, algorithm: str, phase: str) -> dict[str, object]:
    return {
        "context_id": f"e96-{phase}|{level}|{seed:03d}|{mode}",
        "solution_id": solution_id,
        "level": level,
        "seed": f"{seed:03d}",
        "mode": mode,
        "algorithm": algorithm,
        "generation": phase,
    }


def _stable_seed(base: int, level: str, seed: int, mode: str, solution_id: str, salt: str) -> int:
    text = f"{level}|{seed:03d}|{mode}|{solution_id}|{salt}"
    return base + sum((index + 1) * ord(char) for index, char in enumerate(text))


def _write_rows(path: Path, rows: tuple[dict[str, object], ...], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _total_delay(rows: tuple[dict[str, object], ...]) -> int:
    return sum(int(float(row.get("repair_delay", 0) or 0)) for row in rows)


def _bool(value: object) -> bool:
    return str(value) in {"True", "true", "1", "1.0"}


def _int(value: object) -> int:
    return int(float(value)) if value not in {"", None} else 0


def _float(value: object) -> float:
    return float(value) if value not in {"", None} else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 10) if denominator else 0.0


def _avg(values: object) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
