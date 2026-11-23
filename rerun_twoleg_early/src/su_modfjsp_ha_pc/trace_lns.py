from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from su_modfjsp_ha_pc.generator import generate_instance
from su_modfjsp_ha_pc.metrics import evaluate_schedule
from su_modfjsp_ha_pc.models import Instance, Job, Machine, Operation
from su_modfjsp_ha_pc.moea import MoeaConfig, _Genome, _evaluate_genome, _ordered_operations
from su_modfjsp_ha_pc.rules import (
    _available_operations,
    _choose_agv,
    _first_resource_event_feasible_start,
    _first_transport_feasible_start,
    _schedule_transport,
    _shortest_path_edges,
)
from su_modfjsp_ha_pc.schedule import ScheduleResult, ScheduledOperation
from su_modfjsp_ha_pc.structural import realize_structural_events


TRACE_LNS_RESULT_FIELDS = (
    "level",
    "mode",
    "seed",
    "sample_id",
    "move_type",
    "base_total_delay",
    "repaired_total_delay",
    "delay_delta",
    "delay_reduced",
    "base_objective_vector",
    "repaired_objective_vector",
    "objective_relation",
    "objectives_nonworsened",
    "strict_success",
    "candidate_accepted",
    "neighborhood_size",
    "selected_trace_delay",
    "changed_operation_start_count",
    "changed_machine_count",
    "base_repair_types",
    "repaired_repair_types",
)

TRACE_LNS_SUMMARY_FIELDS = (
    "level",
    "mode",
    "move_type",
    "trials",
    "delay_reduction_rate",
    "objective_nonworsening_rate",
    "both_success_rate",
    "strict_success_rate",
    "mean_delay_delta",
    "mean_base_total_delay",
    "mean_repaired_total_delay",
    "mean_neighborhood_size",
    "mean_selected_trace_delay",
    "mean_changed_operation_start_count",
    "mean_changed_machine_count",
)


@dataclass(frozen=True)
class TraceLnsConfig:
    top_records: int = 8
    neighbor_radius: int = 1
    priority_boost: float = 6.0
    process_weight: float = 1.0
    energy_weight: float = 0.01
    path_aware: bool = True
    event_aware: bool = True
    acceptance_policy: str = "raw"


@dataclass(frozen=True)
class TraceLnsRepairResult:
    schedule: ScheduleResult
    trace_rows: tuple[dict[str, object], ...]
    neighborhood_operation_ids: tuple[str, ...]


def select_trace_lns_neighborhood(
    instance: Instance,
    schedule: ScheduleResult,
    trace_rows: tuple[dict[str, object], ...],
    *,
    top_records: int = 8,
    neighbor_radius: int = 1,
) -> tuple[str, ...]:
    if top_records <= 0:
        raise ValueError("top_records must be positive")
    if neighbor_radius < 0:
        raise ValueError("neighbor_radius must be non-negative")

    base_rank = {operation.operation_id: index for index, operation in enumerate(schedule.operations)}
    selected = set(_top_trace_operation_ids(trace_rows, top_records))
    if not selected:
        return ()

    job_sequences = {
        job.job_id: tuple(operation.operation_id for operation in sorted(job.operations, key=lambda item: item.op_index))
        for job in instance.jobs
    }
    machine_sequences: dict[str, tuple[str, ...]] = {}
    for machine in instance.machines:
        machine_sequences[machine.machine_id] = tuple(
            operation.operation_id
            for operation in sorted(
                (item for item in schedule.operations if item.machine_id == machine.machine_id),
                key=lambda item: (item.start, item.end, item.operation_id),
            )
        )

    expanded = set(selected)
    for operation_id in tuple(selected):
        for sequence in job_sequences.values():
            _add_neighbors(expanded, sequence, operation_id, neighbor_radius)
        for sequence in machine_sequences.values():
            _add_neighbors(expanded, sequence, operation_id, neighbor_radius)

    return tuple(sorted(expanded, key=lambda operation_id: (base_rank.get(operation_id, 10**9), operation_id)))


def apply_trace_lns_repair(
    instance: Instance,
    base_schedule: ScheduleResult,
    trace_rows: tuple[dict[str, object], ...],
    *,
    config: TraceLnsConfig | None = None,
    repair_process_context: dict[str, object] | None = None,
) -> TraceLnsRepairResult:
    resolved = config or TraceLnsConfig()
    neighborhood = select_trace_lns_neighborhood(
        instance,
        base_schedule,
        trace_rows,
        top_records=resolved.top_records,
        neighbor_radius=resolved.neighbor_radius,
    )
    if not neighborhood:
        return TraceLnsRepairResult(base_schedule, (), ())

    repair_rows: list[dict[str, object]] = []
    schedule = _schedule_trace_lns(
        instance,
        base_schedule,
        trace_rows,
        neighborhood=frozenset(neighborhood),
        config=resolved,
        repair_process_log=repair_rows,
        repair_process_context=repair_process_context,
    )
    return TraceLnsRepairResult(schedule, tuple(repair_rows), neighborhood)


def apply_trace_lns_repair_with_neighborhood(
    instance: Instance,
    base_schedule: ScheduleResult,
    trace_rows: tuple[dict[str, object], ...],
    *,
    neighborhood_operation_ids: tuple[str, ...],
    config: TraceLnsConfig | None = None,
    repair_process_context: dict[str, object] | None = None,
) -> TraceLnsRepairResult:
    resolved = config or TraceLnsConfig()
    selected = frozenset(neighborhood_operation_ids)
    ordered_neighborhood = tuple(
        operation.operation_id for operation in base_schedule.operations if operation.operation_id in selected
    )
    if not ordered_neighborhood:
        return TraceLnsRepairResult(base_schedule, (), ())

    repair_rows: list[dict[str, object]] = []
    schedule = _schedule_trace_lns(
        instance,
        base_schedule,
        trace_rows,
        neighborhood=frozenset(ordered_neighborhood),
        config=resolved,
        repair_process_log=repair_rows,
        repair_process_context=repair_process_context,
    )
    return TraceLnsRepairResult(schedule, tuple(repair_rows), ordered_neighborhood)


def run_trace_lns_controllability(
    *,
    levels: tuple[str, ...],
    size: str,
    start_seed: int,
    end_seed: int,
    samples_per_context: int,
    genome_seed: int,
    trace_lns_config: TraceLnsConfig | None = None,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    if start_seed > end_seed:
        raise ValueError("start_seed must be <= end_seed")
    if samples_per_context <= 0:
        raise ValueError("samples_per_context must be positive")
    resolved = trace_lns_config or TraceLnsConfig()
    if resolved.acceptance_policy not in {"raw", "pareto-safe"}:
        raise ValueError("acceptance_policy must be raw or pareto-safe")
    rows: list[dict[str, object]] = []
    for level in levels:
        for seed in range(start_seed, end_seed + 1):
            raw_instance = generate_instance(level, size, seed)
            scenarios = [(raw_instance, "raw")]
            if level in {"L4", "L5"}:
                scenarios.append((realize_structural_events(raw_instance).instance, "realized"))
            for instance, mode in scenarios:
                operations = _ordered_operations(instance)
                rng = random.Random(_context_seed(genome_seed, level, mode, seed))
                for sample_id in range(samples_per_context):
                    base_genome = _random_genome(instance, operations, rng)
                    base, base_trace = _evaluate_with_trace(
                        instance,
                        operations,
                        base_genome,
                        uid=sample_id * 10,
                        generation=0,
                        level=level,
                        mode=mode,
                        seed=seed,
                    )
                    if not base_trace:
                        continue
                    repaired = apply_trace_lns_repair(
                        instance,
                        base.schedule,
                        base_trace,
                        config=resolved,
                        repair_process_context={
                            "context_id": f"e61-trace-lns|{instance.instance_id}|{mode}",
                            "solution_id": f"trace-lns-{sample_id:05d}",
                            "level": level,
                            "seed": f"{seed:03d}",
                            "mode": mode,
                            "algorithm": "trace-lns",
                            "generation": 0,
                            "eval_uid": sample_id * 10 + 1,
                        },
                    )
                    raw_metrics = evaluate_schedule(instance, repaired.schedule)
                    raw_vector = tuple(float(raw_metrics[objective]) for objective in instance.objectives)
                    candidate_accepted = True
                    accepted_schedule = repaired.schedule
                    accepted_trace = repaired.trace_rows
                    accepted_vector = raw_vector
                    if resolved.acceptance_policy == "pareto-safe" and not _objectives_nonworsened(
                        base.objective_vector,
                        raw_vector,
                    ):
                        candidate_accepted = False
                        accepted_schedule = base.schedule
                        accepted_trace = base_trace
                        accepted_vector = base.objective_vector
                    rows.append(
                        _trace_lns_row(
                            level=level,
                            mode=mode,
                            seed=seed,
                            sample_id=sample_id,
                            move_type="trace-lns"
                            if resolved.acceptance_policy == "raw"
                            else "trace-lns-pareto-safe",
                            base_schedule=base.schedule,
                            repaired_schedule=accepted_schedule,
                            base_vector=base.objective_vector,
                            repaired_vector=accepted_vector,
                            base_trace=base_trace,
                            repaired_trace=accepted_trace,
                            neighborhood=repaired.neighborhood_operation_ids,
                            candidate_accepted=candidate_accepted,
                        )
                    )
    result_rows = tuple(rows)
    return result_rows, summarize_trace_lns_controllability(result_rows)


def summarize_trace_lns_controllability(rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["level"]), str(row["mode"]), str(row["move_type"]))].append(row)
        grouped[("ALL", "ALL", str(row["move_type"]))].append(row)
    summary = []
    for (level, mode, move_type), group in sorted(grouped.items()):
        trials = len(group)
        delay_reduced = sum(_bool(row["delay_reduced"]) for row in group)
        nonworsened = sum(_bool(row["objectives_nonworsened"]) for row in group)
        both = sum(_bool(row["delay_reduced"]) and _bool(row["objectives_nonworsened"]) for row in group)
        strict = sum(_bool(row["strict_success"]) for row in group)
        summary.append(
            {
                "level": level,
                "mode": mode,
                "move_type": move_type,
                "trials": trials,
                "delay_reduction_rate": _ratio(delay_reduced, trials),
                "objective_nonworsening_rate": _ratio(nonworsened, trials),
                "both_success_rate": _ratio(both, trials),
                "strict_success_rate": _ratio(strict, trials),
                "mean_delay_delta": round(_avg(float(row["delay_delta"]) for row in group), 10),
                "mean_base_total_delay": round(_avg(float(row["base_total_delay"]) for row in group), 10),
                "mean_repaired_total_delay": round(_avg(float(row["repaired_total_delay"]) for row in group), 10),
                "mean_neighborhood_size": round(_avg(float(row["neighborhood_size"]) for row in group), 10),
                "mean_selected_trace_delay": round(_avg(float(row["selected_trace_delay"]) for row in group), 10),
                "mean_changed_operation_start_count": round(
                    _avg(float(row["changed_operation_start_count"]) for row in group),
                    10,
                ),
                "mean_changed_machine_count": round(_avg(float(row["changed_machine_count"]) for row in group), 10),
            }
        )
    return tuple(summary)


def write_trace_lns_outputs(
    rows_path: Path,
    summary_path: Path,
    rows: tuple[dict[str, object], ...],
    summary: tuple[dict[str, object], ...],
) -> None:
    _write_rows(rows_path, rows, TRACE_LNS_RESULT_FIELDS)
    _write_rows(summary_path, summary, TRACE_LNS_SUMMARY_FIELDS)


def _schedule_trace_lns(
    instance: Instance,
    base_schedule: ScheduleResult,
    trace_rows: tuple[dict[str, object], ...],
    *,
    neighborhood: frozenset[str],
    config: TraceLnsConfig,
    repair_process_log: list[dict[str, object]],
    repair_process_context: dict[str, object] | None,
) -> ScheduleResult:
    machine_by_id = {machine.machine_id: machine for machine in instance.machines}
    base_rank = {operation.operation_id: index for index, operation in enumerate(base_schedule.operations)}
    base_machine = {operation.operation_id: operation.machine_id for operation in base_schedule.operations}
    pressure = _trace_delay_by_operation(trace_rows)
    max_pressure = max(pressure.values(), default=1.0)

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
        job, operation = min(
            available,
            key=lambda item: (
                _trace_lns_operation_score(item[1], base_rank, pressure, max_pressure, neighborhood, config),
                item[1].operation_id,
            ),
        )
        machine_id = _trace_lns_machine_choice(
            instance=instance,
            machine_by_id=machine_by_id,
            job=job,
            operation=operation,
            base_machine=base_machine,
            neighborhood=neighborhood,
            machine_available=machine_available,
            job_ready=job_ready,
            agv_available=agv_available,
            agv_node=agv_node,
            edge_reservations=edge_reservations,
            config=config,
        )
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
                path_aware=config.path_aware,
                event_aware=config.event_aware,
                move_index=len(transport_moves),
                repair_process_log=repair_process_log,
                repair_process_context=repair_process_context,
            )
            transport_moves.append(move)
            transport_end = move.end

        start = max(job_ready[job.job_id], machine_available[machine_id], transport_end)
        if config.event_aware:
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
        algorithm="trace-lns-repair",
        path_aware=config.path_aware,
        operations=tuple(scheduled_operations),
        transports=tuple(transport_moves),
        event_aware=config.event_aware,
    )


def _trace_lns_operation_score(
    operation: Operation,
    base_rank: dict[str, int],
    pressure: dict[str, float],
    max_pressure: float,
    neighborhood: frozenset[str],
    config: TraceLnsConfig,
) -> float:
    rank = float(base_rank.get(operation.operation_id, 10**9))
    if operation.operation_id not in neighborhood:
        return rank
    normalized = pressure.get(operation.operation_id, 0.0) / max(max_pressure, 1.0)
    neighbor_bonus = 0.2 if normalized == 0.0 else normalized
    return rank - config.priority_boost * neighbor_bonus


def _trace_lns_machine_choice(
    *,
    instance: Instance,
    machine_by_id: dict[str, Machine],
    job: Job,
    operation: Operation,
    base_machine: dict[str, str],
    neighborhood: frozenset[str],
    machine_available: dict[str, int],
    job_ready: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
    config: TraceLnsConfig,
) -> str:
    original_machine = base_machine.get(operation.operation_id, "")
    if operation.operation_id not in neighborhood and original_machine in operation.eligible_machines:
        return original_machine
    return min(
        operation.eligible_machines,
        key=lambda machine_id: (
            _candidate_machine_score(
                instance=instance,
                machine=machine_by_id[machine_id],
                job=job,
                operation=operation,
                machine_available=machine_available,
                job_ready=job_ready,
                agv_available=agv_available,
                agv_node=agv_node,
                edge_reservations=edge_reservations,
                config=config,
            ),
            machine_id,
        ),
    )


def _candidate_machine_score(
    *,
    instance: Instance,
    machine: Machine,
    job: Job,
    operation: Operation,
    machine_available: dict[str, int],
    job_ready: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
    config: TraceLnsConfig,
) -> float:
    transport_end = job_ready[job.job_id]
    transport_delay = 0
    if instance.agvs and instance.path_edges:
        transport_end, transport_delay = _estimate_transport_end_and_delay(
            instance=instance,
            job=job,
            destination=machine,
            agv_available=agv_available,
            agv_node=agv_node,
            edge_reservations=edge_reservations,
            path_aware=config.path_aware,
            event_aware=config.event_aware,
        )
    earliest_start = max(job_ready[job.job_id], machine_available[machine.machine_id], transport_end)
    final_start = earliest_start
    if config.event_aware:
        final_start = _first_resource_event_feasible_start(
            earliest_start,
            operation.processing_time[machine.machine_id],
            machine.machine_id,
            "machine_breakdown",
            instance,
        )
    machine_delay = final_start - earliest_start
    completion = final_start + operation.processing_time[machine.machine_id]
    return (
        completion
        + config.process_weight * (transport_delay + machine_delay)
        + config.energy_weight * operation.processing_energy[machine.machine_id]
    )


def _estimate_transport_end_and_delay(
    *,
    instance: Instance,
    job: Job,
    destination: Machine,
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
    path_aware: bool,
    event_aware: bool,
) -> tuple[int, int]:
    edge_by_id = {edge.edge_id: edge for edge in instance.path_edges}
    agv = _choose_agv(instance.agvs, agv_available)
    origin_node = agv_node[agv.agv_id]
    route_edges = _shortest_path_edges(instance.path_edges, origin_node, destination.node_id)
    route_length = sum(edge_by_id[edge_id].length for edge_id in route_edges)
    travel_time = max(1, math.ceil(route_length / agv.speed))
    earliest_start = max(job.release_time, agv_available[agv.agv_id])
    final_start = earliest_start
    if path_aware or event_aware:
        final_start = _first_transport_feasible_start(
            earliest_start=earliest_start,
            travel_time=travel_time,
            route_edges=route_edges,
            agv_id=agv.agv_id,
            instance=instance,
            edge_reservations=edge_reservations,
            check_capacity=path_aware,
            check_events=event_aware,
        )
    return final_start + travel_time, final_start - earliest_start


def _trace_lns_row(
    *,
    level: str,
    mode: str,
    seed: int,
    sample_id: int,
    move_type: str,
    base_schedule: ScheduleResult,
    repaired_schedule: ScheduleResult,
    base_vector: tuple[float, ...],
    repaired_vector: tuple[float, ...],
    base_trace: tuple[dict[str, object], ...],
    repaired_trace: tuple[dict[str, object], ...],
    neighborhood: tuple[str, ...],
    candidate_accepted: bool,
) -> dict[str, object]:
    base_delay = _total_delay(base_trace)
    repaired_delay = _total_delay(repaired_trace)
    relation = _objective_relation(base_vector, repaired_vector)
    objectives_nonworsened = relation in {"repaired-dominates-base", "equal"}
    return {
        "level": level,
        "mode": mode,
        "seed": f"{seed:03d}",
        "sample_id": sample_id,
        "move_type": move_type,
        "base_total_delay": base_delay,
        "repaired_total_delay": repaired_delay,
        "delay_delta": repaired_delay - base_delay,
        "delay_reduced": str(repaired_delay < base_delay),
        "base_objective_vector": _format_vector(base_vector),
        "repaired_objective_vector": _format_vector(repaired_vector),
        "objective_relation": relation,
        "objectives_nonworsened": str(objectives_nonworsened),
        "strict_success": str(repaired_delay < base_delay and relation == "repaired-dominates-base"),
        "candidate_accepted": str(candidate_accepted),
        "neighborhood_size": len(neighborhood),
        "selected_trace_delay": _selected_trace_delay(base_trace, frozenset(neighborhood)),
        "changed_operation_start_count": _changed_operation_start_count(base_schedule, repaired_schedule),
        "changed_machine_count": _changed_machine_count(base_schedule, repaired_schedule),
        "base_repair_types": _type_counts(base_trace),
        "repaired_repair_types": _type_counts(repaired_trace),
    }


def _evaluate_with_trace(
    instance: Instance,
    operations: tuple[Operation, ...],
    genome: _Genome,
    *,
    uid: int,
    generation: int,
    level: str,
    mode: str,
    seed: int,
) -> tuple[object, tuple[dict[str, object], ...]]:
    trace_rows: list[dict[str, object]] = []
    config = MoeaConfig(
        population_size=2,
        max_evaluations=2,
        seed=seed,
        path_aware=True,
        event_aware=True,
        repair_process_log=trace_rows,
        repair_process_context={
            "context_id": f"e61-base|{instance.instance_id}|{mode}",
            "level": level,
            "seed": f"{seed:03d}",
            "mode": mode,
        },
    )
    evaluated = _evaluate_genome(uid, generation, genome, instance, operations, "nsga2", config)
    return evaluated, tuple(trace_rows)


def _top_trace_operation_ids(rows: tuple[dict[str, object], ...], top_records: int) -> tuple[str, ...]:
    ordered = sorted(rows, key=lambda row: (-_float(row.get("repair_delay", 0)), str(row.get("operation_id", ""))))
    ids = []
    for row in ordered:
        operation_id = str(row.get("operation_id", ""))
        if operation_id and operation_id not in ids:
            ids.append(operation_id)
        if len(ids) >= top_records:
            break
    return tuple(ids)


def _trace_delay_by_operation(rows: tuple[dict[str, object], ...]) -> dict[str, float]:
    pressure: dict[str, float] = defaultdict(float)
    for row in rows:
        operation_id = str(row.get("operation_id", ""))
        if operation_id:
            pressure[operation_id] += _float(row.get("repair_delay", 0))
    return dict(pressure)


def _add_neighbors(selected: set[str], sequence: tuple[str, ...], operation_id: str, radius: int) -> None:
    if operation_id not in sequence:
        return
    index = sequence.index(operation_id)
    left = max(0, index - radius)
    right = min(len(sequence), index + radius + 1)
    selected.update(sequence[left:right])


def _random_genome(instance: Instance, operations: tuple[Operation, ...], rng: random.Random) -> _Genome:
    gene_count = len(operations) + len(instance.machines) + 3
    return _Genome(tuple(rng.random() for _ in range(gene_count)))


def _objective_relation(base: tuple[float, ...], repaired: tuple[float, ...]) -> str:
    repaired_no_worse = all(left <= right for left, right in zip(repaired, base))
    base_no_worse = all(right <= left for left, right in zip(repaired, base))
    if repaired_no_worse and any(left < right for left, right in zip(repaired, base)):
        return "repaired-dominates-base"
    if base_no_worse and any(right < left for left, right in zip(repaired, base)):
        return "base-dominates-repaired"
    if repaired_no_worse and base_no_worse:
        return "equal"
    return "mixed"


def _objectives_nonworsened(base: tuple[float, ...], repaired: tuple[float, ...]) -> bool:
    return all(left <= right for left, right in zip(repaired, base))


def _total_delay(rows: tuple[dict[str, object], ...]) -> int:
    return sum(int(float(row.get("repair_delay", 0) or 0)) for row in rows)


def _selected_trace_delay(rows: tuple[dict[str, object], ...], operation_ids: frozenset[str]) -> int:
    return sum(
        int(float(row.get("repair_delay", 0) or 0))
        for row in rows
        if str(row.get("operation_id", "")) in operation_ids
    )


def _changed_operation_start_count(base: ScheduleResult, repaired: ScheduleResult) -> int:
    repaired_by_id = {operation.operation_id: operation for operation in repaired.operations}
    return sum(
        1
        for operation in base.operations
        if operation.operation_id in repaired_by_id and operation.start != repaired_by_id[operation.operation_id].start
    )


def _changed_machine_count(base: ScheduleResult, repaired: ScheduleResult) -> int:
    repaired_by_id = {operation.operation_id: operation for operation in repaired.operations}
    return sum(
        1
        for operation in base.operations
        if operation.operation_id in repaired_by_id and operation.machine_id != repaired_by_id[operation.operation_id].machine_id
    )


def _type_counts(rows: tuple[dict[str, object], ...]) -> str:
    counts = Counter(str(row.get("repair_type", "")) for row in rows if row.get("repair_type", ""))
    return "|".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _format_vector(vector: tuple[float, ...]) -> str:
    return ";".join(f"{value:.6f}" for value in vector)


def _write_rows(path: Path, rows: tuple[dict[str, object], ...], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _context_seed(base: int, level: str, mode: str, seed: int) -> int:
    return base + seed * 1009 + sum(ord(ch) for ch in f"{level}:{mode}") * 17


def _bool(value: object) -> bool:
    return str(value) == "True"


def _float(value: object) -> float:
    return float(value) if value not in {"", None} else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 10)


def _avg(values: object) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
