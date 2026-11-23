from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from su_modfjsp_ha_pc.models import DynamicEvent, Instance
from su_modfjsp_ha_pc.schedule import ScheduleResult, ScheduledOperation, TransportMove


ACTIVE_SET_FIELDS = (
    "context_id",
    "solution_id",
    "level",
    "seed",
    "mode",
    "algorithm",
    "generation",
    "eval_uid",
    "active_type",
    "entity_id",
    "operation_id",
    "job_id",
    "move_id",
    "agv_id",
    "machine_id",
    "edge_id",
    "start",
    "end",
    "event_id",
    "pressure_score",
    "reason",
)

ACTIVE_SUMMARY_FIELDS = (
    "level",
    "mode",
    "seed",
    "algorithm",
    "evaluated_solutions",
    "solutions_with_active_set",
    "active_set_nonempty_rate",
    "active_rows",
    "late_job_only_solutions",
    "late_job_only_rate",
    "edge_capacity_rows",
    "agv_overlap_rows",
    "agv_event_rows",
    "path_event_rows",
    "machine_event_rows",
    "late_job_rows",
    "actionable_rows",
    "actionable_rate",
)


def build_active_set_rows(
    instance: Instance,
    schedule: ScheduleResult,
    solution_id: str,
    context_id: str,
    *,
    level: str = "",
    seed: str = "",
    mode: str = "",
    algorithm: str = "",
    generation: int | str = "",
    eval_uid: int | str = "",
    max_active_records: int = 12,
) -> tuple[dict[str, object], ...]:
    rows = [
        *_edge_capacity_rows(instance, schedule),
        *_resource_overlap_rows(schedule.transports, resource="agv"),
        *_event_rows(instance, schedule),
        *_late_job_rows(instance, schedule),
    ]
    rows = sorted(
        rows,
        key=lambda row: (
            -float(row["pressure_score"]),
            _active_type_rank(str(row["active_type"])),
            str(row.get("entity_id", "")),
            str(row.get("operation_id", "")),
            str(row.get("move_id", "")),
        ),
    )[:max_active_records]
    return tuple(
        {
            "context_id": context_id,
            "solution_id": solution_id,
            "level": level,
            "seed": seed,
            "mode": mode,
            "algorithm": algorithm,
            "generation": generation,
            "eval_uid": eval_uid,
            **row,
        }
        for row in rows
    )


def summarize_active_set_rows(
    rows: tuple[dict[str, object], ...],
    evaluated_rows: tuple[dict[str, object], ...] = (),
) -> tuple[dict[str, object], ...]:
    evaluated_by_context: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for row in evaluated_rows:
        key = (
            str(row.get("level", "")),
            str(row.get("mode", "")),
            str(row.get("seed", "")),
            str(row.get("algorithm", "")),
        )
        evaluated_by_context[key].add(str(row.get("solution_id", "")))

    active_by_context: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("level", "")),
            str(row.get("mode", "")),
            str(row.get("seed", "")),
            str(row.get("algorithm", "")),
        )
        active_by_context[key].append(row)

    all_keys = sorted(set(evaluated_by_context) | set(active_by_context))
    summary = []
    for level, mode, seed, algorithm in all_keys:
        group = active_by_context.get((level, mode, seed, algorithm), [])
        evaluated_solutions = evaluated_by_context.get((level, mode, seed, algorithm), set())
        active_solution_types: dict[str, set[str]] = defaultdict(set)
        for row in group:
            active_solution_types[str(row.get("solution_id", ""))].add(str(row.get("active_type", "")))
        solutions_with_active_set = len(active_solution_types)
        late_job_only = sum(1 for types in active_solution_types.values() if types == {"late-job"})
        active_rows = len(group)
        actionable_rows = sum(1 for row in group if _is_actionable(row))
        summary.append(
            {
                "level": level,
                "mode": mode,
                "seed": seed,
                "algorithm": algorithm,
                "evaluated_solutions": len(evaluated_solutions),
                "solutions_with_active_set": solutions_with_active_set,
                "active_set_nonempty_rate": _ratio(solutions_with_active_set, len(evaluated_solutions)),
                "active_rows": active_rows,
                "late_job_only_solutions": late_job_only,
                "late_job_only_rate": _ratio(late_job_only, solutions_with_active_set),
                "edge_capacity_rows": _count_type(group, "edge-capacity"),
                "agv_overlap_rows": _count_type(group, "agv-overlap"),
                "agv_event_rows": _count_type(group, "agv-event"),
                "path_event_rows": _count_type(group, "path-event"),
                "machine_event_rows": _count_type(group, "machine-event"),
                "late_job_rows": _count_type(group, "late-job"),
                "actionable_rows": actionable_rows,
                "actionable_rate": _ratio(actionable_rows, active_rows),
            }
        )
    return tuple(summary)


def write_active_set_outputs(
    trace_path: Path,
    summary_path: Path,
    rows: tuple[dict[str, object], ...],
    evaluated_rows: tuple[dict[str, object], ...],
) -> None:
    _write_rows(trace_path, rows, ACTIVE_SET_FIELDS)
    _write_rows(summary_path, summarize_active_set_rows(rows, evaluated_rows), ACTIVE_SUMMARY_FIELDS)


def _edge_capacity_rows(instance: Instance, schedule: ScheduleResult) -> tuple[dict[str, object], ...]:
    capacities = {edge.edge_id: edge.capacity for edge in instance.path_edges}
    rows = []
    for edge_id, capacity in capacities.items():
        moves = [move for move in schedule.transports if edge_id in move.edge_ids]
        for left, right, active in _overloaded_windows(moves, capacity):
            pressure = len(active) - capacity
            for move in active:
                rows.append(
                    _base_row(
                        active_type="edge-capacity",
                        entity_id=edge_id,
                        operation_id=move.operation_id,
                        job_id=move.job_id,
                        move_id=move.move_id,
                        agv_id=move.agv_id,
                        edge_id=edge_id,
                        start=left,
                        end=right,
                        pressure_score=1.0 + pressure,
                        reason=f"edge {edge_id} has {len(active)} active moves over capacity {capacity}",
                    )
                )
    return tuple(rows)


def _resource_overlap_rows(moves: tuple[TransportMove, ...], resource: str) -> tuple[dict[str, object], ...]:
    if resource != "agv":
        return ()
    grouped: dict[str, list[TransportMove]] = defaultdict(list)
    for move in moves:
        grouped[move.agv_id].append(move)
    rows = []
    for agv_id, agv_moves in grouped.items():
        for left, right, active in _overloaded_windows(agv_moves, 1):
            for move in active:
                rows.append(
                    _base_row(
                        active_type="agv-overlap",
                        entity_id=agv_id,
                        operation_id=move.operation_id,
                        job_id=move.job_id,
                        move_id=move.move_id,
                        agv_id=agv_id,
                        start=left,
                        end=right,
                        pressure_score=float(len(active)),
                        reason=f"AGV {agv_id} has overlapping transport moves",
                    )
                )
    return tuple(rows)


def _event_rows(instance: Instance, schedule: ScheduleResult) -> tuple[dict[str, object], ...]:
    rows = []
    for event in instance.events:
        if event.event_type == "machine_breakdown":
            rows.extend(_machine_event_rows(event, schedule.operations))
        elif event.event_type == "agv_failure":
            rows.extend(_agv_event_rows(event, schedule.transports))
        elif event.event_type in {"path_blockage", "congestion_spike"}:
            rows.extend(_path_event_rows(event, schedule.transports))
    return tuple(rows)


def _machine_event_rows(event: DynamicEvent, operations: tuple[ScheduledOperation, ...]) -> tuple[dict[str, object], ...]:
    rows = []
    event_start, event_end = _event_window(event)
    for operation in operations:
        if operation.machine_id == event.affected_entity and _overlaps(operation.start, operation.end, event_start, event_end):
            rows.append(
                _base_row(
                    active_type="machine-event",
                    entity_id=event.affected_entity,
                    operation_id=operation.operation_id,
                    job_id=operation.job_id,
                    machine_id=operation.machine_id,
                    start=max(operation.start, event_start),
                    end=min(operation.end, event_end),
                    event_id=event.event_id,
                    pressure_score=1.0 + float(event.severity),
                    reason=f"operation overlaps {event.event_type}",
                )
            )
    return tuple(rows)


def _agv_event_rows(event: DynamicEvent, moves: tuple[TransportMove, ...]) -> tuple[dict[str, object], ...]:
    rows = []
    event_start, event_end = _event_window(event)
    for move in moves:
        if move.agv_id == event.affected_entity and _overlaps(move.start, move.end, event_start, event_end):
            rows.append(
                _base_row(
                    active_type="agv-event",
                    entity_id=event.affected_entity,
                    operation_id=move.operation_id,
                    job_id=move.job_id,
                    move_id=move.move_id,
                    agv_id=move.agv_id,
                    start=max(move.start, event_start),
                    end=min(move.end, event_end),
                    event_id=event.event_id,
                    pressure_score=1.0 + float(event.severity),
                    reason=f"transport overlaps {event.event_type}",
                )
            )
    return tuple(rows)


def _path_event_rows(event: DynamicEvent, moves: tuple[TransportMove, ...]) -> tuple[dict[str, object], ...]:
    rows = []
    event_start, event_end = _event_window(event)
    for move in moves:
        if event.affected_entity in move.edge_ids and _overlaps(move.start, move.end, event_start, event_end):
            rows.append(
                _base_row(
                    active_type="path-event",
                    entity_id=event.affected_entity,
                    operation_id=move.operation_id,
                    job_id=move.job_id,
                    move_id=move.move_id,
                    agv_id=move.agv_id,
                    edge_id=event.affected_entity,
                    start=max(move.start, event_start),
                    end=min(move.end, event_end),
                    event_id=event.event_id,
                    pressure_score=1.0 + float(event.severity),
                    reason=f"transport uses event-affected edge during {event.event_type}",
                )
            )
    return tuple(rows)


def _late_job_rows(instance: Instance, schedule: ScheduleResult) -> tuple[dict[str, object], ...]:
    operations_by_job: dict[str, list[ScheduledOperation]] = defaultdict(list)
    for operation in schedule.operations:
        operations_by_job[operation.job_id].append(operation)
    rows = []
    for job in instance.jobs:
        operations = operations_by_job.get(job.job_id, [])
        if not operations:
            continue
        completion = max(operation.end for operation in operations)
        tardiness = completion - job.due_date
        if tardiness <= 0:
            continue
        for operation in operations:
            rows.append(
                _base_row(
                    active_type="late-job",
                    entity_id=job.job_id,
                    operation_id=operation.operation_id,
                    job_id=job.job_id,
                    machine_id=operation.machine_id,
                    start=operation.start,
                    end=operation.end,
                    pressure_score=1.0 + min(10.0, tardiness / 10.0),
                    reason=f"job tardiness {tardiness}",
                )
            )
    return tuple(rows)


def _overloaded_windows(items: Iterable[object], capacity: int) -> tuple[tuple[int, int, tuple[object, ...]], ...]:
    item_tuple = tuple(items)
    if capacity <= 0 or not item_tuple:
        return ()
    time_points = sorted({point for item in item_tuple for point in (int(item.start), int(item.end))})
    windows = []
    for left, right in zip(time_points, time_points[1:]):
        if left >= right:
            continue
        active = tuple(item for item in item_tuple if int(item.start) < right and int(item.end) > left)
        if len(active) > capacity:
            windows.append((left, right, active))
    return tuple(windows)


def _base_row(
    *,
    active_type: str,
    entity_id: str,
    operation_id: str = "",
    job_id: str = "",
    move_id: str = "",
    agv_id: str = "",
    machine_id: str = "",
    edge_id: str = "",
    start: int | str = "",
    end: int | str = "",
    event_id: str = "",
    pressure_score: float = 0.0,
    reason: str = "",
) -> dict[str, object]:
    return {
        "active_type": active_type,
        "entity_id": entity_id,
        "operation_id": operation_id,
        "job_id": job_id,
        "move_id": move_id,
        "agv_id": agv_id,
        "machine_id": machine_id,
        "edge_id": edge_id,
        "start": start,
        "end": end,
        "event_id": event_id,
        "pressure_score": round(float(pressure_score), 10),
        "reason": reason,
    }


def _active_type_rank(active_type: str) -> int:
    order = {
        "agv-event": 0,
        "path-event": 1,
        "machine-event": 2,
        "edge-capacity": 3,
        "agv-overlap": 4,
        "late-job": 5,
    }
    return order.get(active_type, 99)


def _event_window(event: DynamicEvent) -> tuple[int, int]:
    return int(event.event_time), int(event.event_time + event.duration)


def _overlaps(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    if right_start >= right_end:
        return False
    return left_start < right_end and right_start < left_end


def _count_type(rows: list[dict[str, object]], active_type: str) -> int:
    return sum(1 for row in rows if str(row.get("active_type", "")) == active_type)


def _is_actionable(row: dict[str, object]) -> bool:
    return any(str(row.get(key, "")) for key in ("operation_id", "move_id", "agv_id", "machine_id", "edge_id"))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 10)


def _write_rows(path: Path, rows: tuple[dict[str, object], ...], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
