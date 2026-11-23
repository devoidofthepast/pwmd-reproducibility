from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


REPAIR_PROCESS_FIELDS = (
    "context_id",
    "solution_id",
    "level",
    "seed",
    "mode",
    "algorithm",
    "generation",
    "eval_uid",
    "repair_type",
    "entity_id",
    "operation_id",
    "job_id",
    "move_id",
    "agv_id",
    "machine_id",
    "edge_id",
    "earliest_start",
    "final_start",
    "repair_delay",
    "event_id",
    "route_edges",
    "reason",
)

REPAIR_PROCESS_SUMMARY_FIELDS = (
    "level",
    "mode",
    "seed",
    "algorithm",
    "evaluated_solutions",
    "solutions_with_repair_delay",
    "repair_process_nonempty_rate",
    "repair_rows",
    "total_repair_delay",
    "avg_repair_delay_per_evaluated",
    "avg_repair_delay_per_repaired",
    "path_capacity_rows",
    "path_event_rows",
    "agv_event_rows",
    "machine_event_rows",
    "path_capacity_delay",
    "path_event_delay",
    "agv_event_delay",
    "machine_event_delay",
)


def summarize_repair_process_rows(
    rows: tuple[dict[str, object], ...],
    evaluated_rows: tuple[dict[str, object], ...] = (),
) -> tuple[dict[str, object], ...]:
    evaluated_by_context: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for row in evaluated_rows:
        key = _context_key(row)
        evaluated_by_context[key].add(str(row.get("solution_id", "")))

    rows_by_context: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_context[_context_key(row)].append(row)

    summary = []
    for key in sorted(set(evaluated_by_context) | set(rows_by_context)):
        level, mode, seed, algorithm = key
        group = rows_by_context.get(key, [])
        evaluated = evaluated_by_context.get(key, set())
        repaired_solutions = {str(row.get("solution_id", "")) for row in group}
        total_delay = sum(_int(row.get("repair_delay", 0)) for row in group)
        summary.append(
            {
                "level": level,
                "mode": mode,
                "seed": seed,
                "algorithm": algorithm,
                "evaluated_solutions": len(evaluated),
                "solutions_with_repair_delay": len(repaired_solutions),
                "repair_process_nonempty_rate": _ratio(len(repaired_solutions), len(evaluated)),
                "repair_rows": len(group),
                "total_repair_delay": total_delay,
                "avg_repair_delay_per_evaluated": _ratio(total_delay, len(evaluated)),
                "avg_repair_delay_per_repaired": _ratio(total_delay, len(repaired_solutions)),
                "path_capacity_rows": _count_type(group, "path-capacity"),
                "path_event_rows": _count_type(group, "path-event"),
                "agv_event_rows": _count_type(group, "agv-event"),
                "machine_event_rows": _count_type(group, "machine-event"),
                "path_capacity_delay": _delay_by_type(group, "path-capacity"),
                "path_event_delay": _delay_by_type(group, "path-event"),
                "agv_event_delay": _delay_by_type(group, "agv-event"),
                "machine_event_delay": _delay_by_type(group, "machine-event"),
            }
        )
    return tuple(summary)


def write_repair_process_outputs(
    trace_path: Path,
    summary_path: Path,
    rows: tuple[dict[str, object], ...],
    evaluated_rows: tuple[dict[str, object], ...],
) -> None:
    _write_rows(trace_path, rows, REPAIR_PROCESS_FIELDS)
    _write_rows(summary_path, summarize_repair_process_rows(rows, evaluated_rows), REPAIR_PROCESS_SUMMARY_FIELDS)


def _write_rows(path: Path, rows: tuple[dict[str, object], ...], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _context_key(row: dict[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row.get("level", "")),
        str(row.get("mode", "")),
        str(row.get("seed", "")),
        str(row.get("algorithm", "")),
    )


def _count_type(rows: list[dict[str, object]], repair_type: str) -> int:
    return sum(1 for row in rows if str(row.get("repair_type", "")) == repair_type)


def _delay_by_type(rows: list[dict[str, object]], repair_type: str) -> int:
    return sum(_int(row.get("repair_delay", 0)) for row in rows if str(row.get("repair_type", "")) == repair_type)


def _int(value: object) -> int:
    if value in {"", None}:
        return 0
    return int(float(value))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 10)
