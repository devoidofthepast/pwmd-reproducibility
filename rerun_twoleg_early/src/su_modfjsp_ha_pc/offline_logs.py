from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from su_modfjsp_ha_pc.preferences import simplex_grid


OBJECTIVE_BY_LEVEL = {
    "L1": ("makespan", "tardiness", "energy", "machine_idle_time"),
    "L2": ("makespan", "tardiness", "energy"),
    "L3": ("makespan", "tardiness", "energy", "path_conflict_delay"),
    "L4": ("makespan", "tardiness", "energy", "path_conflict_delay", "stability"),
    "L5": ("makespan", "tardiness", "energy", "path_conflict_delay", "stability"),
}

VIOLATION_KEYS = (
    "precedence_violations",
    "machine_overlap_violations",
    "agv_overlap_violations",
    "edge_capacity_violations",
    "machine_unavailability_violations",
    "agv_unavailability_violations",
    "path_event_violations",
)

STRUCTURAL_KEYS = (
    "structural_new_jobs",
    "structural_rework_operations",
    "structural_route_branches",
    "structural_added_jobs",
    "structural_added_operations",
    "structural_total_updates",
)


def read_evaluation_rows(paths: tuple[Path, ...]) -> tuple[dict[str, str], ...]:
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({**row, "input_file": path.name, "input_stem": path.stem})
    return tuple(rows)


def build_behavior_log_rows(
    rows: list[dict[str, str]] | tuple[dict[str, str], ...],
    dataset_id: str,
    archive_id: str,
    preference_resolution: int,
    range_policy: str = "feasible",
    cell_objective_policy: str = "nondegenerate",
) -> tuple[dict[str, object], ...]:
    if cell_objective_policy not in {"nondegenerate", "all"}:
        raise ValueError("cell_objective_policy must be 'nondegenerate' or 'all'")
    objective_ranges = _objective_ranges(rows, range_policy=range_policy)
    logs = []
    for index, row in enumerate(rows):
        level = row.get("level") or _level_from_instance(row["instance_id"])
        objectives = _active_objectives(row, level)
        objective_vector = {objective: _float(row, objective) for objective in objectives}
        violation_vector = {key: _float(row, key) for key in VIOLATION_KEYS}
        structural_vector = {key: _float(row, key) for key in STRUCTURAL_KEYS}
        constraint_violation = _constraint_violation(row, violation_vector)
        behavior_family = _behavior_family(row)
        mode = row.get("mode") or ("realized" if _bool(row.get("structural_realized", "")) else "raw")
        base_instance_id = row.get("base_instance_id") or row.get("instance_id", "")
        scenario_id = f"{base_instance_id}:{mode}:{'structural' if _bool(row.get('structural_realized', '')) else 'event-log'}"
        preference_cell, preference_score, preference_cell_objectives = _preference_cell(
            objective_vector=objective_vector,
            ranges=objective_ranges[(base_instance_id, mode, tuple(objectives))],
            preference_resolution=preference_resolution,
            cell_objective_policy=cell_objective_policy,
        )
        ranges = objective_ranges[(base_instance_id, mode, tuple(objectives))]
        degenerate_objectives = tuple(objective for objective in objectives if ranges[objective][1] <= ranges[objective][0])
        log_id = f"{dataset_id}-{index:07d}"
        seed = row.get("seed") or _seed_from_instance(row.get("instance_id", ""))
        logs.append(
            {
                "log_id": log_id,
                "dataset_id": dataset_id,
                "archive_id": row.get("input_stem") or archive_id,
                "instance_id": row.get("instance_id", ""),
                "base_instance_id": base_instance_id,
                "level": level,
                "seed": seed,
                "mode": mode,
                "scenario_id": scenario_id,
                "behavior_family": behavior_family,
                "behavior_id": row.get("algorithm_variant") or row.get("algorithm", ""),
                "solution_id": row.get("solution_id", ""),
                "source_preference_id": row.get("preference_id", ""),
                "source_preference": row.get("preference", ""),
                "path_aware": row.get("path_aware", ""),
                "event_aware": row.get("event_aware", ""),
                "structural_realized": str(_bool(row.get("structural_realized", ""))),
                "structural_vector": json.dumps(structural_vector, sort_keys=True),
                "objective_names": ";".join(objectives),
                "preference_cell_objectives": ";".join(preference_cell_objectives),
                "degenerate_objectives": ";".join(degenerate_objectives),
                "objective_vector": json.dumps(objective_vector, sort_keys=True),
                "violation_vector": json.dumps(violation_vector, sort_keys=True),
                "constraint_violation": constraint_violation,
                "feasible": 1.0 if _float(row, "feasible") >= 1.0 and constraint_violation == 0 else 0.0,
                "preference_cell": preference_cell,
                "preference_cell_score": round(preference_score, 10),
                "preference_resolution": preference_resolution,
                "range_policy": range_policy,
                "makespan": objective_vector.get("makespan", 0.0),
                "tardiness": objective_vector.get("tardiness", 0.0),
                "energy": objective_vector.get("energy", 0.0),
                "path_conflict_delay": objective_vector.get("path_conflict_delay", 0.0),
                "stability": objective_vector.get("stability", 0.0),
            }
        )
    return tuple(logs)


def build_preference_coverage(
    logs: tuple[dict[str, object], ...],
    near_feasible_violation: float,
) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for log in logs:
        for family in ("all", str(log["behavior_family"])):
            grouped[
                (
                    str(log["dataset_id"]),
                    str(log["level"]),
                    str(log["mode"]),
                    str(log["objective_names"]),
                    str(log["preference_cell_objectives"]),
                    family,
                )
            ].append(log)

    coverage_rows = []
    for (dataset_id, level, mode, objective_names, preference_cell_objectives, behavior_family), group in sorted(grouped.items()):
        cell_objectives = tuple(item for item in preference_cell_objectives.split(";") if item)
        n_objectives = len(cell_objectives)
        preference_resolution = int(group[0].get("preference_resolution", _coverage_resolution(n_objectives)))
        cells = {_preference_label(preference) for preference in _preference_vectors(n_objectives, preference_resolution)}
        covered = {
            str(log["preference_cell"])
            for log in group
            if float(log["constraint_violation"]) <= near_feasible_violation
        }
        feasible = {str(log["preference_cell"]) for log in group if float(log["feasible"]) == 1.0}
        coverage_rows.append(
            {
                "dataset_id": dataset_id,
                "level": level,
                "mode": mode,
                "objective_names": objective_names,
                "preference_cell_objectives": preference_cell_objectives,
                "degenerate_objectives": group[0].get("degenerate_objectives", ""),
                "behavior_family": behavior_family,
                "log_rows": len(group),
                "preference_cells": len(cells),
                "preference_resolution": preference_resolution,
                "covered_cells": len(covered),
                "feasible_cells": len(feasible),
                "coverage_rate": round(len(covered) / len(cells), 6) if cells else 0.0,
                "feasible_coverage_rate": round(len(feasible) / len(cells), 6) if cells else 0.0,
                "feasible_rows": sum(1 for log in group if float(log["feasible"]) == 1.0),
                "near_feasible_rows": sum(
                    1 for log in group if float(log["constraint_violation"]) <= near_feasible_violation
                ),
            }
        )
    return tuple(coverage_rows)


def write_csv(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0]) if rows else ()
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _objective_ranges(
    rows: list[dict[str, str]] | tuple[dict[str, str], ...],
    range_policy: str,
) -> dict[tuple[str, str, tuple[str, ...]], dict[str, tuple[float, float]]]:
    if range_policy not in {"feasible", "all"}:
        raise ValueError("range_policy must be 'feasible' or 'all'")
    grouped: dict[tuple[str, str, tuple[str, ...]], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        level = row.get("level") or _level_from_instance(row["instance_id"])
        objectives = _active_objectives(row, level)
        mode = row.get("mode") or ("realized" if _bool(row.get("structural_realized", "")) else "raw")
        base_instance_id = row.get("base_instance_id") or row.get("instance_id", "")
        grouped[(base_instance_id, mode, tuple(objectives))].append(row)

    ranges = {}
    for key, group in grouped.items():
        objectives = key[2]
        feasible = [row for row in group if _float(row, "feasible") >= 1.0]
        reference = group if range_policy == "all" else feasible if feasible else group
        ranges[key] = {
            objective: (
                min(_float(row, objective) for row in reference),
                max(_float(row, objective) for row in reference),
            )
            for objective in objectives
        }
    return ranges


def _preference_cell(
    objective_vector: dict[str, float],
    ranges: dict[str, tuple[float, float]],
    preference_resolution: int,
    cell_objective_policy: str,
) -> tuple[str, float, tuple[str, ...]]:
    objectives = tuple(objective_vector)
    if cell_objective_policy == "all":
        cell_objectives = objectives
    else:
        cell_objectives = tuple(objective for objective in objectives if ranges[objective][1] > ranges[objective][0])
    if not cell_objectives:
        cell_objectives = objectives
    normalized = []
    for objective in cell_objectives:
        low, high = ranges[objective]
        span = high - low
        normalized.append(0.0 if span <= 0 else (objective_vector[objective] - low) / span)
    preferences = _preference_vectors(len(cell_objectives), preference_resolution)
    best = min(
        preferences,
        key=lambda preference: max(max(weight, 1e-6) * value for weight, value in zip(preference, normalized)),
    )
    score = max(max(weight, 1e-6) * value for weight, value in zip(best, normalized))
    return _preference_label(best), score, cell_objectives


def _active_objectives(row: dict[str, str], level: str) -> tuple[str, ...]:
    raw = row.get("objectives", "")
    if raw:
        return tuple(item for item in raw.split(";") if item)
    return OBJECTIVE_BY_LEVEL[level]


def _behavior_family(row: dict[str, str]) -> str:
    variant = row.get("algorithm_variant") or row.get("algorithm", "")
    if variant == "preference-rule" or variant.startswith("preference-rule"):
        return "preference-rule"
    return row.get("source") or ("moea" if variant.startswith("moea-") else "rule")


def _constraint_violation(row: dict[str, str], violation_vector: dict[str, float]) -> float:
    raw = row.get("constraint_violation", "")
    if raw:
        return float(raw)
    return sum(violation_vector.values())


def _level_from_instance(instance_id: str) -> str:
    return instance_id.split("_", 1)[0]


def _seed_from_instance(instance_id: str) -> str:
    marker = "_seed"
    if marker not in instance_id:
        return ""
    return instance_id.split(marker, 1)[1][:3]


def _preference_label(preference: tuple[float, ...]) -> str:
    return ";".join(f"{weight:.4f}" for weight in preference)


def _coverage_resolution(n_objectives: int) -> int:
    return 4 if n_objectives >= 4 else 5


def _preference_vectors(n_objectives: int, resolution: int) -> tuple[tuple[float, ...], ...]:
    if n_objectives == 1:
        return ((1.0,),)
    return simplex_grid(n_objectives, resolution)


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else 0.0


def _bool(value: str) -> bool:
    return value in {"True", "true", "1", "1.0"}
