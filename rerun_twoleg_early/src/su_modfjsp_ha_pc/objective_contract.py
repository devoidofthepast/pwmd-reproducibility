from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from su_modfjsp_ha_pc.metrics import evaluate_schedule
from su_modfjsp_ha_pc.path_conflict_neural_dataset import preference_teacher_surrogate_metrics
from su_modfjsp_ha_pc.path_conflict_neural_dataset import (
    _route_blockage_risk,
    _route_capacity_risk,
    _route_stats,
)


OBJECTIVE_NAMES = (
    "makespan",
    "tardiness",
    "energy",
    "route_pressure",
    "event_exposure_pressure",
)

HARD_AUDIT_NAMES = (
    "precedence_violations",
    "machine_overlap_violations",
    "agv_overlap_violations",
    "edge_capacity_violations",
    "machine_unavailability_violations",
    "agv_unavailability_violations",
    "path_event_violations",
    "dynamic_event_violations",
    "feasible",
)

ROUTE_PRESSURE_COEFFICIENTS = {
    "route_bottleneck_exposure": 25.0,
    "route_length_exposure": 0.05,
    "route_event_overlap_exposure": 12.0,
    "route_blockage_exposure": 20.0,
    "route_capacity_exposure": 8.0,
}


@dataclass(frozen=True)
class ObjectiveScales:
    values: dict[str, float]
    quantile: float
    quantile_method: str
    sample_count: int
    source: str
    source_sha256: str

    def __post_init__(self) -> None:
        values = dict(self.values)
        if tuple(values) != OBJECTIVE_NAMES:
            raise ValueError(f"objective scale names/order must be exactly {OBJECTIVE_NAMES}")
        if self.sample_count != 480:
            raise ValueError("objective scales must use exactly 480 training schedules")
        if self.quantile != 0.95:
            raise ValueError("objective scales must use the frozen 0.95 quantile")
        if self.quantile_method != "linear":
            raise ValueError("objective scales must use NumPy's linear quantile method")
        if not self.source.strip():
            raise ValueError("objective scale source must not be empty")
        if len(self.source_sha256) != 64 or any(character not in "0123456789abcdef" for character in self.source_sha256.lower()):
            raise ValueError("objective scale source_sha256 must be a 64-character hexadecimal digest")
        if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
            raise ValueError("all objective scales must be positive finite values")
        object.__setattr__(self, "values", values)

    def to_dict(self) -> dict[str, object]:
        return {
            "objective_names": list(OBJECTIVE_NAMES),
            "values": self.values,
            "quantile": self.quantile,
            "quantile_method": self.quantile_method,
            "sample_count": self.sample_count,
            "source": self.source,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class ObjectiveEvaluation:
    raw: dict[str, float]
    hard_audit: dict[str, float]
    soft_components: dict[str, float]


@dataclass(frozen=True)
class ObjectiveBreakdown:
    raw: dict[str, float]
    normalized: dict[str, float]
    contributions: dict[str, float]
    contribution_rates: dict[str, float]
    weighted_value: float


def evaluate_objective_contract(instance: object, schedule: object) -> ObjectiveEvaluation:
    metrics = evaluate_schedule(instance, schedule)
    proxies = preference_teacher_surrogate_metrics(instance, schedule)
    soft_components = route_pressure_components(instance, schedule)
    reconstructed_route_pressure = sum(
        ROUTE_PRESSURE_COEFFICIENTS[name] * soft_components[name]
        for name in ROUTE_PRESSURE_COEFFICIENTS
    )
    if not math.isclose(
        reconstructed_route_pressure,
        float(proxies["teacher_path_pressure"]),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise RuntimeError("route-pressure components do not reconstruct the frozen proxy")
    raw = {
        "makespan": float(metrics["makespan"]),
        "tardiness": float(metrics["tardiness"]),
        "energy": float(metrics["energy"]),
        "route_pressure": float(proxies["teacher_path_pressure"]),
        "event_exposure_pressure": float(proxies["teacher_stability_pressure"]),
    }
    hard_audit = {name: float(metrics[name]) for name in HARD_AUDIT_NAMES}
    return ObjectiveEvaluation(
        raw=raw,
        hard_audit=hard_audit,
        soft_components=soft_components,
    )


def route_pressure_components(instance: object, schedule: object) -> dict[str, float]:
    components = {name: 0.0 for name in ROUTE_PRESSURE_COEFFICIENTS}
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _legs_overlap_duration, _schedule_move_legs

    move_legs = _schedule_move_legs(instance, schedule)
    for move in schedule.transports:
        route_stats = _route_stats(instance, move.edge_ids, move.start, move.end)
        route_stats = {**route_stats, "route_event_overlap_duration": float(_legs_overlap_duration(instance, move_legs[move.move_id], ""))}
        components["route_bottleneck_exposure"] += float(
            route_stats["route_bottleneck_count"]
        )
        components["route_length_exposure"] += float(route_stats["route_length"])
        components["route_event_overlap_exposure"] += float(
            route_stats["route_event_overlap_duration"]
        )
        components["route_blockage_exposure"] += float(
            _route_blockage_risk(instance, move.edge_ids)
        )
        components["route_capacity_exposure"] += float(
            _route_capacity_risk(instance, move.edge_ids)
        )
    return {name: round(value, 10) for name, value in components.items()}


def objective_breakdown(
    raw: Mapping[str, float],
    preference: tuple[float, ...],
    scales: ObjectiveScales,
) -> ObjectiveBreakdown:
    raw_values = _validate_ordered_values(raw, label="raw objective")
    if len(preference) != len(OBJECTIVE_NAMES):
        raise ValueError(f"preference must contain exactly {len(OBJECTIVE_NAMES)} weights")
    if any(not math.isfinite(weight) or weight < 0.0 for weight in preference):
        raise ValueError("preference weights must be finite and nonnegative")
    if not math.isclose(sum(preference), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("preference weights must sum to 1")

    normalized = {
        name: raw_values[name] / scales.values[name]
        for name in OBJECTIVE_NAMES
    }
    contributions = {
        name: preference[index] * normalized[name]
        for index, name in enumerate(OBJECTIVE_NAMES)
    }
    weighted_value = sum(contributions.values())
    contribution_rates = {
        name: (contributions[name] / weighted_value if weighted_value > 0.0 else 0.0)
        for name in OBJECTIVE_NAMES
    }
    return ObjectiveBreakdown(
        raw=raw_values,
        normalized=normalized,
        contributions=contributions,
        contribution_rates=contribution_rates,
        weighted_value=weighted_value,
    )


def relative_material_threshold(base_value: float) -> float:
    value = float(base_value)
    if not math.isfinite(value):
        raise ValueError("material-threshold base value must be finite")
    return max(1.0e-9, 0.01 * abs(value))


def write_objective_scales(path: Path, scales: ObjectiveScales) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(scales.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_objective_scales(path: Path) -> ObjectiveScales:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tuple(payload.get("objective_names", ())) != OBJECTIVE_NAMES:
        raise ValueError(f"objective scale names/order must be exactly {OBJECTIVE_NAMES}")
    return ObjectiveScales(
        values={name: float(payload["values"][name]) for name in OBJECTIVE_NAMES},
        quantile=float(payload["quantile"]),
        quantile_method=str(payload["quantile_method"]),
        sample_count=int(payload["sample_count"]),
        source=str(payload["source"]),
        source_sha256=str(payload["source_sha256"]),
    )


def _validate_ordered_values(raw: Mapping[str, float], *, label: str) -> dict[str, float]:
    values = dict(raw)
    if tuple(values) != OBJECTIVE_NAMES:
        raise ValueError(f"{label} names/order must be exactly {OBJECTIVE_NAMES}")
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError(f"all {label} values must be finite and nonnegative")
    return values
