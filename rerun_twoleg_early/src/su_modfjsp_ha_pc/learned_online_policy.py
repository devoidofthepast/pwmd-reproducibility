from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from su_modfjsp_ha_pc.models import Instance


OBJECTIVE_KEYS = ("makespan", "tardiness", "energy", "path_conflict_delay", "stability")
ACTION_TEMPLATES = ("speed-path", "energy-machine", "due-recovery", "balanced-diversity")
TRAINING_POLICY_NAME = "offline-row-pairwise-action-value-v0"

_VIOLATION_KEYS = (
    "precedence_violations",
    "machine_overlap_violations",
    "agv_overlap_violations",
    "edge_capacity_violations",
    "machine_unavailability_violations",
    "agv_unavailability_violations",
    "path_event_violations",
)
_STRUCTURAL_TYPES = {
    "route_branch",
    "quality_rework",
    "path_blockage",
    "agv_failure",
    "congestion_spike",
    "machine_breakdown",
}
_TEMPLATE_TAILS = {
    "speed-path": (0.68, 0.08, 0.24),
    "energy-machine": (0.12, 0.78, 0.10),
    "due-recovery": (0.18, 0.08, 0.74),
    "balanced-diversity": (0.40, 0.30, 0.30),
}


@dataclass(frozen=True)
class LearnedOnlineActionValuePolicy:
    weights: tuple[float, ...]
    bias: float
    feature_count: int
    fallback_threshold: float
    training_rows: int
    pairwise_rows: int
    label_mean: float
    template_priors: tuple[float, ...]
    training_policy: str = TRAINING_POLICY_NAME


@dataclass(frozen=True)
class LearnedOnlineActionDecision:
    action_template: str
    target_tail: tuple[float, float, float]
    operation_rate: float
    machine_rate: float
    operation_blend: float
    machine_blend: float
    tail_blend: float
    confidence: float
    fallback: bool


@dataclass(frozen=True)
class _ActionContext:
    level_value: float
    realized: float
    structural_pressure: float
    graph_pressure: float
    front_ratio: float
    constraint_pressure: float
    span_pressure: float
    route_branch_pressure: float
    rework_pressure: float
    added_operation_pressure: float


def read_csv_rows(paths: tuple[Path, ...] | list[Path]) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({key: value for key, value in row.items()})
    return tuple(rows)


def train_learned_online_policy(
    evaluation_rows: tuple[dict[str, str], ...] | list[dict[str, str]],
    constraint_pair_rows: tuple[dict[str, str], ...] | list[dict[str, str]] = (),
    fallback_threshold: float = 0.025,
) -> LearnedOnlineActionValuePolicy:
    if fallback_threshold < 0.0:
        raise ValueError("fallback_threshold must be non-negative")

    rows = tuple(evaluation_rows)
    pair_rows = tuple(constraint_pair_rows)
    ranges = _objective_ranges(rows, pair_rows)
    examples: list[tuple[tuple[float, ...], float, str]] = []

    for row in rows:
        preference = _row_preference(row, _row_objectives(row))
        template = _infer_action_template(row)
        context = _row_context(row, ranges)
        utility = _row_utility(row, ranges, preference)
        examples.append((_features(context, preference, template), utility, template))

    for row in pair_rows:
        examples.extend(_pair_examples(row, ranges))

    if not examples:
        raise ValueError("at least one trainable evaluation row or constraint pair is required")

    feature_count = len(examples[0][0])
    labels = tuple(label for _features_tuple, label, _template in examples)
    label_mean = sum(labels) / len(labels)
    centered = tuple(label - label_mean for label in labels)
    weights, bias = _fit_linear(tuple(features for features, _label, _template in examples), centered)
    template_priors = _template_priors(examples, label_mean)
    return LearnedOnlineActionValuePolicy(
        weights=weights,
        bias=bias,
        feature_count=feature_count,
        fallback_threshold=fallback_threshold,
        training_rows=len(rows),
        pairwise_rows=len(pair_rows),
        label_mean=label_mean,
        template_priors=template_priors,
    )


def choose_learned_online_action(
    policy: LearnedOnlineActionValuePolicy,
    instance: Instance,
    target_direction: tuple[float, ...],
    graph_pressure: float,
    structural_pressure: float,
    front_ratio: float,
    constraint_pressure: float,
    span_pressure: float,
) -> LearnedOnlineActionDecision:
    preference = _safe_simplex(tuple(target_direction[: len(instance.objectives)]))
    context = _runtime_context(
        instance=instance,
        graph_pressure=graph_pressure,
        structural_pressure=structural_pressure,
        front_ratio=front_ratio,
        constraint_pressure=constraint_pressure,
        span_pressure=span_pressure,
    )
    scored = []
    for index, template in enumerate(ACTION_TEMPLATES):
        features = _features(context, preference, template)
        linear_score = policy.bias + sum(weight * value for weight, value in zip(policy.weights, features))
        prior = policy.template_priors[index] if index < len(policy.template_priors) else 0.0
        tail_match = 1.0 - _tail_distance(_preference_tail(preference, instance.objectives), _TEMPLATE_TAILS[template])
        pressure_bonus = _pressure_bonus(template, context)
        scored.append((linear_score + 0.08 * prior + 0.60 * tail_match + pressure_bonus, template))

    scored.sort(reverse=True)
    best_score, best_template = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else best_score
    confidence = max(0.0, best_score - second_score)
    fallback = confidence < policy.fallback_threshold
    target_tail = _decision_tail(best_template, preference, instance.objectives, context, confidence)
    rates = _decision_rates(best_template, context, confidence)
    blends = _decision_blends(best_template, context, confidence)
    return LearnedOnlineActionDecision(
        action_template=best_template,
        target_tail=target_tail,
        operation_rate=rates[0],
        machine_rate=rates[1],
        operation_blend=blends[0],
        machine_blend=blends[1],
        tail_blend=blends[2],
        confidence=confidence,
        fallback=fallback,
    )


def _pair_examples(
    row: dict[str, str],
    ranges: dict[str, tuple[float, float]],
) -> tuple[tuple[tuple[float, ...], float, str], ...]:
    objectives = tuple(item for item in str(row.get("target_objectives", "")).split(";") if item)
    preference = _parse_preference(row.get("preference_cell", ""), len(objectives))
    if not objectives or not preference:
        return ()
    examples = []
    for prefix, label_sign in (("preferred", 1.0), ("rejected", -1.0)):
        objective_vector = _parse_objective_json(row.get(f"{prefix}_objective_vector", ""))
        if not objective_vector:
            continue
        pseudo = {
            "level": row.get("level", ""),
            "mode": row.get("mode", ""),
            "objectives": ";".join(objectives),
            "algorithm_variant": row.get(f"{prefix}_behavior_id", ""),
            "constraint_violation": row.get(f"{prefix}_constraint_violation", "0"),
            "feasible": row.get(f"{prefix}_feasible", "0"),
            "preference": row.get("preference_cell", ""),
        }
        for objective, value in objective_vector.items():
            pseudo[objective] = str(value)
        template = _infer_action_template(pseudo)
        context = _row_context(pseudo, ranges)
        utility = _row_utility(pseudo, ranges, preference) + 0.20 * label_sign
        examples.append((_features(context, preference, template), utility, template))
    return tuple(examples)


def _fit_linear(
    feature_rows: tuple[tuple[float, ...], ...],
    labels: tuple[float, ...],
) -> tuple[tuple[float, ...], float]:
    feature_count = len(feature_rows[0])
    weights = [0.0] * feature_count
    bias = 0.0
    learning_rate = 0.04
    l2 = 0.0008
    for _epoch in range(90):
        for features, label in zip(feature_rows, labels):
            prediction = bias + sum(weight * value for weight, value in zip(weights, features))
            error = prediction - label
            bias -= learning_rate * error
            for index, value in enumerate(features):
                weights[index] -= learning_rate * (error * value + l2 * weights[index])
    return tuple(weights), bias


def _template_priors(
    examples: list[tuple[tuple[float, ...], float, str]],
    label_mean: float,
) -> tuple[float, ...]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for _features_tuple, label, template in examples:
        grouped[template].append(label)
    priors = []
    for template in ACTION_TEMPLATES:
        values = grouped.get(template, ())
        priors.append((sum(values) / len(values) - label_mean) if values else 0.0)
    return tuple(priors)


def _features(
    context: _ActionContext,
    preference: tuple[float, ...],
    template: str,
) -> tuple[float, ...]:
    preference_tail = _preference_tail(preference, OBJECTIVE_KEYS)
    template_tail = _TEMPLATE_TAILS[template]
    one_hot = tuple(1.0 if template == action else 0.0 for action in ACTION_TEMPLATES)
    tail_gap = tuple(abs(left - right) for left, right in zip(preference_tail, template_tail))
    return (
        context.level_value,
        context.realized,
        context.structural_pressure,
        context.graph_pressure,
        context.front_ratio,
        context.constraint_pressure,
        context.span_pressure,
        context.route_branch_pressure,
        context.rework_pressure,
        context.added_operation_pressure,
        *preference_tail,
        preference[3] if len(preference) > 3 else 0.0,
        preference[4] if len(preference) > 4 else 0.0,
        *one_hot,
        *template_tail,
        *tail_gap,
        context.graph_pressure * template_tail[0],
        context.graph_pressure * template_tail[1],
        context.structural_pressure * template_tail[2],
        context.constraint_pressure * template_tail[0],
        context.constraint_pressure * template_tail[2],
        context.span_pressure * one_hot[-1],
    )


def _objective_ranges(
    rows: tuple[dict[str, str], ...],
    pair_rows: tuple[dict[str, str], ...],
) -> dict[str, tuple[float, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for objective in _row_objectives(row):
            values[objective].append(_row_objective(row, objective))
    for row in pair_rows:
        for prefix in ("preferred", "rejected"):
            for objective, value in _parse_objective_json(row.get(f"{prefix}_objective_vector", "")).items():
                values[objective].append(float(value))
    ranges = {}
    for objective in OBJECTIVE_KEYS:
        objective_values = values.get(objective, [0.0])
        ranges[objective] = (min(objective_values), max(objective_values))
    return ranges


def _row_utility(
    row: dict[str, str],
    ranges: dict[str, tuple[float, float]],
    preference: tuple[float, ...],
) -> float:
    objectives = _row_objectives(row)
    if not objectives:
        objectives = OBJECTIVE_KEYS
    normalized = []
    for objective in objectives[: len(preference)]:
        low, high = ranges.get(objective, (0.0, 0.0))
        value = _row_objective(row, objective)
        normalized.append(0.5 if high <= low else (value - low) / (high - low))
    if not normalized:
        return 0.0
    weighted_tchebycheff = max(weight * value for weight, value in zip(preference, normalized))
    feasible_bonus = 0.08 if _float(row.get("feasible", "0")) >= 1.0 else -0.08
    violation_penalty = 0.18 * min(1.0, _constraint_violation(row))
    repair_penalty = 0.04 * _repair_pressure(row, ranges)
    search_bonus = -0.02 * _float(row.get("search_score", "0"))
    return 1.0 - weighted_tchebycheff + feasible_bonus - violation_penalty - repair_penalty + search_bonus


def _row_context(
    row: dict[str, str],
    ranges: dict[str, tuple[float, float]],
) -> _ActionContext:
    structural_total = _float(row.get("structural_total_updates", "0"))
    route_branches = _float(row.get("structural_route_branches", "0"))
    rework = _float(row.get("structural_rework_operations", "0"))
    added_operations = _float(row.get("structural_added_operations", "0"))
    graph_pressure = _normalized_value(_row_objective(row, "path_conflict_delay"), ranges["path_conflict_delay"])
    stability_pressure = _normalized_value(_row_objective(row, "stability"), ranges["stability"])
    return _ActionContext(
        level_value=_level_value(row.get("level", "")),
        realized=1.0 if str(row.get("mode", "")).lower() == "realized" else _bool(row.get("structural_realized", "")),
        structural_pressure=_clip(0.45 * structural_total / 20.0 + 0.55 * stability_pressure),
        graph_pressure=_clip(graph_pressure),
        front_ratio=1.0 if _float(row.get("feasible", "0")) >= 1.0 else 0.0,
        constraint_pressure=_clip(_constraint_violation(row)),
        span_pressure=_clip(_float(row.get("preference_cell_score", row.get("search_score", "0")))),
        route_branch_pressure=_clip(route_branches / 10.0),
        rework_pressure=_clip(rework / 20.0),
        added_operation_pressure=_clip(added_operations / 20.0),
    )


def _runtime_context(
    instance: Instance,
    graph_pressure: float,
    structural_pressure: float,
    front_ratio: float,
    constraint_pressure: float,
    span_pressure: float,
) -> _ActionContext:
    route_branches = sum(1 for event in instance.events if event.event_type == "route_branch")
    rework = sum(1 for event in instance.events if event.event_type == "quality_rework")
    structural_events = [event for event in instance.events if event.event_type in _STRUCTURAL_TYPES]
    event_pressure = (
        sum(event.severity for event in structural_events) / len(structural_events)
        if structural_events
        else structural_pressure
    )
    added_operations = sum(1 for job in instance.jobs for operation in job.operations if operation.op_index > 4)
    return _ActionContext(
        level_value=_level_value(instance.level),
        realized=0.0,
        structural_pressure=_clip(0.5 * structural_pressure + 0.5 * event_pressure),
        graph_pressure=_clip(graph_pressure),
        front_ratio=_clip(front_ratio),
        constraint_pressure=_clip(constraint_pressure),
        span_pressure=_clip(span_pressure),
        route_branch_pressure=_clip(route_branches / max(1, len(instance.events))),
        rework_pressure=_clip(rework / max(1, len(instance.events))),
        added_operation_pressure=_clip(added_operations / max(1, len(instance.jobs) * 2)),
    )


def _decision_tail(
    template: str,
    preference: tuple[float, ...],
    objectives: tuple[str, ...],
    context: _ActionContext,
    confidence: float,
) -> tuple[float, float, float]:
    preference_tail = _preference_tail(preference, objectives)
    template_tail = _TEMPLATE_TAILS[template]
    pressure = max(context.graph_pressure, context.structural_pressure, context.constraint_pressure)
    blend = min(0.78, 0.25 + 0.30 * pressure + min(0.20, confidence))
    return _safe_simplex(
        tuple((1.0 - blend) * left + blend * right for left, right in zip(preference_tail, template_tail))
    )


def _decision_rates(
    template: str,
    context: _ActionContext,
    confidence: float,
) -> tuple[float, float]:
    confidence_bonus = min(0.20, confidence)
    if template == "energy-machine":
        return (
            _clip(0.30 + 0.15 * context.structural_pressure + confidence_bonus),
            _clip(0.58 + 0.20 * context.graph_pressure + confidence_bonus),
        )
    if template == "due-recovery":
        return (
            _clip(0.56 + 0.20 * context.structural_pressure + confidence_bonus),
            _clip(0.34 + 0.12 * context.graph_pressure + confidence_bonus),
        )
    if template == "balanced-diversity":
        return (
            _clip(0.28 + 0.20 * context.span_pressure + confidence_bonus),
            _clip(0.28 + 0.15 * context.span_pressure + confidence_bonus),
        )
    return (
        _clip(0.52 + 0.22 * context.graph_pressure + confidence_bonus),
        _clip(0.42 + 0.18 * context.graph_pressure + confidence_bonus),
    )


def _decision_blends(
    template: str,
    context: _ActionContext,
    confidence: float,
) -> tuple[float, float, float]:
    pressure = max(context.graph_pressure, context.structural_pressure)
    confidence_bonus = min(0.12, confidence)
    if template == "balanced-diversity":
        return (0.12 + confidence_bonus, 0.12 + confidence_bonus, 0.22)
    if template == "energy-machine":
        return (0.16 + confidence_bonus, 0.24 + 0.10 * context.graph_pressure, 0.36 + 0.08 * pressure)
    if template == "due-recovery":
        return (0.24 + 0.10 * context.structural_pressure, 0.15 + confidence_bonus, 0.38 + 0.10 * pressure)
    return (0.22 + 0.10 * context.graph_pressure, 0.18 + confidence_bonus, 0.34 + 0.10 * pressure)


def _pressure_bonus(template: str, context: _ActionContext) -> float:
    if template == "speed-path":
        return 0.05 * context.graph_pressure + 0.03 * context.constraint_pressure
    if template == "energy-machine":
        return 0.03 * (1.0 - context.constraint_pressure)
    if template == "due-recovery":
        return 0.05 * context.structural_pressure + 0.04 * context.rework_pressure
    return 0.04 * context.span_pressure + 0.02 * (1.0 - context.front_ratio)


def _infer_action_template(row: dict[str, str]) -> str:
    text = " ".join(
        str(row.get(key, ""))
        for key in (
            "algorithm",
            "algorithm_variant",
            "selected_algorithm",
            "base_algorithm",
            "search_profile",
            "repair_mode",
            "candidate_source",
            "selected_candidate_source",
            "seed_hint_digest",
            "behavior_id",
        )
    ).lower()
    if any(token in text for token in ("energy", "eco", "battery")):
        return "energy-machine"
    if any(token in text for token in ("due", "tard", "stable", "rework", "event")):
        return "due-recovery"
    if any(token in text for token in ("path", "flow", "buffer", "repair", "agv", "congestion")):
        return "speed-path"
    if any(token in text for token in ("nsga3", "spea2", "moead", "divers")):
        return "balanced-diversity"
    preference = _parse_preference(row.get("preference", row.get("source_preference", "")), 5)
    if preference:
        tail = _preference_tail(preference, OBJECTIVE_KEYS)
        if tail[1] >= max(tail[0], tail[2]):
            return "energy-machine"
        if tail[2] >= max(tail[0], tail[1]):
            return "due-recovery"
        if tail[0] >= max(tail[1], tail[2]):
            return "speed-path"
    return "balanced-diversity"


def _row_objectives(row: dict[str, str]) -> tuple[str, ...]:
    raw = row.get("objectives") or row.get("objective_names") or row.get("target_objectives") or ""
    objectives = tuple(item for item in str(raw).split(";") if item)
    if objectives:
        return tuple("path_conflict_delay" if item == "path_repair_cost" else item for item in objectives)
    return tuple(objective for objective in OBJECTIVE_KEYS if row.get(objective, "") not in {"", None})


def _row_preference(row: dict[str, str], objectives: tuple[str, ...]) -> tuple[float, ...]:
    raw = row.get("preference") or row.get("source_preference") or row.get("preference_cell") or ""
    preference = _parse_preference(raw, len(objectives))
    if preference:
        return preference
    if not objectives:
        objectives = OBJECTIVE_KEYS
    return tuple(1.0 / len(objectives) for _ in objectives)


def _parse_preference(raw: object, expected: int) -> tuple[float, ...]:
    if not raw or expected <= 0:
        return ()
    try:
        values = tuple(float(item) for item in str(raw).replace(",", ";").split(";") if item != "")
    except ValueError:
        return ()
    if len(values) < expected:
        values = values + tuple(0.0 for _ in range(expected - len(values)))
    return _safe_simplex(values[:expected])


def _row_objective(row: dict[str, str], objective: str) -> float:
    vector = _objective_vector(row)
    if objective in vector:
        return vector[objective]
    if objective == "path_conflict_delay" and row.get("path_repair_cost", "") not in {"", None}:
        return _float(row.get("path_repair_cost", "0"))
    if objective == "stability" and row.get("event_recovery_cost", "") not in {"", None}:
        return _float(row.get("event_recovery_cost", "0"))
    return _float(row.get(objective, "0"))


def _objective_vector(row: dict[str, str]) -> dict[str, float]:
    raw = row.get("objective_vector", "")
    if not raw:
        return {}
    parsed = _parse_objective_json(raw)
    if parsed:
        return parsed
    values = []
    try:
        values = [float(item) for item in str(raw).split(";") if item != ""]
    except ValueError:
        return {}
    objectives = _row_objectives(row)
    return {objective: values[index] for index, objective in enumerate(objectives[: len(values)])}


def _parse_objective_json(raw: object) -> dict[str, float]:
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): float(value) for key, value in parsed.items()}


def _constraint_violation(row: dict[str, str]) -> float:
    if row.get("constraint_violation", "") not in {"", None}:
        return _float(row.get("constraint_violation", "0"))
    return sum(_float(row.get(key, "0")) for key in _VIOLATION_KEYS)


def _repair_pressure(row: dict[str, str], ranges: dict[str, tuple[float, float]]) -> float:
    path = _normalized_value(_row_objective(row, "path_conflict_delay"), ranges["path_conflict_delay"])
    stability = _normalized_value(_row_objective(row, "stability"), ranges["stability"])
    return _clip(0.55 * path + 0.45 * stability)


def _preference_tail(
    preference: tuple[float, ...],
    objectives: tuple[str, ...],
) -> tuple[float, float, float]:
    if not preference:
        return (0.4, 0.3, 0.3)
    weights = _safe_simplex(tuple(preference[: len(objectives)]))
    by_objective = {objective: weights[index] for index, objective in enumerate(objectives[: len(weights)])}
    speed_weight = by_objective.get("makespan", 0.0) + by_objective.get("path_conflict_delay", 0.0)
    energy_weight = by_objective.get("energy", 0.0)
    due_weight = by_objective.get("tardiness", 0.0) + by_objective.get("stability", 0.0)
    return _safe_simplex((speed_weight, energy_weight, due_weight))


def _tail_distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(abs(left_value - right_value) for left_value, right_value in zip(left, right)) / 2.0


def _normalized_value(value: float, value_range: tuple[float, float]) -> float:
    low, high = value_range
    return 0.0 if high <= low else _clip((value - low) / (high - low))


def _level_value(raw: object) -> float:
    text = str(raw)
    if text.startswith("L") and text[1:].isdigit():
        return min(1.0, int(text[1:]) / 5.0)
    return 0.0


def _safe_simplex(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        return ()
    clipped = tuple(max(0.0, float(value)) for value in values)
    total = sum(clipped)
    if total <= 0.0 or not math.isfinite(total):
        return tuple(1.0 / len(values) for _ in values)
    return tuple(value / total for value in clipped)


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _float(value: object) -> float:
    if value in {"", None}:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool(value: object) -> float:
    return 1.0 if str(value).lower() in {"1", "true", "yes"} else 0.0
