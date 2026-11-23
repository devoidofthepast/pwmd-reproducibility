from __future__ import annotations

from dataclasses import dataclass

from su_modfjsp_ha_pc.models import Instance
from su_modfjsp_ha_pc.moea import MoeaSeedHint
from su_modfjsp_ha_pc.structural import StructuralRealizationReport


@dataclass(frozen=True)
class StructuralRecoveryAction:
    action_name: str
    event_ids: tuple[str, ...]
    tail_weights: tuple[float, float, float]
    operation_template: str
    machine_template: str
    score: float


def build_structural_recovery_hints(
    instance: Instance,
    report: StructuralRealizationReport,
    preference: tuple[float, ...],
    max_hints: int = 4,
    allowed_action_names: tuple[str, ...] | None = None,
) -> tuple[tuple[MoeaSeedHint, ...], tuple[StructuralRecoveryAction, ...]]:
    if max_hints <= 0:
        raise ValueError("max_hints must be positive")
    if report.total_updates <= 0:
        return (), ()

    preference_tail = _preference_tail(preference, instance.objectives)
    event_ids = tuple(report.affected_events)
    actions: list[StructuralRecoveryAction] = []

    if report.new_jobs:
        actions.append(
            _action(
                "early-feasible-insertion",
                event_ids,
                base_tail=(0.20, 0.10, 0.70),
                preference_tail=preference_tail,
                operation_template="due",
                machine_template="index",
                score=1.00 + 0.08 * report.new_jobs,
            )
        )
    if report.rework_operations:
        actions.append(
            _action(
                "local-rework-buffering",
                event_ids,
                base_tail=(0.15, 0.20, 0.65),
                preference_tail=preference_tail,
                operation_template="due",
                machine_template="energy",
                score=0.96 + 0.05 * report.rework_operations,
            )
        )
    if report.route_branches:
        actions.append(
            _action(
                "branch-aware-alternate-machine",
                event_ids,
                base_tail=(0.55, 0.15, 0.30),
                preference_tail=preference_tail,
                operation_template="processing",
                machine_template="index",
                score=0.92 + 0.06 * report.route_branches,
            )
        )
    if len(actions) > 1:
        actions.append(
            _action(
                "conservative-diversity-recovery",
                event_ids,
                base_tail=(0.40, 0.30, 0.30),
                preference_tail=preference_tail,
                operation_template="index",
                machine_template="index",
                score=0.70 + 0.02 * report.total_updates,
            )
        )

    if allowed_action_names is not None:
        allowed = set(allowed_action_names)
        actions = [action for action in actions if action.action_name in allowed]

    ordered = tuple(sorted(actions, key=lambda action: (-action.score, action.action_name))[:max_hints])
    hints = tuple(
        MoeaSeedHint(
            tail_weights=action.tail_weights,
            operation_template=action.operation_template,
            machine_template=action.machine_template,
        )
        for action in ordered
    )
    return hints, ordered


def structural_recovery_columns(
    actions: tuple[StructuralRecoveryAction, ...],
    recovery_mode: str = "event-triggered",
) -> dict[str, object]:
    if not actions:
        return {
            "structural_recovery_mode": "none",
            "structural_recovery_hint_count": 0,
            "structural_recovery_actions": "",
            "structural_recovery_event_ids": "",
        }
    event_ids = sorted({event_id for action in actions for event_id in action.event_ids})
    return {
        "structural_recovery_mode": recovery_mode,
        "structural_recovery_hint_count": len(actions),
        "structural_recovery_actions": ";".join(action.action_name for action in actions),
        "structural_recovery_event_ids": ";".join(event_ids),
    }


def _action(
    action_name: str,
    event_ids: tuple[str, ...],
    base_tail: tuple[float, float, float],
    preference_tail: tuple[float, float, float],
    operation_template: str,
    machine_template: str,
    score: float,
) -> StructuralRecoveryAction:
    blended = _safe_simplex(tuple(0.70 * base + 0.30 * pref for base, pref in zip(base_tail, preference_tail)))
    return StructuralRecoveryAction(
        action_name=action_name,
        event_ids=event_ids,
        tail_weights=blended,
        operation_template=operation_template,
        machine_template=machine_template,
        score=round(score, 10),
    )


def _preference_tail(preference: tuple[float, ...], objectives: tuple[str, ...]) -> tuple[float, float, float]:
    if not preference:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    weights = _safe_simplex(preference[: len(objectives)])
    by_objective = {objective: weights[index] for index, objective in enumerate(objectives[: len(weights)])}
    return _safe_simplex(
        (
            by_objective.get("makespan", 0.0) + by_objective.get("path_conflict_delay", 0.0),
            by_objective.get("energy", 0.0),
            by_objective.get("tardiness", 0.0) + by_objective.get("stability", 0.0),
        )
    )


def _safe_simplex(values: tuple[float, ...]) -> tuple[float, ...]:
    clipped = tuple(max(0.0, float(value)) for value in values)
    total = sum(clipped)
    if total <= 0.0:
        return tuple(1.0 / len(clipped) for _ in clipped)
    return tuple(value / total for value in clipped)
