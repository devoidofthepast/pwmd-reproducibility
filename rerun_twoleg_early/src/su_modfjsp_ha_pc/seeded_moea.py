from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path

from su_modfjsp_ha_pc.active_set_trace import write_active_set_outputs
from su_modfjsp_ha_pc.generator import generate_instance
from su_modfjsp_ha_pc.metrics import evaluate_schedule
from su_modfjsp_ha_pc.moea import (
    MoeaConfig,
    MoeaSeedHint,
    SUPPORTED_MOEA_ALGORITHMS,
    _Genome,
    _evaluate_genome,
    _initial_population,
    _ordered_operations,
    run_moea_baseline,
)
from su_modfjsp_ha_pc.offline_logs import build_behavior_log_rows, build_preference_coverage, write_csv
from su_modfjsp_ha_pc.repair_process_trace import write_repair_process_outputs
from su_modfjsp_ha_pc.structural import empty_structural_report, realize_structural_events
from su_modfjsp_ha_pc.structural_recovery import build_structural_recovery_hints, structural_recovery_columns
from su_modfjsp_ha_pc.trace_lns import TraceLnsConfig, apply_trace_lns_repair


def read_seed_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def evaluate_seeded_moea(
    seed_rows_path: Path,
    levels: tuple[str, ...],
    size: str,
    start_seed: int,
    end_seed: int,
    algorithms: tuple[str, ...],
    population_size: int,
    max_evaluations: int,
    moea_seed: int,
    include_realized: bool,
    max_seed_preferences: int = 5,
    max_seed_hints: int | None = None,
    seed_initial_quota: int | None = None,
    seed_initial_quota_raw: int | None = None,
    seed_initial_quota_realized: int | None = None,
    seed_neighbor_count: int = 1,
    seed_mode: str = "preference",
    seed_teacher_mask: str = "full",
    online_policy: str = "none",
    online_action_policy: object | None = None,
    online_action_reference_count: int = 8,
    online_action_strength: float = 1.0,
    online_action_gate: str = "none",
    online_action_gate_tolerance: float = 0.0,
    mrgv_mode: str = "none",
    mrgv_lambda: float = 0.30,
    mrgv_diversity_cap: float = 0.05,
    mrgv_tau_min: float = 0.25,
    mrgv_tau_path: float = 0.35,
    mrgv_tau_event: float = 0.35,
    mrgv_tau_agv: float = 0.35,
    mrgv_tau_diversity: float = 0.50,
    mrgv_gate: str = "pareto-safe",
    mrgv_force_operator: str = "none",
    mrgv_admission_policy: str = "all",
    mrgv_context_policy: str = "none",
    mrgv_context_low_lambda: float = 0.15,
    mrgv_context_event_lambda: float = 0.10,
    structural_recovery_mode: str = "none",
    structural_recovery_min_updates: int = 1,
    structural_recovery_base_hint_limit: int | None = None,
    structural_recovery_action_filter: tuple[str, ...] | None = None,
    max_structural_recovery_hints: int = 4,
    source: str = "seeded-moea-e25",
    mrgv_trace_path: Path | None = None,
    active_set_trace_path: Path | None = None,
    active_set_summary_path: Path | None = None,
    active_set_max_records: int = 12,
    repair_process_trace_path: Path | None = None,
    repair_process_summary_path: Path | None = None,
    trace_lns_seed_count: int = 0,
    trace_lns_source_count: int = 16,
    trace_lns_top_records: int = 8,
    trace_lns_neighbor_radius: int = 1,
    trace_lns_priority_boost: float = 3.0,
    trace_lns_process_weight: float = 0.5,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    if start_seed > end_seed:
        raise ValueError("start_seed must be <= end_seed")
    if max_seed_preferences <= 0:
        raise ValueError("max_seed_preferences must be positive")
    if max_seed_hints is not None and max_seed_hints <= 0:
        raise ValueError("max_seed_hints must be positive")
    if seed_initial_quota is not None and seed_initial_quota < 0:
        raise ValueError("seed_initial_quota must be non-negative")
    if seed_initial_quota_raw is not None and seed_initial_quota_raw < 0:
        raise ValueError("seed_initial_quota_raw must be non-negative")
    if seed_initial_quota_realized is not None and seed_initial_quota_realized < 0:
        raise ValueError("seed_initial_quota_realized must be non-negative")
    if seed_neighbor_count <= 0:
        raise ValueError("seed_neighbor_count must be positive")
    if seed_teacher_mask not in {"full", "no-repair-signal", "no-event-signal", "objective-only"}:
        raise ValueError(
            "seed_teacher_mask must be 'full', 'no-repair-signal', 'no-event-signal', or 'objective-only'"
        )
    if online_policy == "learned-online-action-value" and online_action_policy is None:
        raise ValueError("online_action_policy is required for learned-online-action-value")
    if online_action_reference_count <= 0:
        raise ValueError("online_action_reference_count must be positive")
    if not 0.0 <= online_action_strength <= 1.0:
        raise ValueError("online_action_strength must be in [0, 1]")
    if online_action_gate not in {"none", "dominance-indicator"}:
        raise ValueError("online_action_gate must be 'none' or 'dominance-indicator'")
    if online_action_gate_tolerance < 0.0:
        raise ValueError("online_action_gate_tolerance must be non-negative")
    if mrgv_mode not in {"none", "evaluation-counted", "replacement", "dual-offspring"}:
        raise ValueError("mrgv_mode must be 'none', 'evaluation-counted', 'replacement', or 'dual-offspring'")
    if not 0.0 <= mrgv_lambda <= 1.0:
        raise ValueError("mrgv_lambda must be in [0, 1]")
    if not 0.0 <= mrgv_diversity_cap <= 1.0:
        raise ValueError("mrgv_diversity_cap must be in [0, 1]")
    if mrgv_tau_min < 0.0:
        raise ValueError("mrgv_tau_min must be non-negative")
    if mrgv_tau_path < 0.0:
        raise ValueError("mrgv_tau_path must be non-negative")
    if mrgv_tau_event < 0.0:
        raise ValueError("mrgv_tau_event must be non-negative")
    if mrgv_tau_agv < 0.0:
        raise ValueError("mrgv_tau_agv must be non-negative")
    if mrgv_tau_diversity < 0.0:
        raise ValueError("mrgv_tau_diversity must be non-negative")
    if mrgv_gate not in {"none", "pareto-safe", "indicator-safe"}:
        raise ValueError("mrgv_gate must be 'none', 'pareto-safe', or 'indicator-safe'")
    if mrgv_admission_policy not in {"all", "gated"}:
        raise ValueError("mrgv_admission_policy must be 'all' or 'gated'")
    if mrgv_context_policy not in {
        "none",
        "raw-only-all",
        "raw-hybrid",
        "raw-hybrid-low-lambda",
        "raw-hybrid-plus-low-event",
        "raw-hybrid-b4-probe",
    }:
        raise ValueError(
            "mrgv_context_policy must be 'none', 'raw-only-all', 'raw-hybrid', "
            "'raw-hybrid-low-lambda', 'raw-hybrid-plus-low-event', or 'raw-hybrid-b4-probe'"
        )
    if not 0.0 <= mrgv_context_low_lambda <= 1.0:
        raise ValueError("mrgv_context_low_lambda must be in [0, 1]")
    if not 0.0 <= mrgv_context_event_lambda <= 1.0:
        raise ValueError("mrgv_context_event_lambda must be in [0, 1]")
    if mrgv_force_operator not in {
        "none",
        "path",
        "event",
        "agv",
        "diversity",
        "path-pressure-shift",
        "event-repair-resequence",
        "agv-balance-reassignment",
        "conservative-structural-diversity",
    }:
        raise ValueError("mrgv_force_operator must be a supported MRGV operator alias")
    if structural_recovery_mode not in {"none", "event-triggered", "pressure-gated"}:
        raise ValueError("structural_recovery_mode must be 'none', 'event-triggered' or 'pressure-gated'")
    if structural_recovery_min_updates <= 0:
        raise ValueError("structural_recovery_min_updates must be positive")
    if structural_recovery_base_hint_limit is not None and structural_recovery_base_hint_limit < 0:
        raise ValueError("structural_recovery_base_hint_limit must be non-negative")
    if max_structural_recovery_hints <= 0:
        raise ValueError("max_structural_recovery_hints must be positive")
    if (active_set_trace_path is None) != (active_set_summary_path is None):
        raise ValueError("active_set_trace_path and active_set_summary_path must be provided together")
    if active_set_max_records <= 0:
        raise ValueError("active_set_max_records must be positive")
    if (repair_process_trace_path is None) != (repair_process_summary_path is None):
        raise ValueError("repair_process_trace_path and repair_process_summary_path must be provided together")
    if trace_lns_seed_count < 0:
        raise ValueError("trace_lns_seed_count must be non-negative")
    if trace_lns_source_count <= 0:
        raise ValueError("trace_lns_source_count must be positive")
    if seed_mode not in {
        "preference",
        "schedule-aware",
        "learned-schedule-aware",
        "trained-schedule-aware",
        "nonlinear-graph-action-trained-schedule-aware",
        "graph-action-trained-schedule-aware",
        "partial-schedule-trained-schedule-aware",
        "partial-action-trained-schedule-aware",
        "action-trained-schedule-aware",
        "graph-trained-schedule-aware",
        "igsp-greedy",
    }:
        raise ValueError(
            "seed_mode must be 'preference', 'schedule-aware', 'learned-schedule-aware' "
            "'trained-schedule-aware', 'nonlinear-graph-action-trained-schedule-aware', "
            "'graph-action-trained-schedule-aware', "
            "'partial-schedule-trained-schedule-aware', 'partial-action-trained-schedule-aware', "
            "'action-trained-schedule-aware', 'graph-trained-schedule-aware' or 'igsp-greedy'"
        )
    unsupported = tuple(algorithm for algorithm in algorithms if algorithm not in SUPPORTED_MOEA_ALGORITHMS)
    if unsupported:
        raise ValueError(f"unsupported algorithms: {', '.join(unsupported)}")

    seed_rows = read_seed_rows(seed_rows_path)
    seed_hint_rows = _apply_seed_teacher_mask(seed_rows, seed_teacher_mask)
    seed_preferences = _seed_preferences_by_context(
        seed_rows,
        levels=levels,
        start_seed=start_seed,
        end_seed=end_seed,
        include_realized=include_realized,
        max_seed_preferences=max_seed_preferences,
    )
    if structural_recovery_mode in {"event-triggered", "pressure-gated"} and include_realized:
        seed_preferences = _with_event_triggered_realized_contexts(seed_preferences)
    if seed_mode == "schedule-aware":
        seed_hints = _seed_hints_by_context(
            seed_hint_rows,
            levels=levels,
            start_seed=start_seed,
            end_seed=end_seed,
            include_realized=include_realized,
            max_seed_hints=max_seed_hints or max_seed_preferences,
        )
    elif seed_mode == "learned-schedule-aware":
        seed_hints = _learned_seed_hints_by_context(
            seed_hint_rows,
            context_preferences=seed_preferences,
            max_seed_hints=max_seed_hints or max_seed_preferences,
            seed_neighbor_count=seed_neighbor_count,
        )
    elif seed_mode == "trained-schedule-aware":
        seed_hints = _trained_seed_hints_by_context(
            seed_hint_rows,
            context_preferences=seed_preferences,
            size=size,
            max_seed_hints=max_seed_hints or max_seed_preferences,
            seed_neighbor_count=seed_neighbor_count,
        )
    elif seed_mode == "graph-action-trained-schedule-aware":
        seed_hints = _graph_action_trained_seed_hints_by_context(
            seed_hint_rows,
            context_preferences=seed_preferences,
            size=size,
            max_seed_hints=max_seed_hints or max_seed_preferences,
            seed_neighbor_count=seed_neighbor_count,
        )
    elif seed_mode == "nonlinear-graph-action-trained-schedule-aware":
        seed_hints = _nonlinear_graph_action_trained_seed_hints_by_context(
            seed_hint_rows,
            context_preferences=seed_preferences,
            size=size,
            max_seed_hints=max_seed_hints or max_seed_preferences,
            seed_neighbor_count=seed_neighbor_count,
        )
    elif seed_mode == "partial-schedule-trained-schedule-aware":
        seed_hints = _partial_schedule_trained_seed_hints_by_context(
            seed_hint_rows,
            context_preferences=seed_preferences,
            size=size,
            max_seed_hints=max_seed_hints or max_seed_preferences,
            seed_neighbor_count=seed_neighbor_count,
        )
    elif seed_mode == "partial-action-trained-schedule-aware":
        seed_hints = _partial_action_trained_seed_hints_by_context(
            seed_hint_rows,
            context_preferences=seed_preferences,
            max_seed_hints=max_seed_hints or max_seed_preferences,
            seed_neighbor_count=seed_neighbor_count,
        )
    elif seed_mode == "action-trained-schedule-aware":
        seed_hints = _action_trained_seed_hints_by_context(
            seed_hint_rows,
            context_preferences=seed_preferences,
            max_seed_hints=max_seed_hints or max_seed_preferences,
            seed_neighbor_count=seed_neighbor_count,
        )
    elif seed_mode == "graph-trained-schedule-aware":
        seed_hints = _graph_trained_seed_hints_by_context(
            seed_hint_rows,
            context_preferences=seed_preferences,
            size=size,
            max_seed_hints=max_seed_hints or max_seed_preferences,
            seed_neighbor_count=seed_neighbor_count,
        )
    elif seed_mode == "igsp-greedy":
        seed_hints = _igsp_greedy_seed_hints_by_context(
            seed_hint_rows,
            context_preferences=seed_preferences,
            max_seed_hints=max_seed_hints or max_seed_preferences,
        )
    else:
        seed_hints = {}
    rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    active_set_rows: list[dict[str, object]] = []
    active_set_evaluated_rows: list[dict[str, object]] = []
    repair_process_rows: list[dict[str, object]] = []
    repair_process_evaluated_rows: list[dict[str, object]] = []
    for level, seed, mode in sorted(set(seed_preferences) | set(seed_hints)):
        raw_instance = generate_instance(level, size, seed)
        if mode == "realized":
            realized = realize_structural_events(raw_instance)
            instance = realized.instance
            structural_report = realized.report
            structural_realized = True
        else:
            instance = raw_instance
            structural_report = empty_structural_report(raw_instance)
            structural_realized = False
        resolved_seed_initial_quota = _seed_initial_quota_for_mode(
            mode,
            seed_initial_quota=seed_initial_quota,
            seed_initial_quota_raw=seed_initial_quota_raw,
            seed_initial_quota_realized=seed_initial_quota_realized,
        )
        seed_initial_quota_policy = _seed_initial_quota_policy(
            seed_initial_quota=seed_initial_quota,
            seed_initial_quota_raw=seed_initial_quota_raw,
            seed_initial_quota_realized=seed_initial_quota_realized,
        )
        preferences = seed_preferences.get((level, seed, mode), ())
        available_base_hints = seed_hints.get((level, seed, mode), ())
        base_hints = available_base_hints
        recovery_hints = ()
        recovery_actions = ()
        if _should_apply_structural_recovery(
            structural_recovery_mode,
            mode=mode,
            total_updates=structural_report.total_updates,
            min_updates=structural_recovery_min_updates,
        ):
            if structural_recovery_base_hint_limit is not None:
                base_hints = available_base_hints[:structural_recovery_base_hint_limit]
            preference_for_recovery = preferences[0] if preferences else ()
            recovery_hints, recovery_actions = build_structural_recovery_hints(
                instance=instance,
                report=structural_report,
                preference=preference_for_recovery,
                max_hints=max_structural_recovery_hints,
                allowed_action_names=structural_recovery_action_filter,
            )
        hints = base_hints + recovery_hints
        operations = _ordered_operations(instance)
        trace_lns_seed_genes, trace_lns_seed_stats = _trace_lns_seed_genes_for_context(
            instance=instance,
            operations=operations,
            preferences=preferences,
            hints=hints,
            population_size=population_size,
            moea_seed=moea_seed,
            level=level,
            seed=seed,
            mode=mode,
            seed_count=trace_lns_seed_count,
            source_count=trace_lns_source_count,
            config=TraceLnsConfig(
                top_records=trace_lns_top_records,
                neighbor_radius=trace_lns_neighbor_radius,
                priority_boost=trace_lns_priority_boost,
                process_weight=trace_lns_process_weight,
                acceptance_policy="pareto-safe",
            ),
        )
        for algorithm in algorithms:
            action_log: list[dict[str, object]] = []
            context_mrgv = _resolve_mrgv_context(
                level=level,
                mode=mode,
                context_policy=mrgv_context_policy,
                base_mode=mrgv_mode,
                base_lambda=mrgv_lambda,
                base_admission_policy=mrgv_admission_policy,
                base_force_operator=mrgv_force_operator,
                low_lambda=mrgv_context_low_lambda,
                event_lambda=mrgv_context_event_lambda,
            )
            config = MoeaConfig(
                population_size=population_size,
                max_evaluations=max_evaluations,
                seed=moea_seed,
                path_aware=True,
                event_aware=True,
                seed_preferences=preferences,
                seed_hints=hints,
                seed_genes=trace_lns_seed_genes,
                seed_initial_quota=resolved_seed_initial_quota,
                online_policy=online_policy,
                online_action_policy=online_action_policy,
                online_action_log=action_log,
                online_action_reference_count=online_action_reference_count,
                online_action_strength=online_action_strength,
                online_action_gate=online_action_gate,
                online_action_gate_tolerance=online_action_gate_tolerance,
                mrgv_mode=context_mrgv["mrgv_mode"],
                mrgv_lambda=context_mrgv["mrgv_lambda"],
                mrgv_diversity_cap=mrgv_diversity_cap,
                mrgv_tau_min=mrgv_tau_min,
                mrgv_tau_path=mrgv_tau_path,
                mrgv_tau_event=mrgv_tau_event,
                mrgv_tau_agv=mrgv_tau_agv,
                mrgv_tau_diversity=mrgv_tau_diversity,
                mrgv_gate=mrgv_gate,
                mrgv_force_operator=context_mrgv["mrgv_force_operator"],
                mrgv_admission_policy=context_mrgv["mrgv_admission_policy"],
                mrgv_probe_policy=context_mrgv["mrgv_probe_policy"],
                mrgv_log=action_log,
                active_set_log=active_set_rows if active_set_trace_path is not None else None,
                active_set_evaluated_log=active_set_evaluated_rows if active_set_trace_path is not None else None,
                active_set_context={
                    "context_id": f"{source}|{instance.instance_id}|{mode}",
                    "level": level,
                    "seed": f"{seed:03d}",
                    "mode": mode,
                }
                if active_set_trace_path is not None
                else None,
                active_set_max_records=active_set_max_records,
                repair_process_log=repair_process_rows if repair_process_trace_path is not None else None,
                repair_process_evaluated_log=repair_process_evaluated_rows
                if repair_process_trace_path is not None
                else None,
                repair_process_context={
                    "context_id": f"{source}|{instance.instance_id}|{mode}",
                    "level": level,
                    "seed": f"{seed:03d}",
                    "mode": mode,
                }
                if repair_process_trace_path is not None
                else None,
            )
            archive = run_moea_baseline(instance, algorithm=algorithm, config=config)
            action_summary = _summarize_action_log(action_log)
            if mrgv_trace_path is not None:
                trace_rows.extend(
                    _mrgv_ordered_trace_rows(
                        action_log,
                        source=source,
                        level=level,
                        seed=seed,
                        mode=mode,
                        algorithm=algorithm,
                        context_policy=mrgv_context_policy,
                        context_mrgv=context_mrgv,
                    )
                )
            for solution in archive:
                rows.append(
                    {
                        "source": source,
                        "instance_id": instance.instance_id,
                        "base_instance_id": raw_instance.instance_id,
                        "level": level,
                        "seed": f"{seed:03d}",
                        "mode": mode,
                        "objectives": ";".join(instance.objectives),
                        "algorithm": f"{solution.algorithm}|{source}",
                        "algorithm_variant": f"{solution.algorithm}|{source}",
                        "solution_id": solution.solution_id,
                        "generation": solution.generation,
                        "path_aware": str(solution.schedule.path_aware),
                        "event_aware": str(solution.schedule.event_aware),
                        "objective_vector": ";".join(f"{value:.6f}" for value in solution.objective_vector),
                        "constraint_violation": solution.constraint_violation,
                        "genome_signature": solution.genome_signature,
                        "seed_preference_count": len(preferences),
                        "seed_preference_digest": _preference_digest(preferences),
                        "seed_mode": seed_mode,
                        "seed_teacher_mask": seed_teacher_mask,
                        "seed_hint_count": len(hints),
                        "base_seed_hint_count": len(base_hints),
                        "available_base_seed_hint_count": len(available_base_hints),
                        "seed_hint_digest": _hint_digest(hints),
                        **trace_lns_seed_stats,
                        "seed_initial_quota": ""
                        if resolved_seed_initial_quota is None
                        else resolved_seed_initial_quota,
                        "seed_initial_quota_policy": seed_initial_quota_policy,
                        "seed_training_policy": _seed_training_policy(seed_mode, seed_neighbor_count),
                        "online_policy": online_policy,
                        **_online_action_columns(online_action_policy, action_summary),
                        **_mrgv_action_columns(
                            context_mrgv["mrgv_mode"],
                            context_mrgv["mrgv_force_operator"],
                            context_mrgv["mrgv_admission_policy"],
                            mrgv_context_policy,
                            action_summary,
                        ),
                        **structural_recovery_columns(recovery_actions, recovery_mode=structural_recovery_mode),
                        **_structural_columns(structural_realized, structural_report),
                        **solution.metrics,
                    }
                )
    result_rows = tuple(rows)
    if mrgv_trace_path is not None:
        write_csv(mrgv_trace_path, tuple(trace_rows))
    if active_set_trace_path is not None and active_set_summary_path is not None:
        write_active_set_outputs(
            active_set_trace_path,
            active_set_summary_path,
            tuple(active_set_rows),
            tuple(active_set_evaluated_rows),
        )
    if repair_process_trace_path is not None and repair_process_summary_path is not None:
        write_repair_process_outputs(
            repair_process_trace_path,
            repair_process_summary_path,
            tuple(repair_process_rows),
            tuple(repair_process_evaluated_rows),
        )
    return result_rows, summarize_seeded_moea(result_rows)


def _seed_initial_quota_for_mode(
    mode: str,
    *,
    seed_initial_quota: int | None,
    seed_initial_quota_raw: int | None,
    seed_initial_quota_realized: int | None,
) -> int | None:
    if mode == "raw" and seed_initial_quota_raw is not None:
        return seed_initial_quota_raw
    if mode == "realized" and seed_initial_quota_realized is not None:
        return seed_initial_quota_realized
    return seed_initial_quota


def _seed_initial_quota_policy(
    *,
    seed_initial_quota: int | None,
    seed_initial_quota_raw: int | None,
    seed_initial_quota_realized: int | None,
) -> str:
    if seed_initial_quota_raw is None and seed_initial_quota_realized is None:
        return "legacy" if seed_initial_quota is None else f"fixed={seed_initial_quota}"
    fallback = "legacy" if seed_initial_quota is None else str(seed_initial_quota)
    raw = fallback if seed_initial_quota_raw is None else str(seed_initial_quota_raw)
    realized = fallback if seed_initial_quota_realized is None else str(seed_initial_quota_realized)
    return f"mode-aware(raw={raw},realized={realized})"


def _resolve_mrgv_context(
    *,
    level: str,
    mode: str,
    context_policy: str,
    base_mode: str,
    base_lambda: float,
    base_admission_policy: str,
    base_force_operator: str,
    low_lambda: float,
    event_lambda: float,
) -> dict[str, object]:
    if context_policy == "none":
        return {
            "mrgv_mode": base_mode,
            "mrgv_lambda": base_lambda,
            "mrgv_admission_policy": base_admission_policy,
            "mrgv_force_operator": base_force_operator,
            "mrgv_probe_policy": "none",
        }
    if mode == "raw":
        if context_policy == "raw-only-all":
            return _mrgv_context_values("dual-offspring", base_lambda, "all", "none", "none")
        if context_policy == "raw-hybrid-low-lambda":
            return _mrgv_context_values(
                "dual-offspring",
                low_lambda,
                "gated" if level == "L5" else "all",
                "none",
                "none",
            )
        if context_policy == "raw-hybrid-b4-probe":
            return _mrgv_context_values(
                "dual-offspring",
                base_lambda,
                "gated" if level == "L5" else "all",
                "none",
                "b4-local-indicator" if level == "L4" else "none",
            )
        if context_policy in {"raw-hybrid", "raw-hybrid-plus-low-event"}:
            return _mrgv_context_values(
                "dual-offspring",
                base_lambda,
                "gated" if level == "L5" else "all",
                "none",
                "none",
            )
    if mode == "realized" and context_policy == "raw-hybrid-plus-low-event":
        return _mrgv_context_values("dual-offspring", event_lambda, "gated", "event", "none")
    return _mrgv_context_values("none", base_lambda, base_admission_policy, "none", "none")


def _mrgv_context_values(
    mode: str,
    mrgv_lambda: float,
    admission_policy: str,
    force_operator: str,
    probe_policy: str,
) -> dict[str, object]:
    return {
        "mrgv_mode": mode,
        "mrgv_lambda": mrgv_lambda,
        "mrgv_admission_policy": admission_policy,
        "mrgv_force_operator": force_operator,
        "mrgv_probe_policy": probe_policy,
    }


def _should_apply_structural_recovery(
    structural_recovery_mode: str,
    *,
    mode: str,
    total_updates: int,
    min_updates: int,
) -> bool:
    if structural_recovery_mode == "none" or mode != "realized":
        return False
    return total_updates >= min_updates


def _trace_lns_seed_genes_for_context(
    *,
    instance: object,
    operations: tuple,
    preferences: tuple[tuple[float, ...], ...],
    hints: tuple[MoeaSeedHint, ...],
    population_size: int,
    moea_seed: int,
    level: str,
    seed: int,
    mode: str,
    seed_count: int,
    source_count: int,
    config: TraceLnsConfig,
) -> tuple[tuple[tuple[float, ...], ...], dict[str, object]]:
    policy = (
        "none"
        if seed_count == 0
        else (
            f"trace-lns-pareto-safe(top={config.top_records},radius={config.neighbor_radius},"
            f"boost={config.priority_boost},process={config.process_weight})"
        )
    )
    stats = {
        "trace_lns_seed_requested_count": seed_count,
        "trace_lns_seed_source_count": 0 if seed_count == 0 else source_count,
        "trace_lns_seed_accepted_count": 0,
        "trace_lns_seed_used_count": 0,
        "trace_lns_seed_policy": policy,
    }
    if seed_count == 0:
        return (), stats

    rng = random.Random(_trace_lns_context_seed(moea_seed, level, mode, seed))
    source_population = _initial_population(
        instance,
        operations,
        population_size=max(source_count, seed_count, 1),
        rng=rng,
        seed_preferences=preferences,
        seed_hints=hints,
        seed_initial_quota=None,
    )[:source_count]
    candidates: list[tuple[float, float, tuple[float, ...]]] = []
    for uid, genome in enumerate(source_population):
        trace_rows: list[dict[str, object]] = []
        eval_config = MoeaConfig(
            population_size=2,
            max_evaluations=2,
            seed=moea_seed,
            path_aware=True,
            event_aware=True,
            repair_process_log=trace_rows,
            repair_process_context={
                "context_id": f"e62-trace-lns-seed|{instance.instance_id}|{mode}",
                "level": level,
                "seed": f"{seed:03d}",
                "mode": mode,
            },
        )
        base = _evaluate_genome(uid, 0, genome, instance, operations, "nsga2", eval_config)
        if not trace_rows:
            continue
        repaired = apply_trace_lns_repair(
            instance,
            base.schedule,
            tuple(trace_rows),
            config=config,
            repair_process_context={
                "context_id": f"e62-trace-lns-seed|{instance.instance_id}|{mode}",
                "solution_id": f"trace-lns-seed-{uid:05d}",
                "level": level,
                "seed": f"{seed:03d}",
                "mode": mode,
                "algorithm": "trace-lns-seed",
                "generation": 0,
                "eval_uid": uid,
            },
        )
        repaired_metrics = evaluate_schedule(instance, repaired.schedule)
        repaired_vector = tuple(float(repaired_metrics[objective]) for objective in instance.objectives)
        if not _objective_vector_nonworsening(base.objective_vector, repaired_vector):
            continue
        base_delay = _trace_total_delay(tuple(trace_rows))
        repaired_delay = _trace_total_delay(repaired.trace_rows)
        if repaired_delay >= base_delay:
            continue
        gene = _schedule_to_seed_genes(instance, operations, repaired.schedule, genome)
        objective_gain = sum(base_value - repaired_value for base_value, repaired_value in zip(base.objective_vector, repaired_vector))
        candidates.append((repaired_delay - base_delay, -objective_gain, gene))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = tuple(item[2] for item in candidates[:seed_count])
    stats["trace_lns_seed_accepted_count"] = len(candidates)
    stats["trace_lns_seed_used_count"] = len(selected)
    return selected, stats


def _schedule_to_seed_genes(instance: object, operations: tuple, schedule: object, source_genome: _Genome) -> tuple[float, ...]:
    operation_count = len(operations)
    machine_count = len(instance.machines)
    operation_rank = {
        operation.operation_id: index
        for index, operation in enumerate(sorted(schedule.operations, key=lambda item: (item.start, item.end, item.operation_id)))
    }
    operation_scores = tuple(
        operation_rank.get(operation.operation_id, index) / max(1, operation_count - 1)
        for index, operation in enumerate(operations)
    )

    by_machine: dict[str, list[float]] = defaultdict(list)
    for operation in sorted(schedule.operations, key=lambda item: (item.start, item.end, item.operation_id)):
        by_machine[operation.machine_id].append(operation_rank[operation.operation_id] / max(1, operation_count - 1))
    max_use = max((len(values) for values in by_machine.values()), default=1)
    machine_scores = []
    for machine in instance.machines:
        values = by_machine.get(machine.machine_id, [])
        if not values:
            machine_scores.append(1.0)
            continue
        average_rank = sum(values) / len(values)
        frequency_bonus = 1.0 - (len(values) / max_use)
        machine_scores.append(0.5 * average_rank + 0.5 * frequency_bonus)

    tail_start = operation_count + machine_count
    tail = source_genome.genes[tail_start : tail_start + 3]
    if len(tail) != 3:
        tail = (0.5, 0.5, 0.5)
    return tuple(_clip_gene(value) for value in operation_scores + tuple(machine_scores) + tuple(tail))


def _objective_vector_nonworsening(base: tuple[float, ...], candidate: tuple[float, ...]) -> bool:
    return all(candidate_value <= base_value for candidate_value, base_value in zip(candidate, base))


def _trace_total_delay(rows: tuple[dict[str, object], ...]) -> int:
    return sum(int(float(row.get("repair_delay", 0) or 0)) for row in rows)


def _trace_lns_context_seed(base: int, level: str, mode: str, seed: int) -> int:
    return base + seed * 1301 + sum(ord(ch) for ch in f"trace-lns:{level}:{mode}") * 19


def _clip_gene(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def summarize_seeded_moea(rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["level"]), str(row["mode"]))].append(row)
        grouped[("ALL", "ALL")].append(row)

    summaries = []
    for (level, mode), group in sorted(grouped.items()):
        context_rows = _context_representative_rows(group)
        summaries.append(
            {
                "level": level,
                "mode": mode,
                "rows": len(group),
                "feasible_rate": round(_avg(_float(row.get("feasible", 0.0)) for row in group), 6),
                "avg_makespan": round(_avg(_float(row.get("makespan", 0.0)) for row in group), 10),
                "avg_tardiness": round(_avg(_float(row.get("tardiness", 0.0)) for row in group), 10),
                "avg_energy": round(_avg(_float(row.get("energy", 0.0)) for row in group), 10),
                "avg_path_conflict_delay": round(_avg(_float(row.get("path_conflict_delay", 0.0)) for row in group), 10),
                "avg_stability": round(_avg(_float(row.get("stability", 0.0)) for row in group), 10),
                "avg_seed_preference_count": round(_avg(_float(row.get("seed_preference_count", 0.0)) for row in group), 6),
                "avg_seed_hint_count": round(_avg(_float(row.get("seed_hint_count", 0.0)) for row in group), 6),
                "avg_base_seed_hint_count": round(
                    _avg(_float(row.get("base_seed_hint_count", row.get("seed_hint_count", 0.0))) for row in group),
                    6,
                ),
                "avg_trace_lns_seed_used_count": round(
                    _avg(_float(row.get("trace_lns_seed_used_count", 0.0)) for row in group),
                    6,
                ),
                "avg_trace_lns_seed_accepted_count": round(
                    _avg(_float(row.get("trace_lns_seed_accepted_count", 0.0)) for row in group),
                    6,
                ),
                "trace_lns_seed_policy": _unique_join(
                    row.get("trace_lns_seed_policy", "none") for row in context_rows
                ),
                "seed_initial_quota": _unique_join(row.get("seed_initial_quota", "") for row in group),
                "seed_initial_quota_policy": _unique_join(
                    row.get("seed_initial_quota_policy", "") for row in group
                ),
                "seed_teacher_mask": _unique_join(row.get("seed_teacher_mask", "full") for row in group),
                "structural_recovery_mode": _unique_join(
                    row.get("structural_recovery_mode", "") for row in group
                ),
                "avg_structural_recovery_hint_count": round(
                    _avg(_float(row.get("structural_recovery_hint_count", 0.0)) for row in group),
                    6,
                ),
                "unique_algorithms": len({str(row["algorithm"]) for row in group}),
                "online_policy": str(group[0].get("online_policy", "")),
                "online_policy_training_policy": str(group[0].get("online_policy_training_policy", "")),
                "learned_action_training_rows": str(group[0].get("learned_action_training_rows", "")),
                "learned_action_pairwise_rows": str(group[0].get("learned_action_pairwise_rows", "")),
                "learned_action_feature_count": str(group[0].get("learned_action_feature_count", "")),
                "learned_action_fallback_threshold": str(group[0].get("learned_action_fallback_threshold", "")),
                "learned_action_strength": str(group[0].get("learned_action_strength", "")),
                "learned_action_gate": str(group[0].get("learned_action_gate", "")),
                "learned_action_gate_decisions": str(group[0].get("learned_action_gate_decisions", "")),
                "avg_learned_action_gate_acceptance_rate": round(
                    _avg(_float(row.get("learned_action_gate_acceptance_rate", 0.0)) for row in group),
                    6,
                ),
                "learned_action_gate_reason_counts": str(group[0].get("learned_action_gate_reason_counts", "")),
                "avg_learned_action_fallback_rate": round(
                    _avg(_float(row.get("learned_action_fallback_rate", 0.0)) for row in group),
                    6,
                ),
                "mrgv_mode": _unique_join(row.get("mrgv_mode", "") for row in context_rows),
                "mrgv_force_operator": _unique_join(row.get("mrgv_force_operator", "") for row in context_rows),
                "mrgv_candidate_policy": _unique_join(
                    row.get("mrgv_candidate_policy", "") for row in context_rows
                ),
                "mrgv_admission_policy": _unique_join(
                    row.get("mrgv_admission_policy", "") for row in context_rows
                ),
                "mrgv_context_policy": _unique_join(
                    row.get("mrgv_context_policy", "") for row in context_rows
                ),
                "mrgv_probe_policy": _unique_join(row.get("mrgv_probe_policy", "") for row in context_rows),
                "mrgv_probe_final_decision": _unique_join(
                    row.get("mrgv_probe_final_decision", "") for row in context_rows
                ),
                "avg_mrgv_probe_positive_rate": round(
                    _avg(_float(row.get("mrgv_probe_positive_rate", 0.0)) for row in context_rows),
                    6,
                ),
                "mrgv_gate": _unique_join(row.get("mrgv_gate", "") for row in context_rows),
                "mrgv_gate_lane_counts": _merge_count_strings(
                    row.get("mrgv_gate_lane_counts", "") for row in context_rows
                ),
                "mrgv_operator_counts": _merge_count_strings(row.get("mrgv_operator_counts", "") for row in context_rows),
                "mrgv_gate_reason_counts": _merge_count_strings(
                    row.get("mrgv_gate_reason_counts", "") for row in context_rows
                ),
                "avg_mrgv_gate_acceptance_rate": round(
                    _avg(_float(row.get("mrgv_gate_acceptance_rate", 0.0)) for row in context_rows),
                    6,
                ),
                "avg_mrgv_diversity_operator_rate": round(
                    _avg(_float(row.get("mrgv_diversity_operator_rate", 0.0)) for row in context_rows),
                    6,
                ),
                **_mrgv_lifecycle_summary_columns(context_rows),
                **_mrgv_contribution_summary_columns(context_rows),
                "avg_learned_action_confidence": round(
                    _avg(_float(row.get("learned_action_avg_confidence", 0.0)) for row in group),
                    10,
                ),
                "avg_learned_action_unique_target_directions": round(
                    _avg(_float(row.get("learned_action_unique_target_directions", 0.0)) for row in group),
                    6,
                ),
            }
        )
    return tuple(summaries)


def _context_representative_rows(group: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
    representatives: dict[tuple[str, ...], dict[str, object]] = {}
    for row in group:
        key = (
            str(row.get("source", "")),
            str(row.get("base_instance_id", "")),
            str(row.get("level", "")),
            str(row.get("seed", "")),
            str(row.get("mode", "")),
            str(row.get("algorithm_variant", "")),
        )
        representatives.setdefault(key, row)
    return tuple(representatives.values())


def _mrgv_lifecycle_summary_columns(rows: tuple[dict[str, object], ...]) -> dict[str, int]:
    return {
        f"mrgv_{metric}_{group}": sum(
            int(_float(row.get(f"mrgv_{metric}_{group}", 0)))
            for row in rows
        )
        for metric in _MRGV_LIFECYCLE_METRICS
        for group in _MRGV_OPERATOR_GROUPS
    }


def _mrgv_contribution_summary_columns(rows: tuple[dict[str, object], ...]) -> dict[str, object]:
    columns: dict[str, object] = {}
    for group in _MRGV_OPERATOR_GROUPS:
        for metric in _MRGV_CONTRIBUTION_RATE_METRICS:
            columns[f"avg_mrgv_{metric}_{group}"] = round(
                _avg(_float(row.get(f"mrgv_{metric}_{group}", 0.0)) for row in rows),
                6,
            )
        for metric in _MRGV_CONTRIBUTION_FLOAT_METRICS:
            columns[f"avg_mrgv_{metric}_{group}"] = round(
                _avg(_float(row.get(f"mrgv_{metric}_{group}", 0.0)) for row in rows),
                10,
            )
        for metric in _MRGV_CONTRIBUTION_VECTOR_METRICS:
            vector = _mean_vector(
                _parse_vector(str(row.get(f"mrgv_{metric}_{group}", "")))
                for row in rows
            )
            columns[f"avg_mrgv_{metric}_{group}"] = _format_vector(vector)
    return columns


def _merge_count_strings(values) -> str:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        for item in str(value).split("|"):
            if not item or ":" not in item:
                continue
            name, raw_count = item.rsplit(":", maxsplit=1)
            if not name:
                continue
            counts[name] += int(float(raw_count))
    return "|".join(f"{name}:{counts[name]}" for name in sorted(counts))


def _unique_join(values) -> str:
    unique = sorted({str(value) for value in values})
    return ";".join(unique)


def write_seeded_moea_outputs(
    rows_path: Path,
    summary_path: Path,
    rows: tuple[dict[str, object], ...],
    summary: tuple[dict[str, object], ...],
) -> None:
    write_csv(rows_path, rows)
    write_csv(summary_path, summary)


def write_seeded_moea_behavior_outputs(
    behavior_path: Path,
    coverage_path: Path,
    rows: tuple[dict[str, object], ...],
    dataset_id: str,
    archive_id: str,
    preference_resolution: int = 4,
    near_feasible_violation: float = 20.0,
) -> None:
    string_rows = tuple({key: str(value) for key, value in row.items()} for row in rows)
    behavior_logs = build_behavior_log_rows(
        string_rows,
        dataset_id=dataset_id,
        archive_id=archive_id,
        preference_resolution=preference_resolution,
        range_policy="all",
        cell_objective_policy="all",
    )
    coverage_rows = build_preference_coverage(behavior_logs, near_feasible_violation=near_feasible_violation)
    write_csv(behavior_path, behavior_logs)
    write_csv(coverage_path, coverage_rows)


def _seed_preferences_by_context(
    rows: tuple[dict[str, str], ...],
    levels: tuple[str, ...],
    start_seed: int,
    end_seed: int,
    include_realized: bool,
    max_seed_preferences: int,
) -> dict[tuple[str, int, str], tuple[tuple[float, ...], ...]]:
    grouped: dict[tuple[str, int, str], list[tuple[str, tuple[float, ...]]]] = defaultdict(list)
    level_set = set(levels)
    allowed_modes = {"raw", "realized"} if include_realized else {"raw"}
    for row in rows:
        level = str(row.get("level", ""))
        if level not in level_set:
            continue
        seed = int(str(row.get("seed", "0")))
        if seed < start_seed or seed > end_seed:
            continue
        mode = str(row.get("mode", "raw"))
        if mode not in allowed_modes:
            continue
        preference = _parse_preference(str(row.get("preference", "")))
        if not preference:
            continue
        grouped[(level, seed, mode)].append((str(row.get("preference_id", "")), preference))
    return {
        key: _dedupe_preferences(sorted(values, key=lambda item: item[0]))[:max_seed_preferences]
        for key, values in grouped.items()
    }


def _with_event_triggered_realized_contexts(
    preferences_by_context: dict[tuple[str, int, str], tuple[tuple[float, ...], ...]],
) -> dict[tuple[str, int, str], tuple[tuple[float, ...], ...]]:
    result = dict(preferences_by_context)
    for level, seed, mode in tuple(preferences_by_context):
        if mode != "raw":
            continue
        result.setdefault((level, seed, "realized"), preferences_by_context[(level, seed, mode)])
    return result


def _seed_hints_by_context(
    rows: tuple[dict[str, str], ...],
    levels: tuple[str, ...],
    start_seed: int,
    end_seed: int,
    include_realized: bool,
    max_seed_hints: int,
) -> dict[tuple[str, int, str], tuple[MoeaSeedHint, ...]]:
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    level_set = set(levels)
    allowed_modes = {"raw", "realized"} if include_realized else {"raw"}
    for row in rows:
        level = str(row.get("level", ""))
        if level not in level_set:
            continue
        seed = int(str(row.get("seed", "0")))
        if seed < start_seed or seed > end_seed:
            continue
        mode = str(row.get("mode", "raw"))
        if mode not in allowed_modes:
            continue
        grouped[(level, seed, mode)].append(row)
    return {
        key: _build_seed_hints(sorted(values, key=lambda row: str(row.get("preference_id", ""))))[:max_seed_hints]
        for key, values in grouped.items()
    }


def _apply_seed_teacher_mask(
    rows: tuple[dict[str, str], ...],
    seed_teacher_mask: str,
) -> tuple[dict[str, str], ...]:
    if seed_teacher_mask == "full":
        return rows
    return tuple(_masked_seed_row(row, seed_teacher_mask) for row in rows)


def _masked_seed_row(row: dict[str, str], seed_teacher_mask: str) -> dict[str, str]:
    masked = dict(row)
    if seed_teacher_mask in {"no-repair-signal", "objective-only"}:
        masked["path_repair_cost"] = "0.0"
        masked["path_conflict_delay"] = "0.0"
        _mask_signature_tokens(masked, ("path-buffer", "flow-spread"), "masked-path")
    if seed_teacher_mask in {"no-event-signal", "objective-only"}:
        masked["event_recovery_cost"] = "0.0"
        masked["stability"] = "0.0"
        _mask_signature_tokens(masked, ("event-stable",), "masked-event")
    if seed_teacher_mask == "objective-only":
        _mask_profile_fields(masked)
    return masked


def _mask_signature_tokens(row: dict[str, str], tokens: tuple[str, ...], replacement: str) -> None:
    for key in (
        "algorithm",
        "algorithm_variant",
        "selected_algorithm",
        "base_algorithm",
        "search_profile",
        "tree_gp_family",
    ):
        value = str(row.get(key, ""))
        for token in tokens:
            value = value.replace(token, replacement)
        row[key] = value


def _mask_profile_fields(row: dict[str, str]) -> None:
    row["algorithm"] = "objective-only"
    row["algorithm_variant"] = "objective-only"
    row["selected_algorithm"] = "objective-only"
    row["base_algorithm"] = "objective-only"
    row["search_profile"] = "objective-only"
    row["selected_candidate_source"] = "objective-only"
    row["tree_gp_family"] = ""


def _learned_seed_hints_by_context(
    rows: tuple[dict[str, str], ...],
    context_preferences: dict[tuple[str, int, str], tuple[tuple[float, ...], ...]],
    max_seed_hints: int,
    seed_neighbor_count: int,
) -> dict[tuple[str, int, str], tuple[MoeaSeedHint, ...]]:
    result = {}
    for context, preferences in context_preferences.items():
        level, seed, mode = context
        training_rows = _training_rows_for_context(rows, level=level, seed=seed, mode=mode)
        if not training_rows:
            result[context] = ()
            continue
        selected = _nearest_training_rows(
            training_rows=training_rows,
            target_preferences=preferences,
            neighbor_count=seed_neighbor_count,
        )
        result[context] = _build_seed_hints(selected)[:max_seed_hints]
    return result


def _trained_seed_hints_by_context(
    rows: tuple[dict[str, str], ...],
    context_preferences: dict[tuple[str, int, str], tuple[tuple[float, ...], ...]],
    size: str,
    max_seed_hints: int,
    seed_neighbor_count: int,
) -> dict[tuple[str, int, str], tuple[MoeaSeedHint, ...]]:
    result = {}
    for context, preferences in context_preferences.items():
        level, seed, mode = context
        training_rows = _training_rows_for_context(rows, level=level, seed=seed, mode=mode)
        if not training_rows:
            result[context] = ()
            continue
        selected = _trained_training_rows(
            training_rows=training_rows,
            target_preferences=preferences,
            neighbor_count=seed_neighbor_count,
        )
        result[context] = _build_seed_hints(selected)[:max_seed_hints]
    return result


def _igsp_greedy_seed_hints_by_context(
    rows: tuple[dict[str, str], ...],
    context_preferences: dict[tuple[str, int, str], tuple[tuple[float, ...], ...]],
    max_seed_hints: int,
) -> dict[tuple[str, int, str], tuple[MoeaSeedHint, ...]]:
    result = {}
    for context, preferences in context_preferences.items():
        level, seed, mode = context
        training_rows = _training_rows_for_context(rows, level=level, seed=seed, mode=mode)
        if not training_rows:
            result[context] = ()
            continue
        selected = _igsp_greedy_training_rows(
            training_rows=training_rows,
            target_preferences=preferences,
            max_seed_hints=max_seed_hints,
        )
        result[context] = _build_seed_hints(selected)[:max_seed_hints]
    return result


def _graph_trained_seed_hints_by_context(
    rows: tuple[dict[str, str], ...],
    context_preferences: dict[tuple[str, int, str], tuple[tuple[float, ...], ...]],
    size: str,
    max_seed_hints: int,
    seed_neighbor_count: int,
) -> dict[tuple[str, int, str], tuple[MoeaSeedHint, ...]]:
    result = {}
    feature_cache: dict[tuple[str, int, str], tuple[float, ...]] = {}
    for context, preferences in context_preferences.items():
        level, seed, mode = context
        training_rows = _training_rows_for_context(rows, level=level, seed=seed, mode=mode)
        if not training_rows:
            result[context] = ()
            continue
        target_features = _graph_context_features(level, seed, mode, size, feature_cache)
        selected = _graph_trained_training_rows(
            training_rows=training_rows,
            target_preferences=preferences,
            target_features=target_features,
            size=size,
            feature_cache=feature_cache,
            neighbor_count=seed_neighbor_count,
        )
        result[context] = _build_seed_hints(selected)[:max_seed_hints]
    return result


def _action_trained_seed_hints_by_context(
    rows: tuple[dict[str, str], ...],
    context_preferences: dict[tuple[str, int, str], tuple[tuple[float, ...], ...]],
    max_seed_hints: int,
    seed_neighbor_count: int,
) -> dict[tuple[str, int, str], tuple[MoeaSeedHint, ...]]:
    result = {}
    for context, preferences in context_preferences.items():
        level, seed, mode = context
        training_rows = _training_rows_for_context(rows, level=level, seed=seed, mode=mode)
        if not training_rows:
            result[context] = ()
            continue
        selected = _action_trained_training_rows(
            training_rows=training_rows,
            target_preferences=preferences,
            neighbor_count=seed_neighbor_count,
        )
        result[context] = _build_seed_hints(selected)[:max_seed_hints]
    return result


def _partial_action_trained_seed_hints_by_context(
    rows: tuple[dict[str, str], ...],
    context_preferences: dict[tuple[str, int, str], tuple[tuple[float, ...], ...]],
    max_seed_hints: int,
    seed_neighbor_count: int,
) -> dict[tuple[str, int, str], tuple[MoeaSeedHint, ...]]:
    result = {}
    for context, preferences in context_preferences.items():
        level, seed, mode = context
        training_rows = _training_rows_for_context(rows, level=level, seed=seed, mode=mode)
        if not training_rows:
            result[context] = ()
            continue
        selected = _partial_action_trained_training_rows(
            training_rows=training_rows,
            target_preferences=preferences,
            neighbor_count=seed_neighbor_count,
        )
        result[context] = _build_seed_hints(selected)[:max_seed_hints]
    return result


def _partial_schedule_trained_seed_hints_by_context(
    rows: tuple[dict[str, str], ...],
    context_preferences: dict[tuple[str, int, str], tuple[tuple[float, ...], ...]],
    size: str,
    max_seed_hints: int,
    seed_neighbor_count: int,
) -> dict[tuple[str, int, str], tuple[MoeaSeedHint, ...]]:
    result = {}
    feature_cache: dict[tuple[str, int, str], tuple[float, ...]] = {}
    for context, preferences in context_preferences.items():
        level, seed, mode = context
        training_rows = _training_rows_for_context(rows, level=level, seed=seed, mode=mode)
        if not training_rows:
            result[context] = ()
            continue
        target_features = _graph_context_features(level, seed, mode, size, feature_cache)
        selected = _partial_schedule_trained_training_rows(
            training_rows=training_rows,
            target_preferences=preferences,
            target_features=target_features,
            size=size,
            feature_cache=feature_cache,
            neighbor_count=seed_neighbor_count,
        )
        result[context] = _build_seed_hints(selected)[:max_seed_hints]
    return result


def _graph_action_trained_seed_hints_by_context(
    rows: tuple[dict[str, str], ...],
    context_preferences: dict[tuple[str, int, str], tuple[tuple[float, ...], ...]],
    size: str,
    max_seed_hints: int,
    seed_neighbor_count: int,
) -> dict[tuple[str, int, str], tuple[MoeaSeedHint, ...]]:
    result = {}
    feature_cache: dict[tuple[str, int, str], tuple[float, ...]] = {}
    for context, preferences in context_preferences.items():
        level, seed, mode = context
        training_rows = _training_rows_for_context(rows, level=level, seed=seed, mode=mode)
        if not training_rows:
            result[context] = ()
            continue
        target_features = _graph_context_features(level, seed, mode, size, feature_cache)
        selected = _graph_action_trained_training_rows(
            training_rows=training_rows,
            target_preferences=preferences,
            target_features=target_features,
            size=size,
            feature_cache=feature_cache,
            neighbor_count=seed_neighbor_count,
        )
        result[context] = _build_seed_hints(selected)[:max_seed_hints]
    return result


def _nonlinear_graph_action_trained_seed_hints_by_context(
    rows: tuple[dict[str, str], ...],
    context_preferences: dict[tuple[str, int, str], tuple[tuple[float, ...], ...]],
    size: str,
    max_seed_hints: int,
    seed_neighbor_count: int,
) -> dict[tuple[str, int, str], tuple[MoeaSeedHint, ...]]:
    result = {}
    feature_cache: dict[tuple[str, int, str], tuple[float, ...]] = {}
    for context, preferences in context_preferences.items():
        level, seed, mode = context
        training_rows = _training_rows_for_context(rows, level=level, seed=seed, mode=mode)
        if not training_rows:
            result[context] = ()
            continue
        target_features = _graph_context_features(level, seed, mode, size, feature_cache)
        selected = _nonlinear_graph_action_trained_training_rows(
            training_rows=training_rows,
            target_preferences=preferences,
            target_features=target_features,
            size=size,
            feature_cache=feature_cache,
            neighbor_count=seed_neighbor_count,
        )
        result[context] = _build_seed_hints(selected)[:max_seed_hints]
    return result


def _training_rows_for_context(
    rows: tuple[dict[str, str], ...],
    level: str,
    seed: int,
    mode: str,
) -> list[dict[str, str]]:
    same_context = [
        row
        for row in rows
        if str(row.get("level", "")) == level
        and str(row.get("mode", "raw")) == mode
        and int(str(row.get("seed", "0"))) != seed
    ]
    if same_context:
        return same_context
    return [
        row
        for row in rows
        if int(str(row.get("seed", "0"))) != seed
    ]


def _nearest_training_rows(
    training_rows: list[dict[str, str]],
    target_preferences: tuple[tuple[float, ...], ...],
    neighbor_count: int,
) -> list[dict[str, str]]:
    ranked_by_preference = []
    for preference in target_preferences:
        ranked_by_preference.append(
            sorted(
                training_rows,
                key=lambda row: (
                    _preference_distance(preference, _parse_preference(str(row.get("preference", "")))),
                    str(row.get("level", "")),
                    str(row.get("mode", "")),
                    str(row.get("seed", "")),
                    str(row.get("preference_id", "")),
                ),
            )
        )
    selected = []
    seen = set()
    for neighbor_index in range(neighbor_count):
        for ranked in ranked_by_preference:
            if neighbor_index >= len(ranked):
                continue
            row = ranked[neighbor_index]
            key = (
                str(row.get("level", "")),
                str(row.get("mode", "")),
                str(row.get("seed", "")),
                str(row.get("preference_id", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
    return selected


def _trained_training_rows(
    training_rows: list[dict[str, str]],
    target_preferences: tuple[tuple[float, ...], ...],
    neighbor_count: int,
) -> list[dict[str, str]]:
    ranges = _objective_ranges(training_rows)
    ranked_by_preference = []
    for preference in target_preferences:
        ranked_by_preference.append(
            sorted(
                training_rows,
                key=lambda row: (
                    _trained_seed_score(row, ranges, preference),
                    _preference_distance(preference, _parse_preference(str(row.get("preference", "")))),
                    str(row.get("level", "")),
                    str(row.get("mode", "")),
                    str(row.get("seed", "")),
                    str(row.get("preference_id", "")),
                ),
            )
        )
    selected = []
    seen = set()
    for neighbor_index in range(neighbor_count):
        for ranked in ranked_by_preference:
            if neighbor_index >= len(ranked):
                continue
            row = ranked[neighbor_index]
            key = (
                str(row.get("level", "")),
                str(row.get("mode", "")),
                str(row.get("seed", "")),
                str(row.get("preference_id", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
    return selected


def _igsp_greedy_training_rows(
    training_rows: list[dict[str, str]],
    target_preferences: tuple[tuple[float, ...], ...],
    max_seed_hints: int,
) -> list[dict[str, str]]:
    if max_seed_hints <= 0:
        return []
    ranges = _objective_ranges(training_rows)
    anchors = _igsp_anchor_vectors(training_rows, target_preferences, ranges)
    if not anchors:
        return _trained_training_rows(training_rows, target_preferences, max_seed_hints)
    candidate_profiles = {
        _seed_row_key(row): _igsp_candidate_profile(row, ranges, anchors)
        for row in training_rows
    }
    selected: list[dict[str, str]] = []
    selected_keys: set[tuple[str, str, str, str]] = set()
    covered = tuple(0.0 for _anchor in anchors)
    while len(selected) < max_seed_hints and len(selected_keys) < len(candidate_profiles):
        best_row = None
        best_gain = -1e18
        for row in training_rows:
            key = _seed_row_key(row)
            if key in selected_keys:
                continue
            profile = candidate_profiles[key]
            gain = _igsp_marginal_gain(
                profile,
                selected,
                covered,
            )
            if gain > best_gain:
                best_gain = gain
                best_row = row
        if best_row is None:
            break
        selected.append(best_row)
        selected_keys.add(_seed_row_key(best_row))
        affinity = candidate_profiles[_seed_row_key(best_row)]["anchor_affinity"]
        covered = tuple(max(left, right) for left, right in zip(covered, affinity))
    return selected


def _igsp_anchor_vectors(
    rows: list[dict[str, str]],
    target_preferences: tuple[tuple[float, ...], ...],
    ranges: dict[str, tuple[float, float]],
) -> tuple[tuple[float, ...], ...]:
    anchors: list[tuple[float, ...]] = []
    for preference in target_preferences:
        if preference:
            anchors.append(_igsp_preference_anchor(preference))
    for row in sorted(rows, key=lambda item: (_igsp_quality(item, ranges), str(item.get("preference_id", ""))))[:24]:
        vector = _igsp_normalized_objective_vector(row, ranges)
        if vector:
            anchors.append(vector)
    return _dedupe_vectors(anchors)


def _igsp_preference_anchor(preference: tuple[float, ...]) -> tuple[float, ...]:
    if not preference:
        return ()
    total = sum(max(0.0, value) for value in preference)
    normalized = tuple((max(0.0, value) / total) if total > 0.0 else (1.0 / len(preference)) for value in preference)
    return tuple(1.0 - value for value in normalized)


def _igsp_candidate_profile(
    row: dict[str, str],
    ranges: dict[str, tuple[float, float]],
    anchors: tuple[tuple[float, ...], ...],
) -> dict[str, object]:
    vector = _igsp_normalized_objective_vector(row, ranges)
    quality = _igsp_quality(row, ranges)
    risk = _igsp_risk(row, ranges)
    affinity = tuple(_igsp_anchor_affinity(vector, anchor) for anchor in anchors)
    return {
        "anchor_affinity": affinity,
        "quality": quality,
        "risk": risk,
        "template_key": _igsp_template_key(row),
    }


def _igsp_marginal_gain(
    profile: dict[str, object],
    selected: list[dict[str, str]],
    covered: tuple[float, ...],
) -> float:
    affinity = tuple(float(value) for value in profile["anchor_affinity"])
    coverage_gain = sum(max(0.0, right - left) for left, right in zip(covered, affinity))
    quality = float(profile["quality"])
    risk = float(profile["risk"])
    redundancy = _igsp_redundancy_penalty(str(profile["template_key"]), selected)
    return coverage_gain + quality - 0.50 * risk - redundancy


def _igsp_normalized_objective_vector(
    row: dict[str, str],
    ranges: dict[str, tuple[float, float]],
) -> tuple[float, ...]:
    objectives = tuple(item for item in str(row.get("objectives", "")).split(";") if item)
    values = []
    for objective in objectives:
        row_key = "path_repair_cost" if objective == "path_conflict_delay" else objective
        row_key = "event_recovery_cost" if row_key == "stability" else row_key
        low, high = ranges.get(row_key, (0.0, 0.0))
        value = _row_objective(row, row_key)
        values.append(0.5 if high <= low else (value - low) / (high - low))
    return tuple(values)


def _igsp_quality(row: dict[str, str], ranges: dict[str, tuple[float, float]]) -> float:
    vector = _igsp_normalized_objective_vector(row, ranges)
    if not vector:
        return 0.0
    ideal_distance = math.sqrt(sum(value * value for value in vector) / len(vector))
    return 1.0 - min(1.0, ideal_distance)


def _igsp_risk(row: dict[str, str], ranges: dict[str, tuple[float, float]]) -> float:
    vector = _igsp_normalized_objective_vector(row, ranges)
    if not vector:
        return 0.0
    level = str(row.get("level", ""))
    mode = str(row.get("mode", "raw"))
    objectives = tuple(item for item in str(row.get("objectives", "")).split(";") if item)
    values = {objective: vector[index] for index, objective in enumerate(objectives)}
    repair_risk = values.get("path_conflict_delay", values.get("path_repair_cost", 0.0))
    event_risk = values.get("stability", values.get("event_recovery_cost", 0.0))
    context_multiplier = 1.25 if level == "L5" and mode == "raw" else 1.0
    return context_multiplier * (0.60 * repair_risk + 0.40 * event_risk)


def _igsp_anchor_affinity(vector: tuple[float, ...], anchor: tuple[float, ...]) -> float:
    if len(vector) != len(anchor) or not vector:
        return 0.0
    distance = math.sqrt(sum((left - right) ** 2 for left, right in zip(vector, anchor)) / len(vector))
    return math.exp(-3.0 * distance)


def _igsp_redundancy_penalty(template_key: str, selected: list[dict[str, str]]) -> float:
    if not selected:
        return 0.0
    repeat_count = sum(1 for row in selected if _igsp_template_key(row) == template_key)
    return 0.15 * repeat_count


def _igsp_template_key(row: dict[str, str]) -> str:
    preference = _parse_preference(str(row.get("preference", "")))
    return f"{_operation_template(row, preference)}|{_machine_template(row)}|{_row_signature(row)}"


def _seed_row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        str(row.get("level", "")),
        str(row.get("mode", "")),
        str(row.get("seed", "")),
        str(row.get("preference_id", "")),
    )


def _dedupe_vectors(vectors: list[tuple[float, ...]]) -> tuple[tuple[float, ...], ...]:
    result = []
    seen = set()
    for vector in vectors:
        key = tuple(round(value, 6) for value in vector)
        if key in seen:
            continue
        seen.add(key)
        result.append(vector)
    return tuple(result)


def _partial_action_trained_training_rows(
    training_rows: list[dict[str, str]],
    target_preferences: tuple[tuple[float, ...], ...],
    neighbor_count: int,
) -> list[dict[str, str]]:
    ranges = _objective_ranges(training_rows)
    action_examples = _partial_action_training_examples(training_rows, ranges)
    ranked_by_preference = []
    for preference in target_preferences:
        ranked_by_preference.append(
            sorted(
                training_rows,
                key=lambda row: (
                    _partial_action_seed_score(row, ranges, preference, action_examples),
                    _partial_action_prior(row, preference, action_examples),
                    _preference_distance(preference, _parse_preference(str(row.get("preference", "")))),
                    str(row.get("level", "")),
                    str(row.get("mode", "")),
                    str(row.get("seed", "")),
                    str(row.get("preference_id", "")),
                ),
            )
        )
    selected = []
    seen = set()
    for neighbor_index in range(neighbor_count):
        for ranked in ranked_by_preference:
            if neighbor_index >= len(ranked):
                continue
            row = ranked[neighbor_index]
            key = (
                str(row.get("level", "")),
                str(row.get("mode", "")),
                str(row.get("seed", "")),
                str(row.get("preference_id", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
    return selected


def _action_trained_training_rows(
    training_rows: list[dict[str, str]],
    target_preferences: tuple[tuple[float, ...], ...],
    neighbor_count: int,
) -> list[dict[str, str]]:
    ranges = _objective_ranges(training_rows)
    ranked_by_preference = []
    for preference in target_preferences:
        ranked_by_preference.append(
            sorted(
                training_rows,
                key=lambda row: (
                    _trained_seed_score(row, ranges, preference)
                    + 0.25 * _action_template_distance(row, preference),
                    _action_template_distance(row, preference),
                    _preference_distance(preference, _parse_preference(str(row.get("preference", "")))),
                    str(row.get("level", "")),
                    str(row.get("mode", "")),
                    str(row.get("seed", "")),
                    str(row.get("preference_id", "")),
                ),
            )
        )
    selected = []
    seen = set()
    for neighbor_index in range(neighbor_count):
        for ranked in ranked_by_preference:
            if neighbor_index >= len(ranked):
                continue
            row = ranked[neighbor_index]
            key = (
                str(row.get("level", "")),
                str(row.get("mode", "")),
                str(row.get("seed", "")),
                str(row.get("preference_id", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
    return selected


def _partial_schedule_trained_training_rows(
    training_rows: list[dict[str, str]],
    target_preferences: tuple[tuple[float, ...], ...],
    target_features: tuple[float, ...],
    size: str,
    feature_cache: dict[tuple[str, int, str], tuple[float, ...]],
    neighbor_count: int,
) -> list[dict[str, str]]:
    ranges = _objective_ranges(training_rows)
    schedule_examples = _partial_schedule_training_examples(training_rows, ranges, size, feature_cache)
    ranked_by_preference = []
    for preference in target_preferences:
        ranked_by_preference.append(
            sorted(
                training_rows,
                key=lambda row: (
                    _partial_schedule_seed_score(
                        row,
                        ranges,
                        preference,
                        target_features,
                        schedule_examples,
                        size,
                        feature_cache,
                    ),
                    _partial_schedule_prior(row, preference, target_features, schedule_examples),
                    _partial_schedule_context_distance(
                        target_features,
                        _row_graph_context_features(row, size, feature_cache),
                    ),
                    _preference_distance(preference, _parse_preference(str(row.get("preference", "")))),
                    str(row.get("level", "")),
                    str(row.get("mode", "")),
                    str(row.get("seed", "")),
                    str(row.get("preference_id", "")),
                ),
            )
        )
    selected = []
    seen = set()
    for neighbor_index in range(neighbor_count):
        for ranked in ranked_by_preference:
            if neighbor_index >= len(ranked):
                continue
            row = ranked[neighbor_index]
            key = (
                str(row.get("level", "")),
                str(row.get("mode", "")),
                str(row.get("seed", "")),
                str(row.get("preference_id", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
    return selected


def _graph_action_trained_training_rows(
    training_rows: list[dict[str, str]],
    target_preferences: tuple[tuple[float, ...], ...],
    target_features: tuple[float, ...],
    size: str,
    feature_cache: dict[tuple[str, int, str], tuple[float, ...]],
    neighbor_count: int,
) -> list[dict[str, str]]:
    ranges = _objective_ranges(training_rows)
    model = _fit_graph_action_model(training_rows, ranges, size, feature_cache)
    ranked_by_preference = []
    for preference in target_preferences:
        ranked_by_preference.append(
            sorted(
                training_rows,
                key=lambda row: (
                    _graph_action_seed_score(
                        row,
                        ranges,
                        preference,
                        target_features,
                        model,
                    ),
                    _graph_action_model_prediction(row, preference, target_features, model),
                    _trained_seed_score(row, ranges, preference),
                    _preference_distance(preference, _parse_preference(str(row.get("preference", "")))),
                    str(row.get("level", "")),
                    str(row.get("mode", "")),
                    str(row.get("seed", "")),
                    str(row.get("preference_id", "")),
                ),
            )
        )
    selected = []
    seen = set()
    for neighbor_index in range(neighbor_count):
        for ranked in ranked_by_preference:
            if neighbor_index >= len(ranked):
                continue
            row = ranked[neighbor_index]
            key = (
                str(row.get("level", "")),
                str(row.get("mode", "")),
                str(row.get("seed", "")),
                str(row.get("preference_id", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
    return selected


def _nonlinear_graph_action_trained_training_rows(
    training_rows: list[dict[str, str]],
    target_preferences: tuple[tuple[float, ...], ...],
    target_features: tuple[float, ...],
    size: str,
    feature_cache: dict[tuple[str, int, str], tuple[float, ...]],
    neighbor_count: int,
) -> list[dict[str, str]]:
    ranges = _objective_ranges(training_rows)
    model = _fit_nonlinear_graph_action_model(training_rows, ranges, size, feature_cache)
    ranked_by_preference = []
    for preference in target_preferences:
        ranked_by_preference.append(
            sorted(
                training_rows,
                key=lambda row: (
                    _nonlinear_graph_action_seed_score(
                        row,
                        ranges,
                        preference,
                        target_features,
                        model,
                    ),
                    _nonlinear_graph_action_model_prediction(row, preference, target_features, model),
                    _trained_seed_score(row, ranges, preference),
                    _preference_distance(preference, _parse_preference(str(row.get("preference", "")))),
                    str(row.get("level", "")),
                    str(row.get("mode", "")),
                    str(row.get("seed", "")),
                    str(row.get("preference_id", "")),
                ),
            )
        )
    selected = []
    seen = set()
    for neighbor_index in range(neighbor_count):
        for ranked in ranked_by_preference:
            if neighbor_index >= len(ranked):
                continue
            row = ranked[neighbor_index]
            key = (
                str(row.get("level", "")),
                str(row.get("mode", "")),
                str(row.get("seed", "")),
                str(row.get("preference_id", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
    return selected


def _graph_trained_training_rows(
    training_rows: list[dict[str, str]],
    target_preferences: tuple[tuple[float, ...], ...],
    target_features: tuple[float, ...],
    size: str,
    feature_cache: dict[tuple[str, int, str], tuple[float, ...]],
    neighbor_count: int,
) -> list[dict[str, str]]:
    ranges = _objective_ranges(training_rows)
    ranked_by_preference = []
    for preference in target_preferences:
        ranked_by_preference.append(
            sorted(
                training_rows,
                key=lambda row: (
                    _trained_seed_score(row, ranges, preference)
                    + 0.10
                    * _graph_context_distance(
                        target_features,
                        _graph_context_features(
                            str(row.get("level", "")),
                            int(str(row.get("seed", "0"))),
                            str(row.get("mode", "raw")),
                            size,
                            feature_cache,
                        ),
                    ),
                    _preference_distance(preference, _parse_preference(str(row.get("preference", "")))),
                    str(row.get("level", "")),
                    str(row.get("mode", "")),
                    str(row.get("seed", "")),
                    str(row.get("preference_id", "")),
                ),
            )
        )
    selected = []
    seen = set()
    for neighbor_index in range(neighbor_count):
        for ranked in ranked_by_preference:
            if neighbor_index >= len(ranked):
                continue
            row = ranked[neighbor_index]
            key = (
                str(row.get("level", "")),
                str(row.get("mode", "")),
                str(row.get("seed", "")),
                str(row.get("preference_id", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
    return selected


def _graph_context_features(
    level: str,
    seed: int,
    mode: str,
    size: str,
    cache: dict[tuple[str, int, str], tuple[float, ...]],
) -> tuple[float, ...]:
    key = (level, seed, mode)
    if key not in cache:
        instance = generate_instance(level, size, seed)
        structural_totals = (0.0, 0.0, 0.0)
        if mode == "realized":
            realized = realize_structural_events(instance)
            instance = realized.instance
            structural_totals = (
                float(realized.report.added_jobs),
                float(realized.report.added_operations),
                float(realized.report.total_updates),
            )
        operation_count = sum(len(job.operations) for job in instance.jobs)
        avg_ops_per_job = operation_count / len(instance.jobs) if instance.jobs else 0.0
        avg_eligible_machines = _avg(
            len(operation.eligible_machines) for job in instance.jobs for operation in job.operations
        )
        bottleneck_edges = sum(1 for edge in instance.path_edges if edge.is_bottleneck)
        low_capacity_edges = sum(1 for edge in instance.path_edges if edge.capacity <= 1)
        avg_edge_capacity = _avg(edge.capacity for edge in instance.path_edges)
        event_pressure = sum(float(event.severity) * max(1.0, float(event.duration)) for event in instance.events)
        cache[key] = (
            len(instance.jobs) / 25.0,
            operation_count / 200.0,
            len(instance.machines) / 12.0,
            len(instance.agvs) / 8.0,
            len(instance.path_nodes) / 30.0,
            len(instance.path_edges) / 60.0,
            bottleneck_edges / 10.0,
            low_capacity_edges / 30.0,
            avg_edge_capacity / 4.0,
            len(instance.events) / 12.0,
            event_pressure / 250.0,
            avg_ops_per_job / 10.0,
            avg_eligible_machines / 8.0,
            structural_totals[0] / 10.0,
            structural_totals[1] / 20.0,
            structural_totals[2] / 20.0,
        )
    return cache[key]


def _graph_context_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        return 1.0
    return sum(abs(left_value - right_value) for left_value, right_value in zip(left, right)) / len(left)


def _trained_seed_score(
    row: dict[str, str],
    ranges: dict[str, tuple[float, float]],
    target_preference: tuple[float, ...],
) -> float:
    objectives = tuple(item for item in str(row.get("objectives", "")).split(";") if item)
    if len(objectives) != len(target_preference) or len(target_preference) < 5:
        return 1e9
    normalized = []
    for objective in objectives:
        row_key = "path_repair_cost" if objective == "path_conflict_delay" else objective
        row_key = "event_recovery_cost" if row_key == "stability" else row_key
        low, high = ranges.get(row_key, (0.0, 0.0))
        value = _row_objective(row, row_key)
        normalized.append(0.5 if high <= low else (value - low) / (high - low))
    weighted_tchebycheff = max(weight * value for weight, value in zip(target_preference, normalized))
    preference_tail = _preference_tail(row, target_preference)
    outcome_tail = _safe_simplex(
        (
            1.0 - normalized[0] + 1.0 - normalized[3],
            1.0 - normalized[2],
            1.0 - normalized[1] + 1.0 - normalized[4],
        )
    )
    tail_mismatch = sum(abs(left - right) for left, right in zip(preference_tail, outcome_tail))
    profile_penalty = _profile_penalty(row, target_preference)
    return weighted_tchebycheff + 0.05 * tail_mismatch + profile_penalty


def _partial_action_training_examples(
    rows: list[dict[str, str]],
    ranges: dict[str, tuple[float, float]],
) -> tuple[tuple[tuple[float, float, float], tuple[str, str], float], ...]:
    examples = []
    for row in rows:
        preference = _parse_preference(str(row.get("preference", "")))
        if not preference:
            continue
        operation_template = _operation_template(row, preference)
        machine_template = _machine_template(row)
        examples.append(
            (
                _preference_tail(row, preference),
                (operation_template, machine_template),
                _trained_seed_score(row, ranges, preference),
            )
        )
    return tuple(examples)


def _partial_action_seed_score(
    row: dict[str, str],
    ranges: dict[str, tuple[float, float]],
    target_preference: tuple[float, ...],
    action_examples: tuple[tuple[tuple[float, float, float], tuple[str, str], float], ...],
) -> float:
    outcome_score = _trained_seed_score(row, ranges, target_preference)
    action_prior = _partial_action_prior(row, target_preference, action_examples)
    preference_distance = _preference_distance(target_preference, _parse_preference(str(row.get("preference", ""))))
    return 0.65 * outcome_score + 0.35 * action_prior + 0.05 * preference_distance


def _partial_action_prior(
    row: dict[str, str],
    target_preference: tuple[float, ...],
    action_examples: tuple[tuple[tuple[float, float, float], tuple[str, str], float], ...],
) -> float:
    if not action_examples:
        return 0.5
    row_preference = _parse_preference(str(row.get("preference", ""))) or target_preference
    action_key = (_operation_template(row, row_preference), _machine_template(row))
    target_tail = _preference_tail(row, target_preference)
    weighted_total = 0.0
    weight_sum = 0.0
    for example_tail, example_action_key, example_score in action_examples:
        if example_action_key != action_key:
            continue
        distance = sum(abs(left - right) for left, right in zip(target_tail, example_tail))
        weight = 1.0 / (0.25 + distance)
        weighted_total += weight * example_score
        weight_sum += weight
    if weight_sum <= 0.0:
        return _avg(example_score for _example_tail, _example_action_key, example_score in action_examples)
    return weighted_total / weight_sum


def _partial_schedule_training_examples(
    rows: list[dict[str, str]],
    ranges: dict[str, tuple[float, float]],
    size: str,
    feature_cache: dict[tuple[str, int, str], tuple[float, ...]],
) -> tuple[tuple[tuple[float, ...], tuple[float, float, float], tuple[float, float, float], tuple[str, str], float], ...]:
    examples = []
    for row in rows:
        preference = _parse_preference(str(row.get("preference", "")))
        if not preference:
            continue
        operation_template = _operation_template(row, preference)
        machine_template = _machine_template(row)
        examples.append(
            (
                _row_graph_context_features(row, size, feature_cache),
                _preference_tail(row, preference),
                _action_template_tail(operation_template, machine_template),
                (operation_template, machine_template),
                _trained_seed_score(row, ranges, preference),
            )
        )
    return tuple(examples)


def _partial_schedule_seed_score(
    row: dict[str, str],
    ranges: dict[str, tuple[float, float]],
    target_preference: tuple[float, ...],
    target_features: tuple[float, ...],
    schedule_examples: tuple[
        tuple[tuple[float, ...], tuple[float, float, float], tuple[float, float, float], tuple[str, str], float],
        ...,
    ],
    size: str,
    feature_cache: dict[tuple[str, int, str], tuple[float, ...]],
) -> float:
    outcome_score = _trained_seed_score(row, ranges, target_preference)
    schedule_prior = _partial_schedule_prior(row, target_preference, target_features, schedule_examples)
    graph_distance = _partial_schedule_context_distance(
        target_features,
        _row_graph_context_features(row, size, feature_cache),
    )
    preference_distance = _preference_distance(target_preference, _parse_preference(str(row.get("preference", ""))))
    return 0.45 * outcome_score + 0.35 * schedule_prior + 0.15 * graph_distance + 0.05 * preference_distance


def _partial_schedule_prior(
    row: dict[str, str],
    target_preference: tuple[float, ...],
    target_features: tuple[float, ...],
    schedule_examples: tuple[
        tuple[tuple[float, ...], tuple[float, float, float], tuple[float, float, float], tuple[str, str], float],
        ...,
    ],
) -> float:
    if not schedule_examples:
        return 0.5
    row_preference = _parse_preference(str(row.get("preference", ""))) or target_preference
    operation_template = _operation_template(row, row_preference)
    machine_template = _machine_template(row)
    action_key = (operation_template, machine_template)
    action_tail = _action_template_tail(operation_template, machine_template)
    target_tail = _preference_tail(row, target_preference)
    weighted_total = 0.0
    weight_sum = 0.0
    for example_features, example_tail, example_action_tail, example_action_key, example_score in schedule_examples:
        graph_distance = _partial_schedule_context_distance(target_features, example_features)
        preference_tail_distance = sum(abs(left - right) for left, right in zip(target_tail, example_tail))
        action_tail_distance = sum(abs(left - right) for left, right in zip(action_tail, example_action_tail))
        action_mismatch = 0.0 if example_action_key == action_key else 0.25
        distance = 0.55 * graph_distance + 0.25 * preference_tail_distance + 0.20 * action_tail_distance + action_mismatch
        weight = 1.0 / (0.05 + distance)
        weighted_total += weight * example_score
        weight_sum += weight
    if weight_sum <= 0.0:
        return _avg(example_score for *_example, example_score in schedule_examples)
    return weighted_total / weight_sum


def _row_graph_context_features(
    row: dict[str, str],
    size: str,
    feature_cache: dict[tuple[str, int, str], tuple[float, ...]],
) -> tuple[float, ...]:
    return _graph_context_features(
        str(row.get("level", "")),
        int(str(row.get("seed", "0"))),
        str(row.get("mode", "raw")),
        size,
        feature_cache,
    )


def _partial_schedule_context_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return min(1.0, 25.0 * _graph_context_distance(left, right))


def _fit_graph_action_model(
    rows: list[dict[str, str]],
    ranges: dict[str, tuple[float, float]],
    size: str,
    feature_cache: dict[tuple[str, int, str], tuple[float, ...]],
) -> tuple[tuple[float, ...], float]:
    examples = []
    for row in rows:
        preference = _parse_preference(str(row.get("preference", "")))
        if not preference:
            continue
        context_features = _row_graph_context_features(row, size, feature_cache)
        features = _graph_action_features(row, preference, context_features)
        label = _trained_seed_score(row, ranges, preference)
        if label >= 1e8:
            continue
        examples.append((features, label))
    if not examples:
        return ((), 0.5)
    label_mean = _avg(label for _features, label in examples)
    feature_count = len(examples[0][0])
    weights = [0.0 for _ in range(feature_count)]
    learning_rate = 0.08
    l2_penalty = 0.02
    for _step in range(120):
        gradients = [0.0 for _ in range(feature_count)]
        for features, label in examples:
            prediction = label_mean + sum(weight * value for weight, value in zip(weights, features))
            error = prediction - label
            for index, value in enumerate(features):
                gradients[index] += error * value
        scale = 1.0 / len(examples)
        for index in range(feature_count):
            penalty = 0.0 if index == 0 else l2_penalty * weights[index]
            weights[index] -= learning_rate * (gradients[index] * scale + penalty)
    return (tuple(weights), label_mean)


def _graph_action_seed_score(
    row: dict[str, str],
    ranges: dict[str, tuple[float, float]],
    target_preference: tuple[float, ...],
    target_features: tuple[float, ...],
    model: tuple[tuple[float, ...], float],
) -> float:
    outcome_score = _trained_seed_score(row, ranges, target_preference)
    learned_score = _graph_action_model_prediction(row, target_preference, target_features, model)
    preference_distance = _preference_distance(target_preference, _parse_preference(str(row.get("preference", ""))))
    return 0.60 * outcome_score + 0.35 * learned_score + 0.05 * preference_distance


def _graph_action_model_prediction(
    row: dict[str, str],
    target_preference: tuple[float, ...],
    target_features: tuple[float, ...],
    model: tuple[tuple[float, ...], float],
) -> float:
    weights, label_mean = model
    if not weights:
        return label_mean
    features = _graph_action_features(row, target_preference, target_features)
    return label_mean + sum(weight * value for weight, value in zip(weights, features))


def _fit_nonlinear_graph_action_model(
    rows: list[dict[str, str]],
    ranges: dict[str, tuple[float, float]],
    size: str,
    feature_cache: dict[tuple[str, int, str], tuple[float, ...]],
) -> tuple[tuple[float, ...], float]:
    examples = []
    for row in rows:
        preference = _parse_preference(str(row.get("preference", "")))
        if not preference:
            continue
        context_features = _row_graph_context_features(row, size, feature_cache)
        features = _nonlinear_graph_action_features(row, preference, context_features)
        label = _trained_seed_score(row, ranges, preference)
        if label >= 1e8:
            continue
        examples.append((features, label))
    if not examples:
        return ((), 0.5)
    label_mean = _avg(label for _features, label in examples)
    feature_count = len(examples[0][0])
    weights = [0.0 for _ in range(feature_count)]
    learning_rate = 0.05
    l2_penalty = 0.03
    for _step in range(160):
        gradients = [0.0 for _ in range(feature_count)]
        for features, label in examples:
            prediction = label_mean + sum(weight * value for weight, value in zip(weights, features))
            error = prediction - label
            for index, value in enumerate(features):
                gradients[index] += error * value
        scale = 1.0 / len(examples)
        for index in range(feature_count):
            penalty = 0.0 if index == 0 else l2_penalty * weights[index]
            weights[index] -= learning_rate * (gradients[index] * scale + penalty)
    return (tuple(weights), label_mean)


def _nonlinear_graph_action_seed_score(
    row: dict[str, str],
    ranges: dict[str, tuple[float, float]],
    target_preference: tuple[float, ...],
    target_features: tuple[float, ...],
    model: tuple[tuple[float, ...], float],
) -> float:
    outcome_score = _trained_seed_score(row, ranges, target_preference)
    learned_score = _nonlinear_graph_action_model_prediction(row, target_preference, target_features, model)
    preference_distance = _preference_distance(target_preference, _parse_preference(str(row.get("preference", ""))))
    return 0.50 * outcome_score + 0.45 * learned_score + 0.05 * preference_distance


def _nonlinear_graph_action_model_prediction(
    row: dict[str, str],
    target_preference: tuple[float, ...],
    target_features: tuple[float, ...],
    model: tuple[tuple[float, ...], float],
) -> float:
    weights, label_mean = model
    if not weights:
        return label_mean
    features = _nonlinear_graph_action_features(row, target_preference, target_features)
    return label_mean + sum(weight * value for weight, value in zip(weights, features))


def _graph_action_features(
    row: dict[str, str],
    target_preference: tuple[float, ...],
    context_features: tuple[float, ...],
) -> tuple[float, ...]:
    row_preference = _parse_preference(str(row.get("preference", ""))) or target_preference
    operation_template = _operation_template(row, row_preference)
    machine_template = _machine_template(row)
    target_tail = _preference_tail(row, target_preference)
    action_tail = _action_template_tail(operation_template, machine_template)
    action_tail_gap = tuple(abs(left - right) for left, right in zip(target_tail, action_tail))
    action_tail_product = tuple(left * right for left, right in zip(target_tail, action_tail))
    context_summary = _graph_action_context_summary(context_features)
    context_action_product = tuple(value * action for value in context_summary for action in action_tail)
    return (
        1.0,
        *context_features,
        *target_preference,
        *target_tail,
        *action_tail,
        *action_tail_gap,
        *action_tail_product,
        1.0 if operation_template == "processing" else 0.0,
        1.0 if operation_template == "energy" else 0.0,
        1.0 if operation_template == "due" else 0.0,
        1.0 if operation_template == "index" else 0.0,
        1.0 if machine_template == "energy" else 0.0,
        1.0 if machine_template == "index" else 0.0,
        *context_summary,
        *context_action_product,
    )


def _nonlinear_graph_action_features(
    row: dict[str, str],
    target_preference: tuple[float, ...],
    context_features: tuple[float, ...],
) -> tuple[float, ...]:
    base = _graph_action_features(row, target_preference, context_features)
    row_preference = _parse_preference(str(row.get("preference", ""))) or target_preference
    operation_template = _operation_template(row, row_preference)
    machine_template = _machine_template(row)
    target_tail = _preference_tail(row, target_preference)
    action_tail = _action_template_tail(operation_template, machine_template)
    action_tail_gap = tuple(abs(left - right) for left, right in zip(target_tail, action_tail))
    context_summary = _graph_action_context_summary(context_features)
    context_target_product = tuple(context * target for context in context_summary for target in target_tail)
    context_gap_product = tuple(context * gap for context in context_summary for gap in action_tail_gap)
    target_action_product = tuple(target * action for target in target_tail for action in action_tail)
    context_target_action = tuple(
        context * target * action
        for context in context_summary
        for target in target_tail
        for action in action_tail
    )
    return (
        *base,
        *(value * value for value in context_summary),
        *(value * value for value in target_tail),
        *(value * value for value in action_tail_gap),
        *context_target_product,
        *context_gap_product,
        *target_action_product,
        *context_target_action,
    )


def _graph_action_context_summary(context_features: tuple[float, ...]) -> tuple[float, float, float]:
    if len(context_features) < 13:
        return (0.0, 0.0, 0.0)
    path_pressure = _clamp01(context_features[6] + context_features[7] + context_features[10])
    operation_pressure = _clamp01(context_features[1] + context_features[11])
    flexibility_pressure = _clamp01(1.0 - context_features[12])
    return (path_pressure, operation_pressure, flexibility_pressure)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _profile_penalty(row: dict[str, str], target_preference: tuple[float, ...]) -> float:
    tail = _safe_simplex(
        (
            target_preference[0] + target_preference[3],
            target_preference[2],
            target_preference[1] + target_preference[4],
        )
    )
    dominant = max(range(len(tail)), key=lambda index: (tail[index], -index))
    signature = _row_signature(row)
    if dominant == 0 and any(token in signature for token in ("fast", "long-op", "path-buffer", "flow-spread")):
        return -0.01
    if dominant == 1 and "energy" in signature:
        return -0.01
    if dominant == 2 and ("due" in signature or "event-stable" in signature):
        return -0.01
    return 0.0


def _action_template_distance(row: dict[str, str], target_preference: tuple[float, ...]) -> float:
    row_preference = _parse_preference(str(row.get("preference", ""))) or target_preference
    operation_template = _operation_template(row, row_preference)
    machine_template = _machine_template(row)
    action_tail = _action_template_tail(operation_template, machine_template)
    target_tail = _preference_tail(row, target_preference)
    return sum(abs(left - right) for left, right in zip(action_tail, target_tail))


def _action_template_tail(operation_template: str, machine_template: str) -> tuple[float, float, float]:
    if operation_template == "processing":
        values = (1.0, 0.0, 0.0)
    elif operation_template == "energy":
        values = (0.0, 1.0, 0.0)
    elif operation_template == "due":
        values = (0.0, 0.0, 1.0)
    else:
        values = (1.0, 1.0, 1.0)
    if machine_template == "energy":
        values = (values[0], values[1] + 0.5, values[2])
    return _safe_simplex(values)


def _preference_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        return 1e9
    return sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right))


def _build_seed_hints(rows: list[dict[str, str]]) -> tuple[MoeaSeedHint, ...]:
    ranges = _objective_ranges(rows)
    hints = []
    seen = set()
    for row in rows:
        preference = _parse_preference(str(row.get("preference", "")))
        tail = _schedule_aware_tail(row, ranges, preference)
        operation_template = _operation_template(row, preference)
        machine_template = _machine_template(row)
        key = (
            tuple(round(value, 6) for value in tail),
            operation_template,
            machine_template,
            str(row.get("preference_id", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        hints.append(
            MoeaSeedHint(
                tail_weights=tail,
                operation_template=operation_template,
                machine_template=machine_template,
            )
        )
    return tuple(hints)


def _objective_ranges(rows: list[dict[str, str]]) -> dict[str, tuple[float, float]]:
    result = {}
    for key in ("makespan", "tardiness", "energy", "path_repair_cost", "event_recovery_cost"):
        values = tuple(_row_objective(row, key) for row in rows)
        result[key] = (min(values), max(values)) if values else (0.0, 0.0)
    return result


def _schedule_aware_tail(
    row: dict[str, str],
    ranges: dict[str, tuple[float, float]],
    preference: tuple[float, ...],
) -> tuple[float, float, float]:
    speed = _achievement(row, ranges, "makespan") + _achievement(row, ranges, "path_repair_cost")
    energy = _achievement(row, ranges, "energy")
    due = _achievement(row, ranges, "tardiness") + _achievement(row, ranges, "event_recovery_cost")
    achievement_tail = _safe_simplex((speed, energy, due))
    preference_tail = _preference_tail(row, preference)
    return _safe_simplex(tuple(0.5 * achieved + 0.5 * preferred for achieved, preferred in zip(achievement_tail, preference_tail)))


def _preference_tail(row: dict[str, str], preference: tuple[float, ...]) -> tuple[float, float, float]:
    objectives = tuple(item for item in str(row.get("objectives", "")).split(";") if item)
    if not objectives or len(preference) != len(objectives):
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    weights = {objective: preference[index] for index, objective in enumerate(objectives)}
    speed = weights.get("makespan", 0.0) + weights.get("path_conflict_delay", 0.0) + weights.get("path_repair_cost", 0.0)
    energy = weights.get("energy", 0.0)
    due = weights.get("tardiness", 0.0) + weights.get("stability", 0.0) + weights.get("event_recovery_cost", 0.0)
    return _safe_simplex((speed, energy, due))


def _achievement(
    row: dict[str, str],
    ranges: dict[str, tuple[float, float]],
    key: str,
) -> float:
    low, high = ranges.get(key, (0.0, 0.0))
    if high <= low:
        return 0.5
    value = _row_objective(row, key)
    return 1.0 - ((value - low) / (high - low))


def _row_objective(row: dict[str, str], key: str) -> float:
    if key == "path_repair_cost":
        return _float(row.get("path_repair_cost", row.get("path_conflict_delay", 0.0)))
    if key == "event_recovery_cost":
        return _float(row.get("event_recovery_cost", row.get("stability", 0.0)))
    return _float(row.get(key, 0.0))


def _operation_template(row: dict[str, str], preference: tuple[float, ...]) -> str:
    signature = _row_signature(row)
    if "energy" in signature:
        return "energy"
    if "due" in signature or "event-stable" in signature:
        return "due"
    if "fast" in signature or "long-op" in signature or "path-buffer" in signature or "flow-spread" in signature:
        return "processing"
    dominant = _dominant_objective(row, preference)
    if dominant == "energy":
        return "energy"
    if dominant in {"tardiness", "stability", "event_recovery_cost"}:
        return "due"
    if dominant in {"makespan", "path_conflict_delay", "path_repair_cost"}:
        return "processing"
    return "index"


def _machine_template(row: dict[str, str]) -> str:
    return "energy" if "energy" in _row_signature(row) else "index"


def _dominant_objective(row: dict[str, str], preference: tuple[float, ...]) -> str:
    objectives = tuple(item for item in str(row.get("objectives", "")).split(";") if item)
    if not preference or len(preference) != len(objectives):
        return ""
    index = min(range(len(preference)), key=lambda item: (-preference[item], item))
    return objectives[index]


def _row_signature(row: dict[str, str]) -> str:
    keys = (
        "algorithm",
        "algorithm_variant",
        "selected_algorithm",
        "base_algorithm",
        "search_profile",
        "selected_candidate_source",
        "tree_gp_family",
    )
    return " ".join(str(row.get(key, "")).lower() for key in keys)


def _safe_simplex(values: tuple[float, ...]) -> tuple[float, ...]:
    clipped = tuple(max(0.0, float(value)) for value in values)
    total = sum(clipped)
    if total <= 0:
        return tuple(1.0 / len(values) for _ in values)
    return tuple(value / total for value in clipped)


def _dedupe_preferences(
    values: list[tuple[str, tuple[float, ...]]],
) -> tuple[tuple[float, ...], ...]:
    result = []
    seen = set()
    for _preference_id, preference in values:
        key = tuple(round(value, 8) for value in preference)
        if key in seen:
            continue
        seen.add(key)
        result.append(preference)
    return tuple(result)


def _parse_preference(raw: str) -> tuple[float, ...]:
    if not raw:
        return ()
    return tuple(float(item) for item in raw.split(";") if item != "")


def _preference_digest(preferences: tuple[tuple[float, ...], ...]) -> str:
    return "|".join(";".join(f"{value:.4f}" for value in preference) for preference in preferences)


def _hint_digest(hints: tuple[MoeaSeedHint, ...]) -> str:
    return "|".join(
        f"{hint.operation_template}/{hint.machine_template}/"
        + ";".join(f"{value:.4f}" for value in hint.tail_weights)
        for hint in hints
    )


def _seed_training_policy(seed_mode: str, seed_neighbor_count: int) -> str:
    if seed_mode == "graph-trained-schedule-aware":
        return f"leave-seed-out-graph-trained-k{seed_neighbor_count}"
    if seed_mode == "nonlinear-graph-action-trained-schedule-aware":
        return f"leave-seed-out-nonlinear-graph-action-trained-k{seed_neighbor_count}"
    if seed_mode == "graph-action-trained-schedule-aware":
        return f"leave-seed-out-graph-action-trained-k{seed_neighbor_count}"
    if seed_mode == "action-trained-schedule-aware":
        return f"leave-seed-out-action-trained-k{seed_neighbor_count}"
    if seed_mode == "partial-schedule-trained-schedule-aware":
        return f"leave-seed-out-partial-schedule-trained-k{seed_neighbor_count}"
    if seed_mode == "partial-action-trained-schedule-aware":
        return f"leave-seed-out-partial-action-trained-k{seed_neighbor_count}"
    if seed_mode == "trained-schedule-aware":
        return f"leave-seed-out-trained-k{seed_neighbor_count}"
    if seed_mode == "learned-schedule-aware":
        return f"leave-seed-out-k{seed_neighbor_count}"
    if seed_mode == "igsp-greedy":
        return "leave-seed-out-igsp-greedy-fixed"
    if seed_mode == "schedule-aware":
        return "same-context-e25-outcome"
    return "preference-only"


def _structural_columns(structural_realized: bool, report: object) -> dict[str, object]:
    return {
        "structural_realized": str(structural_realized),
        "structural_new_jobs": report.new_jobs,
        "structural_rework_operations": report.rework_operations,
        "structural_route_branches": report.route_branches,
        "structural_added_jobs": report.added_jobs,
        "structural_added_operations": report.added_operations,
        "structural_total_updates": report.total_updates,
    }


def _online_action_columns(policy: object | None, action_summary: dict[str, object]) -> dict[str, object]:
    if policy is None:
        return {}
    return {
        "online_policy_training_policy": str(getattr(policy, "training_policy", "")),
        "learned_action_training_rows": getattr(policy, "training_rows", ""),
        "learned_action_pairwise_rows": getattr(policy, "pairwise_rows", ""),
        "learned_action_feature_count": getattr(policy, "feature_count", ""),
        "learned_action_fallback_threshold": getattr(policy, "fallback_threshold", ""),
        "learned_action_strength": action_summary["action_strength"],
        "learned_action_gate": action_summary["gate"],
        "learned_action_gate_decisions": action_summary["gate_decisions"],
        "learned_action_gate_acceptance_rate": action_summary["gate_acceptance_rate"],
        "learned_action_gate_reason_counts": action_summary["gate_reason_counts"],
        "learned_action_template_priors": ";".join(
            f"{value:.6f}" for value in getattr(policy, "template_priors", ())
        ),
        "learned_action_decisions": action_summary["decisions"],
        "learned_action_fallback_rate": action_summary["fallback_rate"],
        "learned_action_avg_confidence": action_summary["avg_confidence"],
        "learned_action_template_counts": action_summary["template_counts"],
        "learned_action_unique_target_directions": action_summary["unique_target_directions"],
        "learned_action_target_direction_counts": action_summary["target_direction_counts"],
    }


_MRGV_OPERATOR_GROUPS = ("path", "event", "agv", "diversity")
_MRGV_LIFECYCLE_METRICS = (
    "generated",
    "noop",
    "gate_pass",
    "evaluated",
    "entered_pool",
    "admitted",
    "survive",
    "archive",
)
_MRGV_CONTRIBUTION_RATE_METRICS = (
    "wasted_eval_rate",
    "survival_per_admitted",
    "archive_per_admitted",
    "dominates_baseline_rate",
    "baseline_dominates_rate",
    "epsilon_dominates_baseline_rate",
    "positive_indicator_rate",
    "grid_cell_change_rate",
    "sparse_grid_cell_rate",
)
_MRGV_CONTRIBUTION_FLOAT_METRICS = ("mean_indicator_delta",)
_MRGV_CONTRIBUTION_VECTOR_METRICS = ("mean_objective_delta",)
_MRGV_TRACE_OBJECTIVE_SLOTS = 5


def _mrgv_ordered_trace_rows(
    action_log: list[dict[str, object]],
    *,
    source: str,
    level: str,
    seed: int,
    mode: str,
    algorithm: str,
    context_policy: str,
    context_mrgv: dict[str, object],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    operator_counts: dict[str, int] = defaultdict(int)
    group_counts: dict[str, int] = defaultdict(int)
    path_pair_index = 0
    run_id = f"{source}|{level}|{seed:03d}|{mode}|{algorithm}"
    global_pair_index = 0
    for item in action_log:
        if "mrgv_operator" not in item:
            continue
        global_pair_index += 1
        operator = str(item.get("mrgv_operator", ""))
        group = _mrgv_operator_group(operator)
        operator_counts[operator] += 1
        group_counts[group] += 1
        if group == "path":
            path_pair_index += 1
        baseline_vector = _parse_vector(str(item.get("baseline_objective_vector", "")))
        mrgv_vector = _parse_vector(str(item.get("mrgv_objective_vector", "")))
        raw_indicator_delta = str(item.get("mrgv_indicator_delta", ""))
        has_indicator_delta = raw_indicator_delta not in {"", "nan", "None"}
        indicator_delta = _float(raw_indicator_delta) if has_indicator_delta else 0.0
        row: dict[str, object] = {
            "run_id": run_id,
            "source": source,
            "level": level,
            "seed": f"{seed:03d}",
            "context": mode,
            "mode": mode,
            "algorithm": algorithm,
            "generation": item.get("generation", ""),
            "eval_index": item.get("eval_count_before", ""),
            "global_pair_index": global_pair_index,
            "operator_pair_index": operator_counts[operator],
            "operator_group_pair_index": group_counts[group],
            "path_pair_index": path_pair_index if group == "path" else "",
            "operator": operator,
            "operator_group": group,
            "mrgv_mode": item.get("mrgv_mode", context_mrgv.get("mrgv_mode", "")),
            "context_policy": context_policy,
            "candidate_policy": item.get("mrgv_candidate_policy", ""),
            "admission_policy": item.get("mrgv_admission_policy", context_mrgv.get("mrgv_admission_policy", "")),
            "gate_type": item.get("mrgv_gate", ""),
            "gate_lane": item.get("mrgv_gate_lane", ""),
            "gate_pass": item.get("mrgv_gate_accepted", ""),
            "gate_reason": item.get("mrgv_gate_reason", ""),
            "probe_policy": item.get("mrgv_probe_policy", ""),
            "probe_phase": item.get("mrgv_probe_phase", ""),
            "probe_pair_index": item.get("mrgv_probe_pair_index", ""),
            "probe_positive": item.get("mrgv_probe_positive", ""),
            "probe_positive_pairs": item.get("mrgv_probe_positive_pairs", ""),
            "probe_positive_rate": item.get("mrgv_probe_positive_rate", ""),
            "probe_decision": item.get("mrgv_probe_decision", ""),
            "probe_admission_override": item.get("mrgv_probe_admission_override", ""),
            "admitted": item.get("mrgv_admitted_to_pool", item.get("mrgv_entered_pool", "")),
            "generated": item.get("mrgv_generated", ""),
            "non_noop": item.get("mrgv_non_noop", ""),
            "evaluated": item.get("mrgv_evaluated", ""),
            "baseline_child_id": item.get("baseline_child_id", ""),
            "mrgv_child_id": item.get("mrgv_child_id", ""),
            "selected_child": item.get("selected_child", ""),
            "baseline_objective_vector": item.get("baseline_objective_vector", ""),
            "mrgv_objective_vector": item.get("mrgv_objective_vector", ""),
            "objective_delta_vector": item.get("mrgv_objective_delta", ""),
            "normalized_objective_delta_vector": _normalized_delta_vector(baseline_vector, mrgv_vector),
            "r_dominates_b": item.get("mrgv_dominates_baseline", ""),
            "b_dominates_r": item.get("baseline_dominates_mrgv", ""),
            "r_epsilon_dominates_b": item.get("mrgv_epsilon_dominates_baseline", ""),
            "baseline_indicator": item.get("baseline_indicator", ""),
            "mrgv_indicator": item.get("mrgv_indicator", ""),
            "local_indicator_delta": item.get("mrgv_indicator_delta", ""),
            "local_indicator_positive": str(has_indicator_delta and indicator_delta >= 0.0),
            "baseline_grid_cell": item.get("baseline_grid_cell", ""),
            "mrgv_grid_cell": item.get("mrgv_grid_cell", ""),
            "current_grid_cell_gain": item.get("mrgv_grid_cell_changed", ""),
            "current_archive_grid_gain": item.get("mrgv_sparse_grid_cell_hit", ""),
            "current_epsilon_delta": item.get("mrgv_epsilon_dominates_baseline", ""),
            "current_grid_hv_proxy_delta": item.get("mrgv_indicator_delta", ""),
            "duplicate_flag": str(item.get("mrgv_non_noop", "") == "False"),
            "noop_flag": str(item.get("mrgv_non_noop", "") == "False"),
            "online_feature_group": "online",
            "survived_selection": item.get("mrgv_survived_selection", ""),
            "entered_archive": item.get("mrgv_entered_archive", ""),
            "final_archive_contribution": "",
            "posthoc_feature_group": "posthoc",
        }
        objective_delta_vector = _parse_vector(str(item.get("mrgv_objective_delta", "")))
        for index in range(1, _MRGV_TRACE_OBJECTIVE_SLOTS + 1):
            row[f"b_obj_{index}"] = baseline_vector[index - 1] if len(baseline_vector) >= index else ""
            row[f"r_obj_{index}"] = mrgv_vector[index - 1] if len(mrgv_vector) >= index else ""
            row[f"delta_obj_{index}"] = objective_delta_vector[index - 1] if len(objective_delta_vector) >= index else ""
        rows.append(row)
    return tuple(rows)


def _normalized_delta_vector(
    baseline_vector: tuple[float, ...],
    mrgv_vector: tuple[float, ...],
) -> str:
    if not baseline_vector or not mrgv_vector or len(baseline_vector) != len(mrgv_vector):
        return ""
    deltas = []
    for baseline, mrgv in zip(baseline_vector, mrgv_vector):
        scale = max(abs(baseline), abs(mrgv), 1.0)
        deltas.append((baseline - mrgv) / scale)
    return ";".join(f"{value:.10f}" for value in deltas)


def _mrgv_action_columns(
    mrgv_mode: str,
    mrgv_force_operator: str,
    mrgv_admission_policy: str,
    mrgv_context_policy: str,
    action_summary: dict[str, object],
) -> dict[str, object]:
    lifecycle = _empty_mrgv_lifecycle_columns()
    contribution = _empty_mrgv_contribution_columns()
    if mrgv_mode == "none":
        return {
            "mrgv_mode": "none",
            "mrgv_force_operator": mrgv_force_operator,
            "mrgv_candidate_policy": "",
            "mrgv_admission_policy": "",
            "mrgv_context_policy": mrgv_context_policy,
            "mrgv_probe_policy": "none",
            "mrgv_probe_final_decision": "",
            "mrgv_probe_pairs": 0,
            "mrgv_probe_positive_pairs": 0,
            "mrgv_probe_positive_rate": 0.0,
            "mrgv_decisions": 0,
            "mrgv_gate": "",
            "mrgv_gate_lane_counts": "",
            "mrgv_gate_acceptance_rate": 0.0,
            "mrgv_operator_counts": "",
            "mrgv_gate_reason_counts": "",
            "mrgv_diversity_operator_rate": 0.0,
            **lifecycle,
            **contribution,
        }
    return {
        "mrgv_mode": mrgv_mode,
        "mrgv_force_operator": mrgv_force_operator,
        "mrgv_candidate_policy": action_summary["mrgv_candidate_policy"],
        "mrgv_admission_policy": action_summary.get("mrgv_admission_policy", mrgv_admission_policy),
        "mrgv_context_policy": mrgv_context_policy,
        "mrgv_probe_policy": action_summary["mrgv_probe_policy"],
        "mrgv_probe_final_decision": action_summary["mrgv_probe_final_decision"],
        "mrgv_probe_pairs": action_summary["mrgv_probe_pairs"],
        "mrgv_probe_positive_pairs": action_summary["mrgv_probe_positive_pairs"],
        "mrgv_probe_positive_rate": action_summary["mrgv_probe_positive_rate"],
        "mrgv_decisions": action_summary["mrgv_decisions"],
        "mrgv_gate": action_summary["mrgv_gate"],
        "mrgv_gate_lane_counts": action_summary["mrgv_gate_lane_counts"],
        "mrgv_gate_acceptance_rate": action_summary["mrgv_gate_acceptance_rate"],
        "mrgv_operator_counts": action_summary["mrgv_operator_counts"],
        "mrgv_gate_reason_counts": action_summary["mrgv_gate_reason_counts"],
        "mrgv_diversity_operator_rate": action_summary["mrgv_diversity_operator_rate"],
        **{key: action_summary[key] for key in lifecycle},
        **{key: action_summary[key] for key in contribution},
    }


def _empty_mrgv_lifecycle_columns() -> dict[str, int]:
    return {
        f"mrgv_{metric}_{group}": 0
        for metric in _MRGV_LIFECYCLE_METRICS
        for group in _MRGV_OPERATOR_GROUPS
    }


def _empty_mrgv_contribution_columns() -> dict[str, object]:
    columns: dict[str, object] = {}
    for group in _MRGV_OPERATOR_GROUPS:
        for metric in _MRGV_CONTRIBUTION_RATE_METRICS + _MRGV_CONTRIBUTION_FLOAT_METRICS:
            columns[f"mrgv_{metric}_{group}"] = 0.0
        for metric in _MRGV_CONTRIBUTION_VECTOR_METRICS:
            columns[f"mrgv_{metric}_{group}"] = ""
    return columns


def _summarize_action_log(action_log: list[dict[str, object]]) -> dict[str, object]:
    if not action_log:
        return {
            "decisions": 0,
            "fallback_rate": 0.0,
            "avg_confidence": 0.0,
            "template_counts": "",
            "unique_target_directions": 0,
            "target_direction_counts": "",
            "action_strength": "",
            "gate": "",
            "gate_decisions": 0,
            "gate_acceptance_rate": 0.0,
            "gate_reason_counts": "",
            "mrgv_decisions": 0,
            "mrgv_candidate_policy": "",
            "mrgv_admission_policy": "",
            "mrgv_probe_policy": "",
            "mrgv_probe_final_decision": "",
            "mrgv_probe_pairs": 0,
            "mrgv_probe_positive_pairs": 0,
            "mrgv_probe_positive_rate": 0.0,
            "mrgv_gate": "",
            "mrgv_gate_lane_counts": "",
            "mrgv_gate_acceptance_rate": 0.0,
            "mrgv_operator_counts": "",
            "mrgv_gate_reason_counts": "",
            "mrgv_diversity_operator_rate": 0.0,
            **_empty_mrgv_lifecycle_columns(),
            **_empty_mrgv_contribution_columns(),
        }
    counts: dict[str, int] = defaultdict(int)
    target_counts: dict[str, int] = defaultdict(int)
    gate_counts: dict[str, int] = defaultdict(int)
    gate_names: set[str] = set()
    mrgv_operator_counts: dict[str, int] = defaultdict(int)
    mrgv_gate_counts: dict[str, int] = defaultdict(int)
    mrgv_gate_lane_counts: dict[str, int] = defaultdict(int)
    mrgv_gate_names: set[str] = set()
    mrgv_candidate_policies: set[str] = set()
    mrgv_admission_policies: set[str] = set()
    mrgv_probe_policies: set[str] = set()
    mrgv_probe_decisions: list[str] = []
    mrgv_probe_pairs = 0
    mrgv_probe_positive_pairs = 0
    mrgv_probe_positive_rate = 0.0
    mrgv_lifecycle = _empty_mrgv_lifecycle_columns()
    mrgv_dominates_baseline_counts: dict[str, int] = defaultdict(int)
    mrgv_baseline_dominates_counts: dict[str, int] = defaultdict(int)
    mrgv_epsilon_dominates_counts: dict[str, int] = defaultdict(int)
    mrgv_positive_indicator_counts: dict[str, int] = defaultdict(int)
    mrgv_grid_cell_change_counts: dict[str, int] = defaultdict(int)
    mrgv_sparse_grid_cell_counts: dict[str, int] = defaultdict(int)
    mrgv_indicator_deltas: dict[str, list[float]] = defaultdict(list)
    mrgv_objective_deltas: dict[str, list[tuple[float, ...]]] = defaultdict(list)
    gate_decisions = 0
    gate_acceptances = 0
    mrgv_decisions = 0
    mrgv_gate_acceptances = 0
    mrgv_diversity_count = 0
    fallback_count = 0
    confidence_total = 0.0
    for item in action_log:
        template = str(item.get("action_template", ""))
        if template:
            counts[template] += 1
        target_direction = str(item.get("target_direction", ""))
        if target_direction:
            target_counts[target_direction] += 1
        fallback_count += 1 if str(item.get("fallback", "")).lower() == "true" else 0
        confidence_total += _float(item.get("confidence", 0.0))
        gate = str(item.get("online_action_gate", ""))
        if gate:
            gate_names.add(gate)
            gate_decisions += 1
            if str(item.get("online_action_gate_accepted", "")).lower() == "true":
                gate_acceptances += 1
            gate_counts[str(item.get("online_action_gate_reason", ""))] += 1
        mrgv_operator = str(item.get("mrgv_operator", ""))
        if mrgv_operator:
            group = _mrgv_operator_group(mrgv_operator)
            mrgv_decisions += 1
            mrgv_operator_counts[mrgv_operator] += 1
            if mrgv_operator == "conservative-structural-diversity":
                mrgv_diversity_count += 1
            if _bool(item.get("mrgv_generated", "")):
                mrgv_lifecycle[f"mrgv_generated_{group}"] += 1
            if _bool(item.get("mrgv_generated", "")) and not _bool(item.get("mrgv_non_noop", "")):
                mrgv_lifecycle[f"mrgv_noop_{group}"] += 1
            if _bool(item.get("mrgv_gate_accepted", "")):
                mrgv_lifecycle[f"mrgv_gate_pass_{group}"] += 1
            if _bool(item.get("mrgv_evaluated", "")):
                mrgv_lifecycle[f"mrgv_evaluated_{group}"] += 1
            if _bool(item.get("mrgv_entered_pool", "")):
                mrgv_lifecycle[f"mrgv_entered_pool_{group}"] += 1
            if _bool(item.get("mrgv_admitted_to_pool", item.get("mrgv_entered_pool", ""))):
                mrgv_lifecycle[f"mrgv_admitted_{group}"] += 1
            if _bool(item.get("mrgv_survived_selection", "")):
                mrgv_lifecycle[f"mrgv_survive_{group}"] += 1
            if _bool(item.get("mrgv_entered_archive", "")):
                mrgv_lifecycle[f"mrgv_archive_{group}"] += 1
            if _bool(item.get("mrgv_dominates_baseline", "")):
                mrgv_dominates_baseline_counts[group] += 1
            if _bool(item.get("baseline_dominates_mrgv", "")):
                mrgv_baseline_dominates_counts[group] += 1
            if _bool(item.get("mrgv_epsilon_dominates_baseline", "")):
                mrgv_epsilon_dominates_counts[group] += 1
            if _bool(item.get("mrgv_positive_indicator_contribution", "")):
                mrgv_positive_indicator_counts[group] += 1
            if _bool(item.get("mrgv_grid_cell_changed", "")):
                mrgv_grid_cell_change_counts[group] += 1
            if _bool(item.get("mrgv_sparse_grid_cell_hit", "")):
                mrgv_sparse_grid_cell_counts[group] += 1
            indicator_delta = str(item.get("mrgv_indicator_delta", ""))
            if indicator_delta:
                mrgv_indicator_deltas[group].append(_float(indicator_delta))
            objective_delta = _parse_vector(str(item.get("mrgv_objective_delta", "")))
            if objective_delta:
                mrgv_objective_deltas[group].append(objective_delta)
            mrgv_gate = str(item.get("mrgv_gate", ""))
            if mrgv_gate:
                mrgv_gate_names.add(mrgv_gate)
            mrgv_gate_lane = str(item.get("mrgv_gate_lane", ""))
            if mrgv_gate_lane:
                mrgv_gate_lane_counts[mrgv_gate_lane] += 1
            mrgv_candidate_policy = str(item.get("mrgv_candidate_policy", ""))
            if mrgv_candidate_policy:
                mrgv_candidate_policies.add(mrgv_candidate_policy)
            mrgv_admission_policy = str(item.get("mrgv_admission_policy", ""))
            if mrgv_admission_policy:
                mrgv_admission_policies.add(mrgv_admission_policy)
            mrgv_probe_policy = str(item.get("mrgv_probe_policy", ""))
            if mrgv_probe_policy:
                mrgv_probe_policies.add(mrgv_probe_policy)
            mrgv_probe_decision = str(item.get("mrgv_probe_decision", ""))
            if mrgv_probe_decision:
                mrgv_probe_decisions.append(mrgv_probe_decision)
            mrgv_probe_pairs = max(mrgv_probe_pairs, _int(item.get("mrgv_probe_pair_index", "")))
            mrgv_probe_positive_pairs = max(
                mrgv_probe_positive_pairs,
                _int(item.get("mrgv_probe_positive_pairs", "")),
            )
            if str(item.get("mrgv_probe_positive_rate", "")):
                mrgv_probe_positive_rate = _float(item.get("mrgv_probe_positive_rate", ""))
            if str(item.get("mrgv_gate_accepted", "")).lower() == "true":
                mrgv_gate_acceptances += 1
            mrgv_gate_counts[str(item.get("mrgv_gate_reason", ""))] += 1
    action_strengths = sorted({str(item.get("action_strength", "")) for item in action_log if item.get("action_strength", "") != ""})
    online_decisions = sum(counts.values())
    return {
        "decisions": online_decisions,
        "fallback_rate": round(fallback_count / online_decisions, 6) if online_decisions else 0.0,
        "avg_confidence": round(confidence_total / online_decisions, 10) if online_decisions else 0.0,
        "template_counts": "|".join(f"{template}:{counts[template]}" for template in sorted(counts)),
        "unique_target_directions": len(target_counts),
        "target_direction_counts": "|".join(
            f"{direction}:{target_counts[direction]}" for direction in sorted(target_counts)
        ),
        "action_strength": "|".join(action_strengths),
        "gate": "|".join(sorted(gate_names)),
        "gate_decisions": gate_decisions,
        "gate_acceptance_rate": round(gate_acceptances / gate_decisions, 6) if gate_decisions else 0.0,
        "gate_reason_counts": "|".join(f"{reason}:{gate_counts[reason]}" for reason in sorted(gate_counts)),
        "mrgv_decisions": mrgv_decisions,
        "mrgv_candidate_policy": "|".join(sorted(mrgv_candidate_policies)),
        "mrgv_admission_policy": "|".join(sorted(mrgv_admission_policies)),
        "mrgv_probe_policy": "|".join(sorted(mrgv_probe_policies)),
        "mrgv_probe_final_decision": mrgv_probe_decisions[-1] if mrgv_probe_decisions else "",
        "mrgv_probe_pairs": mrgv_probe_pairs,
        "mrgv_probe_positive_pairs": mrgv_probe_positive_pairs,
        "mrgv_probe_positive_rate": round(mrgv_probe_positive_rate, 6),
        "mrgv_gate": "|".join(sorted(mrgv_gate_names)),
        "mrgv_gate_lane_counts": "|".join(
            f"{lane}:{mrgv_gate_lane_counts[lane]}"
            for lane in sorted(mrgv_gate_lane_counts)
        ),
        "mrgv_gate_acceptance_rate": round(mrgv_gate_acceptances / mrgv_decisions, 6) if mrgv_decisions else 0.0,
        "mrgv_operator_counts": "|".join(
            f"{operator}:{mrgv_operator_counts[operator]}"
            for operator in sorted(mrgv_operator_counts)
        ),
        "mrgv_gate_reason_counts": "|".join(
            f"{reason}:{mrgv_gate_counts[reason]}"
            for reason in sorted(mrgv_gate_counts)
        ),
        "mrgv_diversity_operator_rate": round(mrgv_diversity_count / mrgv_decisions, 6) if mrgv_decisions else 0.0,
        **mrgv_lifecycle,
        **_mrgv_contribution_columns(
            lifecycle=mrgv_lifecycle,
            dominates_baseline_counts=mrgv_dominates_baseline_counts,
            baseline_dominates_counts=mrgv_baseline_dominates_counts,
            epsilon_dominates_counts=mrgv_epsilon_dominates_counts,
            positive_indicator_counts=mrgv_positive_indicator_counts,
            grid_cell_change_counts=mrgv_grid_cell_change_counts,
            sparse_grid_cell_counts=mrgv_sparse_grid_cell_counts,
            indicator_deltas=mrgv_indicator_deltas,
            objective_deltas=mrgv_objective_deltas,
        ),
    }


def _mrgv_operator_group(operator: str) -> str:
    if operator == "path-pressure-shift":
        return "path"
    if operator == "event-repair-resequence":
        return "event"
    if operator == "agv-balance-reassignment":
        return "agv"
    if operator == "conservative-structural-diversity":
        return "diversity"
    return "path"


def _mrgv_contribution_columns(
    *,
    lifecycle: dict[str, int],
    dominates_baseline_counts: dict[str, int],
    baseline_dominates_counts: dict[str, int],
    epsilon_dominates_counts: dict[str, int],
    positive_indicator_counts: dict[str, int],
    grid_cell_change_counts: dict[str, int],
    sparse_grid_cell_counts: dict[str, int],
    indicator_deltas: dict[str, list[float]],
    objective_deltas: dict[str, list[tuple[float, ...]]],
) -> dict[str, object]:
    columns: dict[str, object] = {}
    for group in _MRGV_OPERATOR_GROUPS:
        evaluated = lifecycle[f"mrgv_evaluated_{group}"]
        admitted = lifecycle[f"mrgv_admitted_{group}"]
        survived = lifecycle[f"mrgv_survive_{group}"]
        archived = lifecycle[f"mrgv_archive_{group}"]
        columns[f"mrgv_wasted_eval_rate_{group}"] = round(_safe_ratio(evaluated - admitted, evaluated), 6)
        columns[f"mrgv_survival_per_admitted_{group}"] = round(_safe_ratio(survived, admitted), 6)
        columns[f"mrgv_archive_per_admitted_{group}"] = round(_safe_ratio(archived, admitted), 6)
        columns[f"mrgv_dominates_baseline_rate_{group}"] = round(
            _safe_ratio(dominates_baseline_counts[group], evaluated),
            6,
        )
        columns[f"mrgv_baseline_dominates_rate_{group}"] = round(
            _safe_ratio(baseline_dominates_counts[group], evaluated),
            6,
        )
        columns[f"mrgv_epsilon_dominates_baseline_rate_{group}"] = round(
            _safe_ratio(epsilon_dominates_counts[group], evaluated),
            6,
        )
        columns[f"mrgv_positive_indicator_rate_{group}"] = round(
            _safe_ratio(positive_indicator_counts[group], evaluated),
            6,
        )
        columns[f"mrgv_grid_cell_change_rate_{group}"] = round(
            _safe_ratio(grid_cell_change_counts[group], evaluated),
            6,
        )
        columns[f"mrgv_sparse_grid_cell_rate_{group}"] = round(
            _safe_ratio(sparse_grid_cell_counts[group], evaluated),
            6,
        )
        columns[f"mrgv_mean_indicator_delta_{group}"] = round(_avg(indicator_deltas[group]), 10)
        columns[f"mrgv_mean_objective_delta_{group}"] = _format_vector(_mean_vector(objective_deltas[group]))
    return columns


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _parse_vector(raw: str) -> tuple[float, ...]:
    if not raw:
        return ()
    return tuple(float(item) for item in raw.split(";") if item)


def _mean_vector(vectors: object) -> tuple[float, ...]:
    items = [tuple(vector) for vector in vectors if vector]
    if not items:
        return ()
    width = min(len(vector) for vector in items)
    return tuple(
        sum(vector[index] for vector in items) / len(items)
        for index in range(width)
    )


def _format_vector(vector: tuple[float, ...]) -> str:
    return ";".join(f"{value:.10f}" for value in vector)


def _bool(value: object) -> bool:
    return str(value).lower() == "true"


def _float(value: object) -> float:
    return float(value) if value not in {"", None} else 0.0


def _int(value: object) -> int:
    return int(float(value)) if value not in {"", None} else 0


def _avg(values: object) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
