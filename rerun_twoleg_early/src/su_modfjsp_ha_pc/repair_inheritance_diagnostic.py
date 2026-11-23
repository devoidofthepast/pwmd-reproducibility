from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from su_modfjsp_ha_pc.generator import generate_instance
from su_modfjsp_ha_pc.metrics import evaluate_schedule
from su_modfjsp_ha_pc.moea import (
    MoeaConfig,
    _Genome,
    _crossover,
    _evaluate_genome,
    _mutate,
    _ordered_operations,
    run_moea_baseline,
)
from su_modfjsp_ha_pc.seeded_moea import _schedule_to_seed_genes
from su_modfjsp_ha_pc.seeded_moea import (
    _apply_seed_teacher_mask,
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
    "source",
    "base_solution_id",
    "method",
    "base_objective_vector",
    "candidate_objective_vector",
    "projection_objective_vector",
    "operator_best_objective_vector",
    "continuation_best_objective_vector",
    "candidate_relation_to_base",
    "projection_relation_to_candidate",
    "projection_relation_to_base",
    "operator_best_relation_to_projection",
    "operator_best_relation_to_candidate",
    "continuation_best_relation_to_projection",
    "continuation_best_relation_to_candidate",
    "base_total_delay",
    "candidate_total_delay",
    "projection_total_delay",
    "operator_best_total_delay",
    "candidate_delay_delta",
    "projection_delay_delta_to_candidate",
    "projection_delay_delta_to_base",
    "projection_gene_l1_delta_from_source",
    "projection_gene_linf_delta_from_source",
    "projection_preserved_candidate",
    "projection_lost_candidate",
    "operator_child_preserved_projection",
    "operator_child_preserved_candidate",
    "continuation_recovered_candidate",
    "trace_neighborhood_size",
    "candidate_neighborhood_size",
    "operator_trials",
    "continuation_evaluations",
)

SUMMARY_FIELDS = (
    "algorithm",
    "level",
    "mode",
    "method",
    "trials",
    "phenotype_success_trials",
    "phenotype_success_rate",
    "projection_preserve_rate",
    "projection_loss_rate",
    "success_projection_preserve_rate",
    "success_projection_loss_rate",
    "operator_child_preserve_projection_rate",
    "operator_child_preserve_candidate_rate",
    "success_operator_child_preserve_candidate_rate",
    "continuation_recover_candidate_rate",
    "success_continuation_recover_candidate_rate",
    "mean_candidate_delay_delta",
    "mean_projection_delay_delta_to_candidate",
    "mean_projection_gene_l1_delta_from_source",
    "mean_projection_gene_linf_delta_from_source",
)


@dataclass(frozen=True)
class RepairInheritanceConfig:
    size: str = "small"
    levels: tuple[str, ...] = ("L4", "L5")
    modes: tuple[str, ...] = ("raw", "realized")
    start_seed: int | None = None
    end_seed: int | None = None
    algorithms: tuple[str, ...] = ("nsga3",)
    sources: tuple[str, ...] = ()
    methods: tuple[str, ...] = ("delay-trace-lns", "delay-topk-lns")
    max_base_rows_per_context: int = 8
    operator_trials: int = 8
    continuation_evaluations: int = 0
    continuation_population_size: int = 16
    population_size: int = 48
    max_evaluations: int = 384
    moea_seed: int = 95
    random_seed: int = 9500
    crossover_rate: float = 0.9
    mutation_rate: float = 0.12
    top_records: int = 8
    neighbor_radius: int = 1
    priority_boost: float = 3.0
    process_weight: float = 0.5
    include_realized: bool = True
    max_seed_preferences: int = 5
    max_seed_hints: int = 5
    seed_initial_quota_raw: int = 24
    seed_initial_quota_realized: int = 8
    seed_neighbor_count: int = 1
    seed_teacher_mask: str = "full"


def run_repair_inheritance_diagnostic(
    evaluations_path: Path,
    *,
    config: RepairInheritanceConfig | None = None,
) -> dict[str, tuple[dict[str, object], ...]]:
    resolved = config or RepairInheritanceConfig()
    rows = _read_rows(evaluations_path)
    diagnostic_rows: list[dict[str, object]] = []
    context_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    seen: set[tuple[str, str, str, str]] = set()

    for row in sorted(rows, key=_row_sort_key):
        if not _include_row(row, resolved):
            continue
        algorithm = _algorithm_from_row(row)
        key = (str(row["level"]), str(row["seed"]), str(row["mode"]))
        dedupe_key = (key[0], key[1], key[2], str(row.get("genome_signature", "")))
        if dedupe_key in seen:
            continue
        if context_counts[key] >= resolved.max_base_rows_per_context:
            continue
        seen.add(dedupe_key)
        context_counts[key] += 1
        diagnostic_rows.extend(_diagnose_base_row(row, algorithm=algorithm, config=resolved))

    diagnostic = tuple(diagnostic_rows)
    return {
        "diagnostic": diagnostic,
        "summary": summarize_repair_inheritance(diagnostic),
    }


def run_repair_inheritance_from_seed_rows(
    seed_rows_path: Path,
    *,
    config: RepairInheritanceConfig | None = None,
) -> dict[str, tuple[dict[str, object], ...]]:
    resolved = config or RepairInheritanceConfig()
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
        if mode == "realized":
            instance = realize_structural_events(raw_instance).instance
        else:
            instance = raw_instance
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
                    _diagnose_base_row(
                        _solution_row(
                            solution=solution,
                            instance_id=instance.instance_id,
                            raw_instance_id=raw_instance.instance_id,
                            level=level,
                            seed=seed,
                            mode=mode,
                            source="e95-generated-base",
                        ),
                        algorithm=algorithm,
                        config=resolved,
                    )
                )

    diagnostic = tuple(diagnostic_rows)
    return {
        "diagnostic": diagnostic,
        "summary": summarize_repair_inheritance(diagnostic),
    }


def summarize_repair_inheritance(rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["algorithm"]),
                str(row["level"]),
                str(row["mode"]),
                str(row["method"]),
            )
        ].append(row)
        grouped[(str(row["algorithm"]), "ALL", "ALL", str(row["method"]))].append(row)

    summaries = []
    for (algorithm, level, mode, method), group in sorted(grouped.items()):
        trials = len(group)
        success_rows = [row for row in group if _is_success_relation(str(row["candidate_relation_to_base"]))]
        continuation_rows = [row for row in group if int(row["continuation_evaluations"]) > 0]
        success_continuation_rows = [row for row in success_rows if int(row["continuation_evaluations"]) > 0]
        summaries.append(
            {
                "algorithm": algorithm,
                "level": level,
                "mode": mode,
                "method": method,
                "trials": trials,
                "phenotype_success_trials": len(success_rows),
                "phenotype_success_rate": _ratio(
                    sum(1 for row in group if _is_success_relation(str(row["candidate_relation_to_base"]))),
                    trials,
                ),
                "projection_preserve_rate": _ratio(
                    sum(1 for row in group if _bool(row["projection_preserved_candidate"])),
                    trials,
                ),
                "projection_loss_rate": _ratio(
                    sum(1 for row in group if _bool(row["projection_lost_candidate"])),
                    trials,
                ),
                "success_projection_preserve_rate": _ratio(
                    sum(1 for row in success_rows if _bool(row["projection_preserved_candidate"])),
                    len(success_rows),
                ),
                "success_projection_loss_rate": _ratio(
                    sum(1 for row in success_rows if _bool(row["projection_lost_candidate"])),
                    len(success_rows),
                ),
                "operator_child_preserve_projection_rate": _ratio(
                    sum(1 for row in group if _bool(row["operator_child_preserved_projection"])),
                    trials,
                ),
                "operator_child_preserve_candidate_rate": _ratio(
                    sum(1 for row in group if _bool(row["operator_child_preserved_candidate"])),
                    trials,
                ),
                "success_operator_child_preserve_candidate_rate": _ratio(
                    sum(1 for row in success_rows if _bool(row["operator_child_preserved_candidate"])),
                    len(success_rows),
                ),
                "continuation_recover_candidate_rate": _ratio(
                    sum(1 for row in continuation_rows if _bool(row["continuation_recovered_candidate"])),
                    len(continuation_rows),
                ),
                "success_continuation_recover_candidate_rate": _ratio(
                    sum(1 for row in success_continuation_rows if _bool(row["continuation_recovered_candidate"])),
                    len(success_continuation_rows),
                ),
                "mean_candidate_delay_delta": round(_avg(float(row["candidate_delay_delta"]) for row in group), 10),
                "mean_projection_delay_delta_to_candidate": round(
                    _avg(float(row["projection_delay_delta_to_candidate"]) for row in group),
                    10,
                ),
                "mean_projection_gene_l1_delta_from_source": round(
                    _avg(float(row["projection_gene_l1_delta_from_source"]) for row in group),
                    10,
                ),
                "mean_projection_gene_linf_delta_from_source": round(
                    _avg(float(row["projection_gene_linf_delta_from_source"]) for row in group),
                    10,
                ),
            }
        )
    return tuple(summaries)


def write_repair_inheritance_outputs(
    *,
    diagnostic_out: Path,
    summary_out: Path,
    rows: dict[str, tuple[dict[str, object], ...]],
) -> None:
    _write_rows(diagnostic_out, rows["diagnostic"], DIAGNOSTIC_FIELDS)
    _write_rows(summary_out, rows["summary"], SUMMARY_FIELDS)


def _diagnose_base_row(
    row: dict[str, str],
    *,
    algorithm: str,
    config: RepairInheritanceConfig,
) -> tuple[dict[str, object], ...]:
    instance = _instance_from_row(row, config.size)
    operations = _ordered_operations(instance)
    source_genome = _genome_from_row(
        row,
        expected_genes=len(operations) + len(instance.machines) + 3,
    )
    base_trace: list[dict[str, object]] = []
    base = _evaluate_genome(
        _stable_seed(config.random_seed, row, "base"),
        _int(row.get("generation", 0)),
        source_genome,
        instance,
        operations,
        algorithm,
        MoeaConfig(
            population_size=2,
            max_evaluations=2,
            seed=config.moea_seed,
            path_aware=_bool(row.get("path_aware", "True")),
            event_aware=_bool(row.get("event_aware", "True")),
            repair_process_log=base_trace,
            repair_process_context=_repair_context(row, algorithm=algorithm, phase="base"),
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

    result = []
    for method in config.methods:
        neighborhood = _neighborhood_for_method(
            method=method,
            operations=tuple(operation.operation_id for operation in base.schedule.operations),
            trace_rows=tuple(base_trace),
            trace_neighborhood=trace_neighborhood,
            seed=_stable_seed(config.random_seed, row, method),
        )
        if not neighborhood:
            continue
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
            repair_process_context=_repair_context(row, algorithm=method, phase="candidate"),
        )
        candidate_metrics = evaluate_schedule(instance, repaired.schedule)
        candidate_vector = tuple(float(candidate_metrics[objective]) for objective in instance.objectives)
        projected_genes = _schedule_to_seed_genes(instance, operations, repaired.schedule, source_genome)
        projected_genome = _Genome(projected_genes)
        projection_trace: list[dict[str, object]] = []
        projection = _evaluate_genome(
            _stable_seed(config.random_seed, row, f"projection:{method}"),
            _int(row.get("generation", 0)) + 1,
            projected_genome,
            instance,
            operations,
            algorithm,
            MoeaConfig(
                population_size=2,
                max_evaluations=2,
                seed=config.moea_seed,
                path_aware=_bool(row.get("path_aware", "True")),
                event_aware=_bool(row.get("event_aware", "True")),
                repair_process_log=projection_trace,
                repair_process_context=_repair_context(row, algorithm=algorithm, phase="projection"),
            ),
        )
        operator_best, operator_best_delay = _best_operator_child(
            instance=instance,
            operations=operations,
            algorithm=algorithm,
            row=row,
            source_genome=source_genome,
            projected_genome=projected_genome,
            config=config,
            method=method,
        )
        continuation_best = _continuation_best(
            instance=instance,
            algorithm=algorithm,
            seed_genes=projected_genes,
            config=config,
        )
        result.append(
            _diagnostic_row(
                row=row,
                algorithm=algorithm,
                method=_canonical_method(method),
                base_vector=base.objective_vector,
                candidate_vector=candidate_vector,
                projection_vector=projection.objective_vector,
                operator_best_vector=operator_best.objective_vector if operator_best is not None else (),
                continuation_best_vector=continuation_best.objective_vector if continuation_best is not None else (),
                source_genes=source_genome.genes,
                projected_genes=projected_genes,
                base_delay=_total_delay(tuple(base_trace)),
                candidate_delay=_total_delay(repaired.trace_rows),
                projection_delay=_total_delay(tuple(projection_trace)),
                operator_best_delay=operator_best_delay,
                trace_neighborhood_size=len(trace_neighborhood),
                candidate_neighborhood_size=len(repaired.neighborhood_operation_ids),
                config=config,
            )
        )
    return tuple(result)


def _best_operator_child(
    *,
    instance: object,
    operations: tuple,
    algorithm: str,
    row: dict[str, str],
    source_genome: _Genome,
    projected_genome: _Genome,
    config: RepairInheritanceConfig,
    method: str,
) -> tuple[object | None, int]:
    if config.operator_trials <= 0:
        return None, 0
    rng = random.Random(_stable_seed(config.random_seed, row, f"operator:{method}"))
    best = None
    best_delay = 0
    for index in range(config.operator_trials):
        child = _crossover(projected_genome, source_genome, rng, config.crossover_rate)
        child = _mutate(child, rng, config.mutation_rate)
        trace_rows: list[dict[str, object]] = []
        evaluated = _evaluate_genome(
            _stable_seed(config.random_seed, row, f"operator:{method}:{index}"),
            _int(row.get("generation", 0)) + 2,
            child,
            instance,
            operations,
            algorithm,
            MoeaConfig(
                population_size=2,
                max_evaluations=2,
                seed=config.moea_seed,
                path_aware=_bool(row.get("path_aware", "True")),
                event_aware=_bool(row.get("event_aware", "True")),
                repair_process_log=trace_rows,
                repair_process_context=_repair_context(row, algorithm=algorithm, phase="operator-child"),
            ),
        )
        if best is None or _vector_sum(evaluated.objective_vector) < _vector_sum(best.objective_vector):
            best = evaluated
            best_delay = _total_delay(tuple(trace_rows))
    return best, best_delay


def _continuation_best(
    *,
    instance: object,
    algorithm: str,
    seed_genes: tuple[float, ...],
    config: RepairInheritanceConfig,
) -> object | None:
    if config.continuation_evaluations <= 0:
        return None
    archive = run_moea_baseline(
        instance,
        algorithm=algorithm,
        config=MoeaConfig(
            population_size=config.continuation_population_size,
            max_evaluations=config.continuation_evaluations,
            seed=config.moea_seed,
            path_aware=True,
            event_aware=True,
            seed_genes=(seed_genes,),
        ),
    )
    return min(archive, key=lambda item: _vector_sum(item.objective_vector), default=None)


def _diagnostic_row(
    *,
    row: dict[str, str],
    algorithm: str,
    method: str,
    base_vector: tuple[float, ...],
    candidate_vector: tuple[float, ...],
    projection_vector: tuple[float, ...],
    operator_best_vector: tuple[float, ...],
    continuation_best_vector: tuple[float, ...],
    source_genes: tuple[float, ...],
    projected_genes: tuple[float, ...],
    base_delay: int,
    candidate_delay: int,
    projection_delay: int,
    operator_best_delay: int,
    trace_neighborhood_size: int,
    candidate_neighborhood_size: int,
    config: RepairInheritanceConfig,
) -> dict[str, object]:
    candidate_relation = _relation(base_vector, candidate_vector, "base", "candidate")
    projection_relation_to_candidate = _relation(candidate_vector, projection_vector, "candidate", "projection")
    projection_relation_to_base = _relation(base_vector, projection_vector, "base", "projection")
    operator_relation_to_projection = (
        _relation(projection_vector, operator_best_vector, "projection", "operator")
        if operator_best_vector
        else "not-run"
    )
    operator_relation_to_candidate = (
        _relation(candidate_vector, operator_best_vector, "candidate", "operator")
        if operator_best_vector
        else "not-run"
    )
    continuation_relation_to_projection = (
        _relation(projection_vector, continuation_best_vector, "projection", "continuation")
        if continuation_best_vector
        else "not-run"
    )
    continuation_relation_to_candidate = (
        _relation(candidate_vector, continuation_best_vector, "candidate", "continuation")
        if continuation_best_vector
        else "not-run"
    )
    gene_deltas = tuple(abs(left - right) for left, right in zip(projected_genes, source_genes))
    projection_preserved = _is_preserve_relation(projection_relation_to_candidate, contender="projection")
    projection_lost = projection_relation_to_candidate == "candidate-dominates-projection"
    return {
        "algorithm": algorithm,
        "level": row["level"],
        "seed": row["seed"],
        "mode": row["mode"],
        "source": row.get("source", ""),
        "base_solution_id": row.get("solution_id", ""),
        "method": method,
        "base_objective_vector": _format_vector(base_vector),
        "candidate_objective_vector": _format_vector(candidate_vector),
        "projection_objective_vector": _format_vector(projection_vector),
        "operator_best_objective_vector": _format_vector(operator_best_vector),
        "continuation_best_objective_vector": _format_vector(continuation_best_vector),
        "candidate_relation_to_base": candidate_relation,
        "projection_relation_to_candidate": projection_relation_to_candidate,
        "projection_relation_to_base": projection_relation_to_base,
        "operator_best_relation_to_projection": operator_relation_to_projection,
        "operator_best_relation_to_candidate": operator_relation_to_candidate,
        "continuation_best_relation_to_projection": continuation_relation_to_projection,
        "continuation_best_relation_to_candidate": continuation_relation_to_candidate,
        "base_total_delay": base_delay,
        "candidate_total_delay": candidate_delay,
        "projection_total_delay": projection_delay,
        "operator_best_total_delay": operator_best_delay,
        "candidate_delay_delta": candidate_delay - base_delay,
        "projection_delay_delta_to_candidate": projection_delay - candidate_delay,
        "projection_delay_delta_to_base": projection_delay - base_delay,
        "projection_gene_l1_delta_from_source": round(sum(gene_deltas), 10),
        "projection_gene_linf_delta_from_source": round(max(gene_deltas, default=0.0), 10),
        "projection_preserved_candidate": str(projection_preserved),
        "projection_lost_candidate": str(projection_lost),
        "operator_child_preserved_projection": str(
            _is_preserve_relation(operator_relation_to_projection, contender="operator")
        ),
        "operator_child_preserved_candidate": str(
            _is_preserve_relation(operator_relation_to_candidate, contender="operator")
        ),
        "continuation_recovered_candidate": str(
            _is_preserve_relation(continuation_relation_to_candidate, contender="continuation")
        ),
        "trace_neighborhood_size": trace_neighborhood_size,
        "candidate_neighborhood_size": candidate_neighborhood_size,
        "operator_trials": config.operator_trials,
        "continuation_evaluations": config.continuation_evaluations,
    }


def _include_row(row: dict[str, str], config: RepairInheritanceConfig) -> bool:
    if not row.get("genome_genes", "") and not row.get("genome_signature", ""):
        return False
    if config.sources and str(row.get("source", "")) not in config.sources:
        return False
    if config.levels and str(row.get("level", "")) not in config.levels:
        return False
    if config.modes and str(row.get("mode", "")) not in config.modes:
        return False
    seed = _int(row.get("seed", 0))
    if config.start_seed is not None and seed < config.start_seed:
        return False
    if config.end_seed is not None and seed > config.end_seed:
        return False
    algorithm = _algorithm_from_row(row)
    return not config.algorithms or algorithm in config.algorithms


def _instance_from_row(row: dict[str, str], size: str):
    raw = generate_instance(str(row["level"]), size, _int(row["seed"]))
    if str(row.get("mode", "raw")) == "realized":
        return realize_structural_events(raw).instance
    return raw


def _neighborhood_for_method(
    *,
    method: str,
    operations: tuple[str, ...],
    trace_rows: tuple[dict[str, object], ...],
    trace_neighborhood: tuple[str, ...],
    seed: int,
) -> tuple[str, ...]:
    canonical = _canonical_method(method)
    if canonical == "delay-trace-lns":
        return trace_neighborhood
    if canonical == "delay-topk-lns":
        return _delay_topk_neighborhood(operations, trace_rows, len(trace_neighborhood))
    if canonical == "random-lns":
        return _random_neighborhood(operations, len(trace_neighborhood), seed)
    raise ValueError(f"unsupported repair inheritance method: {method}")


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
    raise ValueError(f"unsupported repair inheritance method: {method}")


def _delay_topk_neighborhood(
    operation_ids: tuple[str, ...],
    trace_rows: tuple[dict[str, object], ...],
    target_size: int,
) -> tuple[str, ...]:
    pressure = _trace_delay_by_operation(trace_rows)
    selected = {
        operation_id
        for operation_id, _delay in sorted(
            pressure.items(),
            key=lambda item: (-item[1], item[0]),
        )[:target_size]
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


def _relation(
    reference: tuple[float, ...],
    contender: tuple[float, ...],
    reference_name: str,
    contender_name: str,
) -> str:
    if not reference or not contender:
        return "not-run"
    contender_no_worse = all(left <= right for left, right in zip(contender, reference))
    reference_no_worse = all(right <= left for left, right in zip(contender, reference))
    if contender_no_worse and any(left < right for left, right in zip(contender, reference)):
        return f"{contender_name}-dominates-{reference_name}"
    if reference_no_worse and any(right < left for left, right in zip(contender, reference)):
        return f"{reference_name}-dominates-{contender_name}"
    if contender_no_worse and reference_no_worse:
        return "equal"
    return "mixed"


def _is_success_relation(relation: str) -> bool:
    return relation in {"candidate-dominates-base", "equal"}


def _is_preserve_relation(relation: str, *, contender: str) -> bool:
    return relation in {f"{contender}-dominates-candidate", f"{contender}-dominates-projection", "equal"}


def _canonical_method(method: str) -> str:
    if method in {"trace-lns", "delay-trace-lns"}:
        return "delay-trace-lns"
    if method in {"delay-topk-lns", "delay-top-k-lns"}:
        return "delay-topk-lns"
    if method == "random-lns":
        return "random-lns"
    raise ValueError(f"unsupported repair inheritance method: {method}")


def _algorithm_from_row(row: dict[str, str]) -> str:
    raw = f"{row.get('algorithm', '')}|{row.get('algorithm_variant', '')}".lower()
    if "nsga3" in raw:
        return "nsga3"
    if "spea2" in raw:
        return "spea2"
    if "moead" in raw:
        return "moead"
    return "nsga2"


def _solution_row(
    *,
    solution: object,
    instance_id: str,
    raw_instance_id: str,
    level: str,
    seed: int,
    mode: str,
    source: str,
) -> dict[str, str]:
    return {
        "source": source,
        "instance_id": instance_id,
        "base_instance_id": raw_instance_id,
        "level": level,
        "seed": f"{seed:03d}",
        "mode": mode,
        "objectives": ";".join(solution.metrics.keys()),
        "algorithm": solution.algorithm,
        "algorithm_variant": solution.algorithm,
        "solution_id": solution.solution_id,
        "generation": str(solution.generation),
        "path_aware": str(solution.schedule.path_aware),
        "event_aware": str(solution.schedule.event_aware),
        "objective_vector": _format_vector(solution.objective_vector),
        "constraint_violation": str(solution.constraint_violation),
        "genome_signature": solution.genome_signature,
        "genome_genes": _format_vector(solution.genome_genes),
        "feasible": str(solution.metrics.get("feasible", "")),
    }


def _genome_from_row(row: dict[str, str], *, expected_genes: int) -> _Genome:
    if row.get("genome_genes", ""):
        return _genome_from_signature(str(row["genome_genes"]), expected_genes=expected_genes)
    return _genome_from_signature(str(row.get("genome_signature", "")), expected_genes=expected_genes)


def _genome_from_signature(signature: str, *, expected_genes: int) -> _Genome:
    genes = tuple(float(item) for item in signature.split(";") if item)
    if len(genes) != expected_genes:
        raise ValueError(f"genome_signature has {len(genes)} genes, expected {expected_genes}")
    return _Genome(genes)


def _repair_context(row: dict[str, str], *, algorithm: str, phase: str) -> dict[str, object]:
    return {
        "context_id": f"e95-{phase}|{row.get('level', '')}|{row.get('seed', '')}|{row.get('mode', '')}",
        "solution_id": row.get("solution_id", ""),
        "level": row.get("level", ""),
        "seed": row.get("seed", ""),
        "mode": row.get("mode", ""),
        "algorithm": algorithm,
        "generation": row.get("generation", ""),
    }


def _row_sort_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("level", "")),
        str(row.get("seed", "")),
        str(row.get("mode", "")),
        str(row.get("source", "")),
        str(row.get("solution_id", "")),
    )


def _read_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def _write_rows(path: Path, rows: tuple[dict[str, object], ...], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_vector(vector: tuple[float, ...]) -> str:
    return ";".join(f"{value:.6f}" for value in vector)


def _total_delay(rows: tuple[dict[str, object], ...]) -> int:
    return sum(int(float(row.get("repair_delay", 0) or 0)) for row in rows)


def _vector_sum(vector: tuple[float, ...]) -> float:
    return sum(float(value) for value in vector)


def _stable_seed(base: int, row: dict[str, str], salt: str) -> int:
    text = f"{row.get('level', '')}|{row.get('seed', '')}|{row.get('mode', '')}|{row.get('solution_id', '')}|{salt}"
    return base + sum((index + 1) * ord(char) for index, char in enumerate(text))


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
