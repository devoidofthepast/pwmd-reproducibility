from __future__ import annotations

import math
import random
from dataclasses import dataclass

from su_modfjsp_ha_pc.active_set_trace import build_active_set_rows
from su_modfjsp_ha_pc.metrics import evaluate_schedule
from su_modfjsp_ha_pc.models import Instance, Job, Machine, Operation
from su_modfjsp_ha_pc.rules import _schedule_greedy
from su_modfjsp_ha_pc.schedule import ScheduleResult


SUPPORTED_MOEA_ALGORITHMS = ("nsga2", "nsga3", "spea2", "moead")
_CONSTRAINT_KEYS = (
    "precedence_violations",
    "machine_overlap_violations",
    "agv_overlap_violations",
    "edge_capacity_violations",
    "machine_unavailability_violations",
    "agv_unavailability_violations",
    "path_event_violations",
)


@dataclass(frozen=True)
class MoeaSeedHint:
    tail_weights: tuple[float, float, float]
    operation_template: str = "index"
    machine_template: str = "index"


@dataclass(frozen=True)
class MoeaConfig:
    population_size: int = 24
    max_evaluations: int = 96
    seed: int = 0
    path_aware: bool = True
    event_aware: bool = True
    crossover_rate: float = 0.9
    mutation_rate: float = 0.12
    archive_size: int | None = None
    seed_preferences: tuple[tuple[float, ...], ...] = ()
    seed_hints: tuple[MoeaSeedHint, ...] = ()
    seed_genes: tuple[tuple[float, ...], ...] = ()
    seed_initial_quota: int | None = None
    online_policy: str = "none"
    online_action_policy: object | None = None
    online_action_log: list[dict[str, object]] | None = None
    online_action_reference_count: int = 8
    online_action_strength: float = 1.0
    online_action_gate: str = "none"
    online_action_gate_tolerance: float = 0.0
    mrgv_mode: str = "none"
    mrgv_lambda: float = 0.30
    mrgv_diversity_cap: float = 0.05
    mrgv_tau_min: float = 0.25
    mrgv_tau_path: float = 0.35
    mrgv_tau_event: float = 0.35
    mrgv_tau_agv: float = 0.35
    mrgv_tau_diversity: float = 0.50
    mrgv_gate: str = "pareto-safe"
    mrgv_force_operator: str = "none"
    mrgv_admission_policy: str = "all"
    mrgv_probe_policy: str = "none"
    mrgv_log: list[dict[str, object]] | None = None
    active_set_log: list[dict[str, object]] | None = None
    active_set_evaluated_log: list[dict[str, object]] | None = None
    active_set_context: dict[str, object] | None = None
    active_set_max_records: int = 12
    repair_process_log: list[dict[str, object]] | None = None
    repair_process_evaluated_log: list[dict[str, object]] | None = None
    repair_process_context: dict[str, object] | None = None
    evaluated_solution_log: list[object] | None = None


@dataclass(frozen=True)
class MoeaSolution:
    algorithm: str
    solution_id: str
    generation: int
    schedule: ScheduleResult
    metrics: dict[str, float]
    objective_vector: tuple[float, ...]
    constraint_violation: float
    genome_signature: str
    genome_genes: tuple[float, ...] = ()


@dataclass(frozen=True)
class _Genome:
    genes: tuple[float, ...]


@dataclass(frozen=True)
class _Evaluated:
    uid: int
    generation: int
    genome: _Genome
    schedule: ScheduleResult
    metrics: dict[str, float]
    objective_vector: tuple[float, ...]
    constraint_violation: float


@dataclass(frozen=True)
class _OnlineGraphActionState:
    action: str
    target_direction: tuple[float, ...]
    target_tail: tuple[float, float, float]
    graph_pressure: float
    structural_pressure: float
    front_ratio: float
    constraint_pressure: float
    span_pressure: float


@dataclass(frozen=True)
class _MRGVDiagnosis:
    score_path: float
    score_event: float
    score_agv: float
    score_diversity: float
    target_direction: tuple[float, ...]
    target_tail: tuple[float, float, float]
    sparse_grid_cells: frozenset[str]
    diversity_gate_open: bool


@dataclass(frozen=True)
class _MRGVDecision:
    operator: str
    reason: str


@dataclass
class _MRGVProbeState:
    policy: str
    pair_limit: int
    path_pairs: int = 0
    positive_pairs: int = 0
    decision: str = "probing"


def run_moea_baseline(
    instance: Instance,
    algorithm: str,
    config: MoeaConfig | None = None,
) -> tuple[MoeaSolution, ...]:
    """Run a compact MOEA baseline over a shared greedy repair decoder."""

    if algorithm not in SUPPORTED_MOEA_ALGORITHMS:
        raise ValueError(f"algorithm must be one of {', '.join(SUPPORTED_MOEA_ALGORITHMS)}")
    resolved_config = config or MoeaConfig()
    _validate_config(resolved_config)

    operations = _ordered_operations(instance)
    if not operations:
        return ()

    rng = random.Random(resolved_config.seed + SUPPORTED_MOEA_ALGORITHMS.index(algorithm) * 10_000)
    reference_directions = _reference_directions(
        n_objectives=len(instance.objectives),
        target_count=resolved_config.population_size,
        rng=rng,
        preferences=instance.preferences,
    )
    population = _initial_population(
        instance,
        operations,
        resolved_config.population_size,
        rng,
        seed_preferences=resolved_config.seed_preferences,
        seed_hints=resolved_config.seed_hints,
        seed_genes=resolved_config.seed_genes,
        seed_initial_quota=resolved_config.seed_initial_quota,
    )

    evaluated: list[_Evaluated] = []
    all_evaluated: list[_Evaluated] = []
    uid = 0
    for genome in population[: resolved_config.max_evaluations]:
        evaluated.append(_evaluate_genome(uid, 0, genome, instance, operations, algorithm, resolved_config))
        uid += 1
    all_evaluated.extend(evaluated)
    evaluations = len(evaluated)
    generation = 0
    mrgv_probe_state = _mrgv_probe_state(resolved_config)

    while evaluations < resolved_config.max_evaluations:
        generation += 1
        online_state = (
            _online_graph_action_state(
                evaluated=evaluated,
                instance=instance,
                operations=operations,
                reference_directions=reference_directions,
            )
            if resolved_config.online_policy == "graph-action-adaptive"
            else None
        )
        online_states = (
            _online_graph_action_candidate_states(
                evaluated=evaluated,
                instance=instance,
                operations=operations,
                reference_directions=reference_directions,
                limit=resolved_config.online_action_reference_count,
            )
            if resolved_config.online_policy == "learned-online-action-value"
            else ()
        )
        if resolved_config.mrgv_mode in {"evaluation-counted", "dual-offspring"}:
            offspring_evaluated: list[_Evaluated] = []
            while len(offspring_evaluated) < resolved_config.population_size and evaluations < resolved_config.max_evaluations:
                parent_a = _tournament(evaluated, rng)
                parent_b = _tournament(evaluated, rng)
                baseline_child = _crossover(parent_a.genome, parent_b.genome, rng, resolved_config.crossover_rate)
                baseline_child = _mutate(baseline_child, rng, resolved_config.mutation_rate)
                eval_count_before = evaluations
                baseline_evaluated = _evaluate_genome(
                    uid,
                    generation,
                    baseline_child,
                    instance,
                    operations,
                    algorithm,
                    resolved_config,
                )
                uid += 1
                evaluations += 1
                selected_evaluated = baseline_evaluated
                candidate_evaluated: _Evaluated | None = None
                candidate_enters_pool = False
                if (
                    evaluations < resolved_config.max_evaluations
                    and _mrgv_probe_allows_attempt(mrgv_probe_state)
                    and rng.random() < resolved_config.mrgv_lambda
                ):
                    mrgv_attempts = len(resolved_config.mrgv_log or ())
                    diversity_attempts = sum(
                        1
                        for entry in (resolved_config.mrgv_log or ())
                        if entry.get("mrgv_operator") == "conservative-structural-diversity"
                    )
                    diagnosis = _mrgv_diagnosis(
                        evaluated=evaluated,
                        instance=instance,
                        operations=operations,
                        generation=generation,
                        diversity_attempts=diversity_attempts,
                        mrgv_attempts=mrgv_attempts,
                        config=resolved_config,
                    )
                    decision = _select_mrgv_operator(
                        diagnosis,
                        mode=_instance_mode(instance),
                        config=resolved_config,
                    )
                    candidate_child = _mutate_mrgv(
                        genome=baseline_child,
                        rng=rng,
                        operator=decision.operator,
                        diagnosis=diagnosis,
                        operation_count=len(operations),
                        machine_count=len(instance.machines),
                        operations=operations,
                        machines=instance.machines,
                        jobs=instance.jobs,
                    )
                    if decision.operator == "baseline" or _same_genome(candidate_child, baseline_child):
                        _append_mrgv_log(
                            resolved_config.mrgv_log,
                            context_id=instance.instance_id,
                            generation=generation,
                            eval_count_before=eval_count_before,
                            mode=resolved_config.mrgv_mode,
                            mrgv_lambda=resolved_config.mrgv_lambda,
                            decision=decision,
                            diagnosis=diagnosis,
                            gate=resolved_config.mrgv_gate,
                            gate_lane=_mrgv_gate_lane(decision, resolved_config.mrgv_gate),
                            candidate_policy=_mrgv_candidate_policy(
                                resolved_config.mrgv_mode,
                                resolved_config.mrgv_admission_policy,
                            ),
                            admission_policy=resolved_config.mrgv_admission_policy,
                            accepted=False,
                            gate_reason="unchanged-or-baseline",
                            baseline=baseline_evaluated,
                            candidate=None,
                            selected=baseline_evaluated if resolved_config.mrgv_mode == "evaluation-counted" else None,
                            baseline_indicator=None,
                            candidate_indicator=None,
                            baseline_grid_cell=_mrgv_grid_cell(baseline_evaluated, evaluated),
                            candidate_grid_cell="",
                            selected_child="baseline-only" if resolved_config.mrgv_mode == "dual-offspring" else None,
                            generated=decision.operator != "baseline",
                            non_noop=False,
                            evaluated=False,
                            entered_pool=False,
                        )
                    else:
                        candidate_evaluated = _evaluate_genome(
                            uid,
                            generation,
                            candidate_child,
                            instance,
                            operations,
                            algorithm,
                            resolved_config,
                        )
                        uid += 1
                        evaluations += 1
                        accepted, reason, baseline_score, candidate_score, baseline_cell, candidate_cell = _evaluate_mrgv_gate(
                            baseline=baseline_evaluated,
                            candidate=candidate_evaluated,
                            current_population=evaluated,
                            diagnosis=diagnosis,
                            gate=resolved_config.mrgv_gate,
                        )
                        if accepted:
                            selected_evaluated = candidate_evaluated
                        mrgv_entered_pool = _mrgv_enters_pool(
                            mode=resolved_config.mrgv_mode,
                            admission_policy=resolved_config.mrgv_admission_policy,
                            accepted=accepted,
                        )
                        probe_info = _update_mrgv_probe_state(
                            mrgv_probe_state,
                            decision=decision,
                            baseline_indicator=baseline_score,
                            candidate_indicator=candidate_score,
                        )
                        if probe_info.get("probe_admission_override", "") != "":
                            mrgv_entered_pool = bool(probe_info["probe_admission_override"])
                        candidate_enters_pool = mrgv_entered_pool
                        _append_mrgv_log(
                            resolved_config.mrgv_log,
                            context_id=instance.instance_id,
                            generation=generation,
                            eval_count_before=eval_count_before,
                            mode=resolved_config.mrgv_mode,
                            mrgv_lambda=resolved_config.mrgv_lambda,
                            decision=decision,
                            diagnosis=diagnosis,
                            gate=resolved_config.mrgv_gate,
                            gate_lane=_mrgv_gate_lane(decision, resolved_config.mrgv_gate),
                            candidate_policy=_mrgv_candidate_policy(
                                resolved_config.mrgv_mode,
                                resolved_config.mrgv_admission_policy,
                            ),
                            admission_policy=resolved_config.mrgv_admission_policy,
                            accepted=accepted,
                            gate_reason=reason,
                            baseline=baseline_evaluated,
                            candidate=candidate_evaluated,
                            selected=selected_evaluated if resolved_config.mrgv_mode == "evaluation-counted" else None,
                            baseline_indicator=baseline_score,
                            candidate_indicator=candidate_score,
                            baseline_grid_cell=baseline_cell,
                            candidate_grid_cell=candidate_cell,
                            selected_child=_mrgv_selected_child_label(
                                mode=resolved_config.mrgv_mode,
                                entered_pool=mrgv_entered_pool,
                            ),
                            generated=True,
                            non_noop=True,
                            evaluated=True,
                            entered_pool=mrgv_entered_pool,
                            probe_info=probe_info,
                        )
                if resolved_config.mrgv_mode == "dual-offspring":
                    offspring_evaluated.append(baseline_evaluated)
                    if candidate_evaluated is not None and candidate_enters_pool:
                        offspring_evaluated.append(candidate_evaluated)
                else:
                    offspring_evaluated.append(selected_evaluated)
        elif (
            online_states
            and resolved_config.online_policy == "learned-online-action-value"
            and resolved_config.online_action_gate == "dominance-indicator"
        ):
            offspring_evaluated: list[_Evaluated] = []
            while len(offspring_evaluated) < resolved_config.population_size and evaluations < resolved_config.max_evaluations:
                parent_a = _tournament(evaluated, rng)
                parent_b = _tournament(evaluated, rng)
                baseline_child = _crossover(parent_a.genome, parent_b.genome, rng, resolved_config.crossover_rate)
                baseline_child = _mutate(baseline_child, rng, resolved_config.mutation_rate)
                baseline_evaluated = _evaluate_genome(
                    uid,
                    generation,
                    baseline_child,
                    instance,
                    operations,
                    algorithm,
                    resolved_config,
                )
                uid += 1
                evaluations += 1
                selected_evaluated = baseline_evaluated
                if evaluations < resolved_config.max_evaluations:
                    decision_context: dict[str, object] = {}
                    candidate_child = _mutate_learned_online_action_value(
                        baseline_child,
                        rng,
                        base_mutation_rate=resolved_config.mutation_rate,
                        states=online_states,
                        operation_count=len(operations),
                        machine_count=len(instance.machines),
                        operations=operations,
                        machines=instance.machines,
                        jobs=instance.jobs,
                        instance=instance,
                        policy=resolved_config.online_action_policy,
                        action_log=resolved_config.online_action_log,
                        action_strength=resolved_config.online_action_strength,
                        generation=generation,
                        decision_context=decision_context,
                    )
                    if _same_genome(candidate_child, baseline_child):
                        _annotate_online_action_gate(
                            resolved_config.online_action_log,
                            gate=resolved_config.online_action_gate,
                            accepted=False,
                            reason="unchanged-or-fallback",
                        )
                    else:
                        candidate_evaluated = _evaluate_genome(
                            uid,
                            generation,
                            candidate_child,
                            instance,
                            operations,
                            algorithm,
                            resolved_config,
                        )
                        uid += 1
                        evaluations += 1
                        accepted, reason, baseline_score, candidate_score = _passes_online_action_gate(
                            baseline=baseline_evaluated,
                            candidate=candidate_evaluated,
                            current_population=evaluated,
                            target_direction=tuple(decision_context.get("target_direction", ())),
                            tolerance=resolved_config.online_action_gate_tolerance,
                        )
                        _annotate_online_action_gate(
                            resolved_config.online_action_log,
                            gate=resolved_config.online_action_gate,
                            accepted=accepted,
                            reason=reason,
                            baseline_constraint=baseline_evaluated.constraint_violation,
                            candidate_constraint=candidate_evaluated.constraint_violation,
                            baseline_indicator=baseline_score,
                            candidate_indicator=candidate_score,
                        )
                        if accepted:
                            selected_evaluated = candidate_evaluated
                offspring_evaluated.append(selected_evaluated)
        else:
            offspring: list[_Genome] = []
            while len(offspring) < resolved_config.population_size and evaluations + len(offspring) < resolved_config.max_evaluations:
                parent_a = _tournament(evaluated, rng)
                parent_b = _tournament(evaluated, rng)
                child = _crossover(parent_a.genome, parent_b.genome, rng, resolved_config.crossover_rate)
                child = _mutate(child, rng, resolved_config.mutation_rate)
                if resolved_config.mrgv_mode == "replacement" and rng.random() < resolved_config.mrgv_lambda:
                    mrgv_attempts = len(resolved_config.mrgv_log or ())
                    diversity_attempts = sum(
                        1
                        for entry in (resolved_config.mrgv_log or ())
                        if entry.get("mrgv_operator") == "conservative-structural-diversity"
                    )
                    diagnosis = _mrgv_diagnosis(
                        evaluated=evaluated,
                        instance=instance,
                        operations=operations,
                        generation=generation,
                        diversity_attempts=diversity_attempts,
                        mrgv_attempts=mrgv_attempts,
                        config=resolved_config,
                    )
                    decision = _select_mrgv_operator(
                        diagnosis,
                        mode=_instance_mode(instance),
                        config=resolved_config,
                    )
                    candidate_child = _mutate_mrgv(
                        genome=child,
                        rng=rng,
                        operator=decision.operator,
                        diagnosis=diagnosis,
                        operation_count=len(operations),
                        machine_count=len(instance.machines),
                        operations=operations,
                        machines=instance.machines,
                        jobs=instance.jobs,
                    )
                    accepted = decision.operator != "baseline" and not _same_genome(candidate_child, child)
                    if accepted:
                        child = candidate_child
                    _append_mrgv_replacement_log(
                        resolved_config.mrgv_log,
                        context_id=instance.instance_id,
                        generation=generation,
                        eval_count_before=evaluations + len(offspring),
                        mrgv_lambda=resolved_config.mrgv_lambda,
                        decision=decision,
                        diagnosis=diagnosis,
                        selected_child_id=uid + len(offspring),
                        accepted=accepted,
                    )
                if online_state is not None and resolved_config.online_policy == "graph-action-adaptive":
                    child = _mutate_online_graph_action(
                        child,
                        rng,
                        base_mutation_rate=resolved_config.mutation_rate,
                        state=online_state,
                        operation_count=len(operations),
                        machine_count=len(instance.machines),
                        operations=operations,
                        machines=instance.machines,
                        jobs=instance.jobs,
                    )
                elif online_states and resolved_config.online_policy == "learned-online-action-value":
                    child = _mutate_learned_online_action_value(
                        child,
                        rng,
                        base_mutation_rate=resolved_config.mutation_rate,
                        states=online_states,
                        operation_count=len(operations),
                        machine_count=len(instance.machines),
                        operations=operations,
                        machines=instance.machines,
                        jobs=instance.jobs,
                        instance=instance,
                        policy=resolved_config.online_action_policy,
                        action_log=resolved_config.online_action_log,
                        action_strength=resolved_config.online_action_strength,
                        generation=generation,
                    )
                offspring.append(child)

            offspring_evaluated = [
                _evaluate_genome(uid + index, generation, genome, instance, operations, algorithm, resolved_config)
                for index, genome in enumerate(offspring)
            ]
            uid += len(offspring_evaluated)
            evaluations += len(offspring_evaluated)
        all_evaluated.extend(offspring_evaluated)
        evaluated = _select_next(
            evaluated + offspring_evaluated,
            resolved_config.population_size,
            algorithm,
            reference_directions,
        )
        _annotate_mrgv_survival(resolved_config.mrgv_log, generation, {item.uid for item in evaluated})

    archive_size = resolved_config.archive_size or resolved_config.population_size
    archive = _select_archive(all_evaluated, archive_size, algorithm, reference_directions)
    _annotate_mrgv_archive(resolved_config.mrgv_log, {item.uid for item in archive})
    if resolved_config.evaluated_solution_log is not None:
        resolved_config.evaluated_solution_log.extend(
            _solution_from_evaluated(solution, algorithm)
            for solution in all_evaluated
        )
    return tuple(
        _solution_from_evaluated(solution, algorithm)
        for solution in archive
    )


def _solution_from_evaluated(solution: _Evaluated, algorithm: str) -> MoeaSolution:
    return MoeaSolution(
        algorithm=f"moea-{algorithm}",
        solution_id=f"{algorithm}-{solution.uid:05d}",
        generation=solution.generation,
        schedule=solution.schedule,
        metrics=solution.metrics,
        objective_vector=solution.objective_vector,
        constraint_violation=solution.constraint_violation,
        genome_signature=_genome_signature(solution.genome),
        genome_genes=solution.genome.genes,
    )


def _validate_config(config: MoeaConfig) -> None:
    if config.population_size <= 0:
        raise ValueError("population_size must be positive")
    if config.max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")
    if config.max_evaluations < config.population_size:
        raise ValueError("max_evaluations must be greater than or equal to population_size")
    if config.seed_initial_quota is not None and config.seed_initial_quota < 0:
        raise ValueError("seed_initial_quota must be non-negative")
    if not 0.0 <= config.crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be in [0, 1]")
    if not 0.0 <= config.mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be in [0, 1]")
    if config.online_policy not in {"none", "graph-action-adaptive", "learned-online-action-value"}:
        raise ValueError("online_policy must be 'none', 'graph-action-adaptive', or 'learned-online-action-value'")
    if config.online_policy == "learned-online-action-value" and config.online_action_policy is None:
        raise ValueError("online_action_policy is required for learned-online-action-value")
    if config.online_action_reference_count <= 0:
        raise ValueError("online_action_reference_count must be positive")
    if not 0.0 <= config.online_action_strength <= 1.0:
        raise ValueError("online_action_strength must be in [0, 1]")
    if config.online_action_gate not in {"none", "dominance-indicator"}:
        raise ValueError("online_action_gate must be 'none' or 'dominance-indicator'")
    if config.online_action_gate_tolerance < 0.0:
        raise ValueError("online_action_gate_tolerance must be non-negative")
    if config.mrgv_mode not in {"none", "evaluation-counted", "replacement", "dual-offspring"}:
        raise ValueError("mrgv_mode must be 'none', 'evaluation-counted', 'replacement', or 'dual-offspring'")
    if not 0.0 <= config.mrgv_lambda <= 1.0:
        raise ValueError("mrgv_lambda must be in [0, 1]")
    if not 0.0 <= config.mrgv_diversity_cap <= 1.0:
        raise ValueError("mrgv_diversity_cap must be in [0, 1]")
    if config.mrgv_tau_min < 0.0:
        raise ValueError("mrgv_tau_min must be non-negative")
    if config.mrgv_tau_path < 0.0:
        raise ValueError("mrgv_tau_path must be non-negative")
    if config.mrgv_tau_event < 0.0:
        raise ValueError("mrgv_tau_event must be non-negative")
    if config.mrgv_tau_agv < 0.0:
        raise ValueError("mrgv_tau_agv must be non-negative")
    if config.mrgv_tau_diversity < 0.0:
        raise ValueError("mrgv_tau_diversity must be non-negative")
    if config.mrgv_gate not in {"none", "pareto-safe", "indicator-safe"}:
        raise ValueError("mrgv_gate must be 'none', 'pareto-safe', or 'indicator-safe'")
    if config.mrgv_admission_policy not in {"all", "gated"}:
        raise ValueError("mrgv_admission_policy must be 'all' or 'gated'")
    if config.mrgv_probe_policy not in {"none", "b4-local-indicator"}:
        raise ValueError("mrgv_probe_policy must be 'none' or 'b4-local-indicator'")
    _normalize_mrgv_force_operator(config.mrgv_force_operator)


def _ordered_operations(instance: Instance) -> tuple[Operation, ...]:
    return tuple(operation for job in instance.jobs for operation in job.operations)


def _initial_population(
    instance: Instance,
    operations: tuple[Operation, ...],
    population_size: int,
    rng: random.Random,
    seed_preferences: tuple[tuple[float, ...], ...] = (),
    seed_hints: tuple[MoeaSeedHint, ...] = (),
    seed_genes: tuple[tuple[float, ...], ...] = (),
    seed_initial_quota: int | None = None,
) -> list[_Genome]:
    gene_count = len(operations) + len(instance.machines) + 3
    seeded = _direct_seed_genomes(seed_genes, gene_count) + _seed_genomes(
        instance,
        operations,
        seed_preferences=seed_preferences,
        seed_hints=seed_hints,
    )
    seeded_quota = population_size if seed_initial_quota is None else min(population_size, seed_initial_quota)
    population = seeded[:seeded_quota]
    while len(population) < population_size:
        population.append(_Genome(tuple(rng.random() for _ in range(gene_count))))
    return population


def _direct_seed_genomes(seed_genes: tuple[tuple[float, ...], ...], gene_count: int) -> list[_Genome]:
    return [
        _Genome(tuple(_clip(value) for value in genes))
        for genes in seed_genes
        if len(genes) == gene_count
    ]


def _seed_genomes(
    instance: Instance,
    operations: tuple[Operation, ...],
    seed_preferences: tuple[tuple[float, ...], ...] = (),
    seed_hints: tuple[MoeaSeedHint, ...] = (),
) -> list[_Genome]:
    max_due = max((job.due_date for job in instance.jobs), default=1)
    job_by_id = {job.job_id: job for job in instance.jobs}
    max_processing = max((min(operation.processing_time.values()) for operation in operations), default=1)
    max_energy = max((min(operation.processing_energy.values()) for operation in operations), default=1.0)
    templates = []
    operation_scores = {
        "index": tuple(index / max(1, len(operations) - 1) for index, _ in enumerate(operations)),
        "processing": tuple(min(operation.processing_time.values()) / max_processing for operation in operations),
        "due": tuple(job_by_id[operation.job_id].due_date / max_due for operation in operations),
        "energy": tuple(min(operation.processing_energy.values()) / max_energy for operation in operations),
    }
    machine_scores = {
        "index": tuple(index / max(1, len(instance.machines) - 1) for index, _ in enumerate(instance.machines)),
        "energy": tuple(
            machine.energy_rate / max(1.0, max(item.energy_rate for item in instance.machines))
            for machine in instance.machines
        ),
    }
    for hint in seed_hints:
        operation_score = operation_scores.get(hint.operation_template, operation_scores["index"])
        machine_score = machine_scores.get(hint.machine_template, machine_scores["index"])
        templates.append(
            _Genome(
                tuple(
                    _clip(value)
                    for value in operation_score + machine_score + _safe_simplex(tuple(hint.tail_weights))
                )
            )
        )
    default_weight_tails = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.5, 0.5, 0.5),
    )
    preference_weight_tails = tuple(
        _tail_from_seed_preference(preference, instance.objectives)
        for preference in seed_preferences
    )
    weight_tails = preference_weight_tails + default_weight_tails
    for op_score in tuple(operation_scores.values()):
        for machine_score in tuple(machine_scores.values()):
            for tail in weight_tails:
                templates.append(_Genome(tuple(_clip(value) for value in op_score + machine_score + tail)))
    return templates


def _tail_from_seed_preference(
    preference: tuple[float, ...],
    objectives: tuple[str, ...],
) -> tuple[float, float, float]:
    if not preference:
        return (0.5, 0.5, 0.5)
    usable = preference[: len(objectives)]
    weights = _safe_simplex(tuple(float(value) for value in usable))
    by_objective = {
        objective: weights[index]
        for index, objective in enumerate(objectives[: len(weights)])
    }
    speed_weight = by_objective.get("makespan", 0.0) + by_objective.get("path_conflict_delay", 0.0)
    energy_weight = by_objective.get("energy", 0.0)
    due_weight = by_objective.get("tardiness", 0.0) + by_objective.get("stability", 0.0)
    return _safe_simplex((speed_weight, energy_weight, due_weight))


def _evaluate_genome(
    uid: int,
    generation: int,
    genome: _Genome,
    instance: Instance,
    operations: tuple[Operation, ...],
    algorithm: str,
    config: MoeaConfig,
) -> _Evaluated:
    operation_index = {operation.operation_id: index for index, operation in enumerate(operations)}
    machine_index = {machine.machine_id: index for index, machine in enumerate(instance.machines)}
    max_processing = max((max(operation.processing_time.values()) for operation in operations), default=1)
    max_energy = max((max(operation.processing_energy.values()) for operation in operations), default=1.0)
    max_due = max((job.due_date for job in instance.jobs), default=1)
    speed_weight, energy_weight, due_weight = _tail_weights(genome, len(operations), len(instance.machines))

    def select_operation(available: tuple[tuple[Job, Operation], ...]) -> tuple[Job, Operation]:
        return min(
            available,
            key=lambda item: (
                _operation_gene(genome, operation_index[item[1].operation_id])
                + due_weight * item[0].due_date / max_due,
                item[1].operation_id,
            ),
        )

    def select_machine(machine_by_id: dict[str, Machine], job: Job, operation: Operation) -> str:
        del job
        return min(
            operation.eligible_machines,
            key=lambda candidate: (
                speed_weight * operation.processing_time[candidate] / max_processing
                + energy_weight * operation.processing_energy[candidate] / max_energy
                + 0.25 * _machine_gene(genome, len(operations), machine_index[candidate]),
                candidate,
            ),
        )

    repair_context = _repair_process_context(config, instance, uid, generation, algorithm)
    schedule = _schedule_greedy(
        instance=instance,
        algorithm=f"moea-{algorithm}",
        path_aware=config.path_aware,
        event_aware=config.event_aware,
        select_operation=select_operation,
        select_machine=select_machine,
        repair_process_log=config.repair_process_log,
        repair_process_context=repair_context,
    )
    metrics = evaluate_schedule(instance, schedule)
    objective_vector = tuple(float(metrics[objective]) for objective in instance.objectives)
    constraint_violation = sum(float(metrics[key]) for key in _CONSTRAINT_KEYS)
    _append_active_set_trace(
        config,
        instance=instance,
        schedule=schedule,
        uid=uid,
        generation=generation,
        algorithm=algorithm,
    )
    return _Evaluated(
        uid=uid,
        generation=generation,
        genome=genome,
        schedule=schedule,
        metrics=metrics,
        objective_vector=objective_vector,
        constraint_violation=constraint_violation,
    )


def _repair_process_context(
    config: MoeaConfig,
    instance: Instance,
    uid: int,
    generation: int,
    algorithm: str,
) -> dict[str, object] | None:
    if config.repair_process_log is None and config.repair_process_evaluated_log is None:
        return None
    context = config.repair_process_context or {}
    solution_id = f"{algorithm}-{uid:05d}"
    row = {
        "context_id": str(context.get("context_id", instance.instance_id)),
        "solution_id": solution_id,
        "level": str(context.get("level", instance.level)),
        "seed": str(context.get("seed", instance.seed)),
        "mode": str(context.get("mode", "raw")),
        "algorithm": f"moea-{algorithm}",
        "generation": generation,
        "eval_uid": uid,
    }
    if config.repair_process_evaluated_log is not None:
        config.repair_process_evaluated_log.append(row)
    return row


def _append_active_set_trace(
    config: MoeaConfig,
    *,
    instance: Instance,
    schedule: ScheduleResult,
    uid: int,
    generation: int,
    algorithm: str,
) -> None:
    if config.active_set_log is None and config.active_set_evaluated_log is None:
        return
    context = config.active_set_context or {}
    solution_id = f"{algorithm}-{uid:05d}"
    evaluated_row = {
        "context_id": str(context.get("context_id", instance.instance_id)),
        "solution_id": solution_id,
        "level": str(context.get("level", instance.level)),
        "seed": str(context.get("seed", instance.seed)),
        "mode": str(context.get("mode", "raw")),
        "algorithm": f"moea-{algorithm}",
        "generation": generation,
        "eval_uid": uid,
    }
    if config.active_set_evaluated_log is not None:
        config.active_set_evaluated_log.append(evaluated_row)
    if config.active_set_log is None:
        return
    config.active_set_log.extend(
        build_active_set_rows(
            instance,
            schedule,
            solution_id=solution_id,
            context_id=evaluated_row["context_id"],
            level=evaluated_row["level"],
            seed=evaluated_row["seed"],
            mode=evaluated_row["mode"],
            algorithm=evaluated_row["algorithm"],
            generation=generation,
            eval_uid=uid,
            max_active_records=config.active_set_max_records,
        )
    )


def _operation_gene(genome: _Genome, index: int) -> float:
    return genome.genes[index]


def _machine_gene(genome: _Genome, operation_count: int, index: int) -> float:
    return genome.genes[operation_count + index]


def _tail_weights(genome: _Genome, operation_count: int, machine_count: int) -> tuple[float, float, float]:
    offset = operation_count + machine_count
    return genome.genes[offset], genome.genes[offset + 1], genome.genes[offset + 2]


def _tournament(population: list[_Evaluated], rng: random.Random) -> _Evaluated:
    left = rng.choice(population)
    right = rng.choice(population)
    if _dominates(left, right):
        return left
    if _dominates(right, left):
        return right
    return left if sum(left.objective_vector) <= sum(right.objective_vector) else right


def _crossover(left: _Genome, right: _Genome, rng: random.Random, crossover_rate: float) -> _Genome:
    if rng.random() > crossover_rate:
        return left
    genes = []
    for left_value, right_value in zip(left.genes, right.genes):
        if rng.random() < 0.5:
            genes.append(left_value)
        else:
            alpha = rng.random()
            genes.append(alpha * left_value + (1.0 - alpha) * right_value)
    return _Genome(tuple(_clip(value) for value in genes))


def _mutate(genome: _Genome, rng: random.Random, mutation_rate: float) -> _Genome:
    return _Genome(
        tuple(
            _clip(value + rng.gauss(0.0, 0.18)) if rng.random() < mutation_rate else value
            for value in genome.genes
        )
    )


def _online_graph_action_state(
    evaluated: list[_Evaluated],
    instance: Instance,
    operations: tuple[Operation, ...],
    reference_directions: tuple[tuple[float, ...], ...],
) -> _OnlineGraphActionState:
    target_direction = _least_covered_reference_direction(evaluated, instance, reference_directions)
    return _online_graph_action_state_for_direction(
        target_direction=target_direction,
        evaluated=evaluated,
        instance=instance,
        operations=operations,
    )


def _online_graph_action_candidate_states(
    evaluated: list[_Evaluated],
    instance: Instance,
    operations: tuple[Operation, ...],
    reference_directions: tuple[tuple[float, ...], ...],
    limit: int,
) -> tuple[_OnlineGraphActionState, ...]:
    n_objectives = len(instance.objectives)
    candidates: list[tuple[float, ...]] = []
    candidates.extend(_least_covered_reference_directions(evaluated, instance, reference_directions, limit=max(2, limit // 2)))
    candidates.extend(_objective_anchor_directions(instance.objectives))
    candidates.append(tuple(1.0 / n_objectives for _ in range(n_objectives)))
    candidates.extend(_dynamic_priority_reference_directions(instance, reference_directions, limit=max(1, limit // 4)))
    directions = _unique_reference_directions(candidates, n_objectives=n_objectives, limit=limit)
    return tuple(
        _online_graph_action_state_for_direction(
            target_direction=direction,
            evaluated=evaluated,
            instance=instance,
            operations=operations,
        )
        for direction in directions
    )


def _online_graph_action_state_for_direction(
    target_direction: tuple[float, ...],
    evaluated: list[_Evaluated],
    instance: Instance,
    operations: tuple[Operation, ...],
) -> _OnlineGraphActionState:
    graph_pressure = _graph_pressure(instance, operations)
    structural_pressure = _structural_pressure(instance)
    front_ratio = len(_nondominated_front(evaluated)) / len(evaluated) if evaluated else 0.0
    constraint_pressure = _constraint_pressure(evaluated)
    span_pressure = _span_pressure(evaluated)
    target_tail = _online_target_tail(
        target_direction=target_direction,
        objectives=instance.objectives,
        graph_pressure=graph_pressure,
        structural_pressure=structural_pressure,
    )
    if constraint_pressure > 0.0 or graph_pressure + structural_pressure >= 0.85:
        action = "graph-repair"
    elif front_ratio < 0.35 or span_pressure > 0.65:
        action = "diversity-probe"
    else:
        action = "reference-gap"
    return _OnlineGraphActionState(
        action=action,
        target_direction=target_direction,
        target_tail=target_tail,
        graph_pressure=graph_pressure,
        structural_pressure=structural_pressure,
        front_ratio=front_ratio,
        constraint_pressure=constraint_pressure,
        span_pressure=span_pressure,
    )


def _mutate_online_graph_action(
    genome: _Genome,
    rng: random.Random,
    base_mutation_rate: float,
    state: _OnlineGraphActionState,
    operation_count: int,
    machine_count: int,
    operations: tuple[Operation, ...] = (),
    machines: tuple[Machine, ...] = (),
    jobs: tuple[Job, ...] = (),
) -> _Genome:
    genes = list(genome.genes)
    if not genes:
        return genome

    op_targets = _online_operation_targets(state, operations, jobs, machine_count)
    machine_targets = _online_machine_targets(state, machines)
    operation_rate = min(0.85, base_mutation_rate + 0.10 + 0.25 * state.graph_pressure)
    machine_rate = min(0.85, base_mutation_rate + 0.08 + 0.20 * state.graph_pressure)
    operation_blend = 0.18 + 0.12 * state.structural_pressure
    machine_blend = 0.16 + 0.10 * state.graph_pressure

    for index in range(min(operation_count, len(genes))):
        if op_targets and rng.random() < operation_rate:
            target = op_targets[index]
            genes[index] = _clip((1.0 - operation_blend) * genes[index] + operation_blend * target + rng.gauss(0.0, 0.035))

    machine_offset = operation_count
    for index in range(machine_count):
        gene_index = machine_offset + index
        if gene_index >= len(genes):
            break
        if machine_targets and rng.random() < machine_rate:
            target = machine_targets[index]
            genes[gene_index] = _clip((1.0 - machine_blend) * genes[gene_index] + machine_blend * target + rng.gauss(0.0, 0.03))

    tail_offset = operation_count + machine_count
    if tail_offset + 2 < len(genes):
        tail_blend = 0.30 + 0.15 * state.graph_pressure + 0.10 * state.structural_pressure
        if state.action == "diversity-probe":
            tail_blend = 0.22
        for index, target in enumerate(state.target_tail):
            gene_index = tail_offset + index
            genes[gene_index] = _clip((1.0 - tail_blend) * genes[gene_index] + tail_blend * target + rng.gauss(0.0, 0.02))

    if state.action == "diversity-probe":
        span = max(1, operation_count + machine_count)
        for _ in range(max(1, span // 8)):
            index = rng.randrange(min(span, len(genes)))
            genes[index] = _clip(genes[index] + rng.gauss(0.0, 0.24))

    return _Genome(tuple(genes))


def _mutate_learned_online_action_value(
    genome: _Genome,
    rng: random.Random,
    base_mutation_rate: float,
    states: tuple[_OnlineGraphActionState, ...],
    operation_count: int,
    machine_count: int,
    operations: tuple[Operation, ...],
    machines: tuple[Machine, ...],
    jobs: tuple[Job, ...],
    instance: Instance,
    policy: object,
    action_log: list[dict[str, object]] | None = None,
    action_strength: float = 1.0,
    generation: int = 0,
    decision_context: dict[str, object] | None = None,
) -> _Genome:
    from su_modfjsp_ha_pc.learned_online_policy import choose_learned_online_action

    state = _select_learned_online_state(
        genome,
        rng,
        states,
        operation_count,
        machine_count,
        exploration_rate=0.30 * action_strength,
    )
    decision = choose_learned_online_action(
        policy=policy,
        instance=instance,
        target_direction=state.target_direction,
        graph_pressure=state.graph_pressure,
        structural_pressure=state.structural_pressure,
        front_ratio=state.front_ratio,
        constraint_pressure=state.constraint_pressure,
        span_pressure=state.span_pressure,
    )
    if decision_context is not None:
        decision_context.update(
            {
                "target_direction": state.target_direction,
                "target_tail": decision.target_tail,
                "action_template": decision.action_template,
                "fallback": decision.fallback,
            }
        )
    if action_log is not None:
        action_log.append(
            {
                "generation": generation,
                "action_template": decision.action_template,
                "confidence": round(decision.confidence, 10),
                "fallback": str(decision.fallback),
                "state_action": state.action,
                "target_direction": ";".join(f"{value:.6f}" for value in state.target_direction),
                "target_tail": ";".join(f"{value:.6f}" for value in decision.target_tail),
                "action_strength": round(action_strength, 6),
            }
        )
    if decision.fallback or action_strength <= 0.0:
        return genome

    genes = list(genome.genes)
    if not genes:
        return genome

    op_targets = _learned_online_operation_targets(
        action_template=decision.action_template,
        operations=operations,
        jobs=jobs,
        machine_count=machine_count,
    )
    machine_targets = _learned_online_machine_targets(
        action_template=decision.action_template,
        machines=machines,
    )
    operation_rate = _blend_rate(base_mutation_rate, decision.operation_rate, action_strength, floor_bonus=0.01)
    machine_rate = _blend_rate(base_mutation_rate, decision.machine_rate, action_strength, floor_bonus=0.008)
    operation_blend = decision.operation_blend * action_strength
    machine_blend = decision.machine_blend * action_strength
    tail_blend = decision.tail_blend * action_strength
    operation_noise = 0.032 * action_strength
    machine_noise = 0.028 * action_strength
    tail_noise = 0.018 * action_strength

    for index in range(min(operation_count, len(genes))):
        if op_targets and rng.random() < operation_rate:
            target = op_targets[index]
            genes[index] = _clip(
                (1.0 - operation_blend) * genes[index]
                + operation_blend * target
                + rng.gauss(0.0, operation_noise)
            )

    machine_offset = operation_count
    for index in range(machine_count):
        gene_index = machine_offset + index
        if gene_index >= len(genes):
            break
        if machine_targets and rng.random() < machine_rate:
            target = machine_targets[index]
            genes[gene_index] = _clip(
                (1.0 - machine_blend) * genes[gene_index]
                + machine_blend * target
                + rng.gauss(0.0, machine_noise)
            )

    tail_offset = operation_count + machine_count
    if tail_offset + 2 < len(genes):
        for index, target in enumerate(decision.target_tail):
            gene_index = tail_offset + index
            genes[gene_index] = _clip(
                (1.0 - tail_blend) * genes[gene_index]
                + tail_blend * target
                + rng.gauss(0.0, tail_noise)
            )

    if decision.action_template == "balanced-diversity" and rng.random() < action_strength:
        span = max(1, operation_count + machine_count)
        perturbations = max(1, int((span // 10) * action_strength))
        for _ in range(perturbations):
            index = rng.randrange(min(span, len(genes)))
            genes[index] = _clip(genes[index] + rng.gauss(0.0, 0.20 * action_strength))

    return _Genome(tuple(genes))


def _mrgv_diagnosis(
    evaluated: list[_Evaluated],
    instance: Instance,
    operations: tuple[Operation, ...],
    generation: int,
    diversity_attempts: int,
    mrgv_attempts: int,
    config: MoeaConfig,
) -> _MRGVDiagnosis:
    graph_pressure = _graph_pressure(instance, operations)
    structural_pressure = _structural_pressure(instance)
    path_event_pressure = _event_type_pressure(instance, {"path_blockage", "route_branch", "congestion_spike"})
    repair_event_pressure = _event_type_pressure(instance, {"new_job", "machine_breakdown", "quality_rework"})
    agv_event_pressure = _event_type_pressure(instance, {"agv_failure", "congestion_spike"})
    constraint_pressure = _constraint_pressure(evaluated)
    span_pressure = _span_pressure(evaluated)
    path_delay = _mean_float(item.metrics.get("path_conflict_delay", 0.0) for item in evaluated)
    stability = _mean_float(item.metrics.get("stability", 0.0) for item in evaluated)
    edge_violations = _mean_float(item.metrics.get("edge_capacity_violations", 0.0) for item in evaluated)
    agv_violations = _mean_float(item.metrics.get("agv_overlap_violations", 0.0) for item in evaluated)
    event_violations = _mean_float(item.metrics.get("path_event_violations", 0.0) for item in evaluated)
    machine_unavailability = _mean_float(
        item.metrics.get("machine_unavailability_violations", 0.0)
        for item in evaluated
    )
    agv_unavailability = _mean_float(item.metrics.get("agv_unavailability_violations", 0.0) for item in evaluated)

    score_path = _clip(
        0.35 * graph_pressure
        + 0.35 * path_event_pressure
        + 0.15 * _pressure(edge_violations)
        + 0.10 * _pressure(path_delay)
        + 0.05 * constraint_pressure
    )
    score_event = _clip(
        0.35 * repair_event_pressure
        + 0.20 * structural_pressure
        + 0.20 * _pressure(event_violations)
        + 0.15 * _pressure(machine_unavailability)
        + 0.10 * constraint_pressure
    )
    score_agv = _clip(
        0.30 * graph_pressure
        + 0.35 * agv_event_pressure
        + 0.15 * _pressure(agv_violations)
        + 0.15 * _pressure(agv_unavailability)
        + 0.05 * constraint_pressure
    )
    sparse_cells = frozenset(_sparse_mrgv_grid_cells(evaluated))
    sparse_ratio = min(1.0, len(sparse_cells) / max(1, len(evaluated))) if evaluated else 1.0
    score_diversity = _clip(
        0.35 * span_pressure
        + 0.35 * sparse_ratio
        + 0.20 * constraint_pressure
        + 0.10 * (1.0 if generation >= 3 else 0.0)
    )

    target_direction = _mrgv_target_direction(instance.objectives, score_path, score_event, score_agv)
    diversity_gate_open = (
        generation >= 3
        and mrgv_attempts > 0
        and diversity_attempts / mrgv_attempts < config.mrgv_diversity_cap
    )
    return _MRGVDiagnosis(
        score_path=score_path,
        score_event=score_event,
        score_agv=score_agv,
        score_diversity=score_diversity,
        target_direction=target_direction,
        target_tail=_online_target_tail(target_direction, instance.objectives, score_path, score_event),
        sparse_grid_cells=sparse_cells,
        diversity_gate_open=diversity_gate_open,
    )


def _select_mrgv_operator(diagnosis: _MRGVDiagnosis, mode: str, config: MoeaConfig) -> _MRGVDecision:
    forced_operator = _normalize_mrgv_force_operator(config.mrgv_force_operator)
    if forced_operator != "none":
        return _MRGVDecision(forced_operator, "forced-operator")
    scores = {
        "path-pressure-shift": diagnosis.score_path,
        "event-repair-resequence": diagnosis.score_event,
        "agv-balance-reassignment": diagnosis.score_agv,
        "conservative-structural-diversity": diagnosis.score_diversity,
    }
    if max(scores.values()) < config.mrgv_tau_min:
        return _MRGVDecision("baseline", "below-tau-min")
    if diagnosis.score_diversity >= max(diagnosis.score_path, diagnosis.score_event, diagnosis.score_agv):
        if diagnosis.score_diversity >= config.mrgv_tau_diversity and diagnosis.diversity_gate_open:
            return _MRGVDecision("conservative-structural-diversity", "diversity-gate-open")
        return _select_mrgv_targeted_fallback(diagnosis, mode, config, scores)
    if mode == "realized" and diagnosis.score_event >= config.mrgv_tau_event:
        return _MRGVDecision("event-repair-resequence", "realized-event-pressure")
    if diagnosis.score_path >= config.mrgv_tau_path:
        return _MRGVDecision("path-pressure-shift", "path-pressure")
    if diagnosis.score_agv >= config.mrgv_tau_agv:
        return _MRGVDecision("agv-balance-reassignment", "agv-pressure")
    return _MRGVDecision("baseline", "no-operator-threshold-met")


def _select_mrgv_targeted_fallback(
    diagnosis: _MRGVDiagnosis,
    mode: str,
    config: MoeaConfig,
    scores: dict[str, float],
) -> _MRGVDecision:
    if mode == "realized" and diagnosis.score_event >= config.mrgv_tau_min:
        return _MRGVDecision("event-repair-resequence", "diversity-cap-closed-realized-event")
    if diagnosis.score_agv >= config.mrgv_tau_agv and diagnosis.score_agv >= diagnosis.score_path - 0.05:
        return _MRGVDecision("agv-balance-reassignment", "diversity-cap-closed-agv-pressure")
    if diagnosis.score_path >= config.mrgv_tau_min:
        return _MRGVDecision("path-pressure-shift", "diversity-cap-closed-path-pressure")
    best_targeted = max(
        ("path-pressure-shift", "event-repair-resequence", "agv-balance-reassignment"),
        key=lambda operator: scores[operator],
    )
    return _MRGVDecision(best_targeted, "diversity-cap-closed-use-best-targeted")


def _normalize_mrgv_force_operator(value: str) -> str:
    aliases = {
        "none": "none",
        "path": "path-pressure-shift",
        "event": "event-repair-resequence",
        "agv": "agv-balance-reassignment",
        "diversity": "conservative-structural-diversity",
        "path-pressure-shift": "path-pressure-shift",
        "event-repair-resequence": "event-repair-resequence",
        "agv-balance-reassignment": "agv-balance-reassignment",
        "conservative-structural-diversity": "conservative-structural-diversity",
    }
    if value not in aliases:
        raise ValueError(
            "mrgv_force_operator must be 'none', 'path', 'event', 'agv', 'diversity', "
            "or a concrete MRGV operator name"
        )
    return aliases[value]


def _mrgv_gate_lane(decision: _MRGVDecision, gate: str) -> str:
    if gate == "none":
        return "none"
    if gate == "indicator-safe":
        return "diversity-indicator-safe"
    if decision.operator == "conservative-structural-diversity":
        return "diversity-indicator-safe"
    return "repair-pareto-safe"


def _mrgv_candidate_policy(mode: str, admission_policy: str = "all") -> str:
    if mode == "dual-offspring":
        return f"dual-offspring-{admission_policy}"
    if mode == "replacement":
        return "replacement"
    return "one-selected"


def _mrgv_probe_state(config: MoeaConfig) -> _MRGVProbeState | None:
    if config.mrgv_probe_policy == "none":
        return None
    return _MRGVProbeState(policy=config.mrgv_probe_policy, pair_limit=4)


def _mrgv_probe_allows_attempt(state: _MRGVProbeState | None) -> bool:
    return state is None or state.decision != "disabled"


def _update_mrgv_probe_state(
    state: _MRGVProbeState | None,
    *,
    decision: _MRGVDecision,
    baseline_indicator: float,
    candidate_indicator: float,
) -> dict[str, object]:
    if state is None:
        return {}
    if state.policy != "b4-local-indicator":
        return {}
    if decision.operator != "path-pressure-shift":
        return {
            "mrgv_probe_policy": state.policy,
            "mrgv_probe_phase": state.decision,
            "mrgv_probe_pair_index": "",
            "mrgv_probe_positive": "",
            "mrgv_probe_positive_pairs": state.positive_pairs,
            "mrgv_probe_positive_rate": _safe_divide(state.positive_pairs, state.pair_limit),
            "mrgv_probe_decision": state.decision,
            "probe_admission_override": "",
        }
    if state.decision != "probing":
        return {
            "mrgv_probe_policy": state.policy,
            "mrgv_probe_phase": state.decision,
            "mrgv_probe_pair_index": "",
            "mrgv_probe_positive": "",
            "mrgv_probe_positive_pairs": state.positive_pairs,
            "mrgv_probe_positive_rate": _safe_divide(state.positive_pairs, state.pair_limit),
            "mrgv_probe_decision": state.decision,
            "probe_admission_override": "",
        }

    local_indicator_positive = (
        math.isfinite(baseline_indicator)
        and math.isfinite(candidate_indicator)
        and candidate_indicator <= baseline_indicator
    )
    state.path_pairs += 1
    if local_indicator_positive:
        state.positive_pairs += 1
    if state.path_pairs >= state.pair_limit:
        state.decision = "enabled" if state.positive_pairs / state.pair_limit >= 0.5 else "disabled"
    return {
        "mrgv_probe_policy": state.policy,
        "mrgv_probe_phase": "probing",
        "mrgv_probe_pair_index": state.path_pairs,
        "mrgv_probe_positive": str(local_indicator_positive),
        "mrgv_probe_positive_pairs": state.positive_pairs,
        "mrgv_probe_positive_rate": _safe_divide(state.positive_pairs, state.pair_limit),
        "mrgv_probe_decision": state.decision,
        "probe_admission_override": local_indicator_positive,
    }


def _mrgv_enters_pool(*, mode: str, admission_policy: str, accepted: bool) -> bool:
    if mode == "dual-offspring":
        return admission_policy == "all" or accepted
    return accepted


def _mrgv_selected_child_label(*, mode: str, entered_pool: bool) -> object | None:
    if mode != "dual-offspring":
        return None
    return "both" if entered_pool else "baseline-only"


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mutate_mrgv(
    genome: _Genome,
    rng: random.Random,
    operator: str,
    diagnosis: _MRGVDiagnosis,
    operation_count: int,
    machine_count: int,
    operations: tuple[Operation, ...],
    machines: tuple[Machine, ...],
    jobs: tuple[Job, ...],
) -> _Genome:
    if operator == "baseline":
        return genome
    genes = list(genome.genes)
    if not genes:
        return genome

    target_state = _OnlineGraphActionState(
        action="graph-repair",
        target_direction=diagnosis.target_direction,
        target_tail=diagnosis.target_tail,
        graph_pressure=diagnosis.score_path,
        structural_pressure=diagnosis.score_event,
        front_ratio=0.0,
        constraint_pressure=max(diagnosis.score_path, diagnosis.score_event, diagnosis.score_agv),
        span_pressure=diagnosis.score_diversity,
    )
    machine_state = _OnlineGraphActionState(
        action="reference-gap",
        target_direction=diagnosis.target_direction,
        target_tail=diagnosis.target_tail,
        graph_pressure=diagnosis.score_agv,
        structural_pressure=diagnosis.score_event,
        front_ratio=0.0,
        constraint_pressure=max(diagnosis.score_path, diagnosis.score_event, diagnosis.score_agv),
        span_pressure=diagnosis.score_diversity,
    )
    op_targets = _online_operation_targets(target_state, operations, jobs, machine_count)
    machine_targets = _online_machine_targets(machine_state, machines)

    if operator == "path-pressure-shift":
        operation_rate = 0.20 + 0.35 * diagnosis.score_path
        machine_rate = 0.16 + 0.25 * diagnosis.score_path
        operation_blend = 0.14 + 0.16 * diagnosis.score_path
        machine_blend = 0.12 + 0.14 * diagnosis.score_path
        tail = _safe_simplex((0.58, 0.18, 0.24))
    elif operator == "event-repair-resequence":
        operation_rate = 0.18 + 0.34 * diagnosis.score_event
        machine_rate = 0.14 + 0.24 * diagnosis.score_event
        operation_blend = 0.16 + 0.18 * diagnosis.score_event
        machine_blend = 0.10 + 0.12 * diagnosis.score_event
        tail = _safe_simplex((0.20, 0.28, 0.52))
    elif operator == "agv-balance-reassignment":
        operation_rate = 0.08 + 0.16 * diagnosis.score_agv
        machine_rate = 0.24 + 0.36 * diagnosis.score_agv
        operation_blend = 0.08 + 0.08 * diagnosis.score_agv
        machine_blend = 0.18 + 0.18 * diagnosis.score_agv
        tail = _safe_simplex((0.42, 0.42, 0.16))
    elif operator == "conservative-structural-diversity":
        operation_rate = 0.10
        machine_rate = 0.10
        operation_blend = 0.08
        machine_blend = 0.08
        tail = diagnosis.target_tail
    else:
        return genome

    for index in range(min(operation_count, len(genes))):
        if op_targets and rng.random() < operation_rate:
            genes[index] = _clip(
                (1.0 - operation_blend) * genes[index]
                + operation_blend * op_targets[index]
                + rng.gauss(0.0, 0.025)
            )

    machine_offset = operation_count
    for index in range(machine_count):
        gene_index = machine_offset + index
        if gene_index >= len(genes):
            break
        if machine_targets and rng.random() < machine_rate:
            genes[gene_index] = _clip(
                (1.0 - machine_blend) * genes[gene_index]
                + machine_blend * machine_targets[index]
                + rng.gauss(0.0, 0.022)
            )

    tail_offset = operation_count + machine_count
    if tail_offset + 2 < len(genes):
        for index, target in enumerate(tail):
            gene_index = tail_offset + index
            genes[gene_index] = _clip(0.72 * genes[gene_index] + 0.28 * target + rng.gauss(0.0, 0.016))

    if operator == "conservative-structural-diversity":
        span = min(max(1, operation_count + machine_count), len(genes))
        for _ in range(max(1, span // 14)):
            index = rng.randrange(span)
            genes[index] = _clip(genes[index] + rng.gauss(0.0, 0.12))

    return _Genome(tuple(genes))


def _passes_mrgv_gate(
    baseline: _Evaluated,
    candidate: _Evaluated,
    current_population: list[_Evaluated],
    diagnosis: _MRGVDiagnosis,
) -> tuple[bool, str, float, float, str, str]:
    baseline_violation = _mrgv_violation_vector(baseline)
    candidate_violation = _mrgv_violation_vector(candidate)
    baseline_cell = _mrgv_grid_cell(baseline, current_population)
    candidate_cell = _mrgv_grid_cell(candidate, current_population)
    if _lex_strict_less(candidate_violation, baseline_violation):
        return True, "violation-vector-improved", math.nan, math.nan, baseline_cell, candidate_cell
    if _lex_strict_less(baseline_violation, candidate_violation):
        return False, "violation-vector-worse", math.nan, math.nan, baseline_cell, candidate_cell
    if _dominates(candidate, baseline):
        return True, "dominates-baseline", math.nan, math.nan, baseline_cell, candidate_cell
    if _dominates(baseline, candidate):
        return False, "dominated-by-baseline", math.nan, math.nan, baseline_cell, candidate_cell

    weights = diagnosis.target_direction or tuple(1.0 / len(candidate.objective_vector) for _ in candidate.objective_vector)
    front = list(current_population) + [baseline, candidate]
    ideal, nadir = _ideal_nadir(front)
    baseline_score = _tchebycheff(baseline.objective_vector, weights, ideal, nadir)
    candidate_score = _tchebycheff(candidate.objective_vector, weights, ideal, nadir)
    if candidate_score <= baseline_score:
        return True, "indicator-improved", baseline_score, candidate_score, baseline_cell, candidate_cell
    if _lex_less_equal(candidate_violation, baseline_violation) and candidate_cell in diagnosis.sparse_grid_cells:
        return True, "sparse-grid-cell", baseline_score, candidate_score, baseline_cell, candidate_cell
    return False, "not-pareto-safe", baseline_score, candidate_score, baseline_cell, candidate_cell


def _evaluate_mrgv_gate(
    baseline: _Evaluated,
    candidate: _Evaluated,
    current_population: list[_Evaluated],
    diagnosis: _MRGVDiagnosis,
    gate: str,
) -> tuple[bool, str, float, float, str, str]:
    if gate == "none":
        baseline_cell = _mrgv_grid_cell(baseline, current_population)
        candidate_cell = _mrgv_grid_cell(candidate, current_population)
        return True, "gate-disabled", math.nan, math.nan, baseline_cell, candidate_cell
    if gate == "indicator-safe":
        return _passes_mrgv_indicator_gate(
            baseline=baseline,
            candidate=candidate,
            current_population=current_population,
            diagnosis=diagnosis,
        )
    return _passes_mrgv_gate(
        baseline=baseline,
        candidate=candidate,
        current_population=current_population,
        diagnosis=diagnosis,
    )


def _passes_mrgv_indicator_gate(
    baseline: _Evaluated,
    candidate: _Evaluated,
    current_population: list[_Evaluated],
    diagnosis: _MRGVDiagnosis,
) -> tuple[bool, str, float, float, str, str]:
    baseline_cell = _mrgv_grid_cell(baseline, current_population)
    candidate_cell = _mrgv_grid_cell(candidate, current_population)
    weights = diagnosis.target_direction or tuple(1.0 / len(candidate.objective_vector) for _ in candidate.objective_vector)
    front = list(current_population) + [baseline, candidate]
    ideal, nadir = _ideal_nadir(front)
    baseline_score = _tchebycheff(baseline.objective_vector, weights, ideal, nadir)
    candidate_score = _tchebycheff(candidate.objective_vector, weights, ideal, nadir)
    if candidate_score <= baseline_score:
        return True, "indicator-improved", baseline_score, candidate_score, baseline_cell, candidate_cell
    if candidate_cell in diagnosis.sparse_grid_cells:
        return True, "sparse-grid-cell", baseline_score, candidate_score, baseline_cell, candidate_cell
    return False, "indicator-not-improved", baseline_score, candidate_score, baseline_cell, candidate_cell


def _append_mrgv_log(
    mrgv_log: list[dict[str, object]] | None,
    *,
    context_id: str,
    generation: int,
    eval_count_before: int,
    mode: str,
    mrgv_lambda: float,
    decision: _MRGVDecision,
    diagnosis: _MRGVDiagnosis,
    gate: str,
    gate_lane: str,
    candidate_policy: str,
    admission_policy: str,
    accepted: bool,
    gate_reason: str,
    baseline: _Evaluated | None,
    candidate: _Evaluated | None,
    selected: _Evaluated | None,
    baseline_indicator: float | None,
    candidate_indicator: float | None,
    baseline_grid_cell: str,
    candidate_grid_cell: str,
    selected_child: object | None = None,
    generated: bool = False,
    non_noop: bool = False,
    evaluated: bool = False,
    entered_pool: bool = False,
    probe_info: dict[str, object] | None = None,
) -> None:
    if mrgv_log is None:
        return
    resolved_probe_info = probe_info or {}
    mrgv_log.append(
        {
            "context_id": context_id,
            "level": "",
            "mode": "",
            "seed": "",
            "algorithm": "" if selected is None else selected.schedule.algorithm,
            "generation": generation,
            "eval_count_before": eval_count_before,
            "mrgv_mode": mode,
            "mrgv_lambda": round(mrgv_lambda, 6),
            "mrgv_operator": decision.operator,
            "mrgv_operator_reason": decision.reason,
            "mrgv_score_path": round(diagnosis.score_path, 10),
            "mrgv_score_event": round(diagnosis.score_event, 10),
            "mrgv_score_agv": round(diagnosis.score_agv, 10),
            "mrgv_score_diversity": round(diagnosis.score_diversity, 10),
            "mrgv_gate": gate,
            "mrgv_gate_lane": gate_lane,
            "mrgv_candidate_policy": candidate_policy,
            "mrgv_admission_policy": admission_policy,
            "mrgv_probe_policy": resolved_probe_info.get("mrgv_probe_policy", ""),
            "mrgv_probe_phase": resolved_probe_info.get("mrgv_probe_phase", ""),
            "mrgv_probe_pair_index": resolved_probe_info.get("mrgv_probe_pair_index", ""),
            "mrgv_probe_positive": resolved_probe_info.get("mrgv_probe_positive", ""),
            "mrgv_probe_positive_pairs": resolved_probe_info.get("mrgv_probe_positive_pairs", ""),
            "mrgv_probe_positive_rate": resolved_probe_info.get("mrgv_probe_positive_rate", ""),
            "mrgv_probe_decision": resolved_probe_info.get("mrgv_probe_decision", ""),
            "mrgv_probe_admission_override": resolved_probe_info.get("probe_admission_override", ""),
            "mrgv_gate_accepted": str(accepted),
            "mrgv_gate_reason": gate_reason,
            "baseline_child_id": "" if baseline is None else baseline.uid,
            "mrgv_child_id": "" if candidate is None else candidate.uid,
            "selected_child": selected.uid if selected_child is None and selected is not None else selected_child,
            "baseline_violation_vector": "" if baseline is None else _format_vector(_mrgv_violation_vector(baseline)),
            "mrgv_violation_vector": "" if candidate is None else _format_vector(_mrgv_violation_vector(candidate)),
            "baseline_objective_vector": "" if baseline is None else _format_vector(baseline.objective_vector),
            "mrgv_objective_vector": "" if candidate is None else _format_vector(candidate.objective_vector),
            "baseline_indicator": _format_optional_float(baseline_indicator),
            "mrgv_indicator": _format_optional_float(candidate_indicator),
            "mrgv_indicator_delta": _format_indicator_delta(baseline_indicator, candidate_indicator),
            "mrgv_positive_indicator_contribution": str(_positive_indicator_contribution(gate_reason)),
            "baseline_grid_cell": baseline_grid_cell,
            "mrgv_grid_cell": candidate_grid_cell,
            "mrgv_grid_cell_changed": str(_mrgv_grid_cell_changed(baseline_grid_cell, candidate_grid_cell)),
            "mrgv_sparse_grid_cell_hit": str(candidate_grid_cell in diagnosis.sparse_grid_cells),
            "mrgv_dominates_baseline": str(
                False if baseline is None or candidate is None else _dominates(candidate, baseline)
            ),
            "baseline_dominates_mrgv": str(
                False if baseline is None or candidate is None else _dominates(baseline, candidate)
            ),
            "mrgv_epsilon_dominates_baseline": str(
                False if baseline is None or candidate is None else _epsilon_dominates(candidate, baseline)
            ),
            "mrgv_objective_delta": "" if baseline is None or candidate is None else _format_objective_delta(baseline, candidate),
            "mrgv_generated": str(generated),
            "mrgv_non_noop": str(non_noop),
            "mrgv_evaluated": str(evaluated),
            "mrgv_entered_pool": str(entered_pool),
            "mrgv_admitted_to_pool": str(entered_pool),
            "mrgv_survived_selection": "False",
            "mrgv_entered_archive": "False",
        }
    )


def _append_mrgv_replacement_log(
    mrgv_log: list[dict[str, object]] | None,
    *,
    context_id: str,
    generation: int,
    eval_count_before: int,
    mrgv_lambda: float,
    decision: _MRGVDecision,
    diagnosis: _MRGVDiagnosis,
    selected_child_id: int,
    accepted: bool,
) -> None:
    if mrgv_log is None:
        return
    mrgv_log.append(
        {
            "context_id": context_id,
            "level": "",
            "mode": "",
            "seed": "",
            "algorithm": "",
            "generation": generation,
            "eval_count_before": eval_count_before,
            "mrgv_mode": "replacement",
            "mrgv_lambda": round(mrgv_lambda, 6),
            "mrgv_operator": decision.operator,
            "mrgv_operator_reason": decision.reason,
            "mrgv_score_path": round(diagnosis.score_path, 10),
            "mrgv_score_event": round(diagnosis.score_event, 10),
            "mrgv_score_agv": round(diagnosis.score_agv, 10),
            "mrgv_score_diversity": round(diagnosis.score_diversity, 10),
            "mrgv_gate": "replacement",
            "mrgv_gate_lane": "replacement",
            "mrgv_candidate_policy": "replacement",
            "mrgv_admission_policy": "replacement",
            "mrgv_gate_accepted": str(accepted),
            "mrgv_gate_reason": "single-child-replacement",
            "baseline_child_id": "",
            "mrgv_child_id": selected_child_id,
            "selected_child": selected_child_id,
            "baseline_violation_vector": "",
            "mrgv_violation_vector": "",
            "baseline_objective_vector": "",
            "mrgv_objective_vector": "",
            "baseline_indicator": "",
            "mrgv_indicator": "",
            "mrgv_indicator_delta": "",
            "mrgv_positive_indicator_contribution": "False",
            "baseline_grid_cell": "",
            "mrgv_grid_cell": "",
            "mrgv_grid_cell_changed": "False",
            "mrgv_sparse_grid_cell_hit": "False",
            "mrgv_dominates_baseline": "False",
            "baseline_dominates_mrgv": "False",
            "mrgv_epsilon_dominates_baseline": "False",
            "mrgv_objective_delta": "",
            "mrgv_generated": str(decision.operator != "baseline"),
            "mrgv_non_noop": str(accepted),
            "mrgv_evaluated": str(accepted),
            "mrgv_entered_pool": str(accepted),
            "mrgv_admitted_to_pool": str(accepted),
            "mrgv_survived_selection": "False",
            "mrgv_entered_archive": "False",
        }
    )


def _annotate_mrgv_survival(
    mrgv_log: list[dict[str, object]] | None,
    generation: int,
    survivor_ids: set[int],
) -> None:
    if mrgv_log is None:
        return
    for entry in mrgv_log:
        if int(entry.get("generation", -1)) != generation:
            continue
        child_id = _optional_int(entry.get("mrgv_child_id", ""))
        if child_id is not None:
            entry["mrgv_survived_selection"] = str(child_id in survivor_ids)


def _annotate_mrgv_archive(
    mrgv_log: list[dict[str, object]] | None,
    archive_ids: set[int],
) -> None:
    if mrgv_log is None:
        return
    for entry in mrgv_log:
        child_id = _optional_int(entry.get("mrgv_child_id", ""))
        if child_id is not None:
            entry["mrgv_entered_archive"] = str(child_id in archive_ids)


def _optional_int(value: object) -> int | None:
    if value in {"", None}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _passes_online_action_gate(
    baseline: _Evaluated,
    candidate: _Evaluated,
    current_population: list[_Evaluated],
    target_direction: tuple[float, ...],
    tolerance: float,
) -> tuple[bool, str, float, float]:
    if candidate.constraint_violation < baseline.constraint_violation:
        return True, "constraint-improved", math.nan, math.nan
    if candidate.constraint_violation > baseline.constraint_violation:
        return False, "constraint-worse", math.nan, math.nan
    if _dominates(candidate, baseline):
        return True, "dominates-baseline", math.nan, math.nan
    if _dominates(baseline, candidate):
        return False, "dominated-by-baseline", math.nan, math.nan

    raw_weights = tuple(target_direction[: len(candidate.objective_vector)])
    weights = (
        _safe_simplex(raw_weights)
        if raw_weights
        else tuple(1.0 / len(candidate.objective_vector) for _ in candidate.objective_vector)
    )
    front = list(current_population) + [baseline, candidate]
    ideal, nadir = _ideal_nadir(front)
    baseline_score = _tchebycheff(baseline.objective_vector, weights, ideal, nadir)
    candidate_score = _tchebycheff(candidate.objective_vector, weights, ideal, nadir)
    if candidate_score <= baseline_score - tolerance:
        return True, "indicator-improved", baseline_score, candidate_score
    return False, "indicator-not-improved", baseline_score, candidate_score


def _annotate_online_action_gate(
    action_log: list[dict[str, object]] | None,
    gate: str,
    accepted: bool,
    reason: str,
    baseline_constraint: float | None = None,
    candidate_constraint: float | None = None,
    baseline_indicator: float | None = None,
    candidate_indicator: float | None = None,
) -> None:
    if action_log is None or not action_log:
        return
    entry = action_log[-1]
    entry["online_action_gate"] = gate
    entry["online_action_gate_accepted"] = str(accepted)
    entry["online_action_gate_reason"] = reason
    entry["online_action_gate_baseline_constraint"] = _format_optional_float(baseline_constraint)
    entry["online_action_gate_candidate_constraint"] = _format_optional_float(candidate_constraint)
    entry["online_action_gate_baseline_indicator"] = _format_optional_float(baseline_indicator)
    entry["online_action_gate_candidate_indicator"] = _format_optional_float(candidate_indicator)


def _format_optional_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.10f}"


def _format_indicator_delta(baseline_indicator: float | None, candidate_indicator: float | None) -> str:
    if (
        baseline_indicator is None
        or candidate_indicator is None
        or math.isnan(baseline_indicator)
        or math.isnan(candidate_indicator)
    ):
        return ""
    return f"{baseline_indicator - candidate_indicator:.10f}"


def _positive_indicator_contribution(gate_reason: str) -> bool:
    return gate_reason in {
        "indicator-improved",
        "sparse-grid-cell",
        "dominates-baseline",
        "violation-vector-improved",
    }


def _mrgv_grid_cell_changed(baseline_grid_cell: str, candidate_grid_cell: str) -> bool:
    return bool(baseline_grid_cell and candidate_grid_cell and baseline_grid_cell != candidate_grid_cell)


def _epsilon_dominates(left: _Evaluated, right: _Evaluated, epsilon: float = 0.01) -> bool:
    if left.constraint_violation > right.constraint_violation:
        return False
    no_worse = all(
        left_value <= right_value + epsilon * max(1.0, abs(right_value))
        for left_value, right_value in zip(left.objective_vector, right.objective_vector)
    )
    better = left.constraint_violation < right.constraint_violation or any(
        left_value < right_value - epsilon * max(1.0, abs(right_value))
        for left_value, right_value in zip(left.objective_vector, right.objective_vector)
    )
    return no_worse and better


def _format_objective_delta(baseline: _Evaluated, candidate: _Evaluated) -> str:
    return _format_vector(
        tuple(
            baseline_value - candidate_value
            for baseline_value, candidate_value in zip(baseline.objective_vector, candidate.objective_vector)
        )
    )


def _format_vector(values: tuple[float, ...]) -> str:
    return ";".join(f"{value:.10f}" for value in values)


def _mean_float(values: object) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _pressure(value: float) -> float:
    return _clip(value / (1.0 + abs(value)))


def _event_type_pressure(instance: Instance, event_types: set[str]) -> float:
    if not instance.events:
        return 0.0
    events = [event for event in instance.events if event.event_type in event_types]
    if not events:
        return 0.0
    event_share = len(events) / len(instance.events)
    severity = sum(event.severity for event in events) / len(events)
    return _clip(0.45 * event_share + 0.55 * severity)


def _instance_mode(instance: Instance) -> str:
    return "realized" if "realized" in instance.instance_id else "raw"


def _mrgv_target_direction(
    objectives: tuple[str, ...],
    score_path: float,
    score_event: float,
    score_agv: float,
) -> tuple[float, ...]:
    weights = []
    for objective in objectives:
        if objective in {"makespan", "path_conflict_delay"}:
            weights.append(1.0 + score_path + 0.5 * score_agv)
        elif objective in {"tardiness", "stability"}:
            weights.append(1.0 + score_event)
        elif objective == "energy":
            weights.append(1.0 + 0.6 * score_agv)
        else:
            weights.append(1.0)
    return _safe_simplex(tuple(weights))


def _mrgv_violation_vector(evaluated: _Evaluated) -> tuple[float, ...]:
    dynamic_event = float(evaluated.metrics.get("path_event_violations", 0.0))
    edge_capacity = float(evaluated.metrics.get("edge_capacity_violations", 0.0))
    machine_overlap = float(evaluated.metrics.get("machine_overlap_violations", 0.0))
    agv_overlap = float(evaluated.metrics.get("agv_overlap_violations", 0.0))
    path_stability = float(evaluated.metrics.get("path_conflict_delay", 0.0)) + float(evaluated.metrics.get("stability", 0.0))
    infeasible = 1.0 if evaluated.constraint_violation > 0.0 else 0.0
    return (infeasible, dynamic_event, edge_capacity, machine_overlap, agv_overlap, path_stability)


def _lex_less_equal(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return all(left_value <= right_value for left_value, right_value in zip(left, right))


def _lex_strict_less(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return _lex_less_equal(left, right) and any(
        left_value < right_value
        for left_value, right_value in zip(left, right)
    )


def _mrgv_grid_cell(evaluated: _Evaluated, current_population: list[_Evaluated]) -> str:
    front = list(current_population) + [evaluated]
    ideal, nadir = _ideal_nadir(front)
    normalized = _normalize_vector(evaluated.objective_vector, ideal, nadir)
    return ";".join(str(min(4, int(value * 5.0))) for value in normalized)


def _sparse_mrgv_grid_cells(evaluated: list[_Evaluated]) -> tuple[str, ...]:
    if not evaluated:
        return ()
    ideal, nadir = _ideal_nadir(evaluated)
    counts: dict[str, int] = {}
    for item in evaluated:
        normalized = _normalize_vector(item.objective_vector, ideal, nadir)
        cell = ";".join(str(min(4, int(value * 5.0))) for value in normalized)
        counts[cell] = counts.get(cell, 0) + 1
    return tuple(cell for cell, count in counts.items() if count <= 1)


def _same_genome(left: _Genome, right: _Genome) -> bool:
    return len(left.genes) == len(right.genes) and all(
        abs(left_value - right_value) <= 1e-12
        for left_value, right_value in zip(left.genes, right.genes)
    )


def _blend_rate(base_rate: float, decision_rate: float, strength: float, floor_bonus: float) -> float:
    target_rate = base_rate + strength * max(0.0, decision_rate - base_rate)
    return min(0.92, max(base_rate + floor_bonus * strength, target_rate))


def _select_learned_online_state(
    genome: _Genome,
    rng: random.Random,
    states: tuple[_OnlineGraphActionState, ...],
    operation_count: int,
    machine_count: int,
    exploration_rate: float = 0.30,
) -> _OnlineGraphActionState:
    if not states:
        raise ValueError("at least one learned online state is required")
    if len(states) == 1:
        return states[0]
    if exploration_rate > 0.0 and rng.random() < exploration_rate:
        return rng.choice(states)
    tail = _safe_simplex(_tail_weights(genome, operation_count, machine_count))
    return min(
        states,
        key=lambda state: (
            sum(abs(left - right) for left, right in zip(tail, state.target_tail)),
            -_direction_tail_spread(state.target_direction),
            state.target_direction,
        ),
    )


def _direction_tail_spread(direction: tuple[float, ...]) -> float:
    if not direction:
        return 0.0
    return max(direction) - min(direction)


def _learned_online_operation_targets(
    action_template: str,
    operations: tuple[Operation, ...],
    jobs: tuple[Job, ...],
    machine_count: int,
) -> tuple[float, ...]:
    if not operations:
        return ()
    job_by_id = {job.job_id: job for job in jobs}
    max_processing = max((min(operation.processing_time.values()) for operation in operations), default=1)
    max_energy = max((min(operation.processing_energy.values()) for operation in operations), default=1.0)
    max_due = max((job.due_date for job in jobs), default=1)
    targets = []
    for index, operation in enumerate(operations):
        job = job_by_id[operation.job_id]
        processing = min(operation.processing_time.values()) / max_processing
        energy = min(operation.processing_energy.values()) / max_energy
        due = job.due_date / max_due
        index_score = index / max(1, len(operations) - 1)
        eligibility_pressure = 1.0 - (len(operation.eligible_machines) - 1) / max(1, machine_count - 1)
        if action_template == "energy-machine":
            target = 0.58 * energy + 0.20 * processing + 0.12 * due + 0.10 * index_score
        elif action_template == "due-recovery":
            target = 0.54 * due + 0.24 * processing + 0.12 * eligibility_pressure + 0.10 * index_score
        elif action_template == "balanced-diversity":
            target = 0.30 * index_score + 0.25 * processing + 0.25 * due + 0.20 * energy
        else:
            target = 0.46 * processing + 0.22 * eligibility_pressure + 0.18 * due + 0.14 * index_score
        targets.append(_clip(target))
    return tuple(targets)


def _learned_online_machine_targets(
    action_template: str,
    machines: tuple[Machine, ...],
) -> tuple[float, ...]:
    if not machines:
        return ()
    max_energy = max(machine.energy_rate for machine in machines)
    targets = []
    for index, machine in enumerate(machines):
        energy = machine.energy_rate / max(1.0, max_energy)
        index_score = index / max(1, len(machines) - 1)
        if action_template == "energy-machine":
            target = 0.78 * energy + 0.22 * index_score
        elif action_template == "speed-path":
            target = 0.35 * energy + 0.65 * index_score
        elif action_template == "due-recovery":
            target = 0.45 * energy + 0.55 * index_score
        else:
            target = 0.50 * energy + 0.50 * (1.0 - index_score)
        targets.append(_clip(target))
    return tuple(targets)


def _least_covered_reference_direction(
    evaluated: list[_Evaluated],
    instance: Instance,
    reference_directions: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
    directions = _least_covered_reference_directions(evaluated, instance, reference_directions, limit=1)
    if directions:
        return directions[0]
    n_objectives = len(instance.objectives)
    return tuple(1.0 / n_objectives for _ in range(n_objectives))


def _least_covered_reference_directions(
    evaluated: list[_Evaluated],
    instance: Instance,
    reference_directions: tuple[tuple[float, ...], ...],
    limit: int,
) -> tuple[tuple[float, ...], ...]:
    n_objectives = len(instance.objectives)
    directions = tuple(
        _safe_simplex(tuple(direction[:n_objectives]))
        for direction in reference_directions
        if len(direction) >= n_objectives
    )
    if not directions:
        return (tuple(1.0 / n_objectives for _ in range(n_objectives)),)
    if not evaluated:
        return tuple(directions[:limit])

    front = _nondominated_front(evaluated) or evaluated
    ideal, nadir = _ideal_nadir(front)
    counts = {index: 0 for index in range(len(directions))}
    for item in front:
        niche, _ = _associate_reference(item.objective_vector, directions, ideal, nadir)
        counts[niche] += 1
    ranked = sorted(
        enumerate(directions),
        key=lambda item: (
            counts[item[0]],
            -_direction_dynamic_priority(item[1], instance.objectives),
            item[0],
        ),
    )
    return tuple(direction for _index, direction in ranked[:limit])


def _objective_anchor_directions(objectives: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    anchors = []
    for index in range(len(objectives)):
        direction = [0.0] * len(objectives)
        direction[index] = 1.0
        anchors.append(tuple(direction))
    return tuple(anchors)


def _dynamic_priority_reference_directions(
    instance: Instance,
    reference_directions: tuple[tuple[float, ...], ...],
    limit: int,
) -> tuple[tuple[float, ...], ...]:
    n_objectives = len(instance.objectives)
    directions = tuple(
        _safe_simplex(tuple(direction[:n_objectives]))
        for direction in reference_directions
        if len(direction) >= n_objectives
    )
    ranked = sorted(
        directions,
        key=lambda direction: (-_direction_dynamic_priority(direction, instance.objectives), direction),
    )
    return tuple(ranked[:limit])


def _unique_reference_directions(
    directions: list[tuple[float, ...]],
    n_objectives: int,
    limit: int,
) -> tuple[tuple[float, ...], ...]:
    unique = []
    seen = set()
    for direction in directions:
        if len(direction) < n_objectives:
            continue
        normalized = _safe_simplex(tuple(direction[:n_objectives]))
        key = tuple(round(value, 6) for value in normalized)
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
        if len(unique) >= limit:
            break
    if not unique:
        unique.append(tuple(1.0 / n_objectives for _ in range(n_objectives)))
    return tuple(unique)


def _graph_pressure(instance: Instance, operations: tuple[Operation, ...]) -> float:
    if not instance.path_edges:
        return 0.0
    edge_count = len(instance.path_edges)
    node_count = max(1, len(instance.path_nodes))
    edge_density = min(1.0, edge_count / max(1, node_count - 1))
    bottleneck_share = sum(1 for edge in instance.path_edges if edge.is_bottleneck) / edge_count
    blockage_pressure = sum(edge.blockage_probability for edge in instance.path_edges) / edge_count
    agv_contention = min(1.0, len(operations) / max(1, len(instance.agvs) * 20))
    return _clip(
        0.20 * edge_density
        + 0.25 * bottleneck_share
        + 0.20 * blockage_pressure
        + 0.35 * agv_contention
    )


def _structural_pressure(instance: Instance) -> float:
    if not instance.events:
        return 0.0
    structural_types = {
        "route_branch",
        "quality_rework",
        "path_blockage",
        "agv_failure",
        "congestion_spike",
        "machine_breakdown",
    }
    structural = [event for event in instance.events if event.event_type in structural_types]
    if not structural:
        return 0.0
    event_share = len(structural) / len(instance.events)
    severity = sum(event.severity for event in structural) / len(structural)
    return _clip(0.45 * event_share + 0.55 * severity)


def _constraint_pressure(evaluated: list[_Evaluated]) -> float:
    if not evaluated:
        return 0.0
    return _clip(sum(min(1.0, item.constraint_violation) for item in evaluated) / len(evaluated))


def _span_pressure(evaluated: list[_Evaluated]) -> float:
    if len(evaluated) < 2:
        return 1.0
    n_objectives = len(evaluated[0].objective_vector)
    relative_spans = []
    for index in range(n_objectives):
        values = [item.objective_vector[index] for item in evaluated]
        high = max(values)
        low = min(values)
        scale = max(1.0, max(abs(value) for value in values))
        relative_spans.append((high - low) / scale)
    return _clip(1.0 - sum(relative_spans) / len(relative_spans))


def _online_target_tail(
    target_direction: tuple[float, ...],
    objectives: tuple[str, ...],
    graph_pressure: float,
    structural_pressure: float,
) -> tuple[float, float, float]:
    base_tail = _tail_from_seed_preference(target_direction, objectives)
    graph_tail = _safe_simplex(
        (
            0.50 + 0.35 * graph_pressure,
            0.08,
            0.30 + 0.35 * structural_pressure,
        )
    )
    blend = min(0.65, 0.20 + 0.35 * graph_pressure + 0.25 * structural_pressure)
    return _safe_simplex(
        tuple(
            (1.0 - blend) * base_value + blend * graph_value
            for base_value, graph_value in zip(base_tail, graph_tail)
        )
    )


def _online_operation_targets(
    state: _OnlineGraphActionState,
    operations: tuple[Operation, ...],
    jobs: tuple[Job, ...],
    machine_count: int,
) -> tuple[float, ...]:
    if not operations:
        return ()
    job_by_id = {job.job_id: job for job in jobs}
    max_processing = max((min(operation.processing_time.values()) for operation in operations), default=1)
    max_energy = max((min(operation.processing_energy.values()) for operation in operations), default=1.0)
    max_due = max((job.due_date for job in jobs), default=1)
    targets = []
    for index, operation in enumerate(operations):
        job = job_by_id[operation.job_id]
        processing = min(operation.processing_time.values()) / max_processing
        energy = min(operation.processing_energy.values()) / max_energy
        due = job.due_date / max_due
        index_score = index / max(1, len(operations) - 1)
        eligibility_pressure = 1.0 - (len(operation.eligible_machines) - 1) / max(1, machine_count - 1)
        constrained = 1.0 - eligibility_pressure
        if state.action == "graph-repair":
            target = 0.30 * processing + 0.25 * due + 0.15 * energy + 0.20 * constrained + 0.10 * index_score
        elif state.target_tail[1] >= max(state.target_tail[0], state.target_tail[2]):
            target = 0.60 * energy + 0.25 * processing + 0.15 * index_score
        elif state.target_tail[2] >= state.target_tail[0]:
            target = 0.55 * due + 0.30 * processing + 0.15 * index_score
        else:
            target = 0.55 * processing + 0.25 * due + 0.20 * index_score
        targets.append(_clip(target))
    return tuple(targets)


def _online_machine_targets(
    state: _OnlineGraphActionState,
    machines: tuple[Machine, ...],
) -> tuple[float, ...]:
    if not machines:
        return ()
    max_energy = max(machine.energy_rate for machine in machines)
    targets = []
    for index, machine in enumerate(machines):
        energy = machine.energy_rate / max(1.0, max_energy)
        index_score = index / max(1, len(machines) - 1)
        if state.target_tail[1] >= max(state.target_tail[0], state.target_tail[2]):
            target = 0.75 * energy + 0.25 * index_score
        elif state.action == "graph-repair":
            target = 0.55 * energy + 0.45 * index_score
        else:
            target = 0.40 * energy + 0.60 * index_score
        targets.append(_clip(target))
    return tuple(targets)


def _direction_dynamic_priority(direction: tuple[float, ...], objectives: tuple[str, ...]) -> float:
    by_objective = {
        objective: direction[index]
        for index, objective in enumerate(objectives[: len(direction)])
    }
    return (
        0.40 * by_objective.get("path_conflict_delay", 0.0)
        + 0.35 * by_objective.get("stability", 0.0)
        + 0.15 * by_objective.get("makespan", 0.0)
        + 0.10 * by_objective.get("tardiness", 0.0)
    )


def _select_next(
    candidates: list[_Evaluated],
    population_size: int,
    algorithm: str,
    reference_directions: tuple[tuple[float, ...], ...],
) -> list[_Evaluated]:
    if algorithm == "spea2":
        return _select_spea2(candidates, population_size)
    if algorithm == "nsga3":
        return _select_nsga3(candidates, population_size, reference_directions)
    if algorithm == "moead":
        return _select_moead(candidates, population_size, reference_directions)
    return _select_nsga2(candidates, population_size)


def _select_archive(
    candidates: list[_Evaluated],
    archive_size: int,
    algorithm: str,
    reference_directions: tuple[tuple[float, ...], ...],
) -> list[_Evaluated]:
    first_front = _nondominated_front(candidates)
    if len(first_front) <= archive_size:
        return sorted(first_front, key=lambda item: (item.constraint_violation, item.objective_vector, item.uid))
    return _select_next(first_front, archive_size, algorithm, reference_directions)


def _select_nsga2(candidates: list[_Evaluated], population_size: int) -> list[_Evaluated]:
    selected: list[_Evaluated] = []
    for front in _non_dominated_sort(candidates):
        if len(selected) + len(front) <= population_size:
            selected.extend(front)
            continue
        crowding = _crowding_distance(front)
        selected.extend(sorted(front, key=lambda item: (-crowding[item.uid], item.uid))[: population_size - len(selected)])
        break
    return selected


def _select_nsga3(
    candidates: list[_Evaluated],
    population_size: int,
    reference_directions: tuple[tuple[float, ...], ...],
) -> list[_Evaluated]:
    selected: list[_Evaluated] = []
    for front in _non_dominated_sort(candidates):
        if len(selected) + len(front) <= population_size:
            selected.extend(front)
            continue
        selected.extend(_niching_select(front, selected, population_size - len(selected), reference_directions))
        break
    return selected


def _select_spea2(candidates: list[_Evaluated], population_size: int) -> list[_Evaluated]:
    fitness = _spea2_fitness(candidates)
    return sorted(candidates, key=lambda item: (fitness[item.uid], item.constraint_violation, item.objective_vector, item.uid))[
        :population_size
    ]


def _select_moead(
    candidates: list[_Evaluated],
    population_size: int,
    reference_directions: tuple[tuple[float, ...], ...],
) -> list[_Evaluated]:
    selected: dict[int, _Evaluated] = {}
    ideal, nadir = _ideal_nadir(candidates)
    for weights in reference_directions:
        best = min(
            candidates,
            key=lambda item: (
                item.constraint_violation,
                _tchebycheff(item.objective_vector, weights, ideal, nadir),
                item.uid,
            ),
        )
        selected[best.uid] = best
        if len(selected) >= population_size:
            break
    if len(selected) < population_size:
        for item in _select_nsga2(candidates, population_size):
            selected[item.uid] = item
            if len(selected) >= population_size:
                break
    return list(selected.values())


def _non_dominated_sort(candidates: list[_Evaluated]) -> list[list[_Evaluated]]:
    remaining = list(candidates)
    fronts: list[list[_Evaluated]] = []
    while remaining:
        front = _nondominated_front(remaining)
        fronts.append(front)
        front_ids = {item.uid for item in front}
        remaining = [item for item in remaining if item.uid not in front_ids]
    return fronts


def _nondominated_front(candidates: list[_Evaluated]) -> list[_Evaluated]:
    return [
        candidate
        for candidate in candidates
        if not any(_dominates(other, candidate) for other in candidates if other.uid != candidate.uid)
    ]


def _dominates(left: _Evaluated, right: _Evaluated) -> bool:
    if left.constraint_violation != right.constraint_violation:
        return left.constraint_violation < right.constraint_violation
    return all(
        left_value <= right_value
        for left_value, right_value in zip(left.objective_vector, right.objective_vector)
    ) and any(
        left_value < right_value
        for left_value, right_value in zip(left.objective_vector, right.objective_vector)
    )


def _crowding_distance(front: list[_Evaluated]) -> dict[int, float]:
    if not front:
        return {}
    if len(front) <= 2:
        return {item.uid: math.inf for item in front}
    distance = {item.uid: 0.0 for item in front}
    n_objectives = len(front[0].objective_vector)
    for objective_index in range(n_objectives):
        ordered = sorted(front, key=lambda item: item.objective_vector[objective_index])
        distance[ordered[0].uid] = math.inf
        distance[ordered[-1].uid] = math.inf
        span = ordered[-1].objective_vector[objective_index] - ordered[0].objective_vector[objective_index]
        if span <= 0:
            continue
        for previous_item, item, next_item in zip(ordered, ordered[1:], ordered[2:]):
            if math.isinf(distance[item.uid]):
                continue
            distance[item.uid] += (
                next_item.objective_vector[objective_index] - previous_item.objective_vector[objective_index]
            ) / span
    return distance


def _spea2_fitness(candidates: list[_Evaluated]) -> dict[int, float]:
    strength = {
        item.uid: sum(1 for other in candidates if item.uid != other.uid and _dominates(item, other))
        for item in candidates
    }
    raw = {
        item.uid: sum(strength[other.uid] for other in candidates if item.uid != other.uid and _dominates(other, item))
        for item in candidates
    }
    k_index = max(0, int(math.sqrt(len(candidates))) - 1)
    fitness = {}
    for item in candidates:
        distances = sorted(_objective_distance(item, other) for other in candidates if item.uid != other.uid)
        density = 1.0 / (distances[min(k_index, len(distances) - 1)] + 2.0) if distances else 0.0
        fitness[item.uid] = raw[item.uid] + density + item.constraint_violation * 1000.0
    return fitness


def _niching_select(
    front: list[_Evaluated],
    already_selected: list[_Evaluated],
    slots: int,
    reference_directions: tuple[tuple[float, ...], ...],
) -> list[_Evaluated]:
    if slots <= 0:
        return []
    ideal, nadir = _ideal_nadir(front + already_selected)
    niche_counts = {index: 0 for index in range(len(reference_directions))}
    for item in already_selected:
        niche, _ = _associate_reference(item.objective_vector, reference_directions, ideal, nadir)
        niche_counts[niche] += 1
    pool = list(front)
    chosen: list[_Evaluated] = []
    while pool and len(chosen) < slots:
        associated = [
            (*_associate_reference(item.objective_vector, reference_directions, ideal, nadir), item)
            for item in pool
        ]
        niche, _, item = min(
            associated,
            key=lambda entry: (niche_counts[entry[0]], entry[1], entry[2].constraint_violation, entry[2].uid),
        )
        chosen.append(item)
        niche_counts[niche] += 1
        pool = [candidate for candidate in pool if candidate.uid != item.uid]
    return chosen


def _associate_reference(
    vector: tuple[float, ...],
    reference_directions: tuple[tuple[float, ...], ...],
    ideal: tuple[float, ...],
    nadir: tuple[float, ...],
) -> tuple[int, float]:
    normalized = _normalize_vector(vector, ideal, nadir)
    best_index = 0
    best_distance = math.inf
    for index, direction in enumerate(reference_directions):
        distance = _perpendicular_distance(normalized, direction)
        if distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index, best_distance


def _reference_directions(
    n_objectives: int,
    target_count: int,
    rng: random.Random,
    preferences: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    directions: list[tuple[float, ...]] = []
    directions.extend(tuple(vector) for vector in preferences if len(vector) == n_objectives)
    for index in range(n_objectives):
        unit = [0.0] * n_objectives
        unit[index] = 1.0
        directions.append(tuple(unit))
    while len(directions) < target_count:
        values = [rng.random() for _ in range(n_objectives)]
        total = sum(values) or 1.0
        directions.append(tuple(value / total for value in values))
    unique: list[tuple[float, ...]] = []
    seen = set()
    for direction in directions:
        normalized = _safe_simplex(direction)
        key = tuple(round(value, 6) for value in normalized)
        if key not in seen:
            seen.add(key)
            unique.append(normalized)
        if len(unique) >= target_count:
            break
    return tuple(unique)


def _ideal_nadir(candidates: list[_Evaluated]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    n_objectives = len(candidates[0].objective_vector)
    ideal = tuple(min(item.objective_vector[index] for item in candidates) for index in range(n_objectives))
    nadir = tuple(max(item.objective_vector[index] for item in candidates) for index in range(n_objectives))
    return ideal, nadir


def _tchebycheff(
    vector: tuple[float, ...],
    weights: tuple[float, ...],
    ideal: tuple[float, ...],
    nadir: tuple[float, ...],
) -> float:
    normalized = _normalize_vector(vector, ideal, nadir)
    return max(max(weight, 1e-6) * abs(value) for weight, value in zip(weights, normalized))


def _normalize_vector(
    vector: tuple[float, ...],
    ideal: tuple[float, ...],
    nadir: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(
        (value - ideal_value) / (nadir_value - ideal_value)
        if nadir_value > ideal_value
        else 0.0
        for value, ideal_value, nadir_value in zip(vector, ideal, nadir)
    )


def _perpendicular_distance(vector: tuple[float, ...], direction: tuple[float, ...]) -> float:
    direction_norm = math.sqrt(sum(value * value for value in direction)) or 1.0
    projection = sum(value * weight for value, weight in zip(vector, direction)) / direction_norm
    projected = tuple(projection * weight / direction_norm for weight in direction)
    return math.sqrt(sum((value - projected_value) ** 2 for value, projected_value in zip(vector, projected)))


def _objective_distance(left: _Evaluated, right: _Evaluated) -> float:
    return math.sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(left.objective_vector, right.objective_vector)
        )
    )


def _safe_simplex(values: tuple[float, ...]) -> tuple[float, ...]:
    clipped = tuple(max(0.0, value) for value in values)
    total = sum(clipped)
    if total <= 0:
        return tuple(1.0 / len(values) for _ in values)
    return tuple(value / total for value in clipped)


def _genome_signature(genome: _Genome) -> str:
    return ";".join(f"{value:.4f}" for value in genome.genes[:12])


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))
