from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from su_modfjsp_ha_pc.generator import generate_instance
from su_modfjsp_ha_pc.metrics import evaluate_schedule
from su_modfjsp_ha_pc.neural_operation_rollout import (
    _RolloutState,
    _candidate_completion,
    _decode_with_operation_picker,
    _detail_row,
    _objective_values,
    _ordered_detail,
    _ordered_pairwise,
    _ordered_summary,
    _pairwise_rows,
    _summary_rows,
    decode_rule_operation_rollout,
)
from su_modfjsp_ha_pc.offline_logs import write_csv
from su_modfjsp_ha_pc.path_conflict_neural_dataset import (
    CANDIDATE_FIELDS,
    _attach_group_critical_path_regrets,
    _best_preference_action,
    _candidate_row,
    _complete_schedule_after_forced_operation,
    _explicit_preference_id,
    _explicit_teacher_preferences,
    _operation_stage_lookahead,
    decode_preference_conditioned_teacher_schedule,
    preference_teacher_surrogate_metrics,
)
from su_modfjsp_ha_pc.path_conflict_operation_sequence_policy import (
    OPERATION_SEQUENCE_FEATURE_FIELDS,
    OperationGroup,
    _operation_sequence_features,
    fit_standardizer,
)
from su_modfjsp_ha_pc.rules import _available_operations
from su_modfjsp_ha_pc.structural import realize_structural_events


OBJECTIVE_NAMES = ("makespan", "tardiness", "energy", "path_conflict_delay", "stability")

E115_EXTRA_LABEL_FIELDS = (
    "future_value_label_available",
    "future_energy_after_action",
    "future_path_conflict_delay_after_action",
    "future_stability_after_action",
    "future_energy_regret_to_group_best",
    "future_path_conflict_delay_regret_to_group_best",
    "future_stability_regret_to_group_best",
    "future_weighted_value_after_action",
)

DATASET_FIELDS = CANDIDATE_FIELDS + E115_EXTRA_LABEL_FIELDS

TRAIN_FIELDS = (
    "method",
    "train_groups",
    "train_rows",
    "label_rows",
    "epochs",
    "hidden_size",
    "attention_heads",
    "layers",
    "final_loss",
)

TRACE_FIELDS = (
    "level",
    "seed",
    "mode",
    "target_preference_id",
    "step_index",
    "available_count",
    "candidate_operation_id",
    "selected_by_value_model",
    "selected_by_lookahead",
    "predicted_weighted_value",
    "predicted_makespan",
    "predicted_tardiness",
    "predicted_energy",
    "predicted_path_conflict_delay",
    "predicted_stability",
    "lookahead_completion",
)


@dataclass(frozen=True)
class ObjectiveValueGroup:
    group_id: str
    rows: tuple[dict[str, object], ...]
    feature_rows: tuple[tuple[float, ...], ...]
    targets: tuple[tuple[float, ...], ...]
    label_available: bool


@dataclass(frozen=True)
class TargetScaler:
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def transform_one(self, values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple((value - mean) / scale for value, mean, scale in zip(values, self.means, self.scales))

    def inverse_one(self, values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(value * scale + mean for value, mean, scale in zip(values, self.means, self.scales))


@dataclass(frozen=True)
class OracleValueArtifact:
    model: object
    feature_standardizer: object
    target_scaler: TargetScaler
    train_groups: int
    train_rows: int
    label_rows: int
    final_loss: float


def build_oracle_value_policy_rows(
    *,
    levels: tuple[str, ...],
    modes: tuple[str, ...],
    size: str,
    train_start_seed: int,
    train_end_seed: int,
    eval_start_seed: int,
    eval_end_seed: int,
    max_preferences: int,
    epochs: int,
    hidden_size: int,
    attention_heads: int,
    layers: int,
    random_state: int,
    max_train_groups_per_context: int = 0,
) -> dict[str, tuple[dict[str, object], ...]]:
    dataset_rows = build_oracle_value_dataset_rows(
        levels=levels,
        modes=modes,
        size=size,
        start_seed=train_start_seed,
        end_seed=train_end_seed,
        max_preferences=max_preferences,
        max_groups_per_context=max_train_groups_per_context,
    )
    artifact = train_oracle_value_artifact(
        dataset_rows,
        epochs=epochs,
        hidden_size=hidden_size,
        attention_heads=attention_heads,
        layers=layers,
        random_state=random_state,
    )
    detail_rows = []
    trace_rows = []
    for level in levels:
        for seed in range(eval_start_seed, eval_end_seed + 1):
            raw_instance = generate_instance(level, size, seed)
            for mode in modes:
                instance = realize_structural_events(raw_instance).instance if mode == "realized" else raw_instance
                preferences = _explicit_teacher_preferences(len(instance.objectives), max_preferences=max_preferences)
                for preference_index, preference in enumerate(preferences):
                    preference_id = _explicit_preference_id(preference_index, preference)
                    teacher = decode_preference_conditioned_teacher_schedule(
                        instance,
                        preference,
                        algorithm="e115-teacher-rollout",
                    )
                    teacher_ops = tuple(operation.operation_id for operation in teacher.operations)
                    schedules = {
                        "teacher": teacher,
                        "oracle-value-neural": decode_oracle_value_neural_rollout(
                            instance=instance,
                            preference=preference,
                            artifact=artifact,
                            level=level,
                            seed=seed,
                            mode=mode,
                            preference_id=preference_id,
                            trace_rows=trace_rows,
                        ),
                        "lookahead-rule": decode_rule_operation_rollout(
                            instance=instance,
                            preference=preference,
                            rule="lookahead",
                            random_seed=random_state + seed + preference_index,
                        ),
                        "spt-rule": decode_rule_operation_rollout(
                            instance=instance,
                            preference=preference,
                            rule="spt",
                            random_seed=random_state + seed + preference_index,
                        ),
                        "random-rule": decode_rule_operation_rollout(
                            instance=instance,
                            preference=preference,
                            rule="random",
                            random_seed=random_state + seed * 1000 + preference_index,
                        ),
                    }
                    for method, schedule in schedules.items():
                        detail_rows.append(
                            _detail_row(
                                instance=instance,
                                schedule=schedule,
                                teacher_operations=teacher_ops,
                                level=level,
                                seed=seed,
                                mode=mode,
                                preference_id=preference_id,
                                preference=preference,
                                method=method,
                            )
                        )
    return {
        "dataset": tuple(_ordered_dataset(row) for row in dataset_rows),
        "train": (
            {
                "method": "oracle-value-neural",
                "train_groups": artifact.train_groups,
                "train_rows": artifact.train_rows,
                "label_rows": artifact.label_rows,
                "epochs": epochs,
                "hidden_size": hidden_size,
                "attention_heads": attention_heads,
                "layers": layers,
                "final_loss": round(artifact.final_loss, 10),
            },
        ),
        "detail": tuple(_ordered_detail(row) for row in detail_rows),
        "summary": _summary_rows(tuple(detail_rows)),
        "pairwise": _pairwise_rows(tuple(detail_rows), candidate_method="oracle-value-neural"),
        "trace": tuple(_ordered_trace(row) for row in trace_rows),
    }


def build_oracle_value_dataset_rows(
    *,
    levels: tuple[str, ...],
    modes: tuple[str, ...],
    size: str,
    start_seed: int,
    end_seed: int,
    max_preferences: int,
    max_groups_per_context: int = 0,
) -> tuple[dict[str, object], ...]:
    rows = []
    candidate_uid = 0
    for level in levels:
        for seed in range(start_seed, end_seed + 1):
            raw_instance = generate_instance(level, size, seed)
            for mode in modes:
                instance = realize_structural_events(raw_instance).instance if mode == "realized" else raw_instance
                preferences = _explicit_teacher_preferences(len(instance.objectives), max_preferences=max_preferences)
                for preference_index, preference in enumerate(preferences):
                    preference_id = _explicit_preference_id(preference_index, preference)
                    schedule = decode_preference_conditioned_teacher_schedule(
                        instance,
                        preference,
                        algorithm="e115-oracle-value-label-teacher",
                    )
                    context_rows = _labeled_operation_rows_for_schedule(
                        instance=instance,
                        schedule=schedule,
                        preference=preference,
                        preference_id=preference_id,
                        level=level,
                        seed=seed,
                        mode=mode,
                        candidate_uid_start=candidate_uid,
                        max_groups_per_context=max_groups_per_context,
                    )
                    rows.extend(context_rows)
                    candidate_uid += len(context_rows)
    return tuple(_ordered_dataset(row) for row in rows)


def train_oracle_value_artifact(
    rows: tuple[dict[str, object], ...],
    *,
    epochs: int,
    hidden_size: int,
    attention_heads: int,
    layers: int,
    random_state: int,
) -> OracleValueArtifact:
    groups = build_objective_value_groups(tuple(row for row in rows if row.get("stage") == "operation"))
    train_groups = tuple(group for group in groups if group.label_available)
    if not train_groups:
        raise ValueError("oracle value dataset must contain labeled operation groups")
    feature_standardizer = fit_standardizer(tuple(feature for group in train_groups for feature in group.feature_rows))
    target_scaler = fit_target_scaler(tuple(target for group in train_groups for target in group.targets))
    model, final_loss = _train_value_model(
        train_groups,
        feature_standardizer=feature_standardizer,
        target_scaler=target_scaler,
        epochs=epochs,
        hidden_size=hidden_size,
        attention_heads=attention_heads,
        layers=layers,
        random_state=random_state,
    )
    return OracleValueArtifact(
        model=model,
        feature_standardizer=feature_standardizer,
        target_scaler=target_scaler,
        train_groups=len(train_groups),
        train_rows=sum(len(group.rows) for group in train_groups),
        label_rows=sum(
            1
            for group in train_groups
            for row in group.rows
            if _float(row.get("future_value_label_available", 0.0)) > 0.0
        ),
        final_loss=final_loss,
    )


def build_objective_value_groups(rows: tuple[dict[str, object], ...]) -> tuple[ObjectiveValueGroup, ...]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("decision_group_id", "")), []).append(row)
    groups = []
    for group_id, group_rows in sorted(grouped.items()):
        sorted_rows = tuple(sorted(group_rows, key=lambda row: str(row.get("candidate_operation_id", ""))))
        feature_rows = tuple(_operation_sequence_features(row, index, sorted_rows) for index, row in enumerate(sorted_rows))
        label_available = all(_float(row.get("future_value_label_available", 0.0)) > 0.0 for row in sorted_rows)
        targets = tuple(_target_vector(row) for row in sorted_rows)
        groups.append(
            ObjectiveValueGroup(
                group_id=group_id,
                rows=sorted_rows,
                feature_rows=feature_rows,
                targets=targets,
                label_available=label_available,
            )
        )
    return tuple(groups)


def decode_oracle_value_neural_rollout(
    *,
    instance: object,
    preference: tuple[float, ...],
    artifact: OracleValueArtifact,
    level: str,
    seed: int,
    mode: str,
    preference_id: str,
    trace_rows: list[dict[str, object]] | None = None,
) -> object:
    def pick_operation(state: _RolloutState, available: tuple[tuple[object, object], ...], step_index: int):
        rows = _operation_candidate_rows_for_state(
            instance=instance,
            state=state,
            available=available,
            preference=preference,
            level=level,
            seed=seed,
            mode=mode,
            preference_id=preference_id,
            step_index=step_index,
            total_operations=state.total_operations,
        )
        scored = score_oracle_value_rows(rows, artifact=artifact, preference=preference)
        chosen = min(
            scored,
            key=lambda row: (
                float(row.get("predicted_weighted_value", 0.0)),
                _candidate_completion(row),
                str(row.get("candidate_operation_id", "")),
            ),
        )
        if trace_rows is not None:
            trace_rows.extend(
                _trace_rows_for_decision(
                    scored,
                    level=level,
                    seed=seed,
                    mode=mode,
                    preference_id=preference_id,
                    step_index=step_index,
                    chosen_operation_id=str(chosen.get("candidate_operation_id", "")),
                )
            )
        operation_id = str(chosen["candidate_operation_id"])
        for job, operation in available:
            if operation.operation_id == operation_id:
                return job, operation
        raise ValueError(f"oracle-value neural policy selected unavailable operation {operation_id}")

    return _decode_with_operation_picker(
        instance=instance,
        preference=preference,
        algorithm="e115-oracle-value-neural-rollout",
        picker=pick_operation,
    )


def score_oracle_value_rows(
    rows: tuple[dict[str, object], ...],
    *,
    artifact: OracleValueArtifact,
    preference: tuple[float, ...],
) -> tuple[dict[str, object], ...]:
    import torch

    sorted_rows = tuple(sorted(rows, key=lambda row: str(row.get("candidate_operation_id", ""))))
    feature_rows = tuple(_operation_sequence_features(row, index, sorted_rows) for index, row in enumerate(sorted_rows))
    group = ObjectiveValueGroup(
        group_id=str(sorted_rows[0].get("decision_group_id", "e115-rollout")),
        rows=sorted_rows,
        feature_rows=feature_rows,
        targets=tuple((0.0,) * len(OBJECTIVE_NAMES) for _row in sorted_rows),
        label_available=False,
    )
    x, valid_mask, _targets, _preferences, _best_indices = _group_tensors(
        (group,),
        artifact.feature_standardizer,
        artifact.target_scaler,
    )
    artifact.model.eval()
    with torch.no_grad():
        predicted_norm = artifact.model(x, valid_mask)[0]
    scored = []
    for row_index, row in enumerate(sorted_rows):
        predicted = artifact.target_scaler.inverse_one(
            tuple(float(predicted_norm[row_index, index].item()) for index in range(len(OBJECTIVE_NAMES)))
        )
        copy = dict(row)
        weighted = 0.0
        for index, objective in enumerate(OBJECTIVE_NAMES):
            value = max(0.0, float(predicted[index]))
            copy[f"predicted_{objective}"] = value
            if index < len(preference):
                weighted += float(preference[index]) * value
        copy["predicted_weighted_value"] = weighted
        scored.append(copy)
    return tuple(scored)


def write_oracle_value_policy_outputs(out_dir: Path, rows: dict[str, tuple[dict[str, object], ...]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "e115_oracle_value_dataset.csv", rows["dataset"])
    write_csv(out_dir / "e115_oracle_value_train.csv", rows["train"])
    write_csv(out_dir / "e115_oracle_value_detail.csv", rows["detail"])
    write_csv(out_dir / "e115_oracle_value_summary.csv", rows["summary"])
    write_csv(out_dir / "e115_oracle_value_pairwise.csv", rows["pairwise"])
    write_csv(out_dir / "e115_oracle_value_trace.csv", rows["trace"])


def fit_target_scaler(targets: tuple[tuple[float, ...], ...]) -> TargetScaler:
    if not targets:
        raise ValueError("cannot fit target scaler with no targets")
    width = len(targets[0])
    means = []
    scales = []
    for column in range(width):
        values = [row[column] for row in targets]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scale = variance ** 0.5
        means.append(mean)
        scales.append(scale if scale > 1.0e-9 else 1.0)
    return TargetScaler(tuple(means), tuple(scales))


def _labeled_operation_rows_for_schedule(
    *,
    instance: object,
    schedule: object,
    preference: tuple[float, ...],
    preference_id: str,
    level: str,
    seed: int,
    mode: str,
    candidate_uid_start: int,
    max_groups_per_context: int,
) -> tuple[dict[str, object], ...]:
    operation_by_id = {operation.operation_id: operation for job in instance.jobs for operation in job.operations}
    job_by_id = {job.job_id: job for job in instance.jobs}
    machine_available = {machine.machine_id: 0 for machine in instance.machines}
    job_ready = {job.job_id: job.release_time for job in instance.jobs}
    next_op_index = {job.job_id: 0 for job in instance.jobs}
    agv_available = {agv.agv_id: 0 for agv in instance.agvs}
    agv_node = {agv.agv_id: agv.start_node_id for agv in instance.agvs}
    edge_reservations: dict[str, list[tuple[int, int]]] = {edge.edge_id: [] for edge in instance.path_edges}
    total_operations = max(1, len(schedule.operations))
    selected_steps = _selected_label_steps(total_operations, max_groups_per_context)
    rows = []
    transport_by_operation = {move.operation_id: move for move in schedule.transports}

    for step_index, scheduled in enumerate(schedule.operations):
        operation = operation_by_id[scheduled.operation_id]
        job = job_by_id[operation.job_id]
        available = _available_operations(instance.jobs, next_op_index)
        available_ids = {item[1].operation_id for item in available}
        if operation.operation_id not in available_ids:
            available = available + ((job, operation),)
        group_id = f"e115-{level}-{seed:03d}-{mode}-{preference_id}-{step_index:03d}-operation"
        group_rows = _operation_group_rows(
            instance=instance,
            available=available,
            preference=preference,
            preference_id=preference_id,
            level=level,
            seed=seed,
            mode=mode,
            group_id=group_id,
            candidate_uid_start=candidate_uid_start + len(rows),
            step_index=step_index,
            selected_operation_id=operation.operation_id,
            total_operations=total_operations,
            machine_available=machine_available,
            job_ready=job_ready,
            agv_available=agv_available,
            agv_node=agv_node,
            edge_reservations=edge_reservations,
        )
        if step_index in selected_steps:
            prefix_operation_ids = {item.operation_id for item in schedule.operations[:step_index]}
            group_rows = _attach_future_value_labels(
                instance=instance,
                rows=group_rows,
                available=available,
                preference=preference,
                machine_available=machine_available,
                job_ready=job_ready,
                next_op_index=next_op_index,
                agv_available=agv_available,
                agv_node=agv_node,
                edge_reservations=edge_reservations,
                scheduled_prefix=tuple(schedule.operations[:step_index]),
                transport_prefix=tuple(move for move in schedule.transports if move.operation_id in prefix_operation_ids),
            )
        rows.extend(group_rows)
        _append_scheduled_teacher_step(
            instance=instance,
            scheduled=scheduled,
            transport=transport_by_operation.get(operation.operation_id),
            machine_available=machine_available,
            job_ready=job_ready,
            next_op_index=next_op_index,
            agv_available=agv_available,
            agv_node=agv_node,
            edge_reservations=edge_reservations,
        )
    return tuple(_ordered_dataset(row) for row in rows)


def _operation_candidate_rows_for_state(
    *,
    instance: object,
    state: _RolloutState,
    available: tuple[tuple[object, object], ...],
    preference: tuple[float, ...],
    level: str,
    seed: int,
    mode: str,
    preference_id: str,
    step_index: int,
    total_operations: int,
) -> tuple[dict[str, object], ...]:
    group_id = f"e115-rollout-{level}-{seed:03d}-{mode}-{preference_id}-{step_index:03d}-operation"
    return _operation_group_rows(
        instance=instance,
        available=available,
        preference=preference,
        preference_id=preference_id,
        level=level,
        seed=seed,
        mode=mode,
        group_id=group_id,
        candidate_uid_start=step_index * 1000,
        step_index=step_index,
        selected_operation_id="",
        total_operations=total_operations,
        machine_available=state.machine_available,
        job_ready=state.job_ready,
        agv_available=state.agv_available,
        agv_node=state.agv_node,
        edge_reservations=state.edge_reservations,
    )


def _operation_group_rows(
    *,
    instance: object,
    available: tuple[tuple[object, object], ...],
    preference: tuple[float, ...],
    preference_id: str,
    level: str,
    seed: int,
    mode: str,
    group_id: str,
    candidate_uid_start: int,
    step_index: int,
    selected_operation_id: str,
    total_operations: int,
    machine_available: dict[str, int],
    job_ready: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
) -> tuple[dict[str, object], ...]:
    rows = []
    for index, (job, operation) in enumerate(available):
        best = _operation_stage_lookahead(
            instance=instance,
            job=job,
            operation=operation,
            machine_available=machine_available,
            job_ready_time=job_ready[job.job_id],
            agv_available=agv_available,
            agv_node=agv_node,
            edge_reservations=edge_reservations,
        )
        rows.append(
            _candidate_row(
                instance=instance,
                candidate_uid=candidate_uid_start + index,
                split="train",
                level=level,
                seed=seed,
                mode=mode,
                algorithm="e115-oracle-value-policy",
                teacher_family="e115-preference-teacher",
                source_solution_id=preference_id,
                teacher_schedule_id=f"e115-{level}-{seed:03d}-{mode}-{preference_id}",
                target_preference_id=preference_id,
                decision_group_id=group_id,
                stage="operation",
                target_preference=preference,
                step_index=step_index,
                group_size=len(available),
                selected_action=bool(selected_operation_id and operation.operation_id == selected_operation_id),
                available_operation_count=len(available),
                total_operations=total_operations,
                job=job,
                operation=operation,
                job_ready_time=job_ready[job.job_id],
                machine=best["machine"],
                machine_available_time=machine_available[best["machine"].machine_id],
                agv=best.get("agv"),
                agv_available_time=best.get("agv_available_time", 0),
                route=best.get("route", ()),
                route_start=best.get("route_start", job_ready[job.job_id]),
                route_end=best.get("route_end", job_ready[job.job_id]),
                route_wait_time=best.get("route_wait_time", 0.0),
                candidate_has_transport=bool(best.get("has_transport", False)),
            )
        )
    return tuple(_zero_extra_labels(row) for row in _attach_group_critical_path_regrets(tuple(rows)))


def _attach_future_value_labels(
    *,
    instance: object,
    rows: tuple[dict[str, object], ...],
    available: tuple[tuple[object, object], ...],
    preference: tuple[float, ...],
    machine_available: dict[str, int],
    job_ready: dict[str, int],
    next_op_index: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
    scheduled_prefix: tuple[object, ...],
    transport_prefix: tuple[object, ...],
) -> tuple[dict[str, object], ...]:
    operation_by_id = {operation.operation_id: (job, operation) for job, operation in available}
    outcomes = {}
    for row in rows:
        operation_id = str(row.get("candidate_operation_id", ""))
        job, operation = operation_by_id[operation_id]
        schedule = _complete_schedule_after_forced_operation(
            instance=instance,
            preference=preference,
            forced_job=job,
            forced_operation=operation,
            machine_available=machine_available,
            job_ready=job_ready,
            next_op_index=next_op_index,
            agv_available=agv_available,
            agv_node=agv_node,
            edge_reservations=edge_reservations,
            scheduled_prefix=scheduled_prefix,
            transport_prefix=transport_prefix,
        )
        metrics = evaluate_schedule(instance, schedule)
        surrogates = preference_teacher_surrogate_metrics(instance, schedule)
        outcomes[operation_id] = _objective_values(instance, metrics, surrogates)
    best_by_objective = {
        objective: min(float(values.get(objective, 0.0)) for values in outcomes.values())
        for objective in OBJECTIVE_NAMES
    }
    patched = []
    for row in rows:
        operation_id = str(row.get("candidate_operation_id", ""))
        values = outcomes[operation_id]
        copy = dict(row)
        copy["future_value_label_available"] = 1.0
        weighted_value = 0.0
        weighted_regret = 0.0
        for index, objective in enumerate(OBJECTIVE_NAMES):
            value = float(values.get(objective, 0.0))
            regret = max(0.0, value - best_by_objective[objective])
            copy[_future_value_field(objective)] = value
            copy[_future_regret_field(objective)] = regret
            if index < len(preference):
                weighted_value += float(preference[index]) * value
                weighted_regret += float(preference[index]) * regret
        copy["future_weighted_value_after_action"] = weighted_value
        copy["future_weighted_completion_regret"] = weighted_regret
        patched.append(copy)
    return tuple(_ordered_dataset(row) for row in patched)


def _train_value_model(
    groups: tuple[ObjectiveValueGroup, ...],
    *,
    feature_standardizer: object,
    target_scaler: TargetScaler,
    epochs: int,
    hidden_size: int,
    attention_heads: int,
    layers: int,
    random_state: int,
) -> tuple[object, float]:
    import torch
    from torch import nn

    torch.manual_seed(random_state)
    x, valid_mask, targets, preferences, best_indices = _group_tensors(groups, feature_standardizer, target_scaler)
    model = _ObjectiveValueRanker(
        input_size=len(OPERATION_SEQUENCE_FEATURE_FIELDS),
        hidden_size=hidden_size,
        attention_heads=attention_heads,
        layers=layers,
        output_size=len(OBJECTIVE_NAMES),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006, weight_decay=1.0e-4)
    loss_fn = nn.MSELoss(reduction="none")
    rank_loss_fn = nn.CrossEntropyLoss()
    target_means = torch.tensor(target_scaler.means, dtype=torch.float32)
    target_scales = torch.tensor(target_scaler.scales, dtype=torch.float32)
    final_loss = 0.0
    for _epoch in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        predicted = model(x, valid_mask)
        loss_values = loss_fn(predicted, targets).mean(dim=2)
        value_loss = loss_values.masked_select(valid_mask).mean()
        predicted_original = predicted * target_scales + target_means
        predicted_weighted = (predicted_original * preferences.unsqueeze(1)).sum(dim=2)
        rank_scores = (-predicted_weighted).masked_fill(~valid_mask, -1.0e9)
        loss = value_loss + 0.20 * rank_loss_fn(rank_scores, best_indices)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    model.eval()
    return model, final_loss


class _ObjectiveValueRanker:
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        attention_heads: int,
        layers: int,
        output_size: int,
    ) -> None:
        from torch import nn

        class Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input = nn.Linear(input_size, hidden_size)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=attention_heads,
                    dim_feedforward=hidden_size * 2,
                    dropout=0.05,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
                self.value = nn.Linear(hidden_size, output_size)

            def forward(self, tensor, valid_mask):
                hidden = self.input(tensor)
                encoded = self.encoder(hidden, src_key_padding_mask=~valid_mask)
                return self.value(encoded)

        self.network = Network()

    def parameters(self):
        return self.network.parameters()

    def train(self) -> None:
        self.network.train()

    def eval(self) -> None:
        self.network.eval()

    def __call__(self, tensor, valid_mask):
        return self.network(tensor, valid_mask)


def _group_tensors(groups: tuple[ObjectiveValueGroup, ...], feature_standardizer: object, target_scaler: TargetScaler):
    import torch

    max_len = max(len(group.rows) for group in groups)
    width = len(OPERATION_SEQUENCE_FEATURE_FIELDS)
    target_width = len(OBJECTIVE_NAMES)
    x = torch.zeros((len(groups), max_len, width), dtype=torch.float32)
    valid_mask = torch.zeros((len(groups), max_len), dtype=torch.bool)
    targets = torch.zeros((len(groups), max_len, target_width), dtype=torch.float32)
    preferences = torch.zeros((len(groups), target_width), dtype=torch.float32)
    best_indices = torch.zeros((len(groups),), dtype=torch.long)
    for group_index, group in enumerate(groups):
        preference = _group_preference(group)
        preferences[group_index] = torch.tensor(preference, dtype=torch.float32)
        weighted_targets = tuple(_weighted_target(target, preference) for target in group.targets)
        best_indices[group_index] = int(min(range(len(weighted_targets)), key=lambda index: (weighted_targets[index], index)))
        for row_index, feature_row in enumerate(group.feature_rows):
            x[group_index, row_index] = torch.tensor(feature_standardizer.transform_one(feature_row), dtype=torch.float32)
            valid_mask[group_index, row_index] = True
            targets[group_index, row_index] = torch.tensor(
                target_scaler.transform_one(group.targets[row_index]),
                dtype=torch.float32,
            )
    return x, valid_mask, targets, preferences, best_indices


def _append_scheduled_teacher_step(
    *,
    instance: object,
    scheduled: object,
    transport: object | None,
    machine_available: dict[str, int],
    job_ready: dict[str, int],
    next_op_index: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
) -> None:
    if transport is not None:
        from su_modfjsp_ha_pc.rules import job_pickup_of, move_leg_usage

        for _legs, _ls, _le in move_leg_usage(instance, transport, job_pickup_of(instance, agv_node, transport.job_id)):
            for edge_id in dict.fromkeys(_legs):
                edge_reservations.setdefault(edge_id, []).append((_ls, _le))
        agv_available[transport.agv_id] = transport.end
        agv_node[transport.agv_id] = transport.to_node
        agv_node["__job__:" + str(transport.job_id)] = transport.to_node
    del instance
    machine_available[scheduled.machine_id] = scheduled.end
    job_ready[scheduled.job_id] = scheduled.end
    next_op_index[scheduled.job_id] += 1


def _trace_rows_for_decision(
    scored_rows: tuple[dict[str, object], ...],
    *,
    level: str,
    seed: int,
    mode: str,
    preference_id: str,
    step_index: int,
    chosen_operation_id: str,
) -> tuple[dict[str, object], ...]:
    lookahead = min(
        scored_rows,
        key=lambda row: (
            _candidate_completion(row),
            float(row.get("candidate_processing_time", 0.0)),
            str(row.get("candidate_operation_id", "")),
        ),
    )
    lookahead_id = str(lookahead.get("candidate_operation_id", ""))
    return tuple(
        {
            "level": level,
            "seed": seed,
            "mode": mode,
            "target_preference_id": preference_id,
            "step_index": step_index,
            "available_count": len(scored_rows),
            "candidate_operation_id": row.get("candidate_operation_id", ""),
            "selected_by_value_model": 1 if str(row.get("candidate_operation_id", "")) == chosen_operation_id else 0,
            "selected_by_lookahead": 1 if str(row.get("candidate_operation_id", "")) == lookahead_id else 0,
            "predicted_weighted_value": round(float(row.get("predicted_weighted_value", 0.0)), 10),
            "predicted_makespan": round(float(row.get("predicted_makespan", 0.0)), 10),
            "predicted_tardiness": round(float(row.get("predicted_tardiness", 0.0)), 10),
            "predicted_energy": round(float(row.get("predicted_energy", 0.0)), 10),
            "predicted_path_conflict_delay": round(float(row.get("predicted_path_conflict_delay", 0.0)), 10),
            "predicted_stability": round(float(row.get("predicted_stability", 0.0)), 10),
            "lookahead_completion": round(_candidate_completion(row), 10),
        }
        for row in scored_rows
    )


def _target_vector(row: dict[str, object]) -> tuple[float, ...]:
    return tuple(_float(row.get(_future_value_field(objective), 0.0)) for objective in OBJECTIVE_NAMES)


def _group_preference(group: ObjectiveValueGroup) -> tuple[float, ...]:
    if not group.rows:
        return tuple(0.0 for _objective in OBJECTIVE_NAMES)
    return tuple(_float(group.rows[0].get(f"target_preference_{index}", 0.0)) for index in range(len(OBJECTIVE_NAMES)))


def _weighted_target(target: tuple[float, ...], preference: tuple[float, ...]) -> float:
    return sum(float(weight) * float(value) for weight, value in zip(preference, target))


def _future_value_field(objective: str) -> str:
    return f"future_{objective}_after_action"


def _future_regret_field(objective: str) -> str:
    return f"future_{objective}_regret_to_group_best"


def _selected_label_steps(total_operations: int, max_groups_per_context: int) -> set[int]:
    if max_groups_per_context <= 0 or max_groups_per_context >= total_operations:
        return set(range(total_operations))
    if max_groups_per_context == 1:
        return {0}
    return {
        round(index * (total_operations - 1) / (max_groups_per_context - 1))
        for index in range(max_groups_per_context)
    }


def _zero_extra_labels(row: dict[str, object]) -> dict[str, object]:
    copy = dict(row)
    for field in E115_EXTRA_LABEL_FIELDS:
        copy.setdefault(field, 0.0)
    return copy


def _ordered_dataset(row: dict[str, object]) -> dict[str, object]:
    return {field: row.get(field, "") for field in DATASET_FIELDS}


def _ordered_trace(row: dict[str, object]) -> dict[str, object]:
    return {field: row.get(field, "") for field in TRACE_FIELDS}


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
