from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from su_modfjsp_ha_pc.generator import generate_instance
from su_modfjsp_ha_pc.neural_operation_rollout import (
    _RolloutState,
    _candidate_completion,
    _detail_row,
    _pairwise_rows,
    _summary_rows,
    decode_rule_operation_rollout,
)
from su_modfjsp_ha_pc.offline_logs import write_csv
from su_modfjsp_ha_pc.oracle_value_policy import (
    DATASET_FIELDS,
    OBJECTIVE_NAMES,
    TargetScaler,
    _append_scheduled_teacher_step,
    _future_regret_field,
    _future_value_field,
    _ordered_dataset,
    _selected_label_steps,
    _target_vector,
    _weighted_target,
    fit_target_scaler,
)
from su_modfjsp_ha_pc.path_conflict_neural_dataset import (
    _explicit_preference_id,
    _explicit_teacher_preferences,
    decode_preference_conditioned_teacher_schedule,
)
from su_modfjsp_ha_pc.path_conflict_operation_sequence_policy import (
    OPERATION_SEQUENCE_FEATURE_FIELDS,
    _operation_sequence_features,
    fit_standardizer,
)
from su_modfjsp_ha_pc.joint_operation_machine_value_policy import (
    _append_fixed_joint_action,
    _attach_future_joint_value_labels,
    _joint_group_rows,
    _resolve_joint_candidate,
)
from su_modfjsp_ha_pc.rules import _available_operations
from su_modfjsp_ha_pc.schedule import ScheduleResult, ScheduledOperation, TransportMove
from su_modfjsp_ha_pc.structural import realize_structural_events


TOKEN_ID_FIELDS = (
    "token_id",
    "candidate_id",
    "decision_group_id",
    "candidate_operation_id",
    "candidate_machine_id",
    "token_type",
    "token_entity_id",
)

TOKEN_NUMERIC_FIELDS = (
    "token_index_fraction",
    "token_type_operation",
    "token_type_machine",
    "token_type_agv",
    "token_type_path_summary",
    "token_status_scheduled",
    "token_status_available",
    "token_status_future",
    "token_is_candidate_operation",
    "token_is_candidate_machine",
    "token_time_available",
    "token_job_due_slack",
    "token_remaining_ops",
    "token_remaining_work",
    "token_processing_time_min",
    "token_processing_time_mean",
    "token_processing_time_max",
    "token_processing_energy_min",
    "token_processing_energy_mean",
    "token_eligible_machine_count",
    "token_machine_energy_rate",
    "token_machine_available_time",
    "token_agv_available_time",
    "token_route_wait_time",
    "token_route_length",
    "token_route_bottleneck_count",
    "token_event_pressure",
    "token_reservation_count",
)

TOKEN_FIELDS = TOKEN_ID_FIELDS + TOKEN_NUMERIC_FIELDS

TRAIN_FIELDS = (
    "method",
    "train_groups",
    "train_rows",
    "token_rows",
    "label_rows",
    "epochs",
    "hidden_size",
    "attention_heads",
    "layers",
    "final_loss",
)

TRACE_FIELDS = (
    "method",
    "level",
    "seed",
    "mode",
    "target_preference_id",
    "step_index",
    "available_count",
    "candidate_operation_id",
    "candidate_machine_id",
    "selected_by_attention",
    "selected_by_lookahead",
    "in_safe_set",
    "completion_delta_to_lookahead",
    "predicted_weighted_value",
    "predicted_makespan",
    "predicted_tardiness",
    "predicted_energy",
    "predicted_path_conflict_delay",
    "predicted_stability",
    "lookahead_completion",
)


@dataclass(frozen=True)
class StateActionAttentionDataset:
    rows: tuple[dict[str, object], ...]
    token_rows: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class StateActionAttentionGroup:
    group_id: str
    rows: tuple[dict[str, object], ...]
    candidate_feature_rows: tuple[tuple[float, ...], ...]
    token_feature_rows: tuple[tuple[tuple[float, ...], ...], ...]
    targets: tuple[tuple[float, ...], ...]
    label_available: bool


@dataclass(frozen=True)
class StateActionAttentionArtifact:
    model: object
    candidate_standardizer: object
    token_scaler: TargetScaler
    target_scaler: TargetScaler
    train_groups: int
    train_rows: int
    token_rows: int
    label_rows: int
    final_loss: float


def build_state_action_attention_value_policy_rows(
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
    completion_delta_cap: float = 15.0,
) -> dict[str, tuple[dict[str, object], ...]]:
    dataset = build_state_action_attention_dataset(
        levels=levels,
        modes=modes,
        size=size,
        start_seed=train_start_seed,
        end_seed=train_end_seed,
        max_preferences=max_preferences,
        max_groups_per_context=max_train_groups_per_context,
    )
    artifact = train_state_action_attention_artifact(
        dataset.rows,
        dataset.token_rows,
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
                        algorithm="e118-teacher-rollout",
                    )
                    teacher_ops = tuple(operation.operation_id for operation in teacher.operations)
                    schedules = {
                        "teacher": teacher,
                        "state-action-attention-neural": decode_state_action_attention_rollout(
                            instance=instance,
                            preference=preference,
                            artifact=artifact,
                            level=level,
                            seed=seed,
                            mode=mode,
                            preference_id=preference_id,
                            method="state-action-attention-neural",
                            trace_rows=trace_rows,
                        ),
                        "state-action-attention-safe": decode_state_action_attention_rollout(
                            instance=instance,
                            preference=preference,
                            artifact=artifact,
                            level=level,
                            seed=seed,
                            mode=mode,
                            preference_id=preference_id,
                            method="state-action-attention-safe",
                            completion_delta_cap=completion_delta_cap,
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
                    for method_name, schedule in schedules.items():
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
                                method=method_name,
                            )
                        )

    pairwise = tuple(_pairwise_rows(tuple(detail_rows), candidate_method="state-action-attention-neural")) + tuple(
        _pairwise_rows(tuple(detail_rows), candidate_method="state-action-attention-safe")
    )
    return {
        "dataset": tuple(_ordered_dataset(row) for row in dataset.rows),
        "tokens": tuple(_ordered_token(row) for row in dataset.token_rows),
        "train": (
            {
                "method": "state-action-attention",
                "train_groups": artifact.train_groups,
                "train_rows": artifact.train_rows,
                "token_rows": artifact.token_rows,
                "label_rows": artifact.label_rows,
                "epochs": epochs,
                "hidden_size": hidden_size,
                "attention_heads": attention_heads,
                "layers": layers,
                "final_loss": round(artifact.final_loss, 10),
            },
        ),
        "detail": tuple(detail_rows),
        "summary": _summary_rows(tuple(detail_rows)),
        "pairwise": pairwise,
        "trace": tuple(_ordered_trace(row) for row in trace_rows),
    }


def build_state_action_attention_dataset(
    *,
    levels: tuple[str, ...],
    modes: tuple[str, ...],
    size: str,
    start_seed: int,
    end_seed: int,
    max_preferences: int,
    max_groups_per_context: int = 0,
) -> StateActionAttentionDataset:
    rows = []
    token_rows = []
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
                        algorithm="e118-attention-label-teacher",
                    )
                    context = _labeled_state_action_rows_for_schedule(
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
                    rows.extend(context.rows)
                    token_rows.extend(context.token_rows)
                    candidate_uid += len(context.rows)
    return StateActionAttentionDataset(
        rows=tuple(_ordered_dataset(row) for row in rows),
        token_rows=tuple(_ordered_token(row) for row in token_rows),
    )


def train_state_action_attention_artifact(
    rows: tuple[dict[str, object], ...],
    token_rows: tuple[dict[str, object], ...],
    *,
    epochs: int,
    hidden_size: int,
    attention_heads: int,
    layers: int,
    random_state: int,
) -> StateActionAttentionArtifact:
    groups = build_state_action_attention_groups(tuple(row for row in rows if row.get("stage") == "operation"), token_rows)
    train_groups = tuple(group for group in groups if group.label_available)
    if not train_groups:
        raise ValueError("state-action attention dataset must contain labeled groups")
    candidate_standardizer = fit_standardizer(
        tuple(feature for group in train_groups for feature in group.candidate_feature_rows)
    )
    token_scaler = fit_target_scaler(
        tuple(token for group in train_groups for candidate_tokens in group.token_feature_rows for token in candidate_tokens)
    )
    target_scaler = fit_target_scaler(tuple(target for group in train_groups for target in group.targets))
    model, final_loss = _train_state_action_model(
        train_groups,
        candidate_standardizer=candidate_standardizer,
        token_scaler=token_scaler,
        target_scaler=target_scaler,
        epochs=epochs,
        hidden_size=hidden_size,
        attention_heads=attention_heads,
        layers=layers,
        random_state=random_state,
    )
    return StateActionAttentionArtifact(
        model=model,
        candidate_standardizer=candidate_standardizer,
        token_scaler=token_scaler,
        target_scaler=target_scaler,
        train_groups=len(train_groups),
        train_rows=sum(len(group.rows) for group in train_groups),
        token_rows=sum(len(tokens) for group in train_groups for tokens in group.token_feature_rows),
        label_rows=sum(1 for group in train_groups for row in group.rows if _float(row.get("future_value_label_available", 0.0)) > 0.0),
        final_loss=final_loss,
    )


def build_state_action_attention_groups(
    rows: tuple[dict[str, object], ...],
    token_rows: tuple[dict[str, object], ...],
) -> tuple[StateActionAttentionGroup, ...]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("decision_group_id", "")), []).append(row)
    token_lists: dict[str, list[dict[str, object]]] = {}
    for row in token_rows:
        token_lists.setdefault(str(row.get("candidate_id", "")), []).append(row)
    tokens_by_candidate = {
        candidate_id: tuple(sorted(values, key=lambda item: (_float(item.get("token_index_fraction", 0.0)), str(item.get("token_id", "")))))
        for candidate_id, values in token_lists.items()
    }

    groups = []
    for group_id, group_rows in sorted(grouped.items()):
        sorted_rows = tuple(
            sorted(
                group_rows,
                key=lambda row: (
                    str(row.get("candidate_operation_id", "")),
                    str(row.get("candidate_machine_id", "")),
                ),
            )
        )
        feature_rows = tuple(_operation_sequence_features(row, index, sorted_rows) for index, row in enumerate(sorted_rows))
        token_features = []
        has_tokens = True
        for row in sorted_rows:
            candidate_tokens = tokens_by_candidate.get(str(row.get("candidate_id", "")), ())
            if not candidate_tokens:
                has_tokens = False
            token_features.append(tuple(_token_values(token) for token in candidate_tokens))
        label_available = has_tokens and all(_float(row.get("future_value_label_available", 0.0)) > 0.0 for row in sorted_rows)
        groups.append(
            StateActionAttentionGroup(
                group_id=group_id,
                rows=sorted_rows,
                candidate_feature_rows=feature_rows,
                token_feature_rows=tuple(token_features),
                targets=tuple(_target_vector(row) for row in sorted_rows),
                label_available=label_available,
            )
        )
    return tuple(groups)


def decode_state_action_attention_rollout(
    *,
    instance: object,
    preference: tuple[float, ...],
    artifact: StateActionAttentionArtifact,
    level: str,
    seed: int,
    mode: str,
    preference_id: str,
    method: str,
    completion_delta_cap: float | None = None,
    trace_rows: list[dict[str, object]] | None = None,
) -> ScheduleResult:
    state = _RolloutState(
        machine_available={machine.machine_id: 0 for machine in instance.machines},
        job_ready={job.job_id: job.release_time for job in instance.jobs},
        next_op_index={job.job_id: 0 for job in instance.jobs},
        agv_available={agv.agv_id: 0 for agv in instance.agvs},
        agv_node={agv.agv_id: agv.start_node_id for agv in instance.agvs},
        edge_reservations={edge.edge_id: [] for edge in instance.path_edges},
        total_operations=sum(len(job.operations) for job in instance.jobs),
    )
    scheduled_operations: list[ScheduledOperation] = []
    transport_moves: list[TransportMove] = []
    step_index = 0
    while len(scheduled_operations) < state.total_operations:
        available = _available_operations(instance.jobs, state.next_op_index)
        if not available:
            raise ValueError("no schedulable operations remain")
        group_id = f"e118-rollout-{level}-{seed:03d}-{mode}-{preference_id}-{step_index:03d}-joint"
        rows = _joint_group_rows(
            instance=instance,
            available=available,
            preference=preference,
            preference_id=preference_id,
            level=level,
            seed=seed,
            mode=mode,
            group_id=group_id,
            candidate_uid_start=step_index * 10000,
            step_index=step_index,
            selected_operation_id="",
            selected_machine_id="",
            total_operations=state.total_operations,
            machine_available=state.machine_available,
            job_ready=state.job_ready,
            agv_available=state.agv_available,
            agv_node=state.agv_node,
            edge_reservations=state.edge_reservations,
        )
        tokens = _state_token_rows_for_candidates(
            instance=instance,
            candidate_rows=rows,
            step_index=step_index,
            total_operations=state.total_operations,
            machine_available=state.machine_available,
            job_ready=state.job_ready,
            next_op_index=state.next_op_index,
            agv_available=state.agv_available,
            agv_node=state.agv_node,
            edge_reservations=state.edge_reservations,
        )
        scored = score_state_action_attention_rows(rows, tokens, artifact=artifact, preference=preference)
        lookahead = _lookahead_row(scored)
        lookahead_completion = _candidate_completion(lookahead)
        eligible = scored
        if completion_delta_cap is not None:
            capped = tuple(
                row
                for row in scored
                if _candidate_completion(row) <= lookahead_completion + completion_delta_cap
            )
            if capped:
                eligible = capped
        chosen = min(
            eligible,
            key=lambda row: (
                float(row.get("predicted_weighted_value", 0.0)),
                _candidate_completion(row),
                str(row.get("candidate_operation_id", "")),
                str(row.get("candidate_machine_id", "")),
            ),
        )
        if trace_rows is not None:
            trace_rows.extend(
                _trace_rows_for_decision(
                    scored,
                    method=method,
                    level=level,
                    seed=seed,
                    mode=mode,
                    preference_id=preference_id,
                    step_index=step_index,
                    chosen=chosen,
                    lookahead=lookahead,
                    completion_delta_cap=completion_delta_cap,
                )
            )
        job, operation, machine = _resolve_joint_candidate(instance, available, chosen)
        _append_fixed_joint_action(
            instance=instance,
            job=job,
            operation=operation,
            machine=machine,
            machine_available=state.machine_available,
            job_ready=state.job_ready,
            next_op_index=state.next_op_index,
            agv_available=state.agv_available,
            agv_node=state.agv_node,
            edge_reservations=state.edge_reservations,
            scheduled_operations=scheduled_operations,
            transport_moves=transport_moves,
        )
        step_index += 1
    return ScheduleResult(
        algorithm=f"e118-{method}",
        path_aware=bool(instance.path_edges),
        operations=tuple(scheduled_operations),
        transports=tuple(transport_moves),
        event_aware=bool(instance.events),
    )


def score_state_action_attention_rows(
    rows: tuple[dict[str, object], ...],
    token_rows: tuple[dict[str, object], ...],
    *,
    artifact: StateActionAttentionArtifact,
    preference: tuple[float, ...],
) -> tuple[dict[str, object], ...]:
    import torch

    groups = build_state_action_attention_groups(rows, token_rows)
    if len(groups) != 1:
        raise ValueError(f"expected one state-action rollout group; found {len(groups)}")
    tensors = _state_action_group_tensors(
        groups,
        artifact.candidate_standardizer,
        artifact.token_scaler,
        artifact.target_scaler,
    )
    candidate_x, candidate_mask, token_x, token_mask, _targets, _preferences, _best = tensors
    artifact.model.eval()
    with torch.no_grad():
        predicted_norm = artifact.model(candidate_x, candidate_mask, token_x, token_mask)[0]
    scored = []
    for row_index, row in enumerate(groups[0].rows):
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


def write_state_action_attention_value_policy_outputs(
    out_dir: Path,
    rows: dict[str, tuple[dict[str, object], ...]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "e118_state_action_attention_dataset.csv", rows["dataset"])
    write_csv(out_dir / "e118_state_action_attention_tokens.csv", rows["tokens"])
    write_csv(out_dir / "e118_state_action_attention_train.csv", rows["train"])
    write_csv(out_dir / "e118_state_action_attention_detail.csv", rows["detail"])
    write_csv(out_dir / "e118_state_action_attention_summary.csv", rows["summary"])
    write_csv(out_dir / "e118_state_action_attention_pairwise.csv", rows["pairwise"])
    write_csv(out_dir / "e118_state_action_attention_trace.csv", rows["trace"])


def _labeled_state_action_rows_for_schedule(
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
) -> StateActionAttentionDataset:
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
    token_rows = []
    transport_by_operation = {move.operation_id: move for move in schedule.transports}

    for step_index, scheduled in enumerate(schedule.operations):
        operation = operation_by_id[scheduled.operation_id]
        job = job_by_id[operation.job_id]
        available = _available_operations(instance.jobs, next_op_index)
        available_ids = {item[1].operation_id for item in available}
        if operation.operation_id not in available_ids:
            available = available + ((job, operation),)
        group_id = f"e118-{level}-{seed:03d}-{mode}-{preference_id}-{step_index:03d}-joint"
        group_rows = _joint_group_rows(
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
            selected_machine_id=scheduled.machine_id,
            total_operations=total_operations,
            machine_available=machine_available,
            job_ready=job_ready,
            agv_available=agv_available,
            agv_node=agv_node,
            edge_reservations=edge_reservations,
        )
        if step_index in selected_steps:
            token_rows.extend(
                _state_token_rows_for_candidates(
                    instance=instance,
                    candidate_rows=group_rows,
                    step_index=step_index,
                    total_operations=total_operations,
                    machine_available=machine_available,
                    job_ready=job_ready,
                    next_op_index=next_op_index,
                    agv_available=agv_available,
                    agv_node=agv_node,
                    edge_reservations=edge_reservations,
                )
            )
            prefix_operation_ids = {item.operation_id for item in schedule.operations[:step_index]}
            group_rows = _attach_future_joint_value_labels(
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
    return StateActionAttentionDataset(
        rows=tuple(_ordered_dataset(row) for row in rows),
        token_rows=tuple(_ordered_token(row) for row in token_rows),
    )


def _state_token_rows_for_candidates(
    *,
    instance: object,
    candidate_rows: tuple[dict[str, object], ...],
    step_index: int,
    total_operations: int,
    machine_available: dict[str, int],
    job_ready: dict[str, int],
    next_op_index: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
) -> tuple[dict[str, object], ...]:
    base_tokens = _base_state_tokens(
        instance=instance,
        step_index=step_index,
        total_operations=total_operations,
        machine_available=machine_available,
        job_ready=job_ready,
        next_op_index=next_op_index,
        agv_available=agv_available,
        agv_node=agv_node,
        edge_reservations=edge_reservations,
    )
    rows = []
    for candidate in candidate_rows:
        candidate_id = str(candidate.get("candidate_id", ""))
        candidate_operation_id = str(candidate.get("candidate_operation_id", ""))
        candidate_machine_id = str(candidate.get("candidate_machine_id", ""))
        for token in base_tokens:
            copy = dict(token)
            copy["candidate_id"] = candidate_id
            copy["decision_group_id"] = candidate.get("decision_group_id", "")
            copy["candidate_operation_id"] = candidate_operation_id
            copy["candidate_machine_id"] = candidate_machine_id
            copy["token_is_candidate_operation"] = 1.0 if copy.get("token_entity_id") == candidate_operation_id else 0.0
            copy["token_is_candidate_machine"] = 1.0 if copy.get("token_entity_id") == candidate_machine_id else 0.0
            rows.append(copy)
    return tuple(_ordered_token(row) for row in rows)


def _base_state_tokens(
    *,
    instance: object,
    step_index: int,
    total_operations: int,
    machine_available: dict[str, int],
    job_ready: dict[str, int],
    next_op_index: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
) -> tuple[dict[str, object], ...]:
    tokens = []
    operation_items = [
        (job, operation)
        for job in sorted(instance.jobs, key=lambda item: item.job_id)
        for operation in sorted(job.operations, key=lambda item: item.op_index)
    ]
    machine_count = max(1, len(instance.machines))
    agv_count = max(1, len(instance.agvs))
    denominator = max(1, len(operation_items) + len(instance.machines) + len(instance.agvs))
    token_index = 0
    for job, operation in operation_items:
        scheduled = operation.op_index < next_op_index[job.job_id]
        available = operation.op_index == next_op_index[job.job_id]
        future = operation.op_index > next_op_index[job.job_id]
        processing_values = tuple(float(operation.processing_time[machine_id]) for machine_id in operation.eligible_machines)
        energy_values = tuple(float(operation.processing_energy[machine_id]) for machine_id in operation.eligible_machines)
        ready_lower_bound = _operation_ready_lower_bound(
            job=job,
            operation_index=operation.op_index,
            next_operation_index=next_op_index[job.job_id],
            current_job_ready=float(job_ready[job.job_id]),
        )
        tokens.append(
            _token(
                token_id=f"tok{step_index:03d}-op-{operation.operation_id}",
                token_type="operation",
                token_entity_id=operation.operation_id,
                token_index_fraction=token_index / denominator,
                token_type_operation=1.0,
                token_status_scheduled=1.0 if scheduled else 0.0,
                token_status_available=1.0 if available else 0.0,
                token_status_future=1.0 if future else 0.0,
                token_time_available=ready_lower_bound,
                token_job_due_slack=float(job.due_date) - ready_lower_bound,
                token_remaining_ops=0.0 if scheduled else float(max(0, len(job.operations) - operation.op_index)),
                token_remaining_work=0.0 if scheduled else _remaining_min_processing(job, operation.op_index),
                token_processing_time_min=min(processing_values),
                token_processing_time_mean=sum(processing_values) / len(processing_values),
                token_processing_time_max=max(processing_values),
                token_processing_energy_min=min(energy_values),
                token_processing_energy_mean=sum(energy_values) / len(energy_values),
                token_eligible_machine_count=float(len(operation.eligible_machines)) / machine_count,
                token_event_pressure=_operation_event_pressure(instance, operation, ready_lower_bound),
            )
        )
        token_index += 1
    for machine in sorted(instance.machines, key=lambda item: item.machine_id):
        available_time = float(machine_available[machine.machine_id])
        tokens.append(
            _token(
                token_id=f"tok{step_index:03d}-machine-{machine.machine_id}",
                token_type="machine",
                token_entity_id=machine.machine_id,
                token_index_fraction=token_index / denominator,
                token_type_machine=1.0,
                token_time_available=available_time,
                token_machine_energy_rate=float(machine.energy_rate),
                token_machine_available_time=available_time,
                token_event_pressure=_entity_event_pressure(instance, "machine_breakdown", machine.machine_id, available_time),
                token_reservation_count=float(sum(1 for value in machine_available.values() if value > 0)) / machine_count,
            )
        )
        token_index += 1
    for agv in sorted(instance.agvs, key=lambda item: item.agv_id):
        available_time = float(agv_available[agv.agv_id])
        tokens.append(
            _token(
                token_id=f"tok{step_index:03d}-agv-{agv.agv_id}",
                token_type="agv",
                token_entity_id=agv.agv_id,
                token_index_fraction=token_index / denominator,
                token_type_agv=1.0,
                token_time_available=available_time,
                token_agv_available_time=available_time,
                token_processing_energy_mean=float(getattr(agv, "consumption_rate", 0.0)),
                token_event_pressure=_entity_event_pressure(instance, "agv_failure", agv.agv_id, available_time),
                token_reservation_count=float(sum(1 for value in agv_available.values() if value > 0)) / agv_count,
            )
        )
        token_index += 1
    tokens.append(
        _token(
            token_id=f"tok{step_index:03d}-path-summary",
            token_type="path_summary",
            token_entity_id="path_summary",
            token_index_fraction=token_index / denominator,
            token_type_path_summary=1.0,
            token_route_bottleneck_count=float(sum(1 for edge in instance.path_edges if edge.capacity <= 1)),
            token_event_pressure=float(sum(1 for event in instance.events if event.event_type in {"path_blockage", "congestion_spike"})),
            token_reservation_count=float(sum(len(items) for items in edge_reservations.values())),
        )
    )
    del total_operations, agv_node
    return tuple(tokens)


def _train_state_action_model(
    groups: tuple[StateActionAttentionGroup, ...],
    *,
    candidate_standardizer: object,
    token_scaler: TargetScaler,
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
    tensors = _state_action_group_tensors(groups, candidate_standardizer, token_scaler, target_scaler)
    candidate_x, candidate_mask, token_x, token_mask, targets, preferences, best_indices = tensors
    model = _StateActionCrossAttentionNet(
        candidate_dim=len(OPERATION_SEQUENCE_FEATURE_FIELDS),
        token_dim=len(TOKEN_NUMERIC_FIELDS),
        hidden_size=hidden_size,
        attention_heads=attention_heads,
        layers=layers,
        output_size=len(OBJECTIVE_NAMES),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=1.0e-4)
    loss_fn = nn.MSELoss(reduction="none")
    rank_loss_fn = nn.CrossEntropyLoss()
    target_means = torch.tensor(target_scaler.means, dtype=torch.float32)
    target_scales = torch.tensor(target_scaler.scales, dtype=torch.float32)
    final_loss = 0.0
    for _epoch in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        predicted = model(candidate_x, candidate_mask, token_x, token_mask)
        value_loss = loss_fn(predicted, targets).mean(dim=2).masked_select(candidate_mask).mean()
        predicted_original = predicted * target_scales + target_means
        predicted_weighted = (predicted_original * preferences.unsqueeze(1)).sum(dim=2)
        rank_scores = (-predicted_weighted).masked_fill(~candidate_mask, -1.0e9)
        loss = value_loss + 0.25 * rank_loss_fn(rank_scores, best_indices)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    model.eval()
    return model, final_loss


class _StateActionCrossAttentionNet:
    def __init__(
        self,
        *,
        candidate_dim: int,
        token_dim: int,
        hidden_size: int,
        attention_heads: int,
        layers: int,
        output_size: int,
    ) -> None:
        from torch import nn

        class Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.candidate_input = nn.Linear(candidate_dim, hidden_size)
                candidate_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=attention_heads,
                    dim_feedforward=hidden_size * 2,
                    dropout=0.05,
                    batch_first=True,
                    activation="gelu",
                )
                self.candidate_encoder = nn.TransformerEncoder(candidate_layer, num_layers=layers)
                self.token_input = nn.Linear(token_dim, hidden_size)
                token_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=attention_heads,
                    dim_feedforward=hidden_size * 2,
                    dropout=0.05,
                    batch_first=True,
                    activation="gelu",
                )
                self.token_encoder = nn.TransformerEncoder(token_layer, num_layers=max(1, layers))
                self.cross_attention = nn.MultiheadAttention(hidden_size, attention_heads, dropout=0.05, batch_first=True)
                self.value = nn.Sequential(
                    nn.Linear(hidden_size * 3, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, output_size),
                )

            def forward(self, candidate_x, candidate_mask, token_x, token_mask):
                import torch

                batch_size, candidate_count, token_count, token_width = token_x.shape
                candidate_hidden = self.candidate_input(candidate_x)
                candidate_encoded = self.candidate_encoder(candidate_hidden, src_key_padding_mask=~candidate_mask)
                token_flat = token_x.reshape(batch_size * candidate_count, token_count, token_width)
                token_mask_flat = token_mask.reshape(batch_size * candidate_count, token_count)
                token_hidden = self.token_input(token_flat)
                token_encoded = self.token_encoder(token_hidden, src_key_padding_mask=~token_mask_flat)
                query = candidate_encoded.reshape(batch_size * candidate_count, 1, candidate_encoded.shape[-1])
                attended, _weights = self.cross_attention(
                    query,
                    token_encoded,
                    token_encoded,
                    key_padding_mask=~token_mask_flat,
                    need_weights=False,
                )
                candidate_flat = query.squeeze(1)
                attended_flat = attended.squeeze(1)
                combined = torch.cat((candidate_flat, attended_flat, candidate_flat * attended_flat), dim=1)
                values = self.value(combined).reshape(batch_size, candidate_count, -1)
                return values

        import torch

        self.network = Network()
        self._torch = torch

    def parameters(self):
        return self.network.parameters()

    def train(self) -> None:
        self.network.train()

    def eval(self) -> None:
        self.network.eval()

    def __call__(self, candidate_x, candidate_mask, token_x, token_mask):
        return self.network(candidate_x, candidate_mask, token_x, token_mask)


def _state_action_group_tensors(
    groups: tuple[StateActionAttentionGroup, ...],
    candidate_standardizer: object,
    token_scaler: TargetScaler,
    target_scaler: TargetScaler,
):
    import torch

    max_candidates = max(len(group.rows) for group in groups)
    max_tokens = max(len(tokens) for group in groups for tokens in group.token_feature_rows)
    candidate_width = len(OPERATION_SEQUENCE_FEATURE_FIELDS)
    token_width = len(TOKEN_NUMERIC_FIELDS)
    target_width = len(OBJECTIVE_NAMES)
    candidate_x = torch.zeros((len(groups), max_candidates, candidate_width), dtype=torch.float32)
    candidate_mask = torch.zeros((len(groups), max_candidates), dtype=torch.bool)
    token_x = torch.zeros((len(groups), max_candidates, max_tokens, token_width), dtype=torch.float32)
    token_mask = torch.zeros((len(groups), max_candidates, max_tokens), dtype=torch.bool)
    targets = torch.zeros((len(groups), max_candidates, target_width), dtype=torch.float32)
    preferences = torch.zeros((len(groups), target_width), dtype=torch.float32)
    best_indices = torch.zeros((len(groups),), dtype=torch.long)
    for group_index, group in enumerate(groups):
        preference = _group_preference(group)
        preferences[group_index] = torch.tensor(preference, dtype=torch.float32)
        weighted_targets = tuple(_weighted_target(target, preference) for target in group.targets)
        best_indices[group_index] = int(min(range(len(weighted_targets)), key=lambda index: (weighted_targets[index], index)))
        for candidate_index, feature_row in enumerate(group.candidate_feature_rows):
            candidate_x[group_index, candidate_index] = torch.tensor(
                candidate_standardizer.transform_one(feature_row),
                dtype=torch.float32,
            )
            candidate_mask[group_index, candidate_index] = True
            targets[group_index, candidate_index] = torch.tensor(
                target_scaler.transform_one(group.targets[candidate_index]),
                dtype=torch.float32,
            )
            for token_index, token in enumerate(group.token_feature_rows[candidate_index]):
                token_x[group_index, candidate_index, token_index] = torch.tensor(
                    token_scaler.transform_one(token),
                    dtype=torch.float32,
                )
                token_mask[group_index, candidate_index, token_index] = True
    return candidate_x, candidate_mask, token_x, token_mask, targets, preferences, best_indices


def _trace_rows_for_decision(
    scored_rows: tuple[dict[str, object], ...],
    *,
    method: str,
    level: str,
    seed: int,
    mode: str,
    preference_id: str,
    step_index: int,
    chosen: dict[str, object],
    lookahead: dict[str, object],
    completion_delta_cap: float | None,
) -> tuple[dict[str, object], ...]:
    chosen_key = (str(chosen.get("candidate_operation_id", "")), str(chosen.get("candidate_machine_id", "")))
    lookahead_key = (str(lookahead.get("candidate_operation_id", "")), str(lookahead.get("candidate_machine_id", "")))
    lookahead_completion = _candidate_completion(lookahead)
    return tuple(
        {
            "method": method,
            "level": level,
            "seed": seed,
            "mode": mode,
            "target_preference_id": preference_id,
            "step_index": step_index,
            "available_count": len(scored_rows),
            "candidate_operation_id": row.get("candidate_operation_id", ""),
            "candidate_machine_id": row.get("candidate_machine_id", ""),
            "selected_by_attention": 1
            if (str(row.get("candidate_operation_id", "")), str(row.get("candidate_machine_id", ""))) == chosen_key
            else 0,
            "selected_by_lookahead": 1
            if (str(row.get("candidate_operation_id", "")), str(row.get("candidate_machine_id", ""))) == lookahead_key
            else 0,
            "in_safe_set": 1
            if completion_delta_cap is None or _candidate_completion(row) <= lookahead_completion + completion_delta_cap
            else 0,
            "completion_delta_to_lookahead": round(_candidate_completion(row) - lookahead_completion, 10),
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


def _lookahead_row(rows: tuple[dict[str, object], ...]) -> dict[str, object]:
    return min(
        rows,
        key=lambda row: (
            _candidate_completion(row),
            float(row.get("candidate_processing_time", 0.0)),
            float(row.get("candidate_processing_energy", 0.0)),
            str(row.get("candidate_operation_id", "")),
            str(row.get("candidate_machine_id", "")),
        ),
    )


def _group_preference(group: StateActionAttentionGroup) -> tuple[float, ...]:
    if not group.rows:
        return tuple(0.0 for _objective in OBJECTIVE_NAMES)
    return tuple(_float(group.rows[0].get(f"target_preference_{index}", 0.0)) for index in range(len(OBJECTIVE_NAMES)))


def _token_values(row: dict[str, object]) -> tuple[float, ...]:
    return tuple(_float(row.get(field, 0.0)) for field in TOKEN_NUMERIC_FIELDS)


def _token(**values: object) -> dict[str, object]:
    row = {field: 0.0 for field in TOKEN_NUMERIC_FIELDS}
    row.update(
        {
            "token_id": values.get("token_id", ""),
            "candidate_id": "",
            "decision_group_id": "",
            "candidate_operation_id": "",
            "candidate_machine_id": "",
            "token_type": values.get("token_type", ""),
            "token_entity_id": values.get("token_entity_id", ""),
        }
    )
    for field in TOKEN_NUMERIC_FIELDS:
        if field in values:
            row[field] = values[field]
    return row


def _remaining_min_processing(job: object, start_index: int) -> float:
    total = 0.0
    for operation in getattr(job, "operations", ())[start_index:]:
        total += min(float(value) for value in operation.processing_time.values())
    return total


def _operation_ready_lower_bound(
    *,
    job: object,
    operation_index: int,
    next_operation_index: int,
    current_job_ready: float,
) -> float:
    if operation_index <= next_operation_index:
        return current_job_ready
    return current_job_ready + _remaining_min_processing(job, next_operation_index) - _remaining_min_processing(job, operation_index)


def _operation_event_pressure(instance: object, operation: object, time_point: float) -> float:
    pressure = 0.0
    for machine_id in operation.eligible_machines:
        pressure += _entity_event_pressure(instance, "machine_breakdown", machine_id, time_point)
    return pressure / max(1, len(operation.eligible_machines))


def _entity_event_pressure(instance: object, event_type: str, entity_id: str, time_point: float) -> float:
    pressure = 0.0
    for event in instance.events:
        if event.event_type != event_type or event.affected_entity != entity_id:
            continue
        start = float(event.event_time)
        end = start + float(event.duration)
        if start <= time_point <= end:
            pressure += 1.0 + float(event.severity)
        else:
            pressure += float(event.severity) / (1.0 + abs(time_point - start))
    return pressure


def _ordered_token(row: dict[str, object]) -> dict[str, object]:
    return {field: row.get(field, "") for field in TOKEN_FIELDS}


def _ordered_trace(row: dict[str, object]) -> dict[str, object]:
    return {field: row.get(field, "") for field in TRACE_FIELDS}


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
