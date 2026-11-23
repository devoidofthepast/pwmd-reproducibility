from __future__ import annotations

import csv
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from su_modfjsp_ha_pc.exact_advantage_learnability import (
    ADVANTAGE_FEATURE_FIELDS,
    AdvantageLearningRow,
)
from su_modfjsp_ha_pc.exact_advantage_relational import REGIMES, advantage_scale


RANK_FIELDS = (
    "candidate_processing_time_mean",
    "candidate_operation_completion_lower_bound",
    "candidate_job_tail_completion_lower_bound",
    "candidate_job_projected_tardiness_lower_bound",
    "candidate_job_due_slack_after_tail",
)
TARDINESS_FIELDS = (
    "candidate_job_due_slack_after_tail",
    "candidate_job_projected_tardiness_lower_bound",
    "candidate_job_remaining_processing_after",
    "candidate_job_remaining_ops",
)


@dataclass(frozen=True)
class V4Group:
    group_id: str
    regime: int
    features: np.ndarray
    normalized_advantage: np.ndarray
    raw_advantage: np.ndarray
    scale: float
    material_threshold: float
    pivot: bool
    base_index: int


@dataclass(frozen=True)
class V4Artifact:
    model: object
    scaler: StandardScaler
    feature_width: int
    final_loss: float
    random_state: int


@dataclass(frozen=True)
class V4Predictions:
    normalized_advantage: tuple[tuple[float, ...], ...]
    positive_probability: tuple[tuple[float, ...], ...]
    lower_quantile: tuple[tuple[float, ...], ...]
    embedding: tuple[np.ndarray, ...]


def read_context_groups(path: Path) -> tuple[V4Group, ...]:
    grouped: dict[str, list[AdvantageLearningRow]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = AdvantageLearningRow(
                group_id=raw["group_id"],
                seed=int(raw["seed"]),
                regime=raw["regime"],
                candidate_key=raw["candidate_key"],
                is_base=raw["is_base"].lower() == "true",
                advantage=float(raw["advantage"]),
                material_threshold=float(raw["material_threshold"]),
                features=tuple(float(value) for value in raw["features"].split(";")),
                base_value=float(raw["base_value"]),
            )
            grouped.setdefault(row.group_id, []).append(row)
    return tuple(build_v4_group(tuple(rows)) for rows in grouped.values())


def build_v4_group(rows: tuple[AdvantageLearningRow, ...]) -> V4Group:
    if not rows:
        raise ValueError("group is empty")
    if len({row.group_id for row in rows}) != 1 or len({row.regime for row in rows}) != 1:
        raise ValueError("rows do not form one regime-specific decision group")
    base_indexes = [index for index, row in enumerate(rows) if row.is_base]
    if len(base_indexes) != 1:
        raise ValueError("group must contain exactly one base action")
    raw = np.asarray([row.features for row in rows], dtype=np.float32)
    base = raw[base_indexes[0]]
    median = np.median(raw, axis=0)
    ranks = np.column_stack(
        [_percentile_ranks(raw[:, ADVANTAGE_FEATURE_FIELDS.index(field)]) for field in RANK_FIELDS]
    ).astype(np.float32)
    due_slack = raw[:, ADVANTAGE_FEATURE_FIELDS.index(TARDINESS_FIELDS[0])]
    projected_tardiness = raw[:, ADVANTAGE_FEATURE_FIELDS.index(TARDINESS_FIELDS[1])]
    remaining_work = raw[:, ADVANTAGE_FEATURE_FIELDS.index(TARDINESS_FIELDS[2])]
    remaining_ops = raw[:, ADVANTAGE_FEATURE_FIELDS.index(TARDINESS_FIELDS[3])]
    completion = raw[:, ADVANTAGE_FEATURE_FIELDS.index("candidate_operation_completion_lower_bound")]
    tail_completion = raw[:, ADVANTAGE_FEATURE_FIELDS.index("candidate_job_tail_completion_lower_bound")]
    tardiness = np.column_stack(
        (
            due_slack,
            projected_tardiness,
            remaining_work,
            remaining_ops,
            remaining_work / np.maximum(1.0, np.abs(due_slack)),
            projected_tardiness / np.maximum(1.0, remaining_work),
            completion - completion[base_indexes[0]],
            tail_completion - tail_completion[base_indexes[0]],
            ranks[:, RANK_FIELDS.index("candidate_job_projected_tardiness_lower_bound")],
            ranks[:, RANK_FIELDS.index("candidate_job_due_slack_after_tail")],
        )
    ).astype(np.float32)
    features = np.concatenate((raw, raw - base, raw - median, ranks, tardiness), axis=1).astype(np.float32)
    scale = advantage_scale(rows[0])
    raw_advantage = np.asarray([row.advantage for row in rows], dtype=np.float32)
    threshold = rows[0].material_threshold
    return V4Group(
        group_id=rows[0].group_id,
        regime=REGIMES.index(rows[0].regime),
        features=features,
        normalized_advantage=raw_advantage / scale,
        raw_advantage=raw_advantage,
        scale=scale,
        material_threshold=threshold,
        pivot=bool(np.max(raw_advantage) > threshold),
        base_index=base_indexes[0],
    )


def fit_v4_scaler(groups: tuple[V4Group, ...]) -> StandardScaler:
    if not groups:
        raise ValueError("at least one group is required")
    scaler = StandardScaler()
    for group in groups:
        scaler.partial_fit(group.features)
    return scaler


def train_v4_model(
    groups: tuple[V4Group, ...],
    *,
    epochs: int,
    random_state: int,
    groups_per_batch: int = 64,
    hidden_size: int = 96,
    device: str | None = None,
) -> V4Artifact:
    import torch

    if not groups:
        raise ValueError("at least one group is required")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    scaler = fit_v4_scaler(groups)
    training_groups = tuple(
        replace(group, features=scaler.transform(group.features).astype(np.float32)) for group in groups
    )
    torch.manual_seed(random_state)
    if device == "cuda":
        torch.cuda.manual_seed_all(random_state)
    model = V4GroupContextNetwork(groups[0].features.shape[1], hidden_size, len(REGIMES)).module.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    rng = random.Random(random_state)
    final_loss = 0.0
    indexes = list(range(len(training_groups)))
    for _epoch in range(epochs):
        rng.shuffle(indexes)
        model.train()
        epoch_losses = []
        for start in range(0, len(indexes), groups_per_batch):
            batch = tuple(training_groups[index] for index in indexes[start : start + groups_per_batch])
            tensors = _batch_tensors(batch, scaler=None, device=device)
            optimizer.zero_grad(set_to_none=True)
            predicted, positive_logit, _quantile, _embedding = model(
                tensors["features"], tensors["group_index"], tensors["regime"], len(batch)
            )
            group_index = tensors["group_index"]
            target = tensors["target"]
            group_count = len(batch)
            counts = torch.bincount(group_index, minlength=group_count).to(predicted.dtype)
            regression_rows = torch.nn.functional.smooth_l1_loss(predicted, target, reduction="none")
            classification_rows = torch.nn.functional.binary_cross_entropy_with_logits(
                positive_logit, (target > 0.0).float(), reduction="none"
            )
            regression = torch.zeros(group_count, device=device).index_add_(0, group_index, regression_rows) / counts
            classification = (
                torch.zeros(group_count, device=device).index_add_(0, group_index, classification_rows) / counts
            )

            max_candidates = int(counts.max().item())
            padded_scores = torch.zeros((group_count, max_candidates), device=device)
            padded_target = torch.zeros((group_count, max_candidates), device=device)
            valid = torch.zeros((group_count, max_candidates), dtype=torch.bool, device=device)
            padded_scores[group_index, tensors["local_index"]] = predicted
            padded_target[group_index, tensors["local_index"]] = target
            valid[group_index, tensors["local_index"]] = True
            target_difference = padded_target[:, :, None] - padded_target[:, None, :]
            pair_valid = valid[:, :, None] & valid[:, None, :] & (target_difference.abs() > 1.0e-9)
            score_difference = padded_scores[:, :, None] - padded_scores[:, None, :]
            pair_loss = torch.where(
                pair_valid,
                torch.nn.functional.softplus(-score_difference * torch.sign(target_difference)),
                torch.zeros_like(score_difference),
            )
            ranking = pair_loss.sum(dim=(1, 2)) / pair_valid.sum(dim=(1, 2)).clamp_min(1)
            best = torch.argmax(padded_target.masked_fill(~valid, -torch.inf), dim=1)
            best_loss = torch.nn.functional.cross_entropy(
                padded_scores.masked_fill(~valid, -torch.inf), best, reduction="none"
            )
            losses = regression + 0.25 * classification + 0.50 * ranking + 0.50 * best_loss
            pivot = torch.tensor([group.pivot for group in batch], dtype=torch.bool, device=device)
            parts = []
            if pivot.any():
                parts.append(losses[pivot].mean())
            if (~pivot).any():
                parts.append(losses[~pivot].mean())
            loss = torch.stack(parts).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        final_loss = float(np.mean(epoch_losses))
    model.eval()
    return V4Artifact(model, scaler, groups[0].features.shape[1], final_loss, random_state)


def predict_v4_model(
    artifact: V4Artifact,
    groups: tuple[V4Group, ...],
    *,
    groups_per_batch: int = 128,
    device: str | None = None,
    raw_units: bool = False,
) -> tuple[tuple[float, ...], ...]:
    predictions = predict_v4_details(
        artifact,
        groups,
        groups_per_batch=groups_per_batch,
        device=device,
    ).normalized_advantage
    if not raw_units:
        return predictions
    return tuple(
        tuple(value * group.scale for value in group_predictions)
        for group, group_predictions in zip(groups, predictions)
    )


def predict_v4_details(
    artifact: V4Artifact,
    groups: tuple[V4Group, ...],
    *,
    groups_per_batch: int = 128,
    device: str | None = None,
) -> V4Predictions:
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    artifact.model.to(device).eval()
    advantages = []
    probabilities = []
    quantiles = []
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(groups), groups_per_batch):
            batch = groups[start : start + groups_per_batch]
            tensors = _batch_tensors(batch, scaler=artifact.scaler, device=device)
            normalized, positive, quantile, embedding = artifact.model(
                tensors["features"], tensors["group_index"], tensors["regime"], len(batch)
            )
            offset = 0
            values = normalized.cpu().numpy()
            probability_values = torch.sigmoid(positive).cpu().numpy()
            quantile_values = quantile.cpu().numpy()
            embedding_values = embedding.cpu().numpy()
            for group in batch:
                width = len(group.raw_advantage)
                advantages.append(tuple(float(value) for value in values[offset : offset + width]))
                probabilities.append(
                    tuple(float(value) for value in probability_values[offset : offset + width])
                )
                quantiles.append(
                    tuple(float(value) for value in quantile_values[offset : offset + width])
                )
                embeddings.append(embedding_values[offset : offset + width].copy())
                offset += width
    return V4Predictions(tuple(advantages), tuple(probabilities), tuple(quantiles), tuple(embeddings))


class V4GroupContextNetwork:
    def __init__(self, feature_width: int, hidden_size: int, regimes: int) -> None:
        import torch
        from torch import nn

        class Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(feature_width, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                )
                self.regime_embedding = nn.Embedding(regimes, 16)
                self.contextual = nn.Sequential(
                    nn.Linear(hidden_size * 3 + 16, hidden_size),
                    nn.GELU(),
                )
                self.advantage_heads = nn.ModuleList(nn.Linear(hidden_size, 1) for _ in range(regimes))
                self.positive_heads = nn.ModuleList(nn.Linear(hidden_size, 1) for _ in range(regimes))
                self.quantile_heads = nn.ModuleList(nn.Linear(hidden_size, 1) for _ in range(regimes))

            def forward(self, features, group_index, regime, group_count):
                embedding = self.encoder(features)
                sums = torch.zeros(group_count, embedding.shape[1], device=embedding.device)
                sums.index_add_(0, group_index, embedding)
                counts = torch.bincount(group_index, minlength=group_count).clamp_min(1).to(embedding.dtype)
                means = sums / counts[:, None]
                maxima = torch.full_like(means, -torch.inf)
                maxima.scatter_reduce_(
                    0, group_index[:, None].expand_as(embedding), embedding, reduce="amax", include_self=True
                )
                contextual = self.contextual(
                    torch.cat(
                        (embedding, means[group_index], maxima[group_index], self.regime_embedding(regime)), dim=1
                    )
                )
                advantage = torch.cat([head(contextual) for head in self.advantage_heads], dim=1)
                positive = torch.cat([head(contextual) for head in self.positive_heads], dim=1)
                quantile = torch.cat([head(contextual) for head in self.quantile_heads], dim=1)
                selector = regime[:, None]
                return (
                    advantage.gather(1, selector).squeeze(1),
                    positive.gather(1, selector).squeeze(1),
                    quantile.gather(1, selector).squeeze(1),
                    contextual,
                )

        self.module = Network()


def _batch_tensors(
    groups: tuple[V4Group, ...], *, scaler: StandardScaler | None, device: str
) -> dict[str, object]:
    import torch

    features = np.concatenate(
        [
            group.features if scaler is None else scaler.transform(group.features).astype(np.float32)
            for group in groups
        ]
    )
    target = np.concatenate([group.normalized_advantage for group in groups]).astype(np.float32)
    group_index = np.concatenate(
        [np.full(len(group.raw_advantage), index, dtype=np.int64) for index, group in enumerate(groups)]
    )
    local_index = np.concatenate(
        [np.arange(len(group.raw_advantage), dtype=np.int64) for group in groups]
    )
    regime = np.concatenate(
        [np.full(len(group.raw_advantage), group.regime, dtype=np.int64) for group in groups]
    )
    return {
        "features": torch.tensor(features, device=device),
        "target": torch.tensor(target, device=device),
        "group_index": torch.tensor(group_index, device=device),
        "local_index": torch.tensor(local_index, device=device),
        "regime": torch.tensor(regime, device=device),
    }


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks / max(1, len(values) - 1)
