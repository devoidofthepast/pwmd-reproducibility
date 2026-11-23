from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from sklearn.preprocessing import StandardScaler

from su_modfjsp_ha_pc.exact_advantage_learnability import (
    ADVANTAGE_FEATURE_FIELDS,
    AdvantageLearningRow,
)


REGIMES = ("balanced", "makespan", "tardiness", "energy", "path_conflict_delay", "stability")
ABSOLUTE_CONTEXT_FIELDS = (
    "level_l5",
    "mode_realized",
    "target_preference_0",
    "target_preference_1",
    "target_preference_2",
    "target_preference_3",
    "target_preference_4",
    "step_fraction",
    "available_operation_count",
    "candidate_job_due_date",
    "candidate_processing_time_mean",
    "candidate_operation_completion_lower_bound",
    "candidate_job_projected_tardiness_lower_bound",
    "candidate_future_risk_pressure",
)


@dataclass(frozen=True)
class RelationalArtifact:
    model: object
    scaler: StandardScaler
    feature_width: int
    final_loss: float
    random_state: int


def build_relational_rows(
    rows: tuple[AdvantageLearningRow, ...],
) -> tuple[AdvantageLearningRow, ...]:
    groups = _group_indexes(rows)
    absolute_indexes = tuple(ADVANTAGE_FEATURE_FIELDS.index(field) for field in ABSOLUTE_CONTEXT_FIELDS)
    output = []
    for indexes in groups.values():
        group = np.asarray([rows[index].features for index in indexes], dtype=float)
        base_positions = [position for position, index in enumerate(indexes) if rows[index].is_base]
        if len(base_positions) != 1:
            raise ValueError("each decision group must contain exactly one base action")
        base = group[base_positions[0]]
        median = np.median(group, axis=0)
        for position, index in enumerate(indexes):
            row = rows[index]
            scale = advantage_scale(row)
            features = tuple(group[position] - base) + tuple(group[position] - median)
            features += tuple(group[position, absolute_indexes]) + (scale,)
            output.append((index, replace(row, features=tuple(float(value) for value in features))))
    return tuple(row for _index, row in sorted(output))


def advantage_scale(row: AdvantageLearningRow) -> float:
    if abs(row.base_value) > 0.0:
        return max(1.0, abs(row.base_value))
    if row.material_threshold > 1.0:
        return 100.0 * row.material_threshold
    return 1.0


def train_relational_model(
    rows: tuple[AdvantageLearningRow, ...],
    *,
    epochs: int,
    random_state: int,
    hidden_size: int = 64,
) -> RelationalArtifact:
    import torch
    from torch import nn

    if not rows:
        raise ValueError("at least one row is required")
    x = np.asarray([row.features for row in rows], dtype=np.float32)
    scaler = StandardScaler().fit(x)
    x_tensor = torch.tensor(scaler.transform(x), dtype=torch.float32)
    scale = torch.tensor([advantage_scale(row) for row in rows], dtype=torch.float32)
    target = torch.tensor([row.advantage for row in rows], dtype=torch.float32) / scale
    positive = (target > 0.0).float()
    regime = torch.tensor([REGIMES.index(row.regime) for row in rows], dtype=torch.long)
    group_indexes = tuple(tuple(indexes) for indexes in _group_indexes(rows).values())

    torch.manual_seed(random_state)
    model = _RelationalNetwork(x.shape[1], hidden_size, len(REGIMES)).module
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1.0e-4)
    final_loss = 0.0
    for _epoch in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        normalized_advantage, positive_logit = model(x_tensor, regime)
        group_losses = []
        for indexes in group_indexes:
            index = torch.tensor(indexes, dtype=torch.long)
            predicted = normalized_advantage[index]
            actual = target[index]
            regression = torch.nn.functional.smooth_l1_loss(predicted, actual)
            classification = torch.nn.functional.binary_cross_entropy_with_logits(
                positive_logit[index], positive[index]
            )
            differences = actual[:, None] - actual[None, :]
            mask = differences.abs() > 1.0e-9
            if mask.any():
                predicted_differences = predicted[:, None] - predicted[None, :]
                ranking = torch.nn.functional.softplus(
                    -predicted_differences[mask] * torch.sign(differences[mask])
                ).mean()
            else:
                ranking = predicted.sum() * 0.0
            best = torch.argmax(actual).reshape(1)
            best_candidate = torch.nn.functional.cross_entropy(predicted.reshape(1, -1), best)
            group_losses.append(regression + 0.25 * classification + 0.25 * ranking + 0.25 * best_candidate)
        loss = torch.stack(group_losses).mean()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    model.eval()
    return RelationalArtifact(model, scaler, x.shape[1], final_loss, random_state)


def predict_relational_model(
    artifact: RelationalArtifact,
    rows: tuple[AdvantageLearningRow, ...],
) -> tuple[float, ...]:
    import torch

    if any(len(row.features) != artifact.feature_width for row in rows):
        raise ValueError("feature width differs from training artifact")
    x = np.asarray([row.features for row in rows], dtype=np.float32)
    regime = torch.tensor([REGIMES.index(row.regime) for row in rows], dtype=torch.long)
    artifact.model.eval()
    with torch.no_grad():
        normalized, _positive = artifact.model(
            torch.tensor(artifact.scaler.transform(x), dtype=torch.float32), regime
        )
    return tuple(
        float(value) * advantage_scale(row)
        for value, row in zip(normalized.tolist(), rows)
    )


class _RelationalNetwork:
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
                self.advantage_heads = nn.ModuleList(nn.Linear(hidden_size, 1) for _ in range(regimes))
                self.positive_heads = nn.ModuleList(nn.Linear(hidden_size, 1) for _ in range(regimes))

            def forward(self, x, regime):
                embedding = self.encoder(x)
                advantages = torch.cat([head(embedding) for head in self.advantage_heads], dim=1)
                positives = torch.cat([head(embedding) for head in self.positive_heads], dim=1)
                selected_advantage = advantages.gather(1, regime[:, None]).squeeze(1)
                selected_positive = positives.gather(1, regime[:, None]).squeeze(1)
                return selected_advantage, selected_positive

        self.module = Network()


def _group_indexes(rows: tuple[AdvantageLearningRow, ...]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row.group_id, []).append(index)
    return groups
