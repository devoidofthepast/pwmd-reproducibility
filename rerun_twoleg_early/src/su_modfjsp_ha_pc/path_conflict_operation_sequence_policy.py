from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from su_modfjsp_ha_pc.offline_logs import write_csv
from su_modfjsp_ha_pc.path_conflict_neural_dataset import CANDIDATE_FEATURE_FIELDS


GROUP_FEATURE_FIELDS = (
    "group_candidate_order_fraction",
    "group_ready_rank",
    "group_due_slack_rank",
    "group_processing_rank",
    "group_completion_rank",
    "group_remaining_ops_rank",
    "group_ready_delta_norm",
    "group_due_slack_delta_norm",
    "group_processing_delta_norm",
    "group_completion_delta_norm",
)

OPERATION_SEQUENCE_FEATURE_FIELDS = CANDIDATE_FEATURE_FIELDS + GROUP_FEATURE_FIELDS

SUMMARY_FIELDS = (
    "split",
    "groups",
    "candidate_rows",
    "feature_count",
    "epochs",
    "hidden_size",
    "attention_heads",
    "layers",
    "final_loss",
    "top1_accuracy",
    "random_top1_baseline",
    "mean_selected_rank",
)

PREDICTION_FIELDS = (
    "candidate_id",
    "split",
    "decision_group_id",
    "candidate_operation_id",
    "selected_action",
    "predicted_score",
    "predicted_future_regret",
    "regret_aware_score",
    "group_rank",
    "group_top1_selected",
)


@dataclass(frozen=True)
class Standardizer:
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def transform_one(self, values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple((value - mean) / scale for value, mean, scale in zip(values, self.means, self.scales))


@dataclass(frozen=True)
class OperationGroup:
    group_id: str
    rows: tuple[dict[str, object], ...]
    feature_rows: tuple[tuple[float, ...], ...]
    selected_index: int
    regret_targets: tuple[float, ...]
    regret_label_available: bool


def train_and_evaluate_operation_sequence_policy(
    dataset_path: Path,
    *,
    epochs: int = 220,
    hidden_size: int = 96,
    attention_heads: int = 4,
    layers: int = 2,
    random_state: int = 106,
) -> dict[str, tuple[dict[str, object], ...]]:
    rows = read_candidate_rows(dataset_path)
    operation_rows = tuple(row for row in rows if row.get("stage") == "operation")
    train_rows = tuple(row for row in operation_rows if row.get("split") == "train")
    test_rows = tuple(row for row in operation_rows if row.get("split") == "test")
    if not train_rows:
        raise ValueError("operation sequence dataset must contain at least one train row")
    if not test_rows:
        raise ValueError("operation sequence dataset must contain at least one test row")

    train_groups = build_operation_groups(train_rows)
    test_groups = build_operation_groups(test_rows)
    standardizer = fit_standardizer(tuple(feature for group in train_groups for feature in group.feature_rows))
    model, final_loss = _train_sequence_ranker(
        train_groups,
        standardizer=standardizer,
        epochs=epochs,
        hidden_size=hidden_size,
        attention_heads=attention_heads,
        layers=layers,
        random_state=random_state,
    )

    summary_rows = []
    prediction_rows = []
    for split, groups in (("train", train_groups), ("test", test_groups)):
        scored = _score_groups(model, groups, standardizer=standardizer)
        prediction_rows.extend(_prediction_rows(scored))
        summary_rows.append(
            _summary_row(
                split,
                scored,
                feature_count=len(OPERATION_SEQUENCE_FEATURE_FIELDS),
                epochs=epochs,
                hidden_size=hidden_size,
                attention_heads=attention_heads,
                layers=layers,
                final_loss=final_loss,
            )
        )
    return {
        "summary": tuple(_ordered_summary(row) for row in summary_rows),
        "predictions": tuple(_ordered_prediction(row) for row in prediction_rows),
    }


def read_candidate_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(csv.DictReader(handle))


def build_operation_groups(rows: tuple[dict[str, object], ...]) -> tuple[OperationGroup, ...]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["decision_group_id"])].append(row)
    groups = []
    for group_id, group_rows in sorted(grouped.items()):
        sorted_rows = tuple(sorted(group_rows, key=lambda row: str(row.get("candidate_operation_id", ""))))
        selected = tuple(index for index, row in enumerate(sorted_rows) if _int(row["selected_action"]) == 1)
        if len(selected) != 1:
            raise ValueError(f"operation group {group_id} must contain exactly one selected action; found {len(selected)}")
        feature_rows = tuple(_operation_sequence_features(row, index, sorted_rows) for index, row in enumerate(sorted_rows))
        regret_targets, regret_label_available = _future_regret_targets(sorted_rows)
        groups.append(OperationGroup(group_id, sorted_rows, feature_rows, selected[0], regret_targets, regret_label_available))
    return tuple(groups)


def fit_standardizer(rows: tuple[tuple[float, ...], ...]) -> Standardizer:
    if not rows:
        raise ValueError("cannot fit standardizer with no feature rows")
    width = len(rows[0])
    means = []
    scales = []
    for column in range(width):
        values = [row[column] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scale = variance ** 0.5
        means.append(mean)
        scales.append(scale if scale > 1e-9 else 1.0)
    return Standardizer(tuple(means), tuple(scales))


def write_operation_sequence_outputs(out_dir: Path, rows: dict[str, tuple[dict[str, object], ...]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "e106_operation_sequence_policy_summary.csv", rows["summary"])
    write_csv(out_dir / "e106_operation_sequence_policy_predictions.csv", rows["predictions"])


def _operation_sequence_features(
    row: dict[str, object],
    index: int,
    group_rows: tuple[dict[str, object], ...],
) -> tuple[float, ...]:
    base = tuple(_float(row.get(name, 0.0)) for name in CANDIDATE_FEATURE_FIELDS)
    completion_values = tuple(
        _float(item.get("candidate_machine_start_lower_bound", 0.0)) + _float(item.get("candidate_processing_time", 0.0))
        for item in group_rows
    )
    ready_values = tuple(_float(item.get("candidate_job_ready_time", 0.0)) for item in group_rows)
    slack_values = tuple(_float(item.get("candidate_job_due_slack_at_ready", 0.0)) for item in group_rows)
    processing_values = tuple(_float(item.get("candidate_processing_time_mean", 0.0)) for item in group_rows)
    remaining_values = tuple(_float(item.get("candidate_job_remaining_ops", 0.0)) for item in group_rows)
    own_completion = _float(row.get("candidate_machine_start_lower_bound", 0.0)) + _float(
        row.get("candidate_processing_time", 0.0)
    )
    group_features = (
        index / max(1, len(group_rows) - 1),
        _rank_fraction(ready_values, _float(row.get("candidate_job_ready_time", 0.0))),
        _rank_fraction(slack_values, _float(row.get("candidate_job_due_slack_at_ready", 0.0))),
        _rank_fraction(processing_values, _float(row.get("candidate_processing_time_mean", 0.0))),
        _rank_fraction(completion_values, own_completion),
        _rank_fraction(remaining_values, _float(row.get("candidate_job_remaining_ops", 0.0))),
        _range_delta(ready_values, _float(row.get("candidate_job_ready_time", 0.0))),
        _range_delta(slack_values, _float(row.get("candidate_job_due_slack_at_ready", 0.0))),
        _range_delta(processing_values, _float(row.get("candidate_processing_time_mean", 0.0))),
        _range_delta(completion_values, own_completion),
    )
    return base + group_features


def _future_regret_targets(group_rows: tuple[dict[str, object], ...]) -> tuple[tuple[float, ...], bool]:
    available = any(_float(row.get("future_regret_label_available", 0.0)) > 0.0 for row in group_rows)
    if not available:
        return tuple(0.0 for _ in group_rows), False
    values = tuple(_float(row.get("future_weighted_completion_regret", 0.0)) for row in group_rows)
    high = max(values)
    if high <= 1.0e-9:
        return tuple(0.0 for _ in group_rows), True
    return tuple(value / high for value in values), True


def _train_sequence_ranker(
    groups: tuple[OperationGroup, ...],
    *,
    standardizer: Standardizer,
    epochs: int,
    hidden_size: int,
    attention_heads: int,
    layers: int,
    random_state: int,
) -> tuple[object, float]:
    import torch
    from torch import nn

    torch.manual_seed(random_state)
    x, valid_mask, labels, regret_targets, regret_mask = _group_tensors(groups, standardizer)
    model = _OperationSequenceRanker(
        input_size=len(OPERATION_SEQUENCE_FEATURE_FIELDS),
        hidden_size=hidden_size,
        attention_heads=attention_heads,
        layers=layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    regret_loss_fn = nn.SmoothL1Loss(reduction="none")
    final_loss = 0.0
    for _ in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        scores, predicted_regrets = model(x, valid_mask)
        loss = loss_fn(scores, labels)
        if bool(regret_mask.any()):
            regret_loss = regret_loss_fn(predicted_regrets, regret_targets)
            masked_regret_loss = regret_loss.masked_select(regret_mask).mean()
            loss = loss + 0.35 * masked_regret_loss
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    return model, final_loss


class _OperationSequenceRanker:
    def __init__(self, *, input_size: int, hidden_size: int, attention_heads: int, layers: int) -> None:
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
                self.score = nn.Linear(hidden_size, 1)
                self.regret = nn.Linear(hidden_size, 1)

            def forward(self, tensor, valid_mask):
                hidden = self.input(tensor)
                encoded = self.encoder(hidden, src_key_padding_mask=~valid_mask)
                scores = self.score(encoded).squeeze(-1)
                regrets = nn.functional.softplus(self.regret(encoded).squeeze(-1))
                return scores.masked_fill(~valid_mask, -1.0e9), regrets.masked_fill(~valid_mask, 0.0)

        self.network = Network()

    def parameters(self):
        return self.network.parameters()

    def train(self) -> None:
        self.network.train()

    def eval(self) -> None:
        self.network.eval()

    def __call__(self, tensor, valid_mask):
        return self.network(tensor, valid_mask)


def _group_tensors(groups: tuple[OperationGroup, ...], standardizer: Standardizer):
    import torch

    max_len = max(len(group.rows) for group in groups)
    width = len(OPERATION_SEQUENCE_FEATURE_FIELDS)
    x = torch.zeros((len(groups), max_len, width), dtype=torch.float32)
    valid_mask = torch.zeros((len(groups), max_len), dtype=torch.bool)
    labels = torch.zeros((len(groups),), dtype=torch.long)
    regret_targets = torch.zeros((len(groups), max_len), dtype=torch.float32)
    regret_mask = torch.zeros((len(groups), max_len), dtype=torch.bool)
    for group_index, group in enumerate(groups):
        labels[group_index] = group.selected_index
        for row_index, feature_row in enumerate(group.feature_rows):
            x[group_index, row_index] = torch.tensor(standardizer.transform_one(feature_row), dtype=torch.float32)
            valid_mask[group_index, row_index] = True
            regret_targets[group_index, row_index] = float(group.regret_targets[row_index])
            regret_mask[group_index, row_index] = bool(group.regret_label_available)
    return x, valid_mask, labels, regret_targets, regret_mask


def _score_groups(
    model: object,
    groups: tuple[OperationGroup, ...],
    *,
    standardizer: Standardizer,
) -> tuple[tuple[dict[str, object], ...], ...]:
    import torch

    model.eval()
    x, valid_mask, _labels, _regret_targets, _regret_mask = _group_tensors(groups, standardizer)
    with torch.no_grad():
        scores, predicted_regrets = model(x, valid_mask)
    scored_groups = []
    for group_index, group in enumerate(groups):
        scored = []
        group_scores = tuple(float(scores[group_index, row_index].item()) for row_index in range(len(group.rows)))
        group_regrets = tuple(
            float(predicted_regrets[group_index, row_index].item()) for row_index in range(len(group.rows))
        )
        for row_index, row in enumerate(group.rows):
            predicted_score = group_scores[row_index]
            predicted_regret = group_regrets[row_index]
            score_norm = _range_delta(group_scores, predicted_score)
            regret_norm = _range_delta(group_regrets, predicted_regret)
            scored.append(
                {
                    **row,
                    "predicted_score": predicted_score,
                    "predicted_future_regret": predicted_regret,
                    "regret_aware_score": score_norm - 0.75 * regret_norm,
                }
            )
        scored_groups.append(tuple(scored))
    return tuple(scored_groups)


def _prediction_rows(scored_groups: tuple[tuple[dict[str, object], ...], ...]) -> tuple[dict[str, object], ...]:
    rows = []
    for group in scored_groups:
        ranked = sorted(group, key=lambda row: float(row["predicted_score"]), reverse=True)
        top1_selected = _int(ranked[0]["selected_action"]) == 1
        for rank, row in enumerate(ranked, start=1):
            rows.append(
                {
                    "candidate_id": row.get("candidate_id", ""),
                    "split": row.get("split", ""),
                    "decision_group_id": row.get("decision_group_id", ""),
                    "candidate_operation_id": row.get("candidate_operation_id", ""),
                    "selected_action": row.get("selected_action", ""),
                    "predicted_score": round(float(row["predicted_score"]), 10),
                    "predicted_future_regret": round(float(row.get("predicted_future_regret", 0.0)), 10),
                    "regret_aware_score": round(float(row.get("regret_aware_score", row["predicted_score"])), 10),
                    "group_rank": rank,
                    "group_top1_selected": 1 if top1_selected else 0,
                }
            )
    return tuple(rows)


def _summary_row(
    split: str,
    scored_groups: tuple[tuple[dict[str, object], ...], ...],
    *,
    feature_count: int,
    epochs: int,
    hidden_size: int,
    attention_heads: int,
    layers: int,
    final_loss: float,
) -> dict[str, object]:
    candidate_rows = sum(len(group) for group in scored_groups)
    top1 = 0
    selected_ranks = []
    random_baseline_sum = 0.0
    for group in scored_groups:
        ranked = sorted(group, key=lambda row: float(row["predicted_score"]), reverse=True)
        random_baseline_sum += 1.0 / len(group)
        for rank, row in enumerate(ranked, start=1):
            if _int(row["selected_action"]) == 1:
                selected_ranks.append(rank)
                if rank == 1:
                    top1 += 1
                break
    groups = len(scored_groups)
    return {
        "split": split,
        "groups": groups,
        "candidate_rows": candidate_rows,
        "feature_count": feature_count,
        "epochs": epochs,
        "hidden_size": hidden_size,
        "attention_heads": attention_heads,
        "layers": layers,
        "final_loss": round(final_loss, 10),
        "top1_accuracy": round(top1 / groups, 10) if groups else 0.0,
        "random_top1_baseline": round(random_baseline_sum / groups, 10) if groups else 0.0,
        "mean_selected_rank": round(sum(selected_ranks) / len(selected_ranks), 10) if selected_ranks else 0.0,
    }


def _rank_fraction(values: tuple[float, ...], value: float) -> float:
    if len(values) <= 1:
        return 0.0
    sorted_values = sorted(values)
    rank = sorted_values.index(value)
    return rank / (len(values) - 1)


def _range_delta(values: tuple[float, ...], value: float) -> float:
    if not values:
        return 0.0
    low = min(values)
    high = max(values)
    width = high - low
    if width <= 1e-9:
        return 0.0
    return (value - low) / width


def _ordered_summary(row: dict[str, object]) -> dict[str, object]:
    return {field: row.get(field, "") for field in SUMMARY_FIELDS}


def _ordered_prediction(row: dict[str, object]) -> dict[str, object]:
    return {field: row.get(field, "") for field in PREDICTION_FIELDS}


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train an operation-level graph/sequence ranking policy.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=106)
    args = parser.parse_args(argv)

    rows = train_and_evaluate_operation_sequence_policy(
        args.dataset,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        attention_heads=args.attention_heads,
        layers=args.layers,
        random_state=args.random_state,
    )
    write_operation_sequence_outputs(args.out_dir, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
