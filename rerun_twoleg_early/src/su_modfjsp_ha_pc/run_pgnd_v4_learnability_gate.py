from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from su_modfjsp_ha_pc.exact_advantage_learnability import AdvantageLearningRow, evaluate_predictions
from su_modfjsp_ha_pc.exact_advantage_relational import REGIMES
from su_modfjsp_ha_pc.pgnd_v4_data import expected_context_names
from su_modfjsp_ha_pc.pgnd_v4_model import (
    V4Artifact,
    V4Group,
    V4GroupContextNetwork,
    predict_v4_model,
    read_context_groups,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_groups = _load_groups(args.context_dir, range(100, 120))
    selection_groups = _load_groups(args.context_dir, range(120, 125))

    linear_scaler, linear = _fit_pac_linear(train_groups)
    pac_predictions = _predict_pac_linear(selection_groups, linear_scaler, linear)
    metric_rows = _metric_rows(selection_groups)
    pac_report = _evaluate_v4_predictions(selection_groups, metric_rows, pac_predictions)

    member_reports = []
    member_predictions = []
    for random_state in (320, 321, 322, 323, 324):
        artifact = _load_artifact(args.model_dir / f"pgnd_v4_member_{random_state}.pt")
        grouped_predictions = predict_v4_model(artifact, selection_groups)
        predictions = tuple(value for group in grouped_predictions for value in group)
        member_predictions.append(predictions)
        member_reports.append(_evaluate_v4_predictions(selection_groups, metric_rows, predictions))
    ensemble_predictions = tuple(float(value) for value in np.mean(member_predictions, axis=0))
    ensemble_report = _evaluate_v4_predictions(selection_groups, metric_rows, ensemble_predictions)

    auprcs = [report.overall.positive_auprc for report in member_reports]
    regrets = [report.overall.mean_selected_regret for report in member_reports]
    required_regimes = ("tardiness", "makespan", "path_conflict_delay", "stability")
    conditions = {
        "overall_auprc": ensemble_report.overall.positive_auprc > pac_report.overall.positive_auprc,
        "overall_selected_regret": (
            ensemble_report.overall.mean_selected_regret < pac_report.overall.mean_selected_regret
        ),
        "required_regime_regret": all(
            ensemble_report.by_regime[regime].mean_selected_regret
            < pac_report.by_regime[regime].mean_selected_regret
            for regime in required_regimes
        ),
        "nonnegative_regime_tau": all(
            metric.mean_kendall_tau >= 0.0 for metric in ensemble_report.by_regime.values()
        ),
        "restart_auprc_range": max(auprcs) - min(auprcs) <= 0.05,
        "restart_regret_range": max(regrets) - min(regrets) <= 0.10 * pac_report.overall.mean_selected_regret,
    }
    report = {
        "protocol": "PGND-V4",
        "train_seeds": list(range(100, 120)),
        "selection_seeds": list(range(120, 125)),
        "train_groups": len(train_groups),
        "train_rows": sum(len(group.raw_advantage) for group in train_groups),
        "selection_groups": len(selection_groups),
        "selection_rows": len(metric_rows),
        "prediction_units": "normalized_advantage",
        "pac_linear": asdict(pac_report),
        "members": [asdict(member) for member in member_reports],
        "ensemble": asdict(ensemble_report),
        "restart_stability": {
            "auprc_min": min(auprcs),
            "auprc_max": max(auprcs),
            "auprc_range": max(auprcs) - min(auprcs),
            "regret_min": min(regrets),
            "regret_max": max(regrets),
            "regret_range": max(regrets) - min(regrets),
        },
        "conditions": conditions,
        "learnability_gate_passed": all(conditions.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"conditions": conditions, "passed": report["learnability_gate_passed"]}, indent=2))
    return 0


def _load_groups(context_dir: Path, seeds) -> tuple[V4Group, ...]:
    groups = []
    for seed in seeds:
        for name in expected_context_names(seed):
            groups.extend(read_context_groups(context_dir / f"{name}.csv"))
    return tuple(groups)


def _fit_pac_linear(groups: tuple[V4Group, ...]):
    scaler = StandardScaler()
    for group in groups:
        scaler.partial_fit(group.features)
    x = np.concatenate([scaler.transform(group.features).astype(np.float32) for group in groups])
    y = np.concatenate([group.normalized_advantage for group in groups]).astype(np.float32)
    pivot_count = max(1, sum(group.pivot for group in groups))
    nonpivot_count = max(1, len(groups) - pivot_count)
    weights = np.concatenate(
        [
            np.full(
                len(group.raw_advantage),
                0.5 / (pivot_count if group.pivot else nonpivot_count) / len(group.raw_advantage),
                dtype=np.float32,
            )
            for group in groups
        ]
    )
    model = Ridge(alpha=1.0, solver="lsqr").fit(x, y, sample_weight=weights)
    return scaler, model


def _predict_pac_linear(groups, scaler, model) -> tuple[float, ...]:
    output = []
    for group in groups:
        output.extend(float(value) for value in model.predict(scaler.transform(group.features)))
    return tuple(output)


def _metric_rows(groups: tuple[V4Group, ...]) -> tuple[AdvantageLearningRow, ...]:
    rows = []
    for group in groups:
        regime = REGIMES[group.regime]
        for index, advantage in enumerate(group.raw_advantage):
            rows.append(
                AdvantageLearningRow(
                    group.group_id,
                    0,
                    regime,
                    str(index),
                    advantage == 0.0 and index == int(np.argmin(np.abs(group.raw_advantage))),
                    float(advantage),
                    group.material_threshold,
                    (),
                    group.scale,
                )
            )
    return tuple(rows)


def _load_artifact(path: Path) -> V4Artifact:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(payload["scaler_mean"], dtype=float)
    scaler.scale_ = np.asarray(payload["scaler_scale"], dtype=float)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = int(payload["feature_width"])
    model = V4GroupContextNetwork(
        int(payload["feature_width"]), int(payload["hidden_size"]), len(REGIMES)
    ).module
    model.load_state_dict(payload["model_state"])
    return V4Artifact(
        model=model,
        scaler=scaler,
        feature_width=int(payload["feature_width"]),
        final_loss=float(payload["final_loss"]),
        random_state=int(payload["random_state"]),
    )


def _evaluate_v4_predictions(groups, metric_rows, predictions):
    from dataclasses import replace

    report = evaluate_predictions(metric_rows, predictions)
    offsets = []
    offset = 0
    for group in groups:
        width = len(group.raw_advantage)
        offsets.append((offset, offset + width))
        offset += width

    def pivot_metrics(selected_regime: str | None):
        truth = []
        predicted = []
        for group, (start, end) in zip(groups, offsets):
            if selected_regime is not None and REGIMES[group.regime] != selected_regime:
                continue
            truth.append(bool(np.max(group.raw_advantage) > group.material_threshold))
            normalized_threshold = group.material_threshold / group.scale
            predicted.append(bool(np.max(predictions[start:end]) > normalized_threshold))
        return (
            float(precision_score(truth, predicted, zero_division=0)),
            float(recall_score(truth, predicted, zero_division=0)),
        )

    precision, recall = pivot_metrics(None)
    overall = replace(report.overall, pivot_precision=precision, pivot_recall=recall)
    by_regime = {}
    for regime, metric in report.by_regime.items():
        precision, recall = pivot_metrics(regime)
        by_regime[regime] = replace(metric, pivot_precision=precision, pivot_recall=recall)
    return replace(report, overall=overall, by_regime=by_regime)


if __name__ == "__main__":
    raise SystemExit(main())
