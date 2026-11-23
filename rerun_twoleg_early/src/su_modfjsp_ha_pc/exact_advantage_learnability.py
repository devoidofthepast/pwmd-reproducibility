from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import mean

import numpy as np

from su_modfjsp_ha_pc.oracle_value_policy import OBJECTIVE_NAMES
from su_modfjsp_ha_pc.objective_contract import relative_material_threshold
from su_modfjsp_ha_pc.path_conflict_operation_sequence_policy import (
    OPERATION_SEQUENCE_FEATURE_FIELDS,
    _operation_sequence_features,
)


ADVANTAGE_FEATURE_FIELDS = tuple(
    field for field in OPERATION_SEQUENCE_FEATURE_FIELDS if field != "seed_scaled"
)
ABSOLUTE_RESOURCE_FEATURE_FIELDS = (
    "candidate_job_release_time",
    "candidate_job_due_date",
    "candidate_job_ready_time",
    "candidate_job_due_slack_at_ready",
    "candidate_processing_time_min",
    "candidate_processing_time_mean",
    "candidate_processing_time_max",
    "candidate_machine_energy_rate",
    "candidate_machine_available_time",
    "candidate_processing_time",
    "candidate_processing_energy",
    "candidate_machine_start_lower_bound",
    "candidate_agv_available_time",
    "candidate_operation_completion_lower_bound",
    "candidate_job_remaining_processing_after",
    "candidate_job_tail_completion_lower_bound",
    "candidate_job_due_slack_after_tail",
    "candidate_job_projected_tardiness_lower_bound",
    "candidate_machine_queue_delay",
    "candidate_transport_queue_delay",
    "candidate_completion_regret_to_group_best",
    "candidate_tail_regret_to_group_best",
)
_ADVANTAGE_FEATURE_INDEXES = tuple(
    index for index, field in enumerate(OPERATION_SEQUENCE_FEATURE_FIELDS) if field != "seed_scaled"
)


@dataclass(frozen=True)
class AdvantageLearningRow:
    group_id: str
    seed: int
    regime: str
    candidate_key: str
    is_base: bool
    advantage: float
    material_threshold: float
    features: tuple[float, ...]
    base_value: float = 0.0

    @property
    def positive(self) -> bool:
        return self.advantage > 0.0


@dataclass(frozen=True)
class MetricSlice:
    rows: int
    groups: int
    positive_precision: float
    positive_recall: float
    positive_auprc: float
    specificity: float
    pivot_precision: float
    pivot_recall: float
    top1_accuracy: float
    mean_kendall_tau: float
    mean_selected_regret: float


@dataclass(frozen=True)
class PredictionReport:
    overall: MetricSlice
    by_regime: dict[str, MetricSlice]

    @property
    def positive_recall(self) -> float:
        return self.overall.positive_recall

    @property
    def specificity(self) -> float:
        return self.overall.specificity

    @property
    def mean_selected_regret(self) -> float:
        return self.overall.mean_selected_regret


@dataclass(frozen=True)
class PacLinearArtifact:
    scaler: StandardScaler
    model: Ridge
    feature_width: int


def transform_advantage_features(
    rows: tuple[AdvantageLearningRow, ...],
    *,
    variant: str,
) -> tuple[AdvantageLearningRow, ...]:
    if variant == "baseline":
        return rows
    indexes = tuple(ADVANTAGE_FEATURE_FIELDS.index(name) for name in ABSOLUTE_RESOURCE_FEATURE_FIELDS)
    if variant == "drop_absolute_resource":
        dropped = set(indexes)
        return tuple(
            replace(row, features=tuple(value for index, value in enumerate(row.features) if index not in dropped))
            for row in rows
        )
    if variant != "group_relative_resource":
        raise ValueError(f"unknown feature variant: {variant}")
    transformed = []
    for group_indexes in _group_indexes(rows).values():
        group = np.asarray([rows[index].features for index in group_indexes], dtype=float)
        means = np.mean(group[:, indexes], axis=0)
        scales = np.std(group[:, indexes], axis=0)
        scales[scales <= 1.0e-12] = 1.0
        relative = (group[:, indexes] - means) / scales
        for local_index, row_index in enumerate(group_indexes):
            features = list(rows[row_index].features)
            for column, feature_index in enumerate(indexes):
                features[feature_index] = float(relative[local_index, column])
            transformed.append((row_index, replace(rows[row_index], features=tuple(features))))
    return tuple(row for _index, row in sorted(transformed))


def summarize_label_distribution(
    rows: tuple[AdvantageLearningRow, ...],
) -> dict[str, dict[str, float | int]]:
    if not rows:
        raise ValueError("at least one row is required")
    output = {"overall": _label_distribution_slice(rows)}
    for regime in sorted({row.regime for row in rows}):
        output[regime] = _label_distribution_slice(tuple(row for row in rows if row.regime == regime))
    return output


def summarize_feature_drift(
    train_rows: tuple[AdvantageLearningRow, ...],
    calibration_rows: tuple[AdvantageLearningRow, ...],
    *,
    feature_names: tuple[str, ...] = ADVANTAGE_FEATURE_FIELDS,
) -> tuple[dict[str, float | str], ...]:
    _validate_feature_width(train_rows)
    _validate_feature_width(calibration_rows, expected=len(train_rows[0].features))
    if len(feature_names) != len(train_rows[0].features):
        raise ValueError("feature name count differs from feature width")
    train = np.asarray([row.features for row in train_rows], dtype=float)
    calibration = np.asarray([row.features for row in calibration_rows], dtype=float)
    output = []
    for index, name in enumerate(feature_names):
        train_values = train[:, index]
        calibration_values = calibration[:, index]
        pooled_scale = float(np.sqrt((np.var(train_values) + np.var(calibration_values)) / 2.0))
        mean_shift = float(np.mean(calibration_values) - np.mean(train_values))
        standardized_mean_difference = mean_shift / pooled_scale if pooled_scale > 1.0e-12 else 0.0
        train_min = float(np.min(train_values))
        train_max = float(np.max(train_values))
        outside_train_range = float(np.mean((calibration_values < train_min) | (calibration_values > train_max)))
        output.append(
            {
                "feature": name,
                "train_mean": float(np.mean(train_values)),
                "calibration_mean": float(np.mean(calibration_values)),
                "standardized_mean_difference": standardized_mean_difference,
                "calibration_outside_train_range_rate": outside_train_range,
            }
        )
    return tuple(
        sorted(
            output,
            key=lambda row: (
                abs(float(row["standardized_mean_difference"])),
                float(row["calibration_outside_train_range_rate"]),
            ),
            reverse=True,
        )
    )


def build_learning_rows_from_labeled_candidates(
    candidate_rows: tuple[dict[str, object], ...],
    *,
    material_threshold_mode: str = "legacy",
) -> tuple[AdvantageLearningRow, ...]:
    if material_threshold_mode not in {"legacy", "contract-relative"}:
        raise ValueError("material_threshold_mode must be legacy or contract-relative")
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in candidate_rows:
        if float(row.get("future_value_label_available", 0.0) or 0.0) <= 0.0:
            continue
        grouped.setdefault(str(row.get("decision_group_id", "")), []).append(row)
    output = []
    for group_id, values in sorted(grouped.items()):
        sorted_rows = tuple(
            sorted(
                values,
                key=lambda row: (
                    str(row.get("candidate_operation_id", "")),
                    str(row.get("candidate_machine_id", "")),
                ),
            )
        )
        base_rows = tuple(row for row in sorted_rows if float(row.get("selected_action", 0.0) or 0.0) > 0.0)
        if len(base_rows) != 1:
            raise ValueError(f"group {group_id!r} must contain exactly one selected base action")
        base_value = float(base_rows[0].get("future_weighted_value_after_action", 0.0) or 0.0)
        threshold = (
            relative_material_threshold(base_value)
            if material_threshold_mode == "contract-relative"
            else max(1.0, 0.01 * base_value)
        )
        regime = _regime_from_row(sorted_rows[0])
        seed = int(float(sorted_rows[0].get("seed", 0) or 0))
        for index, row in enumerate(sorted_rows):
            terminal_value = float(row.get("future_weighted_value_after_action", 0.0) or 0.0)
            is_base = row is base_rows[0]
            advantage = 0.0 if is_base else base_value - terminal_value
            features = _operation_sequence_features(row, index, sorted_rows)
            output.append(
                AdvantageLearningRow(
                    group_id=group_id,
                    seed=seed,
                    regime=regime,
                    candidate_key="|".join(
                        (
                            str(row.get("candidate_operation_id", "")),
                            str(row.get("candidate_machine_id", "")),
                            str(row.get("candidate_agv_id", "")),
                            str(row.get("candidate_route_edges", "")),
                        )
                    ),
                    is_base=is_base,
                    advantage=advantage,
                    material_threshold=threshold,
                    features=tuple(
                        float(features[feature_index])
                        for feature_index in _ADVANTAGE_FEATURE_INDEXES
                    ),
                    base_value=base_value,
                )
            )
    return tuple(output)


def build_inference_rows_from_candidates(
    candidate_rows: tuple[dict[str, object], ...],
    *,
    base_operation_id: str,
    base_machine_id: str,
) -> tuple[AdvantageLearningRow, ...]:
    if not candidate_rows:
        raise ValueError("at least one candidate row is required")
    group_ids = {str(row.get("decision_group_id", "")) for row in candidate_rows}
    if len(group_ids) != 1:
        raise ValueError("inference rows must contain one decision group")
    sorted_rows = tuple(
        sorted(
            candidate_rows,
            key=lambda row: (
                str(row.get("candidate_operation_id", "")),
                str(row.get("candidate_machine_id", "")),
            ),
        )
    )
    base_matches = [
        row
        for row in sorted_rows
        if str(row.get("candidate_operation_id", "")) == base_operation_id
        and str(row.get("candidate_machine_id", "")) == base_machine_id
    ]
    if len(base_matches) != 1:
        raise ValueError("exactly one inference candidate must match the base action")
    regime = _regime_from_row(sorted_rows[0])
    seed = int(float(sorted_rows[0].get("seed", 0) or 0))
    output = []
    for index, row in enumerate(sorted_rows):
        features = _operation_sequence_features(row, index, sorted_rows)
        is_base = row is base_matches[0]
        output.append(
            AdvantageLearningRow(
                group_id=next(iter(group_ids)),
                seed=seed,
                regime=regime,
                candidate_key="|".join(
                    (
                        str(row.get("candidate_operation_id", "")),
                        str(row.get("candidate_machine_id", "")),
                        str(row.get("candidate_agv_id", "")),
                        str(row.get("candidate_route_edges", "")),
                    )
                ),
                is_base=is_base,
                advantage=0.0,
                material_threshold=0.0,
                features=tuple(
                    float(features[feature_index]) for feature_index in _ADVANTAGE_FEATURE_INDEXES
                ),
                base_value=0.0,
            )
        )
    return tuple(output)


def split_by_seed_and_group(
    rows: tuple[AdvantageLearningRow, ...],
    *,
    train_seeds: tuple[int, ...],
    calibration_seeds: tuple[int, ...],
) -> dict[str, tuple[AdvantageLearningRow, ...]]:
    train_set = set(train_seeds)
    calibration_set = set(calibration_seeds)
    if train_set & calibration_set:
        raise ValueError("train and calibration seeds overlap")
    groups: dict[str, list[AdvantageLearningRow]] = {}
    for row in rows:
        groups.setdefault(row.group_id, []).append(row)
    split_rows: dict[str, list[AdvantageLearningRow]] = {"train": [], "calibration": [], "unused": []}
    for group_rows in groups.values():
        seeds = {row.seed for row in group_rows}
        if len(seeds) != 1:
            raise ValueError("one decision group spans multiple seeds")
        seed = next(iter(seeds))
        split = "train" if seed in train_set else "calibration" if seed in calibration_set else "unused"
        split_rows[split].extend(group_rows)
    return {key: tuple(value) for key, value in split_rows.items()}


def group_locations(splits: dict[str, tuple[AdvantageLearningRow, ...]]) -> dict[str, set[str]]:
    locations: dict[str, set[str]] = {}
    for split, rows in splits.items():
        for row in rows:
            locations.setdefault(row.group_id, set()).add(split)
    return locations


def evaluate_majority_nonpositive(rows: tuple[AdvantageLearningRow, ...]) -> PredictionReport:
    return evaluate_predictions(rows, tuple(0.0 for _row in rows))


def evaluate_predictions(
    rows: tuple[AdvantageLearningRow, ...],
    predictions: tuple[float, ...],
) -> PredictionReport:
    if len(rows) != len(predictions):
        raise ValueError("row and prediction counts differ")
    if not rows:
        raise ValueError("at least one row is required")
    overall = _metric_slice(rows, predictions)
    regimes = sorted({row.regime for row in rows})
    by_regime = {}
    for regime in regimes:
        indexes = tuple(index for index, row in enumerate(rows) if row.regime == regime)
        by_regime[regime] = _metric_slice(
            tuple(rows[index] for index in indexes),
            tuple(predictions[index] for index in indexes),
        )
    return PredictionReport(overall=overall, by_regime=by_regime)


def fit_pac_linear(rows: tuple[AdvantageLearningRow, ...], *, alpha: float = 1.0) -> PacLinearArtifact:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    _validate_feature_width(rows)
    x = np.asarray([row.features for row in rows], dtype=float)
    y = np.asarray([row.advantage for row in rows], dtype=float)
    weights = _balanced_group_weights(rows)
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=alpha).fit(scaler.transform(x), y, sample_weight=weights)
    return PacLinearArtifact(scaler=scaler, model=model, feature_width=x.shape[1])


def predict_pac_linear(
    artifact: PacLinearArtifact,
    rows: tuple[AdvantageLearningRow, ...],
) -> tuple[float, ...]:
    _validate_feature_width(rows, expected=artifact.feature_width)
    x = np.asarray([row.features for row in rows], dtype=float)
    return tuple(float(value) for value in artifact.model.predict(artifact.scaler.transform(x)))


def _metric_slice(rows: tuple[AdvantageLearningRow, ...], predictions: tuple[float, ...]) -> MetricSlice:
    from scipy.stats import kendalltau
    from sklearn.metrics import average_precision_score, precision_score, recall_score

    truth = np.asarray([row.positive for row in rows], dtype=int)
    predicted_positive = np.asarray([value > 0.0 for value in predictions], dtype=int)
    positive_precision = float(precision_score(truth, predicted_positive, zero_division=0))
    positive_recall = float(recall_score(truth, predicted_positive, zero_division=0))
    specificity = float(recall_score(1 - truth, 1 - predicted_positive, zero_division=0))
    positive_auprc = float(average_precision_score(truth, np.asarray(predictions))) if truth.any() else 0.0

    grouped = _group_indexes(rows)
    actual_pivot = []
    predicted_pivot = []
    top1 = []
    taus = []
    regrets = []
    for indexes in grouped.values():
        actual = np.asarray([rows[index].advantage for index in indexes], dtype=float)
        predicted = np.asarray([predictions[index] for index in indexes], dtype=float)
        threshold = rows[indexes[0]].material_threshold
        actual_pivot.append(bool(np.max(actual) > threshold))
        predicted_pivot.append(bool(np.max(predicted) > threshold))
        actual_best = int(np.argmax(actual))
        predicted_best = int(np.argmax(predicted))
        top1.append(actual_best == predicted_best)
        regrets.append(float(np.max(actual) - actual[predicted_best]))
        tau = kendalltau(actual, predicted).statistic
        taus.append(0.0 if np.isnan(tau) else float(tau))
    pivot_truth = np.asarray(actual_pivot, dtype=int)
    pivot_prediction = np.asarray(predicted_pivot, dtype=int)
    return MetricSlice(
        rows=len(rows),
        groups=len(grouped),
        positive_precision=positive_precision,
        positive_recall=positive_recall,
        positive_auprc=positive_auprc,
        specificity=specificity,
        pivot_precision=float(precision_score(pivot_truth, pivot_prediction, zero_division=0)),
        pivot_recall=float(recall_score(pivot_truth, pivot_prediction, zero_division=0)),
        top1_accuracy=float(mean(top1)),
        mean_kendall_tau=float(mean(taus)),
        mean_selected_regret=float(mean(regrets)),
    )


def _label_distribution_slice(rows: tuple[AdvantageLearningRow, ...]) -> dict[str, float | int]:
    groups = _group_indexes(rows)
    positive_count = sum(row.positive for row in rows)
    material_positive_count = sum(row.advantage > row.material_threshold for row in rows)
    pivot_count = sum(
        any(rows[index].advantage > rows[index].material_threshold for index in indexes)
        for indexes in groups.values()
    )
    candidates_per_group = [len(indexes) for indexes in groups.values()]
    return {
        "rows": len(rows),
        "groups": len(groups),
        "positive_candidates": positive_count,
        "positive_candidate_rate": positive_count / len(rows),
        "material_positive_candidates": material_positive_count,
        "material_positive_candidate_rate": material_positive_count / len(rows),
        "pivot_groups": pivot_count,
        "pivot_group_rate": pivot_count / len(groups),
        "mean_candidates_per_group": float(mean(candidates_per_group)),
    }


def _group_indexes(rows: tuple[AdvantageLearningRow, ...]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row.group_id, []).append(index)
    return groups


def _balanced_group_weights(rows: tuple[AdvantageLearningRow, ...]) -> np.ndarray:
    groups = _group_indexes(rows)
    group_pivot = {
        group_id: any(rows[index].advantage > rows[index].material_threshold for index in indexes)
        for group_id, indexes in groups.items()
    }
    counts = {
        True: max(1, sum(group_pivot.values())),
        False: max(1, len(group_pivot) - sum(group_pivot.values())),
    }
    weights = np.zeros(len(rows), dtype=float)
    for group_id, indexes in groups.items():
        group_weight = 0.5 / counts[group_pivot[group_id]]
        per_row = group_weight / len(indexes)
        for index in indexes:
            weights[index] = per_row
    return weights


def _validate_feature_width(rows: tuple[AdvantageLearningRow, ...], expected: int | None = None) -> None:
    if not rows:
        raise ValueError("at least one row is required")
    widths = {len(row.features) for row in rows}
    if len(widths) != 1:
        raise ValueError("inconsistent feature widths")
    width = next(iter(widths))
    if expected is not None and width != expected:
        raise ValueError(f"expected feature width {expected}; found {width}")


def _regime_from_row(row: dict[str, object]) -> str:
    preference = tuple(float(row.get(f"target_preference_{index}", 0.0) or 0.0) for index in range(len(OBJECTIVE_NAMES)))
    if preference and max(preference) - min(preference) <= 1.0e-9:
        return "balanced"
    return OBJECTIVE_NAMES[max(range(len(preference)), key=lambda index: preference[index])]
