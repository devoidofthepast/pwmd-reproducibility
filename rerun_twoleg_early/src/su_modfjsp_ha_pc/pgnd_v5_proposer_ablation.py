from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from su_modfjsp_ha_pc.exact_advantage_learnability import ADVANTAGE_FEATURE_FIELDS
from su_modfjsp_ha_pc.pgnd_v4_model import V4Artifact, V4Group, predict_v4_model
from su_modfjsp_ha_pc.pgnd_v5_exact_certificate import ProposalSelector


CONTEXT_ONLY_FIELDS = (
    "level_l5",
    "mode_realized",
    "target_preference_0",
    "target_preference_1",
    "target_preference_2",
    "target_preference_3",
    "target_preference_4",
    "stage_operation",
    "stage_machine",
    "stage_transport",
    "step_fraction",
    "available_operation_count",
    "candidate_operation_index",
    "candidate_job_remaining_ops",
    "candidate_eligible_machine_count",
    "group_candidate_order_fraction",
)


@dataclass(frozen=True)
class LinearProposalArtifact:
    scaler: StandardScaler
    model: Ridge
    feature_width: int
    alpha: float


def context_only_feature_indexes() -> tuple[int, ...]:
    raw_width = len(ADVANTAGE_FEATURE_FIELDS)
    raw_indexes = tuple(ADVANTAGE_FEATURE_FIELDS.index(field) for field in CONTEXT_ONLY_FIELDS)
    # V4 features are [raw, raw-base, raw-median, five physics ranks, ten
    # tardiness/completion transforms].  The last fifteen columns are excluded.
    return tuple(index + block * raw_width for block in range(3) for index in raw_indexes)


def subset_v4_group(group: V4Group, feature_indexes: tuple[int, ...]) -> V4Group:
    if not feature_indexes:
        raise ValueError("at least one feature is required")
    if min(feature_indexes) < 0 or max(feature_indexes) >= group.features.shape[1]:
        raise ValueError("feature index is out of range")
    return replace(group, features=group.features[:, feature_indexes].astype(np.float32))


def deterministic_random_proposal_selector() -> ProposalSelector:
    def select(candidate_rows, group, base_index):
        candidate_indexes = tuple(
            index for index in range(len(candidate_rows)) if index != base_index
        )
        if not candidate_indexes:
            return None
        digest = hashlib.sha256(group.group_id.encode("utf-8")).digest()
        return candidate_indexes[int.from_bytes(digest[:8], "big") % len(candidate_indexes)]

    return select


def fit_linear_proposal(
    groups: tuple[V4Group, ...], *, alpha: float = 1.0
) -> LinearProposalArtifact:
    if not groups:
        raise ValueError("at least one training group is required")
    feature_width = groups[0].features.shape[1]
    if any(group.features.shape[1] != feature_width for group in groups):
        raise ValueError("training groups have inconsistent feature widths")
    features = np.concatenate([group.features for group in groups]).astype(np.float64)
    targets = np.concatenate([group.normalized_advantage for group in groups]).astype(np.float64)
    weights = np.concatenate(
        [np.full(len(group.raw_advantage), 1.0 / len(group.raw_advantage)) for group in groups]
    )
    scaler = StandardScaler().fit(features)
    model = Ridge(alpha=alpha).fit(scaler.transform(features), targets, sample_weight=weights)
    return LinearProposalArtifact(scaler, model, feature_width, alpha)


def linear_proposal_selector(artifact: LinearProposalArtifact) -> ProposalSelector:
    def select(candidate_rows, group, base_index):
        if group.features.shape[1] != artifact.feature_width:
            raise ValueError("linear proposer feature width mismatch")
        predictions = artifact.model.predict(artifact.scaler.transform(group.features))
        return _unique_best_nonbase(predictions, len(candidate_rows), base_index)

    return select


def context_only_neural_proposal_selector(
    artifacts: tuple[V4Artifact, ...],
    *,
    feature_indexes: tuple[int, ...] | None = None,
) -> ProposalSelector:
    if not artifacts:
        raise ValueError("at least one model artifact is required")
    indexes = feature_indexes or context_only_feature_indexes()

    def select(candidate_rows, group, base_index):
        reduced = subset_v4_group(group, indexes)
        if any(artifact.feature_width != reduced.features.shape[1] for artifact in artifacts):
            raise ValueError("context-only neural proposer feature width mismatch")
        member_predictions = [predict_v4_model(artifact, (reduced,))[0] for artifact in artifacts]
        mean_prediction = np.mean(np.asarray(member_predictions, dtype=float), axis=0)
        if not np.all(np.isfinite(mean_prediction)):
            return None
        return _unique_best_nonbase(mean_prediction, len(candidate_rows), base_index)

    return select


def _unique_best_nonbase(predictions, candidate_count: int, base_index: int) -> int | None:
    if len(predictions) != candidate_count:
        raise ValueError("prediction count does not match candidate count")
    candidate_indexes = tuple(index for index in range(candidate_count) if index != base_index)
    if not candidate_indexes:
        return None
    best_value = max(float(predictions[index]) for index in candidate_indexes)
    best = tuple(
        index
        for index in candidate_indexes
        if abs(float(predictions[index]) - best_value) <= 1.0e-12
    )
    return best[0] if len(best) == 1 else None
