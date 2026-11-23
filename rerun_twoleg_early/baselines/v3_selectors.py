"""v3 ablation proposers as top-k selectors (same tie and top-k convention as scenario_cert2.neural_topk_selector):
ridge on the full features, the context-only ensemble, and the ensemble without feature group (4)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def _topk(pred, n, base_index, k):
    if not np.all(np.isfinite(pred)):
        return None
    idx = [i for i in range(n) if i != base_index]
    if not idx:
        return None
    best = max(pred[i] for i in idx)
    if sum(abs(pred[i] - best) <= 1.0e-12 for i in idx) > 1:
        return None
    return sorted(idx, key=lambda i: (-pred[i], i))[:k]


def ridge_topk(model_dir, k):
    with (Path(model_dir) / "ridge_linear_full_features.pkl").open("rb") as h:
        art = pickle.load(h)

    def select(rows, group, base_index):
        return _topk(art.model.predict(art.scaler.transform(group.features)), len(rows), base_index, k)
    return select


def _ensemble_topk(paths, indexes, k):
    from su_modfjsp_ha_pc.pgnd_v4_model import predict_v4_model
    from su_modfjsp_ha_pc.pgnd_v5_proposer_ablation import subset_v4_group
    from su_modfjsp_ha_pc.run_pgnd_v4_learnability_gate import _load_artifact
    arts = tuple(_load_artifact(p) for p in paths)

    def select(rows, group, base_index):
        g = subset_v4_group(group, indexes)
        pred = np.mean(np.asarray([predict_v4_model(a, (g,))[0] for a in arts], dtype=float), axis=0)
        return _topk(pred, len(rows), base_index, k)
    return select


def context_only_topk(model_dir, k):
    from su_modfjsp_ha_pc.pgnd_v5_proposer_ablation import context_only_feature_indexes
    from su_modfjsp_ha_pc.run_pgnd_v5_smoke import MEMBER_STATES
    return _ensemble_topk([Path(model_dir) / f"context_only_member_{s}.pt" for s in MEMBER_STATES], context_only_feature_indexes(), k)


def no_route_event_topk(model_dir, k):
    from su_modfjsp_ha_pc.run_pgnd_v5_smoke import MEMBER_STATES
    from su_modfjsp_ha_pc.run_v3_no_route_event_train import no_route_event_feature_indexes
    return _ensemble_topk([Path(model_dir) / f"no_route_event_member_{s}.pt" for s in MEMBER_STATES], no_route_event_feature_indexes(), k)
