"""Development candidates, round 2 (rerun v2 draft, not frozen). Builds on scenario_cert.scenarios.

decode_scen2 adds to the sampled-event certificate:
- rule "mean": accept if the mean scenario value beats the mean base value by the relative margin M (= scenario_cert);
- rule "lcb":  accept if the mean paired gain minus z standard errors still exceeds M times the mean base value;
- rule "maj":  rule "mean" and the proposal beats the base in at least 3/4 of the scenarios;
- top-k nominations: the selector returns up to k candidates; each is evaluated on the same scenarios, and the accepted
  candidate with the largest mean gain is committed.
k = 1, rule "mean" reproduces scenario_cert.decode_scen.
"""
import hashlib
import inspect
import math

import numpy as np

import event_blind as EB
from scenario_cert import scenarios

_src = inspect.getsource(EB.decode)
_src = _src.replace(
    "def decode(*, true, raw, preference, level, seed, mode, preference_id, selector, scales, clock, trace=None, margin=0.0, nogate=False):",
    "def decode_scen2(*, true, raw, preference, level, seed, mode, preference_id, selector, scales, clock, trace=None, margin=0.0, nogate=False, k=8, template=(), rule='mean', z=1.0):")


def _rep(old, new):
    global _src
    assert _src.count(old) == 1, old[:60]
    _src = _src.replace(old, new)


_rep("""                index = selector(ordered, group, group.base_index)
                if index is not None:
                    n_proposals += 1""",
     """                picks = selector(ordered, group, group.base_index)
                picks = [] if picks is None else ([picks] if isinstance(picks, (int, np.integer)) else list(picks))
                if picks:
                    index = picks[0]
                    n_proposals += 1""")
_rep("""                    def value(j, o, m, tag):
                        schedule = _complete_schedule_after_forced_path_aware_joint_action(
                            instance=known,""",
     """                    scen = scenarios(known, template, t, f"scen-{seed}-{level}-{mode}-{preference_id}-{step}", k) if k > 0 else [known]

                    def value(j, o, m, tag):
                        return [value1(inst, j, o, m, tag) for inst in scen]

                    def value1(inst, j, o, m, tag):
                        schedule = _complete_schedule_after_forced_path_aware_joint_action(
                            instance=inst,""")
_rep("                        return _weighted_value(known, schedule, preference, level, seed, mode, preference_id, tag, scales)",
     "                        return _weighted_value(inst, schedule, preference, level, seed, mode, preference_id, tag, scales)")
_rep("""                    base_value = value(base_job, base_op, base_action["machine"], "base")
                    proposal_value = value(job, op, machine, "proposal")
                    n_rollouts += 2
                    if proposal_value < base_value * (1.0 - margin) - 1.0e-12:
                        chosen = (job, op, machine)
                        accepted = True
                        n_accepted += 1
""",
     """                    bvec = value(base_job, base_op, base_action["machine"], "base")
                    best = None
                    for rank, idx in enumerate(picks):
                        cand = (job, op, machine) if rank == 0 else _resolve_joint_candidate(known, available, ordered[idx])
                        pvec = value(*cand, "proposal" if rank == 0 else f"proposal{rank}")
                        if _accept(bvec, pvec, margin, rule, z):
                            gain = sum(bvec) / len(bvec) - sum(pvec) / len(pvec)
                            if best is None or gain > best[0] + 1.0e-12:
                                best = (gain, cand)
                    n_rollouts += (1 + len(picks)) * len(scen)
                    if best is not None:
                        chosen = best[1]
                        accepted = True
                        n_accepted += 1
""")


def _accept(bvec, pvec, margin, rule, z):
    mb, mp = sum(bvec) / len(bvec), sum(pvec) / len(pvec)
    ok = mp < mb * (1.0 - margin) - 1.0e-12
    if rule == "mean" or not ok:
        return ok
    if rule == "maj":
        return sum(p < b - 1.0e-12 for b, p in zip(bvec, pvec)) >= math.ceil(0.75 * len(bvec))
    if rule == "lcb":
        d = [b - p for b, p in zip(bvec, pvec)]
        n = len(d)
        sd = (sum((x - sum(d) / n) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0.0
        return sum(d) / n - z * sd / math.sqrt(n) > margin * mb + 1.0e-12
    raise ValueError(rule)


_ns = dict(EB.__dict__, scenarios=scenarios, _accept=_accept, np=np)
exec(compile(_src, "scenario_cert2.decode_scen2", "exec"), _ns)
decode_scen2 = _ns["decode_scen2"]


def neural_topk_selector(artifacts, k):
    """Top-k of the ensemble mean prediction; k = 1 equals neural_ensemble_proposal_selector (None on a tied best)."""
    from su_modfjsp_ha_pc.pgnd_v4_model import predict_v4_model

    def select(candidate_rows, group, base_index):
        pred = np.mean(np.asarray([predict_v4_model(a, (group,))[0] for a in artifacts], dtype=float), axis=0)
        if not np.all(np.isfinite(pred)):
            return None
        idx = [i for i in range(len(candidate_rows)) if i != base_index]
        if not idx:
            return None
        best = max(pred[i] for i in idx)
        if sum(abs(pred[i] - best) <= 1.0e-12 for i in idx) > 1:
            return None
        return sorted(idx, key=lambda i: (-pred[i], i))[:k]

    return select


def random_topk_selector(k):
    """First pick equals deterministic_random_proposal_selector; later picks drawn from the rest by further hashes."""
    def select(candidate_rows, group, base_index):
        rest = [i for i in range(len(candidate_rows)) if i != base_index]
        out = []
        for r in range(k):
            if not rest:
                break
            key = group.group_id if r == 0 else f"{group.group_id}-{r}"
            pick = rest[int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % len(rest)]
            out.append(pick)
            rest.remove(pick)
        return out or None

    return select
