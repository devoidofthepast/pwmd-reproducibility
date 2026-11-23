"""v3 amendment 2, development step (全部重跑规则_v3修正案2_前沿实验_草案.md section 2): head rule for mixed preference weights.
Arms on synthetic development seeds 630-639, the 30 mixed vectors of the Das-Dennis set (5 objectives, 3 divisions):
  la            G-ECT (rules_v3 'lookahead' chooser, strict clock), the reference;
  pgnd-argmax   PGND as frozen in v3 (head of the largest weight);
  pgnd-mix      PGND whose nomination score is sum_k w_k u^(k)(a) over the five pure-objective heads.
M, K, k from v3_final_config.json. Usage: python dev_v3/dev_headmix.py SEED   (output dev_v3/runs_headmix/syn_SEED.jsonl, resumable)."""
import dataclasses
import itertools
import json
import sys
from pathlib import Path
from time import perf_counter

H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / "src"), str(H / "baselines"), str(H.parent / "twoleg"), str(H / "test_v3")]
seed = int(sys.argv[1])
assert 630 <= seed <= 639, "development seeds only"
sys.argv = sys.argv[:1]
import numpy as np  # noqa: E402
import test_v3 as T  # noqa: E402


def das_dennis(m=5, h=3):
    out = []
    for c in itertools.combinations(range(h + m - 1), m - 1):
        b = [-1, *c, h + m - 1]
        out.append(tuple((b[i + 1] - b[i] - 1) / h for i in range(m)))
    return out


MIXED = [w for w in das_dennis() if sorted(w)[-2] > 0]  # 30 vectors with at least two positive weights
assert len(MIXED) == 30


def mix_topk_selector(artifacts, k, w):
    from su_modfjsp_ha_pc.pgnd_v4_model import predict_v4_model

    def select(candidate_rows, group, base_index):
        per_member = []
        for a in artifacts:
            heads = [np.asarray(predict_v4_model(a, (dataclasses.replace(group, regime=r),))[0], dtype=float) for r in range(1, 6)]
            per_member.append(sum(w[i] * heads[i] for i in range(5)))
        pred = np.mean(np.asarray(per_member), axis=0)
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


def main():
    import rules_v3 as R
    from blind_arms import policy_decode
    from rerun_paths import MODEL_DIR
    from scenario_cert3 import decode_scen3, neural_topk_selector
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _explicit_preference_id
    from su_modfjsp_ha_pc.run_pgnd_v4_learnability_gate import _load_artifact
    from su_modfjsp_ha_pc.run_pgnd_v5_smoke import MEMBER_STATES
    sc = T.C.scales()
    M, K, k = T.CFG["M"], T.CFG["K"], T.CFG["k"]
    members = tuple(_load_artifact(MODEL_DIR / f"pgnd_v4_member_{s}.pt") for s in MEMBER_STATES)

    def scen(case, pi, pref, selector):
        return decode_scen3(true=case["true"], raw=case["raw"], preference=pref, level=case["level"], seed=case["seed"], mode=case["mode"],
                            preference_id=_explicit_preference_id(pi, pref), selector=selector, scales=sc, clock="strict", margin=M,
                            k=K, template=T.template_for("syn", case["level"]))

    arms = [("la", lambda case, pi, pref: (policy_decode(true=case["true"], raw=case["raw"], preference=pref,
                                                         choose=R.chooser("lookahead", seed=case["seed"]), strict=True)[0], {})),
            ("pgnd-argmax", lambda case, pi, pref: scen(case, pi, pref, neural_topk_selector(members, k))),
            ("pgnd-mix", lambda case, pi, pref: scen(case, pi, pref, mix_topk_selector(members, k, pref)))]
    out = H / "dev_v3" / "runs_headmix"
    out.mkdir(exist_ok=True)
    path = out / f"syn_{seed}.jsonl"
    done = {(r["case_id"], r["arm"]) for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    with path.open("a") as f:
        for set_name, case in T.cases_for("S", seed):
            for wi, pref in enumerate(MIXED):
                pi = 100 + wi
                cid = f"seed{seed}_{case['level']}_{case['mode']}_w{wi:02d}"
                for arm, run in arms:
                    if (cid, arm) in done:
                        continue
                    t0 = perf_counter()
                    s, info = run(case, pi, pref)
                    rec = dict(case_id=cid, seed=seed, level=case["level"], mode=case["mode"], weight=list(pref), arm=arm,
                               value=T.C.partial_value(case["true"], s.operations, s.transports, pref, sc), wall_s=round(perf_counter() - t0, 3),
                               **T.checks(case["true"], s), **{x: y for x, y in (info or {}).items() if isinstance(y, (int, float))})
                    f.write(json.dumps(rec) + "\n")
                    f.flush()


if __name__ == "__main__":
    main()
