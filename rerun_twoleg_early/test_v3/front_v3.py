"""v3 amendment 2 section 3 (全部重跑规则_v3修正案2_第3节_前沿实验_冻结_20261114.md): front experiment.
Usage:
  python front_v3.py SEED det                  PGND (argmax head), 8 rules, DANIEL, DRL-S, HGT, DQN (largest-weight model)
  python front_v3.py SEED ea --method M --run R   NSGA-II, NSGA-III or MOEA/D, run R in 0..4, seed 1000*R + instance seed
Synthetic test seeds 800-809, L4/L5 x raw/realized (40 instances), the 15 Das-Dennis vectors (5 objectives, 2 divisions),
hidden random events, strict clock. Each record keeps the five raw objectives (evaluate_objective_contract(...).raw), V,
and the test_v3 checks. Output test_v3/front/{group}[_{M}_run{R}]_{SEED}.jsonl, resumable by (instance, weight, arm)."""
import itertools
import json
import sys
from pathlib import Path
from time import perf_counter

H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / "src"), str(H / "baselines"), str(H.parent / "twoleg"), str(H / "test_v3")]
ARGV = list(sys.argv)
sys.argv = sys.argv[:1]
import numpy as np  # noqa: E402
import test_v3 as T  # noqa: E402

NAMES = ("makespan", "tardiness", "energy", "route_pressure", "event_exposure_pressure")


def das_dennis(m=5, h=2):
    out = []
    for c in itertools.combinations(range(h + m - 1), m - 1):
        b = [-1, *c, h + m - 1]
        out.append(tuple((b[i + 1] - b[i] - 1) / h for i in range(m)))
    return out


WEIGHTS = das_dennis()
assert len(WEIGHTS) == 15 and sum(1 for w in WEIGHTS if max(w) == 1.0) == 5


def largest(w):
    """Index of the largest weight, the lowest index on a tie (same rule as the PGND head)."""
    return int(np.argmax(np.asarray(w)))


def pid_index(w, wi):
    """Preference index for the PGND scenario keys: a pure vector keeps its index in the main test set P (1..5), so its
    front run reproduces P exactly (built-in consistency check); mixed vectors get 200 + wi."""
    return 1 + largest(w) if max(w) == 1.0 else 200 + wi


def raw_objectives(instance, s):
    from su_modfjsp_ha_pc.objective_contract import evaluate_objective_contract
    raw = evaluate_objective_contract(instance, s).raw
    return [float(raw[n]) for n in NAMES]


def det_arms():
    import torch
    import cheng_dqn as D
    import daniel_adapter as DA
    import hgt_adapter as HG
    import rules_v3 as R
    import song_adapter as SO
    from blind_arms import policy_decode
    from rerun_paths import MODEL_DIR
    from scenario_cert3 import decode_scen3, neural_topk_selector
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _explicit_preference_id
    from su_modfjsp_ha_pc.run_pgnd_v4_learnability_gate import _load_artifact
    from su_modfjsp_ha_pc.run_pgnd_v5_smoke import MEMBER_STATES
    sc = T.C.scales()
    M, K, k = T.CFG["M"], T.CFG["K"], T.CFG["k"]
    members = tuple(_load_artifact(MODEL_DIR / f"pgnd_v4_member_{s}.pt") for s in MEMBER_STATES)
    H3 = T.H

    def pol(make):
        return lambda case, wi, w: (policy_decode(true=case["true"], raw=case["raw"], preference=w, choose=make(case, w), strict=True)[0], {})

    arms = [("pgnd", lambda case, wi, w: decode_scen3(true=case["true"], raw=case["raw"], preference=w, level=case["level"], seed=case["seed"],
                                                     mode=case["mode"], preference_id=_explicit_preference_id(pid_index(w, wi), w),
                                                     selector=neural_topk_selector(members, k), scales=sc, clock="strict", margin=M, k=K,
                                                     template=T.template_for("syn", case["level"])))]
    for rule in R.RULES:
        arms.append((f"rule-{rule}", pol(lambda case, w, rule=rule: R.chooser(rule, seed=case["seed"]))))
    for name, mod in (("daniel", DA), ("song", SO), ("hgt", HG)):
        m = mod.make_model()
        m.load_state_dict(torch.load(H3 / "drl_v3" / name / f"{name}_selected.pt"))
        m.eval()
        arms.append((name, pol(lambda case, w, mod=mod, m=m: mod.chooser(m))))
    dqn = []
    for i in range(6):
        q = D.network()
        q.load_state_dict(torch.load(H3 / "drl_v3" / "cheng" / "final" / "selected" / f"dqn_pref{i:02d}.pt"))
        q.eval()
        dqn.append(q)
    arms.append(("dqn", pol(lambda case, w: D.chooser(dqn[1 + largest(w)]))))  # model i+1 is the pure objective i
    return arms


def ea_arms(method, run):
    import ea_resched_v3 as E
    sc = T.C.scales()
    return [(f"ea-{method}-run{run}", lambda case, wi, w: E.run_case(case, method, w, sc, T.CFG["ea_budget"], seed=1000 * run + case["seed"]))]


def main():
    import torch
    torch.set_num_threads(1)
    seed, group = int(ARGV[1]), ARGV[2]
    assert 800 <= seed <= 809, "front experiment: synthetic test seeds 800-809"
    if group == "ea":
        method, run = ARGV[ARGV.index("--method") + 1], int(ARGV[ARGV.index("--run") + 1])
        assert method in ("nsga2", "nsga3", "moead") and 0 <= run <= 4
        arms, tag = ea_arms(method, run), f"ea_{method}_run{run}"
    else:
        assert group == "det"
        arms, tag = det_arms(), "det"
    import os
    smoke = os.environ.get("FRONT_SMOKE") == "1"  # first instance, two weights, separate folder
    out = T.H / "test_v3" / ("front_smoke" if smoke else "front")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}_{seed}.jsonl"
    done = {(r["case_id"], r["arm"]) for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    sc = T.C.scales()
    with path.open("a") as f:
        for ci, (_, case) in enumerate(T.cases_for("S", seed)):
            if smoke and ci > 0:
                break
            for wi, w in enumerate(WEIGHTS):
                if smoke and wi not in tuple(int(x) for x in os.environ.get("FRONT_SMOKE_W", "1,7").split(",")):
                    continue
                cid = f"syn_seed{seed}_{case['level']}_{case['mode']}_w{wi:02d}"
                for arm, run in arms:
                    if (cid, arm) in done:
                        continue
                    t0 = perf_counter()
                    s, info = run(case, wi, w)
                    wall = perf_counter() - t0
                    rec = dict(case_id=cid, instance=f"syn_seed{seed}_{case['level']}_{case['mode']}", seed=seed, level=case["level"],
                               mode=case["mode"], weight=list(w), arm=arm, raw=raw_objectives(case["true"], s),
                               value=T.C.partial_value(case["true"], s.operations, s.transports, w, sc), wall_s=round(wall, 3),
                               **T.checks(case["true"], s), **{x: y for x, y in (info or {}).items() if isinstance(y, (int, float)) and x != "wall_s"},
                               **({"ea_wall_s": info["wall_s"]} if info and "wall_s" in info else {}))
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    if rec["route_bad"] or rec["timing_bad"] or rec["hard_viol"] or rec["event_viol"] or rec.get("failures"):
                        with (out / "check_failures.log").open("a") as g:
                            g.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
