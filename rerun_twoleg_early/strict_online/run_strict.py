"""Strict online execution on the synthetic test set (2026-11-22). Arms: G-ECT (rule-lookahead) and PWMD (pgnd), same
models, settings, cases and checks as test_v3.py; only the execution follows baselines/strict_online.py.
Usage: python run_strict.py SEED [--reveal-all] [--first N]
--reveal-all reveals every event from the start and must reproduce the frozen values of test_v3/runs/S exactly."""
import json
import sys
from pathlib import Path
from time import perf_counter

H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / "test_v3"), str(H / "src"), str(H / "baselines"), str(H.parent / "twoleg")]
import test_v3 as T  # noqa: E402
import strict_online as SO  # noqa: E402


def main():
    seed = int(sys.argv[1])
    reveal_all = "--reveal-all" in sys.argv
    first = int(sys.argv[sys.argv.index("--first") + 1]) if "--first" in sys.argv else None
    SO.REVEAL_ALL = reveal_all
    import rules_v3 as R
    from rerun_paths import MODEL_DIR
    from scenario_cert3 import neural_topk_selector
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _explicit_preference_id
    from su_modfjsp_ha_pc.run_pgnd_v4_learnability_gate import _load_artifact
    from su_modfjsp_ha_pc.run_pgnd_v5_smoke import MEMBER_STATES
    sc, prefs = T.C.scales(), T.C.preferences()
    M, K, k = T.CFG["M"], T.CFG["K"], T.CFG["k"]
    members = tuple(_load_artifact(MODEL_DIR / f"pgnd_v4_member_{s}.pt") for s in MEMBER_STATES)
    selector = neural_topk_selector(members, k)

    def run_rule(set_name, case, pi, pref):
        s, _ = SO.policy_decode_online(true=case["true"], raw=case["raw"], preference=pref,
                                       choose=R.chooser("lookahead", seed=case["seed"]), strict=True)
        return s, {}

    def run_pgnd(set_name, case, pi, pref):
        return SO.decode_scen3_online(true=case["true"], raw=case["raw"], preference=pref, level=case["level"], seed=case["seed"],
                                      mode=case["mode"], preference_id=_explicit_preference_id(pi, pref), selector=selector,
                                      scales=sc, clock="strict", margin=M, k=K, template=T.template_for(set_name, case["level"]))

    frozen = {}
    for g in ("rules", "pgnd"):
        f = H / "test_v3" / "runs" / "S" / f"{g}_{seed}.jsonl"
        for line in open(f):
            r = json.loads(line)
            frozen[(r["case_id"], r["arm"])] = r["value"]
    out = Path(__file__).parent / ("check" if reveal_all else "runs")
    out.mkdir(exist_ok=True)
    path = out / f"strict_{seed}.jsonl"
    done = {(r["case_id"], r["arm"]) for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    n = 0
    with path.open("a") as f:
        for set_name, case in T.cases_for("S", seed):
            for pi, pref in enumerate(prefs):
                cid = f"{set_name}_seed{seed}_{case['level']}_{case['mode']}_pref{pi:02d}"
                for arm, run in (("rule-lookahead", run_rule), ("pgnd", run_pgnd)):
                    if (cid, arm) in done:
                        continue
                    t0 = perf_counter()
                    s, info = run(set_name, case, pi, pref)
                    rec = dict(case_id=cid, set=set_name, seed=seed, level=case["level"], mode=case["mode"], pref=pi, arm=arm,
                               value=T.C.partial_value(case["true"], s.operations, s.transports, pref, sc),
                               frozen=frozen.get((cid, arm)), wall_s=round(perf_counter() - t0, 3), **T.checks(case["true"], s),
                               **{kk: v for kk, v in (info or {}).items() if isinstance(v, (int, float))})
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                n += 1
                if first and n >= first:
                    return


if __name__ == "__main__":
    main()
