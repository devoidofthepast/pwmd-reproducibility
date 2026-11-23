"""v3 step 3: EA per-(re)optimisation budget, 10 s vs 40 s, syn seeds 630-631 (dev, rule in 执行记录 step 3), one run per method, idle machine, 12 workers on 14 cores."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / "src"), str(H / "baselines"), str(H.parent / "twoleg")]
def job(args):
    seed, ci, pi, m, b = args
    import drl_v3_common as C, ea_resched_v3 as E
    sc, prefs = C.scales(), C.preferences()
    case = C.instances([seed])[ci]
    s, info = E.run_case(case, m, prefs[pi], sc, budget=b, seed=seed)
    return dict(seed=seed, case=ci, pref=pi, method=m, budget=b, value=C.partial_value(case["true"], s.operations, s.transports, prefs[pi], sc), **info)
if __name__ == "__main__":
    tasks = [(sd, ci, pi, m, b) for b in (10.0, 40.0) for m in ("ga", "nsga2", "nsga3", "moead") for ci in range(4) for pi in range(6) for sd in (630, 631)]
    out = open(H / "dev_v3" / "ea_budget_630_631.jsonl", "a")
    with ProcessPoolExecutor(int(sys.argv[1])) as ex:
        for r in ex.map(job, tasks):
            out.write(json.dumps(r) + "\n"); out.flush()
