"""Diagnosis (validation seeds 640-649, not a test result): DRL-S vs rules, per preference."""
import sys, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / "src"), str(H / "baselines"), str(H.parent / "twoleg")]
def job(arm):
    import numpy as np, torch, drl_v3_common as C, rules_v3 as R, song_adapter as S
    from blind_arms import policy_decode
    torch.set_num_threads(1)
    sc, prefs = C.scales(), C.preferences()
    if arm == "song":
        m = S.make_model(); m.load_state_dict(torch.load(H / "drl_v3/song/song_selected.pt")); m.eval()
        mk = lambda case: S.chooser(m)
    else:
        mk = lambda case: R.chooser(arm, seed=case["seed"])
    out = {}
    for case in C.instances(C.VALID_SEEDS):
        for pi, p in enumerate(prefs):
            s, _ = policy_decode(true=case["true"], raw=case["raw"], preference=p, choose=mk(case), strict=True)
            out.setdefault(pi, []).append(C.partial_value(case["true"], s.operations, s.transports, p, sc))
    return arm, {pi: float(np.mean(v)) for pi, v in out.items()}
if __name__ == "__main__":
    arms = ["lookahead", "gest", "mwkr_eat", "spt", "edd", "fifo", "mwkr", "random", "song"]
    with ProcessPoolExecutor(9) as ex:
        res = dict(ex.map(job, arms))
    json.dump(res, open(H / "dev_v3/diag_drl.json", "w"), indent=1)
    names = ["balanced", "makespan", "tardiness", "energy", "path", "stability"]
    print("arm".ljust(10), " ".join(n[:9].rjust(9) for n in names), "mean".rjust(7))
    for a, d in res.items():
        print(a.ljust(10), " ".join(f"{d[i]:9.4f}" for i in range(6)), f"{sum(d.values())/6:7.4f}")
