"""Pilot (development only, not frozen, not a test result): confidence gate in front of the scenario certificate.
Same decoder, scenarios and parameters as dev_v3.py (M = 0.02, K = 8, k = 2); only the selector is wrapped.
The gate skips the certificate (no rollouts, base action committed) when the ensemble is not confident that the top
nominee improves on the base action:
  gate_vote<q>: fewer than q of the 5 members predict a positive advantage for the top nominee;
  gate_mean<t>: the ensemble-mean prediction of the top nominee is at most t.
Usage: python dev_gate.py SET SEED ARMS [--out DIR]; baseline PGND and la rows come from runs_Kk."""
import os as _pwmd_os
_PWMD_ROOT = _pwmd_os.path.abspath(_pwmd_os.path.join(_pwmd_os.path.dirname(__file__), *['..'] * 2))


def _pwmd_path(*parts):
    return _pwmd_os.path.join(_PWMD_ROOT, *parts)


import json, sys
from pathlib import Path
from time import perf_counter
import numpy as np
H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / "src"), str(H / "baselines"), str(H.parent / "twoleg")]
import torch; torch.set_num_threads(1)
import drl_v3_common as C
from scenario_cert3 import decode_scen3
from run_early import check_schedule
from rerun_paths import MODEL_DIR
from su_modfjsp_ha_pc.metrics import evaluate_feasibility
from su_modfjsp_ha_pc.generator import generate_instance
from su_modfjsp_ha_pc.public_fjsp import load_and_extend_public_fjsp
from su_modfjsp_ha_pc.path_conflict_neural_dataset import _explicit_preference_id
from su_modfjsp_ha_pc.run_pgnd_v4_learnability_gate import _load_artifact
from su_modfjsp_ha_pc.run_pgnd_v5_smoke import MEMBER_STATES
from su_modfjsp_ha_pc.pgnd_v4_model import predict_v4_model
B = _pwmd_path('external_baselines', 'FJSP-DRL-DANIEL', 'data', 'BenchData')
CORES = {"mk1": f"{B}/Brandimarte/BrandimarteMk1.fjs", "edata7": f"{B}/Hurink_edata/HurinkEdata7.fjs"}
S, seed, arms = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else H / "dev_v3" / "runs_gate"
out.mkdir(parents=True, exist_ok=True)
sc, prefs = C.scales(), C.preferences()
arts = tuple(_load_artifact(MODEL_DIR / f"pgnd_v4_member_{s}.pt") for s in MEMBER_STATES[:5])
load = (lambda lv, sd: load_and_extend_public_fjsp(Path(CORES[S]), level=lv, seed=sd, n_agvs=3)) if S in CORES else (lambda lv, sd: generate_instance(lv, "small", sd))
cases = C.instances([seed], core=CORES.get(S))
M, K, k = 0.02, 8, 2


def gated_selector(rule, par, stats):
    def select(candidate_rows, group, base_index):
        mem = np.asarray([predict_v4_model(a, (group,))[0] for a in arts], dtype=float)
        pred = mem.mean(axis=0)
        if not np.all(np.isfinite(pred)):
            return None
        idx = [i for i in range(len(candidate_rows)) if i != base_index]
        if not idx:
            return None
        best = max(pred[i] for i in idx)
        if sum(abs(pred[i] - best) <= 1.0e-12 for i in idx) > 1:
            return None
        picks = sorted(idx, key=lambda i: (-pred[i], i))[:k]
        top = picks[0]
        confident = (int((mem[:, top] > 0).sum()) >= par) if rule == "vote" else (pred[top] > par)
        stats["calls"] += 1
        if not confident:
            stats["skipped"] += 1
            return None
        return picks
    return select


fp = out / f"{S}_{seed}_{'+'.join(arms)}.jsonl"
done = {(r["case_id"], r["arm"]) for r in map(json.loads, fp.read_text().splitlines())} if fp.exists() else set()
f = open(fp, "a")
for case in cases:
    template = load(case["level"], 600).events
    for pi, pref in enumerate(prefs):
        cid = f"seed{seed}_{case['level']}_{case['mode']}_pref{pi:02d}"
        for arm in arms:
            if (cid, arm) in done:
                continue
            rule, par = ("vote", int(arm[len("gate_vote"):])) if arm.startswith("gate_vote") else ("mean", float(arm[len("gate_mean"):]))
            stats = dict(calls=0, skipped=0)
            t0 = perf_counter()
            s, info = decode_scen3(true=case["true"], raw=case["raw"], preference=pref, level=case["level"], seed=seed, mode=case["mode"],
                                   preference_id=_explicit_preference_id(pi, pref), selector=gated_selector(rule, par, stats), scales=sc,
                                   clock="strict", margin=M, k=K, template=template)
            wall = perf_counter() - t0
            rb, tb, _, _ = check_schedule(case["true"], s, "rerun"); rep = evaluate_feasibility(case["true"], s)
            hv = rep.precedence_violations + rep.machine_overlap_violations + rep.agv_overlap_violations + rep.edge_capacity_violations
            f.write(json.dumps(dict(case_id=cid, arm=arm, value=C.partial_value(case["true"], s.operations, s.transports, pref, sc),
                                    wall_s=round(wall, 2), route_bad=rb, timing_bad=tb, hard_viol=hv, event_viol=rep.dynamic_event_violations,
                                    gate_calls=stats["calls"], gate_skipped=stats["skipped"], **info)) + "\n")
            f.flush()
