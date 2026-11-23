"""v3 step 3 (development, not a test result): hidden events (random, strict clock), v3 proposer, scenario certificate
(scenario_cert3). Usage: python dev_v3.py SET SEED ARMS [--members N]
SET in syn, mk1, edata7. ARMS: la, or pgnd_M<m>_K<K>_k<k> (e.g. pgnd_M0.005_K8_k1); --members takes the first N members."""
import os as _pwmd_os
_PWMD_ROOT = _pwmd_os.path.abspath(_pwmd_os.path.join(_pwmd_os.path.dirname(__file__), *['..'] * 2))


def _pwmd_path(*parts):
    return _pwmd_os.path.join(_PWMD_ROOT, *parts)


import json, sys
from pathlib import Path
from time import perf_counter
H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / "src"), str(H / "baselines"), str(H.parent / "twoleg")]
import torch; torch.set_num_threads(1)
import event_blind as EB
import drl_v3_common as C
from scenario_cert3 import decode_scen3, neural_topk_selector
from run_early import check_schedule
from rerun_paths import MODEL_DIR
from su_modfjsp_ha_pc.metrics import evaluate_feasibility
from su_modfjsp_ha_pc.generator import generate_instance
from su_modfjsp_ha_pc.public_fjsp import load_and_extend_public_fjsp
from su_modfjsp_ha_pc.path_conflict_neural_dataset import _explicit_preference_id
from su_modfjsp_ha_pc.run_pgnd_v4_learnability_gate import _load_artifact
from su_modfjsp_ha_pc.run_pgnd_v5_smoke import MEMBER_STATES
B = _pwmd_path('external_baselines', 'FJSP-DRL-DANIEL', 'data', 'BenchData')
CORES = {"mk1": f"{B}/Brandimarte/BrandimarteMk1.fjs", "edata7": f"{B}/Hurink_edata/HurinkEdata7.fjs"}
S, seed, arms = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
nm = int(sys.argv[sys.argv.index("--members") + 1]) if "--members" in sys.argv else 5
out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else H / "dev_v3" / "runs"
out.mkdir(parents=True, exist_ok=True)
sc, prefs = C.scales(), C.preferences()
md = Path(sys.argv[sys.argv.index("--model-dir") + 1]) if "--model-dir" in sys.argv else MODEL_DIR  # sensitivity: 5 training seeds
arts = tuple(_load_artifact(md / f"pgnd_v4_member_{s}.pt") for s in MEMBER_STATES[:nm])
load = (lambda lv, sd: load_and_extend_public_fjsp(Path(CORES[S]), level=lv, seed=sd, n_agvs=3)) if S in CORES else (lambda lv, sd: generate_instance(lv, "small", sd))
cases = C.instances([seed], core=CORES.get(S))
fp = out / f"{S}_{seed}_{'+'.join(arms)}_m{nm}{'_' + md.name if md != MODEL_DIR else ''}.jsonl"
done = {(r["case_id"], r["arm"]) for r in map(json.loads, fp.read_text().splitlines())} if fp.exists() else set()  # resume
f = open(fp, "a")
for case in cases:
    template = load(case["level"], 600).events  # event script of a training seed: types, durations, severities
    for pi, pref in enumerate(prefs):
        for arm in arms:
            if (f"seed{seed}_{case['level']}_{case['mode']}_pref{pi:02d}", arm) in done:
                continue
            if arm == "la":
                sel, M, K, k = None, 0.0, 0, 0
            else:
                _, m, kk, k1 = arm.split("_")
                M, K, k = float(m[1:]), int(kk[1:]), int(k1[1:])
                sel = neural_topk_selector(arts, k)
            t0 = perf_counter()
            s, info = decode_scen3(true=case["true"], raw=case["raw"], preference=pref, level=case["level"], seed=seed, mode=case["mode"],
                                   preference_id=_explicit_preference_id(pi, pref), selector=sel, scales=sc, clock="strict", margin=M,
                                   k=K, template=template)
            wall = perf_counter() - t0
            rb, tb, _, _ = check_schedule(case["true"], s, "rerun"); rep = evaluate_feasibility(case["true"], s)
            hv = rep.precedence_violations + rep.machine_overlap_violations + rep.agv_overlap_violations + rep.edge_capacity_violations
            f.write(json.dumps(dict(case_id=f"seed{seed}_{case['level']}_{case['mode']}_pref{pi:02d}", arm=arm, members=nm,
                                    value=C.partial_value(case["true"], s.operations, s.transports, pref, sc), wall_s=round(wall, 2),
                                    route_bad=rb, timing_bad=tb, hard_viol=hv, event_viol=rep.dynamic_event_violations, **info)) + "\n")
            f.flush()
