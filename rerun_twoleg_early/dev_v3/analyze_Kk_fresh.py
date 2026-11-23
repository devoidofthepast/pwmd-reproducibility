"""K x k selection with Mk1/Edata7 on fresh seeds 770-779 (runs_Kk_fresh) and synthetic 630-639 as before (runs_Kk, runs_M).
Rule as in v3 section 3: the largest mean improvement over the base rule across the three development sets.
Also prints the original selection (Mk1/Edata7 610-619) for comparison."""
import collections
import glob
import json
from pathlib import Path
from statistics import mean

H = Path(__file__).resolve().parent
ARMS = ["pgnd_M0.02_K4_k1", "pgnd_M0.02_K4_k2", "pgnd_M0.02_K8_k1", "pgnd_M0.02_K8_k2"]


def load(files):
    v, bad = collections.defaultdict(dict), 0
    for f in files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                v[r["case_id"]][r["arm"]] = r["value"]
                bad += r["route_bad"] + r["timing_bad"] + r["hard_viol"] + r["event_viol"] + r.get("failures", 0)
    full = {c: x for c, x in v.items() if "la" in x and all(a in x for a in ARMS)}
    return full, bad


def impr(full, a):
    return 100 * (1 - mean(x[a] for x in full.values()) / mean(x["la"] for x in full.values()))


syn, bs = load(glob.glob(str(H / "runs_Kk" / "syn_*.jsonl")) + glob.glob(str(H / "runs_M" / "*.jsonl")))
res = {}
for label, seeds_glob in (("original 610-619", "runs_Kk/{S}_*.jsonl"), ("fresh 770-779", "runs_Kk_fresh/{S}_*.jsonl")):
    per = {"syn": {a: impr(syn, a) for a in ARMS}}
    n = {"syn": len(syn)}
    for S in ("mk1", "edata7"):
        full, b = load(glob.glob(str(H / seeds_glob.format(S=S))))
        n[S] = len(full)
        per[S] = {a: impr(full, a) for a in ARMS} if full else {}
        print(f"[{label}] {S}: cases {len(full)}, check errors {b}")
    if all(per[S] for S in per):
        avg = {a: mean(per[S][a] for S in per) for a in ARMS}
        print(f"[{label}] cases {n}")
        for a in ARMS:
            print(f"   {a}: syn {per['syn'][a]:6.2f}  mk1 {per['mk1'][a]:6.2f}  edata7 {per['edata7'][a]:6.2f}  mean {avg[a]:6.2f}")
        print(f"   selected: {max(avg, key=avg.get)}")
