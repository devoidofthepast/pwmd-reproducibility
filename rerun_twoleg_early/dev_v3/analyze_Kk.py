"""v3 step 3(b) analysis: K in {4,8} x k in {1,2} at M=0.02. Usage: python dev_v3/analyze_Kk.py mk1 [edata7 syn]
Descriptive only; the choice is made when all three sets are complete (rule: v3 section 3)."""
import collections
import contextlib
import glob
import io
import json
import sys

sets = sys.argv[1:] or ["mk1"]
sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import analyze_c1 as A

ARMS = ["pgnd_M0.02_K4_k1", "pgnd_M0.02_K4_k2", "pgnd_M0.02_K8_k1", "pgnd_M0.02_K8_k2"]
for S in sets:
    files = glob.glob(f"dev_v3/runs_Kk/{S}_*.jsonl")
    if S == "syn":
        files += glob.glob("dev_v3/runs_M/*.jsonl")  # syn K8 k1 and la reused from step (a)
    rows = [json.loads(l) for f in files for l in open(f) if l.strip()]
    v = collections.defaultdict(dict)
    meta = collections.defaultdict(lambda: collections.defaultdict(float))
    bad = 0
    for x in rows:
        v[x["case_id"]][x["arm"]] = x["value"]
        bad += x["route_bad"] + x["timing_bad"] + x["hard_viol"] + x["event_viol"] + x.get("failures", 0)
        m = meta[x["arm"]]; m["wall"] += x["wall_s"]; m["roll"] += x["rollouts"]; m["acc"] += x["accepted"]; m["n"] += 1
    full = {c: x for c, x in v.items() if "la" in x and all(a in x for a in ARMS)}
    print(f"== {S}: complete cases {len(full)}, check errors {bad}")
    for a in ARMS:
        c = A.compare(full, a, "la")
        m = meta[a]
        print(f"{a:18s} vs la {c['delta_pct']:6.2f}%  W/T/L {c['WTL']:10s} p={c['p']:.4f} tier {c['tier']}  "
              f"mean wall {m['wall']/m['n']:6.1f}s  mean rollouts {m['roll']/m['n']:7.1f}")
    for a, b in [("pgnd_M0.02_K8_k1", "pgnd_M0.02_K4_k1"), ("pgnd_M0.02_K8_k2", "pgnd_M0.02_K4_k2"),
                 ("pgnd_M0.02_K4_k2", "pgnd_M0.02_K4_k1"), ("pgnd_M0.02_K8_k2", "pgnd_M0.02_K8_k1")]:
        c = A.compare(full, a, b)
        print(f"  {a} vs {b}: {c['delta_pct']:+.2f}%  W/T/L {c['WTL']}  p={c['p']:.4f} tier {c['tier']}")
