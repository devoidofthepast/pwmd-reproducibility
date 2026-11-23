import sys, json, collections, itertools
from statistics import mean
sys.argv = sys.argv[:1]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import analyze_c1 as A
rows = [json.loads(l) for f in __import__("glob").glob("dev_v3/runs_M/*.jsonl") for l in open(f)]
v = collections.defaultdict(dict); bad = 0
for x in rows:
    v[x["case_id"]][x["arm"]] = x["value"]; bad += x["route_bad"] + x["timing_bad"] + x["hard_viol"] + x["event_viol"] + x.get("failures", 0)
arms = ["pgnd_M0_K8_k1", "pgnd_M0.002_K8_k1", "pgnd_M0.005_K8_k1", "pgnd_M0.01_K8_k1", "pgnd_M0.02_K8_k1"]
full = {c: x for c, x in v.items() if "la" in x and all(a in x for a in arms)}
print("cases", len(full), "check errors", bad)
res = {}
for a in arms:
    c = A.compare(full, a, "la"); res[a] = c["delta_pct"]; print(a, c)
thr = 0.95 * res["pgnd_M0_K8_k1"]
ok = [a for a in arms if res[a] >= thr]
sel = max(ok, key=lambda a: float(a.split("_")[1][1:]))
print("threshold", round(thr, 3), "selected", sel)
