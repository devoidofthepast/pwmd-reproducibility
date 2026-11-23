"""Amendment 3: drl23 diagnostic (all sets pooled and per set) and M top-up on dev syn 630-639."""
import glob, json, sys, io, contextlib, collections
from pathlib import Path
H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H)); sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import analyze_c1 as A
def load(pats, blocks=("S", "C", "L")):
    v = collections.defaultdict(lambda: collections.defaultdict(dict)); bad = 0
    for b in blocks:
        for pat in pats:
            for f in glob.glob(str(H / "test_v3" / "runs" / b / f"{pat}.jsonl")):
                for l in open(f):
                    r = json.loads(l); v[r["set"]][r["case_id"].split("_", 1)[1]][r["arm"]] = r["value"]
                    bad += r["route_bad"] + r["timing_bad"] + r["hard_viol"] + r["event_viol"]
    return v, bad
v, bad = load(("rules_*", "drl_*", "drl23_*"))
print("drl23 check errors", bad)
pooled = {}
for st, d in v.items():
    for c, x in d.items(): pooled[f"{st}|{c}"] = x
mean = lambda d, a: sum(x[a] for x in d.values()) / len(d)
for n in ("daniel", "song", "hgt", "dqn"):
    print(f"{n}: orig {mean(pooled, n):.4f}  eq23 {mean(pooled, n + '-eq23'):.4f}  random {mean(pooled, 'rule-random'):.4f}  G-ECT {mean(pooled, 'rule-lookahead'):.4f}")
    for st, d in sorted(v.items()):
        print("   ", st, "eq23 vs random", A.compare(d, n + "-eq23", "rule-random"), "| eq23 vs orig", A.compare(d, n + "-eq23", n)["delta_pct"])
rows = [json.loads(l) for f in glob.glob(str(H / "dev_v3" / "runs_M" / "*.jsonl")) + glob.glob(str(H / "dev_v3" / "runs_M2" / "*.jsonl")) for l in open(f)]
m = collections.defaultdict(dict)
for x in rows: m[x["case_id"]][x["arm"]] = x["value"]
for a in ("pgnd_M0_K8_k1", "pgnd_M0.02_K8_k1", "pgnd_M0.05_K8_k1", "pgnd_M0.1_K8_k1"):
    full = {c: x for c, x in m.items() if a in x and "la" in x}
    print(a, A.compare(full, a, "la"))
