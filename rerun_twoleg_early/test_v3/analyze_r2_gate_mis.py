"""Amendment 3 sections 4 and 6: nocert-gated (S, C, L, per set) and mis (S) under analyze_c1.compare."""
import glob, json, sys, io, contextlib, collections
from pathlib import Path
H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H)); sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import analyze_c1 as A
v = collections.defaultdict(lambda: collections.defaultdict(dict)); bad = 0
for b in ("S", "C", "L"):
    for pat in ("rules_*", "pgnd_*", "gate_*", "mis_*"):
        for f in glob.glob(str(H / "test_v3" / "runs" / b / f"{pat}.jsonl")):
            for l in open(f):
                r = json.loads(l); v[r["set"]][r["case_id"].split("_", 1)[1]][r["arm"]] = r["value"]
                if pat in ("gate_*", "mis_*"):
                    bad += r["route_bad"] + r["timing_bad"] + r["hard_viol"] + r["event_viol"] + r.get("failures", 0)
print("check errors (gate+mis)", bad)
def show(d, a, b):
    c = A.compare(d, a, b); return {k: c[k] for k in c if k in ("n", "delta_pct", "p", "tier", "wtl", "w_t_l", "W/T/L")} or c
for st, d in sorted(v.items()):
    full = {c: x for c, x in d.items() if "nocert-gated" in x}
    print(f"== {st} n={len(full)}")
    print("  gated vs G-ECT   ", A.compare(full, "nocert-gated", "rule-lookahead"))
    print("  nocert vs G-ECT  ", A.compare(full, "pgnd-nocert", "rule-lookahead"))
    print("  pgnd vs gated    ", A.compare(full, "pgnd", "nocert-gated"))
    print("  gated vs nocert  ", A.compare(full, "nocert-gated", "pgnd-nocert"))
d = v["syn"]
mis = sorted({a for x in d.values() for a in x if a.startswith("pgnd-mis")})
full = {c: x for c, x in d.items() if all(a in x for a in mis)}
print(f"== mis S n={len(full)}  pgnd vs G-ECT", A.compare(full, "pgnd", "rule-lookahead"))
for a in mis:
    print(" ", a, "vs G-ECT", A.compare(full, a, "rule-lookahead"))
    print(" ", a, "vs pgnd ", A.compare(full, a, "pgnd"))
