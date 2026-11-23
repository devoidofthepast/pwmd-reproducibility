"""Amendment 3 section 4: choose the heuristic for P-L (larger mean improvement over rule-lookahead on S)."""
import glob, json, sys, io, contextlib, collections
from pathlib import Path
H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H)); sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import analyze_c1 as A
v = collections.defaultdict(dict); bad = 0
for pat in ("rules_*", "pgnd_*", "random_*", "heu_*"):
    for f in glob.glob(str(H / "test_v3" / "runs" / "S" / f"{pat}.jsonl")):
        for l in open(f):
            r = json.loads(l); v[r["case_id"].split("_", 1)[1]][r["arm"]] = r["value"]
            bad += r["route_bad"] + r["timing_bad"] + r["hard_viol"] + r["event_viol"] + r.get("failures", 0)
arms = ("heu-ect", "heu-cost")
full = {c: x for c, x in v.items() if all(a in x for a in arms + ("rule-lookahead", "pgnd", "rnd"))}
print("cases", len(full), "check errors", bad)
res = {}
for a in arms:
    c = A.compare(full, a, "rule-lookahead"); res[a] = c["delta_pct"]; print(a, "vs G-ECT", c)
    print("pgnd vs", a, A.compare(full, "pgnd", a))
print("rnd vs G-ECT", A.compare(full, "rnd", "rule-lookahead"))
choice = max(arms, key=lambda a: res[a]).split("-")[1]
print("choice", choice)
if "--write" in sys.argv[1:] or True:
    pass
