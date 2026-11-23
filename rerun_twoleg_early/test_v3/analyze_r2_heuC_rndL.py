"""Amendment 3 section 4: heu on C (per set) and rndL on L under analyze_c1.compare."""
import glob, json, sys, io, contextlib, collections
from pathlib import Path
H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H)); sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import analyze_c1 as A
v = collections.defaultdict(lambda: collections.defaultdict(dict)); bad = 0
for b, pats in (("C", ("rules_*", "pgnd_*", "heu_*", "random_*")), ("L", ("rules_*", "pgnd_*", "rndL_*"))):
    for pat in pats:
        for f in glob.glob(str(H / "test_v3" / "runs" / b / f"{pat}.jsonl")):
            for l in open(f):
                r = json.loads(l); v[r["set"]][r["case_id"].split("_", 1)[1]][r["arm"]] = r["value"]
                if pat in ("heu_*", "rndL_*"):
                    bad += r["route_bad"] + r["timing_bad"] + r["hard_viol"] + r["event_viol"] + r.get("failures", 0)
print("check errors", bad)
for st in ("mk1", "edata7"):
    d = {c: x for c, x in v[st].items() if "heu-ect" in x and "heu-cost" in x}
    print(f"== {st} n={len(d)}")
    for a in ("heu-ect", "heu-cost"):
        print("  ", a, "vs G-ECT", A.compare(d, a, "rule-lookahead"))
        print("   pgnd vs", a, A.compare(d, "pgnd", a))
    if all("rnd" in x for x in d.values()):
        print("   rnd vs G-ECT", A.compare(d, "rnd", "rule-lookahead"))
d = v["large"]; arm = sorted({a for x in d.values() for a in x if a.startswith("rnd")})
print("== large rnd arms", arm)
d = {c: x for c, x in d.items() if arm[0] in x}
print("  n", len(d), "rnd vs G-ECT", A.compare(d, arm[0], "rule-lookahead"))
print("  pgnd vs rnd", A.compare(d, "pgnd", arm[0]))
print("  pgnd vs G-ECT", A.compare(d, "pgnd", "rule-lookahead"))
