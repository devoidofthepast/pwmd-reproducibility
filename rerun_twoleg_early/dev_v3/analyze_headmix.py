"""Amendment 2 development step: head rule for mixed weights (rule in 全部重跑规则_v3修正案2_前沿实验_草案.md section 2).
Usage: python dev_v3/analyze_headmix.py DIR"""
import collections, glob, json, sys
sys.path.insert(0, ".")
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    import analyze_c1 as A
d = sys.argv[1]
rows, bad_lines = [], 0
for f in glob.glob(f"{d}/syn_*.jsonl"):
    for l in open(f):
        try:
            rows.append(json.loads(l))
        except Exception:
            bad_lines += 1
keys = collections.Counter((r["case_id"], r["arm"]) for r in rows)
print("rows", len(rows), "bad lines", bad_lines, "duplicates", sum(c > 1 for c in keys.values()),
      "check errors", sum(r["route_bad"] + r["timing_bad"] + r["hard_viol"] + r["event_viol"] + r.get("failures", 0) for r in rows))
v = collections.defaultdict(dict)
for r in rows:
    v[r["case_id"]][r["arm"]] = r["value"]
full = {c: x for c, x in v.items() if all(a in x for a in ("la", "pgnd-argmax", "pgnd-mix"))}
print("complete cases", len(full))
res = {}
for a in ("pgnd-argmax", "pgnd-mix"):
    c = A.compare(full, a, "la"); res[a] = c["delta_pct"]
    acc = [r.get("accepted", 0) for r in rows if r["arm"] == a]
    print(f"{a:12s} vs la {c['delta_pct']:6.2f}%  W/T/L {c['WTL']}  p={c['p']:.4f} tier {c['tier']}  mean accepted {sum(acc)/len(acc):.2f}")
c = A.compare(full, "pgnd-mix", "pgnd-argmax")
print(f"pgnd-mix vs pgnd-argmax {c['delta_pct']:+.2f}%  W/T/L {c['WTL']}  p={c['p']:.4f} tier {c['tier']}")
gap = res["pgnd-mix"] - res["pgnd-argmax"]
print(f"difference in improvement over la: {gap:+.2f} pp -> selected rule:", "pgnd-mix" if gap >= 0.5 else "pgnd-argmax")
