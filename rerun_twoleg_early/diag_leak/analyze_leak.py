"""Summary of leak_measure.py output: share of commits and cases whose timing was moved by an unrevealed event window."""
import collections
import glob
import json
from pathlib import Path

D = Path(__file__).parent
rows = {}
for f in sorted(glob.glob(str(D / "leak_*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        rows[(r["case_id"], r["arm"])] = r
mism = [k for k, r in rows.items() if r["frozen"] is None or abs(r["value"] - r["frozen"]) > 1e-9]
print("records", len(rows), "value mismatches vs frozen test runs", len(mism), mism[:3])
g = collections.defaultdict(list)
for r in rows.values():
    g[(r["set"], r["arm"])].append(r)
print(f"{'set':7s} {'arm':15s} {'cases':>5s} {'commits':>7s} {'shifted%':>8s} {'cases w/ shift%':>15s} {'mean delay':>10s} {'max':>5s} {'shadow err':>10s}")
for (s, a), rs in sorted(g.items()):
    c = sum(r["commits"] for r in rs)
    sh = sum(r["shifted"] for r in rs)
    d = sum(r["delay_sum"] for r in rs)
    print(f"{s:7s} {a:15s} {len(rs):5d} {c:7d} {100*sh/c:8.2f} {100*sum(r['shifted']>0 for r in rs)/len(rs):15.1f} "
          f"{(d/sh if sh else 0):10.1f} {max(r['delay_max'] for r in rs):5d} {sum(r['shadow_err'] for r in rs):10d}")
