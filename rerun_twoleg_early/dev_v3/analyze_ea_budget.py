"""EA budget rule: mean V over all rows per budget; lower wins; if relative gap < 1%, take 10 s."""
import json, collections
from statistics import mean
r = [json.loads(l) for l in open("ea_budget_630_631.jsonl") if l.strip()]
by = collections.defaultdict(list); bm = collections.defaultdict(list); w = collections.defaultdict(list)
for x in r:
    by[x["budget"]].append(x["value"]); bm[(x["method"], x["budget"])].append(x["value"]); w[x["budget"]].append(x["wall_s"])
print("rows", len(r))
for b in sorted(by): print(f"budget {b:4.0f}s  n={len(by[b])}  mean V {mean(by[b]):.5f}  mean wall {mean(w[b]):.1f}s")
for k in sorted(bm): print(" ", k, len(bm[k]), round(mean(bm[k]), 5))
if len(by) == 2:
    v10, v40 = mean(by[10.0]), mean(by[40.0]); gap = (v10 - v40) / v10 * 100
    print(f"gap (10s-40s)/10s = {gap:.3f}%  ->  selected {'40 s' if gap >= 1 else '10 s'}")
