"""Front experiment analysis, v3 amendment 2 sec. 3 frozen (全部重跑规则_v3修正案2_第3节_前沿实验_冻结_20261114.md), sec. 5-7.
Per instance: min-max normalisation over all methods and runs (before dedupe/filtering; constant dim -> 0), dedupe + non-dominated
filter per front, exact HV (pymoo 0.6.2, reference point 1.1), IGD+ against Z* = non-dominated union of all methods and runs.
EAs: indicator per run, mean of the 5 runs. PGND vs each method: seed-cluster (4 instances) exact two-sided Wilcoxon, alpha 0.05.
Also checks that pure-weight front runs reproduce the main test set P (pgnd_*/drl_*/rules in runs/S) where the arm exists there.
Output front/front_result.txt and front/front_indicators.json."""
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import HV

D = Path(__file__).resolve().parent
src = open(D.parents[1] / "manuscript_v9_rerun_20261113" / "make_tables_v9.py", encoding="utf-8").read()
a = src.index("def wil(")
b = src.index("\ndef ", a + 1)
ns = {"defaultdict": collections.defaultdict}
exec(src[a:b], ns)  # same exact Wilcoxon as the paper tables
wil = ns["wil"]

recs = [json.loads(l) for f in sorted(glob.glob(str(D / "front" / "det_*.jsonl"))) + sorted(glob.glob(str(D / "front" / "ea_*.jsonl")))
        for l in open(f, encoding="utf-8")]
viol = sum(any(r[k] for k in ("route_bad", "timing_bad", "hard_viol", "event_viol")) or r.get("failures", 0) for r in recs)
fronts = collections.defaultdict(lambda: collections.defaultdict(list))  # inst -> (method, run) -> raw list
for r in recs:
    a = r["arm"]
    key = (a.split("-")[1], int(a.split("run")[1])) if a.startswith("ea-") else (a, 0)
    fronts[r["instance"]][key].append((tuple(r["weight"]), r["raw"]))
bad_size = [(i, k, len(v)) for i, fr in fronts.items() for k, v in fr.items() if len(v) != 15 or len({w for w, _ in v}) != 15]


def nd(P):
    P = np.unique(np.asarray(P, float), axis=0)
    return np.array([p for i, p in enumerate(P) if not any(np.all(q <= p) and np.any(q < p) for j, q in enumerate(P) if j != i)])


def igdp(A, Z):
    return float(np.mean([min(np.sqrt((np.maximum(a - z, 0.0) ** 2).sum()) for a in A) for z in Z]))


hv = HV(ref_point=np.full(5, 1.1))
ind = collections.defaultdict(dict)  # inst -> method -> (HV, IGD+)
nnd = collections.defaultdict(list)
for inst, fr in sorted(fronts.items()):
    allp = np.array([raw for v in fr.values() for _, raw in v], float)
    lo, hi = allp.min(0), allp.max(0)
    sp = hi - lo
    ok = sp > 1e-12

    def norm(P):
        P = np.asarray(P, float)
        return np.where(ok, (P - lo) / np.where(ok, sp, 1.0), 0.0)

    Z = nd(norm(allp))
    per = collections.defaultdict(list)
    for (m, run), v in fr.items():
        A = nd(norm([raw for _, raw in v]))
        A_in = A[np.all(A < 1.1, axis=1)]
        per[m].append((float(hv(A_in)) if len(A_in) else 0.0, igdp(A, Z)))
        nnd[m].append(len(A))
    for m, v in per.items():
        ind[inst][m] = tuple(np.mean(v, axis=0))
methods = sorted({m for v in ind.values() for m in v})
ea_runs = {m: sorted({k[1] for fr in fronts.values() for k in fr if k[0] == m}) for m in methods}

# consistency with P at the pure weights
mism = []
P = collections.defaultdict(dict)
for f in glob.glob(str(D / "runs" / "S" / "*_80?.jsonl")):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        if r.get("set") == "syn" and 1 <= r["pref"] <= 5:
            P[(f"syn_seed{r['seed']}_{r['level']}_{r['mode']}", r["pref"], r["arm"])] = r["value"]
checked = 0
for r in recs:
    if r["arm"].startswith("ea-") or max(r["weight"]) != 1.0:
        continue
    k = (r["instance"], 1 + int(np.argmax(r["weight"])), r["arm"])
    if k in P:
        checked += 1
        if abs(P[k] - r["value"]) > 1e-9:
            mism.append((k, P[k], r["value"]))

out = [f"records {len(recs)}  check violations {viol}  fronts with != 15 weights {len(bad_size)}  instances {len(ind)}",
       f"pure-weight consistency with P: {checked} matched records, {len(mism)} mismatches",
       "EA runs per method: " + str({m: r for m, r in ea_runs.items() if r != [0]}), "",
       f"{'method':14s} {'HV mean':>8s} {'HV sd':>7s} {'IGD+ mean':>9s} {'IGD+ sd':>8s}  nd/15 | PGND vs method: HV d, W/T/L, p, verdict | IGD+ d, W/T/L, p, verdict"]
seeds = sorted({int(i.split("seed")[1][:3]) for i in ind})
res = {}
for m in methods:
    h = np.array([ind[i][m][0] for i in sorted(ind)])
    g = np.array([ind[i][m][1] for i in sorted(ind)])
    line = f"{m:14s} {h.mean():8.4f} {h.std(ddof=1):7.4f} {g.mean():9.4f} {g.std(ddof=1):8.4f}  {np.mean(nnd[m]):5.1f}"
    res[m] = dict(hv_mean=h.mean(), hv_sd=h.std(ddof=1), igd_mean=g.mean(), igd_sd=g.std(ddof=1))
    if m != "pgnd":
        for j, nm in ((0, "HV"), (1, "IGD+")):
            d = {i: (ind[i]["pgnd"][j] - ind[i][m][j]) * (1 if j == 0 else -1) for i in ind}  # >0: PGND better
            cl = [np.mean([d[i] for i in d if f"seed{s}_" in i]) for s in seeds]
            p = wil(cl)
            w, t, l = sum(x > 0 for x in d.values()), sum(x == 0 for x in d.values()), sum(x < 0 for x in d.values())
            v = "+" if p < 0.05 and np.mean(cl) > 0 else "-" if p < 0.05 else "*"
            line += f" | {np.mean(list(d.values())):+.4f} {w}/{t}/{l} p={p:.3f} {v}"
            res[m][nm] = dict(d=float(np.mean(list(d.values()))), w=w, t=t, l=l, p=p, verdict=v)
    out.append(line)
out += ["", "verdict: + PGND significantly better than, - significantly worse than, * comparable to (seed-cluster exact Wilcoxon, 0.05)"]
(D / "front" / "front_result.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
(D / "front" / "front_indicators.json").write_text(json.dumps(dict(per_instance=ind, summary=res, mismatches=[str(x) for x in mism[:20]],
                                                                    bad_size=[str(x) for x in bad_size[:20]]), default=float, indent=1), encoding="utf-8")
print("\n".join(out))
