"""Summarize C-stage batch 1 (c1/): primary 650-659, proposer ablation 680-689, hidden synthetic 740-749.
Delta = 1 - mean V(PGND) / mean V(opponent), as in the original baseline table; p = seed-cluster exact Wilcoxon on per-seed mean
differences; tiers of 全部重跑规则_冻结v1.md section 3."""
import csv
import glob
import itertools
import json
from collections import defaultdict
from statistics import mean, median

C = "c1"
EPS = 1e-12


def wil(d):
    d = [x for x in d if x != 0]; n = len(d)
    if n == 0:
        return 1.0
    o = sorted(range(n), key=lambda i: abs(d[i])); r = [0.0] * n; i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[o[j + 1]]) == abs(d[o[i]]):
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    wo = sum(r[i] for i in range(n) if d[i] > 0); mu = sum(r) / 2
    dist = [sum(r[i] for i in range(n) if s[i]) for s in itertools.product((0, 1), repeat=n)]
    return sum(1 for w in dist if abs(w - mu) >= abs(wo - mu) - 1e-9) / len(dist)


def compare(v, a, b):
    """v[case][method] -> PGND-like arm a against opponent b."""
    cases = [c for c in v if a in v[c] and b in v[c]]
    va, vb = mean(v[c][a] for c in cases), mean(v[c][b] for c in cases)
    w = sum(v[c][a] < v[c][b] - EPS for c in cases); l = sum(v[c][a] > v[c][b] + EPS for c in cases)
    bs = defaultdict(list)
    for c in cases:
        bs[seed_of(c)].append(v[c][b] - v[c][a])
    p = wil([mean(x) for _, x in sorted(bs.items())])
    delta = 100 * (1 - va / vb)
    tier = "A" if p < 0.05 and delta > 0 else ("C" if p < 0.05 and delta < 0 else "B")
    return dict(n=len(cases), V_a=round(va, 4), V_b=round(vb, 4), delta_pct=round(delta, 2), WTL=f"{w}/{len(cases)-w-l}/{l}", p=round(p, 4), tier=tier)


def seed_of(case_id):
    return int(case_id.split("_")[0].replace("seed", ""))


def load_jsonl(folder):
    rows = [json.loads(l) for f in glob.glob(f"{C}/{folder}/seed_*.jsonl") for l in open(f) if l.strip()]
    v = defaultdict(dict)
    for r in rows:
        v[r["case_id"]][r["method"]] = r["value"]
    return rows, v


def frac(la, x):
    return (la - x) / la if la > 0 else 0.0  # as improvement_fraction in the original confirmation


out = {}
rows, v = load_jsonl("primary")
out["primary_checks"] = {k: sum(int(r.get(k) or 0) for r in rows) for k in ("route_bad", "timing_bad", "hard_viol", "invariant_failures")}
out["primary"] = {m: compare(v, "pgnd", m) for m in ("rule-lookahead", "rule-spt", "rule-edd", "rule-fifo", "rule-mwkr", "rule-random", "full-exact")}
pg = [r for r in rows if r["method"] == "pgnd"]; fe = [r for r in rows if r["method"] == "full-exact"]
gains = []
for c in v:
    la, p, f = v[c]["rule-lookahead"], v[c]["pgnd"], v[c]["full-exact"]
    if la - f > EPS:
        gains.append((la - p) / (la - f))
pi = defaultdict(list)
for c in v:
    pi[c.split("_pref")[1]].append(c)
out["primary_detail"] = {
    "pgnd_accepted": sum(r["accepted"] for r in pg), "pgnd_decisions": sum(r["decisions"] for r in pg),
    "pgnd_proposals": sum(r["proposals"] for r in pg),
    "pgnd_rollouts": sum(r["rollouts"] for r in pg), "full_exact_rollouts": sum(r["rollouts"] for r in fe),
    "rollout_saving_pct": round(100 * (1 - sum(r["rollouts"] for r in pg) / sum(r["rollouts"] for r in fe)), 1),
    "runtime_saving_pct_under_load": round(100 * (1 - sum(r["wall_ms"] for r in pg) / sum(r["wall_ms"] for r in fe)), 1),
    "mean_retained_gain": round(mean(gains), 4), "retained_gain_defined": len(gains),
    "improved_over_1pct": sum(frac(v[c]["rule-lookahead"], v[c]["pgnd"]) > 0.01 for c in v),
    "mean_per_case_impr_pct": round(100 * mean(frac(v[c]["rule-lookahead"], v[c]["pgnd"]) for c in v), 1),
    "per_pref_impr_pct": {k: round(100 * (1 - mean(v[c]["pgnd"] for c in cs) / mean(v[c]["rule-lookahead"] for c in cs)), 1) for k, cs in sorted(pi.items())},
}
rows, v = load_jsonl("ablation")
out["ablation_checks"] = {k: sum(int(r.get(k) or 0) for r in rows) for k in ("route_bad", "timing_bad", "hard_viol", "invariant_failures")}
out["ablation_vs_lookahead"] = {m: compare(v, m, "rule-lookahead") for m in ("deterministic-random", "ridge-linear", "context-only-neural", "physics-neural")}
out["ablation_physics_vs"] = {m: compare(v, "physics-neural", m) for m in ("deterministic-random", "ridge-linear", "context-only-neural")}
out["ablation_accepted"] = {m: sum(r.get("accepted", 0) for r in rows if r["method"] == m) for m in ("deterministic-random", "ridge-linear", "context-only-neural", "physics-neural")}
hr = [r for f in glob.glob(f"{C}/hidden_740_749/seed_*.csv") for r in csv.DictReader(open(f))]
out["hidden_checks"] = {k: sum(int(r[k]) for r in hr) for k in ("route_bad", "timing_bad", "hard_viol")} | {"failures": sum(int(r["failures"] or 0) for r in hr)}
v = defaultdict(dict)
for r in hr:
    v[r["case_id"]][r["arm"]] = float(r["value"])
out["hidden"] = {"pgnd0.005@strict vs la@strict": compare(v, "pgnd0.005@strict", "la@strict"),
                 "pgnd@strict vs la@strict": compare(v, "pgnd@strict", "la@strict"),
                 "pgnd@oracle vs la@oracle": compare(v, "pgnd@oracle", "la@oracle")}
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open(f"{C}/summary_batch1.json", "w"), indent=1)
