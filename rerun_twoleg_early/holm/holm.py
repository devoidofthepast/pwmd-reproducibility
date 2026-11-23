"""Holm check of the significance markers (2026-11-22, GPT review point 6). Uses the data loading, case selection and the
seed-cluster exact Wilcoxon of make_tables_v9.py (executed up to its first table write; nothing is written to the
manuscript folder). Two family definitions:
  A (control vs k, Derrac et al. 2011): within one table row, PWMD against the k opponents of that table;
  B (whole table): every marker of a table in one family.
Prints every marker whose significance changes."""
import json
import os
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MS = HERE.parents[1] / "manuscript_v9_rerun_20261113"
src = open(MS / "make_tables_v9.py", encoding="utf-8").read()
os.chdir(MS)
ns = {"__name__": "helpers"}
exec(compile(src[:src.index('open("tables_v9_rules.tex", "w")')], "make_tables_v9_helpers", "exec"), ns)
compare, cases, SETS = ns["compare"], ns["cases"], ns["SETS"]
PREFS = ["Balanced", "Makespan", "Tardiness", "Energy", "Path", "Stability"]

tests = []  # (table, row, label, p, delta)


def add(table, row, label, keys, a, b):
    if not keys:
        return
    c = compare(keys, a, b)
    tests.append(dict(table=table, row=row, label=label, p=c["p"], delta=c["delta"], n=c["n"]))


def method_rows(table, opps, sets, extra_pref=False):
    for S, sname in sets:
        for lev in ("L4", "L5"):
            keys = cases(S, lev, ["pgnd"] + opps)
            for o in opps:
                add(table, f"{sname} {lev}", o, keys, "pgnd", o)
    allk = [k for S, _ in sets for k in cases(S, None, ["pgnd"] + opps)]
    for o in opps:
        add(table, "All", o, allk, "pgnd", o)
    if extra_pref:
        for pi, pn in enumerate(PREFS):
            kp = [k for k in allk if k[1].endswith(f"pref{pi:02d}")]
            for o in opps:
                add(table, pn, o, kp, "pgnd", o)


RULES = ["rule-lookahead", "rule-gest", "rule-mwkr_eat", "rule-spt", "rule-edd", "rule-fifo", "rule-mwkr", "rule-random"]
method_rows("rules", RULES, SETS)
method_rows("drl", ["daniel", "song", "hgt", "dqn"], SETS)
method_rows("ea", ["ea-ga", "ea-nsga2", "ea-nsga3", "ea-moead"], [("syn", "Synthetic")], extra_pref=True)

ABL = ["pgnd-nocert", "nocert-gated", "rnd", "heu-ect", "heu-cost", "abl-ridge", "abl-context-only", "abl-no-route-event"]
for S, sname in SETS:
    for a in ABL:
        keys = cases(S, None, ("pgnd", a, "rule-lookahead"))
        add("ablation", f"{sname} vsGECT", a, keys, a, "rule-lookahead")
        add("ablation", f"{sname} PWMDvs", a, keys, "pgnd", a)


def holm(group):
    m = len(group)
    order = sorted(range(m), key=lambda i: group[i]["p"])
    adj, run = [0.0] * m, 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * group[i]["p"]))
        adj[i] = run
    return adj


for name, keyf in (("A: row family", lambda t: (t["table"], t["row"])), ("B: table family", lambda t: t["table"])):
    fam = defaultdict(list)
    for t in tests:
        fam[keyf(t)].append(t)
    changed = defaultdict(list)
    sig_raw = defaultdict(int)
    for key, g in fam.items():
        for t, pa in zip(g, holm(g)):
            sig_raw[t["table"]] += t["p"] < 0.05
            if (t["p"] < 0.05) != (pa < 0.05):
                changed[t["table"]].append(f"{t['row']} | {t['label']}: p={t['p']:.4f} -> {pa:.4f}, delta={t['delta']:.2f}%")
    print(f"\n===== {name}")
    for tb in ("rules", "drl", "ea", "ablation"):
        n = sum(1 for t in tests if t["table"] == tb)
        print(f"{tb}: tests {n}, significant before {sig_raw[tb]}, lose significance {len(changed[tb])}")
        for line in changed[tb]:
            print("    ", line)
json.dump(tests, open(HERE / "tests.json", "w"), indent=0)
