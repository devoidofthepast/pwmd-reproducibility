"""v9 manuscript tables from the frozen v3 test set P (rerun_twoleg_early/test_v3/runs). Run from this folder.
Statistics: seed-cluster exact two-sided Wilcoxon signed-rank test (clusters = (set, seed)), as in the frozen rules v3 section 6.
Markers (Cai et al., ASOC): '+' PGND significantly better, '-' significantly worse, '*' comparable (p >= 0.05)."""
import glob
import json
from collections import defaultdict
from statistics import mean, pstdev

R = "../rerun_twoleg_early/test_v3/runs"
SETS = [("syn", "Synthetic"), ("mk1", "Mk1"), ("edata7", "Edata7"), ("large", "Larger")]
EPS = 1e-12

rows = [json.loads(l) for f in glob.glob(f"{R}/*/*.jsonl") for l in open(f) if l.strip()]
V = defaultdict(dict)  # (set, case) -> arm -> value
META = defaultdict(list)
for r in rows:
    V[(r["set"], r["case_id"])][r["arm"]] = r["value"]
    META[(r["set"], r["arm"])].append(r)


def wil(d):
    """Exact two-sided Wilcoxon signed-rank p (average ranks for ties, zeros dropped), by dynamic programming."""
    d = [x for x in d if abs(x) > 0]
    n = len(d)
    if n == 0:
        return 1.0
    o = sorted(range(n), key=lambda i: abs(d[i]))
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[o[j + 1]]) == abs(d[o[i]]):
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    r2 = [int(round(2 * x)) for x in r]
    tot = sum(r2)
    dist = {0: 1}
    for x in r2:
        nd = defaultdict(int)
        for s, c in dist.items():
            nd[s] += c
            nd[s + x] += c
        dist = nd
    wo = sum(r2[i] for i in range(n) if d[i] > 0)
    mu = tot / 2
    return sum(c for s, c in dist.items() if abs(s - mu) >= abs(wo - mu) - 1e-9) / 2 ** n


def cases(S, level=None, arms=()):
    out = []
    for (s, c), x in V.items():
        if S != "all" and s != S:
            continue
        if level and f"_{level}_" not in c:
            continue
        if all(a in x for a in arms):
            out.append((s, c))
    return out


def seed_of(c):
    return int(c.split("_seed")[1].split("_")[0])


def compare(keys, a, b):
    """a = proposed, b = opponent; lower V is better."""
    va = [V[k][a] for k in keys]
    vb = [V[k][b] for k in keys]
    w = sum(x < y - EPS for x, y in zip(va, vb))
    l = sum(x > y + EPS for x, y in zip(va, vb))
    cl = defaultdict(list)
    for k in keys:
        cl[(k[0], seed_of(k[1]))].append(V[k][b] - V[k][a])
    p = wil([mean(x) for _, x in sorted(cl.items())])
    delta = 100 * (1 - mean(va) / mean(vb))
    mark = "*" if p >= 0.05 else ("+" if delta > 0 else "-")
    return dict(delta=delta, w=w, t=len(keys) - w - l, l=l, p=p, mark=mark, n=len(keys))


def sd(x):
    return ("+" if x >= 0 else "$-$") + f"{abs(x):.2f}"


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def cell(keys, arm):
    v = [V[k][arm] for k in keys]
    return mean(v), pstdev(v)


def method_table(label, caption, arms, names, ref="pgnd", sets=SETS, note="", panels=None):
    """panels: optional index lists into arms; each is printed as one block of the same table. Bold = best of all arms."""
    panels = panels or [list(range(len(arms)))]
    ncol = max(len(p) for p in panels)
    lines = [r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\scriptsize",
             r"\setlength{\tabcolsep}{2.5pt}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{@{}ll" + "c" * ncol + r"@{}}", r"\toprule"]
    for pi, idx in enumerate(panels):
        parms = [arms[i] for i in idx]
        pad = " &" * (ncol - len(parms))
        if pi:
            lines.append(r"\midrule")
        lines += ["Set & Level & " + " & ".join(names[i] for i in idx) + pad + r" \\", r"\midrule"]
        allkeys = []
        for S, sname in sets:
            for lev in ("L4", "L5"):
                keys = cases(S, lev, arms)
                if not keys:
                    continue
                allkeys += keys
                stats = {a: cell(keys, a) for a in arms}
                best = min(stats[a][0] for a in arms)
                row = []
                for a in parms:
                    m, s = stats[a]
                    txt = f"{m:.4f} ({s:.4f})"
                    if abs(m - best) < 5e-5:
                        txt = rf"\textbf{{{m:.4f}}} ({s:.4f})"
                    if a != ref:
                        txt += compare(keys, ref, a)["mark"].replace("-", "$-$")
                    row.append(txt)
                lines.append(f"{sname if lev == 'L4' else ''} & {lev} & " + " & ".join(row) + pad + r" \\")
        stats = {a: cell(allkeys, a) for a in arms}
        best = min(stats[a][0] for a in arms)
        lines.append(r"\midrule")
        row = []
        wtl = []
        for a in parms:
            m, s = stats[a]
            txt = f"{m:.4f} ({s:.4f})"
            if abs(m - best) < 5e-5:
                txt = rf"\textbf{{{m:.4f}}} ({s:.4f})"
            if a != ref:
                c = compare(allkeys, ref, a)
                txt += c["mark"].replace("-", "$-$")
                wtl.append(f"{c['w']}/{c['t']}/{c['l']}")
            else:
                wtl.append("--")
            row.append(txt)
        lines.append(r"\multicolumn{2}{@{}l}{All} & " + " & ".join(row) + pad + r" \\")
        lines.append(r"\multicolumn{2}{@{}l}{w/t/l} & " + " & ".join(wtl) + pad + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\par\smallskip",
              r"{\raggedright\footnotesize Note: Mean (standard deviation) of $V$; lower is better, and the best mean in each row is in bold. "
              r"The symbols ``+'', ``$-$'', and ``*'' indicate that \pgnd{} performs significantly better than, significantly worse than, "
              r"or comparably to the method, respectively (seed-cluster Wilcoxon signed-rank test, 0.05 level). "
              r"w/t/l counts the cases in which \pgnd{} wins, ties, and loses. " + note + r"\par}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# evolutionary algorithms: mean over the five independent runs of each case
_ea = defaultdict(list)
for r in rows:
    if r["arm"].startswith("ea-"):
        _ea[(r["set"], r["case_id"], r["arm"].rsplit("-run", 1)[0])].append(r["value"])
for (st_, cid, m), vals in _ea.items():
    V[(st_, cid)][m] = sum(vals) / len(vals)
EA_TIME = {m: mean(r["wall_s"] for r in rows if r["arm"].startswith(m + "-run")) for m in ("ea-ga", "ea-nsga2", "ea-nsga3", "ea-moead")}

RULES = ["pgnd", "rule-lookahead", "rule-gest", "rule-mwkr_eat", "rule-spt", "rule-edd", "rule-fifo", "rule-mwkr", "rule-random"]
RULE_NAMES = [r"\pgnd{}", "G-ECT", "G-EST", "MWKR-EAT", "SPT", "EDD", "FIFO", "MWKR", "Random"]
DRL = ["pgnd", "daniel", "song", "hgt", "dqn"]
DRL_NAMES = [r"\pgnd{}", "DANIEL", "DRL-S", "HGT", "DQN"]
ABL = ["pgnd", "pgnd-nocert", "rnd", "abl-ridge", "abl-context-only", "abl-no-route-event"]
ABL_NAMES = [r"\pgnd{}", "w/o certificate", "Random proposer", "Ridge proposer", "Context-only", "w/o route-event"]

open("tables_v9_rules.tex", "w").write(method_table("tab:v9-rules", r"Comparison of \pgnd{} with dispatching rules on the test set.", RULES, RULE_NAMES,
                                                 panels=[[0, 1, 2, 3, 4], [0, 5, 6, 7, 8]]))
open("tables_v9_drl.tex", "w").write(method_table("tab:v9-drl", r"Comparison of \pgnd{} with DRL-based methods on the test set.", DRL, DRL_NAMES))

EA = ["pgnd", "ea-ga", "ea-nsga2", "ea-nsga3", "ea-moead"]
EA_NAMES = [r"\pgnd{}", "GA", "NSGA-II", "NSGA-III", "MOEA/D"]
open("tables_v9_ea.tex", "w").write(method_table("tab:v9-ea", r"Comparison of \pgnd{} with multi-objective evolutionary algorithms on synthetic seeds 800--809.", EA, EA_NAMES, sets=SETS[:1],
    note="Each evolutionary algorithm is run five times independently on every case, and its value is the mean of the five runs. "
         + "Mean computing time per case: " + ", ".join(f"{n} {EA_TIME[m]:.1f}\,s" for m, n in zip(EA[1:], EA_NAMES[1:])) + "."))

# ablation: syn only for all variants; per-set summary for the variants run on several sets
out = []
abl_rows = []
for S, sname in SETS:
    for a, nm in zip(ABL[1:], ABL_NAMES[1:]):
        keys = cases(S, None, ("pgnd", a, "rule-lookahead"))
        if not keys:
            continue
        c = compare(keys, "pgnd", a)
        cb = compare(keys, a, "rule-lookahead")
        acc = [r.get("accepted", 0) for r in META[(S, a)]]
        abl_rows.append((sname, nm, len(keys), mean(V[k][a] for k in keys), cb, c, mean(acc) if acc else 0))
lines = [r"\begin{table}[!htbp]", r"\centering", r"\caption{Results of the ablation experiment.}", r"\label{tab:v9-ablation}",
         r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r"Set & Variant & Cases & Mean $V$ & $\Delta$ vs G-ECT & w/t/l vs G-ECT & $\Delta$ of \pgnd{} & w/t/l of \pgnd{} \\", r"\midrule"]
last = None
for sname, nm, n, mv, cb, c, acc in abl_rows:
    if last is not None and sname != last:
        lines.append(r"\midrule")
    lines.append(f"{sname if sname != last else ''} & {nm} & {n} & {mv:.4f} & {sd(cb['delta'])}\\%{cb['mark'].replace('-', '$-$')} & "
                 f"{cb['w']}/{cb['t']}/{cb['l']} & {sd(c['delta'])}\\%{c['mark'].replace('-', '$-$')} & {c['w']}/{c['t']}/{c['l']} \\\\")
    last = sname
lines += [r"\bottomrule", r"\end{tabular}}", r"\par\smallskip",
          r"{\raggedright\footnotesize Note: $\Delta$ vs G-ECT is the improvement of the variant over the base rule, and $\Delta$ of \pgnd{} is the improvement of \pgnd{} over the variant. "
          r"The markers follow Table~\ref{tab:v9-rules}, with the first method of each comparison as the reference.\par}", r"\end{table}"]
open("tables_v9_ablation.tex", "w").write("\n".join(lines) + "\n")

# known-event reference (syn)
REF = [("ref-rule-known", "G-ECT"), ("ref-pgnd-known", r"\pgnd{}"), ("ref-full-rollout-known", "Rollout over all candidates")]
keys = cases("syn", None, [a for a, _ in REF])
lines = [r"\begin{table}[!htbp]", r"\centering", r"\caption{Results with event outcomes known to the simulator on the synthetic test set.}",
         r"\label{tab:v9-reference}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"Method & Mean $V$ & $\Delta$ vs G-ECT & w/t/l vs G-ECT & Rollouts per case & Accepted per case \\", r"\midrule"]
for a, nm in REF:
    m, s = cell(keys, a)
    ms = [r for r in META[("syn", a)]]
    ro = mean(r.get("unique_rollouts", r.get("rollouts", 0)) for r in ms)
    ac = mean(r.get("accepted", 0) for r in ms) if a != "ref-full-rollout-known" else float("nan")
    if a == "ref-rule-known":
        lines.append(f"{nm} & {m:.4f} & -- & -- & -- & -- \\\\")
    else:
        c = compare(keys, a, "ref-rule-known")
        lines.append(f"{nm} & {m:.4f} & {sd(c['delta'])}\\%{c['mark'].replace('-', '$-$')} & {c['w']}/{c['t']}/{c['l']} & {ro:,.0f} & "
                     + ("--" if a == "ref-full-rollout-known" else f"{ac:.1f}") + r" \\")
c = compare(keys, "ref-pgnd-known", "ref-full-rollout-known")
lines += [r"\bottomrule", r"\end{tabular}}", r"\par\smallskip",
          rf"{{\raggedright\footnotesize Note: {len(keys)} cases. Against rollout over all candidates, \pgnd{{}} gives $\Delta={c['delta']:.2f}\%$ "
          rf"(w/t/l {c['w']}/{c['t']}/{c['l']}, $p{'<0.001' if c['p'] < 0.001 else '=' + format(c['p'], '.3f')}$).\par}}", r"\end{table}"]
open("tables_v9_reference.tex", "w").write("\n".join(lines) + "\n")

# numbers for the text
txt = []
for S, sname in SETS:
    keys = cases(S, None, ("pgnd", "rule-lookahead"))
    c = compare(keys, "pgnd", "rule-lookahead")
    d = [(V[k]["pgnd"] - V[k]["rule-lookahead"]) for k in keys if V[k]["pgnd"] > V[k]["rule-lookahead"] + EPS]
    d.sort()
    med = d[len(d) // 2] if len(d) % 2 else (d[len(d) // 2 - 1] + d[len(d) // 2]) / 2
    ms = META[(S, "pgnd")]
    txt.append(f"{sname}: vs G-ECT {c['delta']:.2f}% w/t/l {c['w']}/{c['t']}/{c['l']} p={fmt_p(c['p'])}; losses {len(d)} median {med:.4f} max {max(d):.4f}; "
               f"per case decisions {mean(r['decisions'] for r in ms):.1f} proposals {mean(r['proposals'] for r in ms):.1f} accepted {mean(r['accepted'] for r in ms):.2f} "
               f"rollouts {mean(r['rollouts'] for r in ms):.0f} unique {mean(r['unique_rollouts'] for r in ms):.0f}")
    for pi, pn in enumerate(["balanced", "makespan", "tardiness", "energy", "path", "stability"]):
        kp = [k for k in keys if k[1].endswith(f"pref{pi:02d}")]
        cp = compare(kp, "pgnd", "rule-lookahead")
        txt.append(f"   {pn}: {cp['delta']:.2f}% {cp['w']}/{cp['t']}/{cp['l']} p={fmt_p(cp['p'])}")
    for mode in ("raw", "realized"):
        km = [k for k in keys if f"_{mode}_" in k[1]]
        cm = compare(km, "pgnd", "rule-lookahead")
        txt.append(f"   {mode}: {cm['delta']:.2f}% p={fmt_p(cm['p'])}")
    for lev in ("L4", "L5"):
        kl = [k for k in keys if f"_{lev}_" in k[1]]
        cl = compare(kl, "pgnd", "rule-lookahead")
        txt.append(f"   {lev}: {cl['delta']:.2f}% p={fmt_p(cl['p'])}")
for S, sname in SETS:
    for a in ["daniel", "song", "hgt", "dqn", "pgnd-nocert", "rule-gest"]:
        keys = cases(S, None, (a, "rule-lookahead"))
        c = compare(keys, a, "rule-lookahead")
        txt.append(f"{sname} {a} vs G-ECT {c['delta']:.2f}% {c['w']}/{c['t']}/{c['l']} p={fmt_p(c['p'])}")
open("numbers_v9.txt", "w").write("\n".join(txt) + "\n")
print("\n".join(txt))
