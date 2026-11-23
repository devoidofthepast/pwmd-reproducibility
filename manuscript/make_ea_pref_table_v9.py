"""Adds the per-preference block to tables_v9_ea.tex (run after make_tables_v9.py, which rewrites that file).
Same statistics and format as the rows above it (make_tables_v9.method_table). Backup of the earlier table: tables_v9_ea_before_pref.tex."""
src = open("make_tables_v9.py", encoding="utf-8").read()
g = {}
exec(compile(src[:src.index("RULES = [")], "mt", "exec"), g)
cases, compare, cell = g["cases"], g["compare"], g["cell"]
EA = ["pgnd", "ea-ga", "ea-nsga2", "ea-nsga3", "ea-moead"]
PREFS = ["Balanced", "Makespan", "Tardiness", "Energy", "Path pressure", "Stability"]
keys = cases("syn", None, EA)
rows = [r"\midrule", r"\multicolumn{7}{@{}l}{\emph{By preference}} \\"]
for pi, name in enumerate(PREFS):
    kp = [k for k in keys if k[1].endswith(f"pref{pi:02d}")]
    st = {a: cell(kp, a) for a in EA}
    best = min(v[0] for v in st.values())
    out = []
    for a in EA:
        m, s = st[a]
        txt = rf"\textbf{{{m:.4f}}} ({s:.4f})" if abs(m - best) < 5e-5 else f"{m:.4f} ({s:.4f})"
        if a != "pgnd":
            txt += compare(kp, "pgnd", a)["mark"].replace("-", "$-$")
        out.append(txt)
    rows.append(rf"\multicolumn{{2}}{{@{{}}l}}{{{name}}} & " + " & ".join(out) + r" \\")
t = open("tables_v9_ea.tex", encoding="utf-8").read()
assert "By preference" not in t
anchor = r"\bottomrule"
t = t.replace(anchor, "\n".join(rows) + "\n" + anchor, 1)
t = t.replace(r"w/t/l counts the cases in which \pgnd{} wins, ties, and loses. ",
              r"w/t/l counts the cases in which \pgnd{} wins, ties, and loses. Each preference row contains 40 cases. ")
open("tables_v9_ea.tex", "w", encoding="utf-8").write(t)
print("\n".join(rows))
