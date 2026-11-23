"""Holm check (2026-11-22, GPT review point 6). Re-executes make_tables_v9.py with its table writes redirected to this
folder and logs every compare() it makes (table label, set, level, arms, p, delta). Nothing in the manuscript folder is
written. Output: p_values.json."""
import builtins
import inspect
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
MS = HERE.parents[1] / "manuscript_v9_rerun_20261113"
LOG = []
_open = builtins.open


def guarded_open(f, mode="r", *a, **k):
    if "w" in mode and str(f).startswith("tables_v9"):
        f = HERE / ("redirected_" + Path(f).name)
    return _open(f, mode, *a, **k)


src = _open(MS / "make_tables_v9.py", encoding="utf-8").read()
ns = {"__name__": "make_tables_v9_holm", "open": guarded_open}
os.chdir(MS)
code = compile(src, str(MS / "make_tables_v9.py"), "exec")

# wrap compare after its definition: execute up to the first table, patch, then the rest
cut = src.index('open("tables_v9_rules.tex", "w")')
exec(compile(src[:cut], "make_tables_v9_part1", "exec"), ns)
_compare = ns["compare"]


def compare(keys, a, b):
    c = _compare(keys, a, b)
    fr = inspect.currentframe().f_back
    loc = fr.f_locals
    LOG.append(dict(table=loc.get("label", f"line{fr.f_lineno}"), line=fr.f_lineno, set=loc.get("S"), level=loc.get("lev"),
                    a=a, b=b, n=c["n"], p=c["p"], delta=round(c["delta"], 3), row="all" if fr.f_lineno == 138 else "row"))
    return c


ns["compare"] = compare
exec(compile("\n" * src[:cut].count("\n") + src[cut:], str(MS / "make_tables_v9.py"), "exec"), ns)
_open(HERE / "p_values.json", "w").write(json.dumps(LOG, indent=0))
print("logged", len(LOG))
