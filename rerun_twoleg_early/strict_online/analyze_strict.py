"""Strict online execution versus the frozen test runs on the synthetic test set, with the paper's statistics
(analyze_c1.compare: seed-cluster exact two-sided Wilcoxon, w/t/l, relative difference)."""
import collections
import contextlib
import glob
import io
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parents[1]
_src = open(H / "analyze_c1.py", encoding="utf-8").read()
_src = _src[:_src.index("\nout = {}")]  # helpers only (wil, compare, seed_of)
A = type("A", (), {})()
_ns = {}
exec(compile(_src, "analyze_c1_helpers", "exec"), _ns)
A.compare = _ns["compare"]

v = collections.defaultdict(dict)
bad = 0
for f in glob.glob(str(Path(__file__).parent / "runs" / "strict_*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        c = r["case_id"].split("_", 1)[1]
        tag = "pgnd" if r["arm"] == "pgnd" else "rule"
        v[c][f"{tag}-online"] = r["value"]
        v[c][f"{tag}-frozen"] = r["frozen"]
        bad += r["route_bad"] + r["timing_bad"] + r["hard_viol"] + r["event_viol"] + r.get("failures", 0)
full = {c: x for c, x in v.items() if len(x) == 4}
print("cases", len(full), "check errors", bad)
for a, b in (("pgnd-online", "rule-online"), ("pgnd-frozen", "rule-frozen"), ("rule-online", "rule-frozen"), ("pgnd-online", "pgnd-frozen")):
    print(f"{a:12s} vs {b:12s}", A.compare(full, a, b))
for k in ("pgnd-online", "rule-online", "pgnd-frozen", "rule-frozen"):
    print(k, "mean V", round(sum(x[k] for x in full.values()) / len(full), 4))
