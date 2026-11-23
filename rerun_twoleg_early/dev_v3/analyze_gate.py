"""Pilot analysis (development): gated variants vs PGND (M=0.02, K=8, k=2) from runs_Kk, per case."""
import glob, json, sys
from pathlib import Path
from scipy.stats import wilcoxon
D = Path(__file__).resolve().parent
S = sys.argv[1] if len(sys.argv) > 1 else "syn"
base = {}
for fp in glob.glob(str(D / "runs_Kk" / f"{S}_*.jsonl")):
    for r in map(json.loads, open(fp)):
        if r["arm"] == "pgnd_M0.02_K8_k2":
            base[r["case_id"]] = r
gate = {}
for fp in glob.glob(str(D / "runs_gate" / f"{S}_*.jsonl")):
    for r in map(json.loads, open(fp)):
        gate.setdefault(r["arm"], {})[r["case_id"]] = r
print(f"{S}: PGND rows {len(base)}")
for arm in sorted(gate):
    ids = [c for c in gate[arm] if c in base]
    if not ids:
        continue
    vb = [base[c]["value"] for c in ids]; vg = [gate[arm][c]["value"] for c in ids]
    w = sum(g < b - 1e-12 for g, b in zip(vg, vb)); l = sum(g > b + 1e-12 for g, b in zip(vg, vb)); t = len(ids) - w - l
    d = 1 - (sum(vg) / len(vg)) / (sum(vb) / len(vb))
    try:
        p = wilcoxon(vg, vb).pvalue if w + l else 1.0
    except ValueError:
        p = 1.0
    rb = sum(base[c]["rollouts"] for c in ids) / len(ids); rg = sum(gate[arm][c]["rollouts"] for c in ids) / len(ids)
    ub = sum(base[c].get("unique_rollouts", base[c]["rollouts"]) for c in ids) / len(ids)
    ug = sum(gate[arm][c].get("unique_rollouts", gate[arm][c]["rollouts"]) for c in ids) / len(ids)
    sk = sum(gate[arm][c]["gate_skipped"] for c in ids) / max(1, sum(gate[arm][c]["gate_calls"] for c in ids))
    acc_b = sum(base[c]["accepted"] for c in ids) / len(ids); acc_g = sum(gate[arm][c]["accepted"] for c in ids) / len(ids)
    bad = sum(gate[arm][c]["route_bad"] + gate[arm][c]["timing_bad"] + gate[arm][c]["hard_viol"] + gate[arm][c]["event_viol"] for c in ids)
    print(f"{arm:12s} n={len(ids):3d}  dV vs PGND {d*100:+.2f}% (+ = gate better)  W/T/L {w}/{t}/{l}  p={p:.3f}  "
          f"rollouts {rg:.0f}/{rb:.0f} ({rg/rb*100:.0f}%)  unique {ug:.0f}/{ub:.0f} ({ug/ub*100:.0f}%)  skipped {sk*100:.0f}%  "
          f"accepted {acc_g:.2f}/{acc_b:.2f}  violations {bad}")
