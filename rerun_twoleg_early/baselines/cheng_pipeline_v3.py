"""v3: Cheng DQN tuning and final training, procedure of cheng_pipeline.py with v3 selection: grid episodes {3000, 10000} x
lr {1e-4, 1e-3}, one run per cell, cell chosen by mean V on validation seeds 640-649 (random events, strict clock); three
runs at the chosen cell; per preference the run with the lowest validation V."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "drl_v3" / "cheng"; ROOT.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONWARNINGS="ignore", OMP_NUM_THREADS="1")
W = sys.argv[1] if len(sys.argv) > 1 else "6"
def run(args, log):
    with open(ROOT / log, "a") as h:
        subprocess.run([sys.executable, str(HERE / "cheng_dqn_v3.py"), *args, "--workers", W], stdout=h, stderr=h, check=True, env=env)
res = []
for ep, lr, tag in (("3000", "0.0001", "3000_1e-4"), ("3000", "0.001", "3000_1e-3"), ("10000", "0.0001", "10000_1e-4"), ("10000", "0.001", "10000_1e-3")):
    d = ROOT / f"grid_{tag}"
    if not (d / "run0" / "dqn_pref05.pt").exists():
        run(["train", "--episodes", ep, "--lr", lr, "--models", str(d)], f"grid_{tag}.log")
    if not (d / "valid.json").exists():
        run(["valid", "--models", str(d)], f"grid_{tag}.log")
    v = json.load(open(d / "valid.json"))
    res.append((tag, ep, lr, sum(x[2] for x in v) / len(v)))
    print("grid", res[-1], flush=True)
best = min(res, key=lambda r: r[3])
json.dump({"grid": res, "selected": best}, open(ROOT / "grid_selection.json", "w"), indent=1)
final = ROOT / "final"
if not (final / "run2" / "dqn_pref05.pt").exists():
    run(["train", "--episodes", best[1], "--lr", best[2], "--runs", "3", "--models", str(final)], "final.log")
if not (final / "valid.json").exists():
    run(["valid", "--models", str(final)], "final.log")
v = json.load(open(final / "valid.json"))
sel = {p: min((x for x in v if x[1] == p), key=lambda x: x[2]) for p in range(6)}
(final / "selected").mkdir(exist_ok=True)
for p, x in sel.items():
    shutil.copy(Path(x[0]) / f"dqn_pref{p:02d}.pt", final / "selected" / f"dqn_pref{p:02d}.pt")
json.dump(sel, open(final / "selected" / "selection.json", "w"), indent=1)
(ROOT / "pipeline.done").write_text(time.ctime())
print("done", best, flush=True)
