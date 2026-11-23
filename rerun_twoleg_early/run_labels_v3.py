"""v3 step 1: proposer labels for training seeds 600-619 with per-seed random events (v3_random_events.py launcher)."""
import itertools, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = HERE / "labels_train_600_619_v3"; LOG = HERE / "label_logs_v3"; LOG.mkdir(exist_ok=True)
SCALES = HERE / "scales_train_600_619_v3" / "objective_scales.json"
env = dict(os.environ, PYTHONPATH=f"{HERE / 'src'};{HERE / 'baselines'};{HERE.parent / 'twoleg'}", PYTHONWARNINGS="ignore", OMP_NUM_THREADS="1")
def one(ctx):
    seed, level, mode, pi = ctx
    name = f"seed{seed}_{level}_{mode}_pref{pi:02d}"
    with open(LOG / f"{name}.log", "w") as h:
        r = subprocess.run([sys.executable, str(HERE / "v3_random_events.py"), "su_modfjsp_ha_pc.run_pgnd_v4_context_labels", "--seed", str(seed), "--role", "train",
                            "--level", level, "--mode", mode, "--preference-index", str(pi), "--output-dir", str(OUT),
                            "--objective-scales", str(SCALES)], stdout=h, stderr=subprocess.STDOUT, env=env, cwd=str(HERE))
    return name, r.returncode
seeds = [int(s) for s in sys.argv[2:]] if len(sys.argv) > 2 else list(range(600, 620))
ctxs = list(itertools.product(seeds, ("L4", "L5"), ("raw", "realized"), range(6)))
workers = int(sys.argv[1])
with ThreadPoolExecutor(workers) as ex:
    for name, rc in ex.map(one, ctxs):
        print(name, rc, flush=True)
