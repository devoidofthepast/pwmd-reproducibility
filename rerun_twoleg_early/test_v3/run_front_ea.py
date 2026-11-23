"""Front experiment, EA group: 3 methods x 5 runs x seeds 800-809 = 150 tasks, 12 single-threaded processes (frozen section 8).
Resumable (front_v3.py skips written rows). Writes front_ea.done only if every task returned 0."""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
H = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONWARNINGS="ignore", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
jobs = [(s, m, r) for m in ("nsga2", "nsga3", "moead") for r in range(5) for s in range(800, 810)]
(H / "front" / "logs").mkdir(parents=True, exist_ok=True)
def run(j):
    s, m, r = j
    with open(H / "front" / "logs" / f"ea_{m}_run{r}_{s}.log", "a") as h:
        return j, subprocess.run([sys.executable, str(H / "front_v3.py"), str(s), "ea", "--method", m, "--run", str(r)], stdout=h, stderr=h, env=env).returncode
with ThreadPoolExecutor(12) as ex:
    res = list(ex.map(run, jobs))
bad = [j for j, rc in res if rc != 0]
print("failed", bad, flush=True)
if not bad:
    (H / "front_ea.done").write_text("ok")
