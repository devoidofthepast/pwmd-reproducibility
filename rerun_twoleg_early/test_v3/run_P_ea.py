"""v3 step 5: test set P, ea group on P-S 800-809 (idle machine), one job per (seed, method), 12 workers as in the budget dev run.
Retries a job that exits with 0xC0000142 (process failed to start); test_v3.py resumes from its output."""
import subprocess, sys, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
H = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONWARNINGS="ignore", OMP_NUM_THREADS="1")
jobs = [("S", s, "ea", m) for m in ("ga", "nsga2", "nsga3", "moead") for s in range(800, 810)]
(H / "logs").mkdir(exist_ok=True)
def run(j):
    b, s, g, m = j
    for _ in range(5):
        with open(H / "logs" / f"{b}_{g}_{m}_{s}.txt", "a") as h:
            rc = subprocess.run([sys.executable, str(H / "test_v3.py"), b, str(s), g, "--method", m], stdout=h, stderr=h, env=env).returncode
        if rc != 3221225794:
            return (b, s, g, m, rc)
        time.sleep(30)
    return (b, s, g, m, rc)
with ThreadPoolExecutor(int(sys.argv[1])) as ex:
    res = list(ex.map(run, jobs))
print(res, flush=True)
(H / "P_ea.done").write_text("ok")
