"""v3 step 4: test set P, every group except ea (EA and timing later, on an idle machine). Heavy jobs first.
Retries a job that exits with 0xC0000142 (process failed to start); test_v3.py resumes from its output."""
import subprocess, sys, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
H = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONWARNINGS="ignore", OMP_NUM_THREADS="1")
jobs = ([("L", s, "pgnd") for s in range(830, 840)] + [("C", s, "pgnd") for s in range(820, 830)]
        + [("S", s, g) for g in ("reference", "ablation", "pgnd") for s in range(800, 820)]
        + [(b, s, g) for g in ("drl", "rules") for b, r in (("S", range(800, 820)), ("C", range(820, 830)), ("L", range(830, 840))) for s in r])
(H / "logs").mkdir(exist_ok=True)
def run(j):
    b, s, g = j
    for _ in range(5):
        with open(H / "logs" / f"{b}_{g}_{s}.txt", "a") as h:
            rc = subprocess.run([sys.executable, str(H / "test_v3.py"), b, str(s), g], stdout=h, stderr=h, env=env).returncode
        if rc != 3221225794:
            return (b, s, g, rc)
        time.sleep(30)
    return (b, s, g, rc)
with ThreadPoolExecutor(int(sys.argv[1])) as ex:
    res = list(ex.map(run, jobs))
print(res, flush=True)
(H / "P_noea.done").write_text("ok")
