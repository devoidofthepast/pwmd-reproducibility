"""v3 amendment 1: random proposer (rnd) on P-S 800-819 and P-C 820-829. Start only after run_P_ea.py has finished.
Retries a job that exits with 0xC0000142 (process failed to start); test_v3.py resumes from its output."""
import subprocess, sys, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
H = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONWARNINGS="ignore", OMP_NUM_THREADS="1")
jobs = [("C", s) for s in range(820, 830)] + [("S", s) for s in range(800, 820)]
(H / "logs").mkdir(exist_ok=True)
def run(j):
    b, s = j
    for _ in range(5):
        with open(H / "logs" / f"{b}_random_{s}.txt", "a") as h:
            rc = subprocess.run([sys.executable, str(H / "test_v3_rnd.py"), b, str(s)], stdout=h, stderr=h, env=env).returncode
        if rc != 3221225794:
            return (b, s, rc)
        time.sleep(30)
    return (b, s, rc)
with ThreadPoolExecutor(int(sys.argv[1])) as ex:
    res = list(ex.map(run, jobs))
print(res, flush=True)
(H / "P_rnd.done").write_text("ok")
