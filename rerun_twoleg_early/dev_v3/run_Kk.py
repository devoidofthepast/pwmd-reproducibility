"""v3 step 3(b): K in {4, 8} x k in {1, 2} at M = 0.02 on syn 630-639 and Mk1/Edata7 610-619 (syn K8 k1 and la reused)."""
import subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
H = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONWARNINGS="ignore", OMP_NUM_THREADS="1")
four = "pgnd_M0.02_K4_k1,pgnd_M0.02_K4_k2,pgnd_M0.02_K8_k1,pgnd_M0.02_K8_k2"
jobs = [(S, s, "la," + four) for S in ("mk1", "edata7") for s in range(610, 620)] + [("syn", s, "pgnd_M0.02_K4_k1,pgnd_M0.02_K4_k2,pgnd_M0.02_K8_k2") for s in range(630, 640)]
def run(j):
    S, s, arms = j
    with open(H / f"log_Kk_{S}_{s}.txt", "w") as h:
        return subprocess.run([sys.executable, str(H / "dev_v3.py"), S, str(s), arms, "--out", str(H / "runs_Kk")], stdout=h, stderr=h, env=env).returncode
with ThreadPoolExecutor(int(sys.argv[1])) as ex:
    print(list(ex.map(run, jobs)))
(H / "Kk.done").write_text("ok")
