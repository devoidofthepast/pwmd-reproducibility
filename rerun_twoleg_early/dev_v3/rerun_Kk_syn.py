"""Relaunch syn 632-639 of step 3(b): these 8 jobs exited at start with 0xC0000142 (18:37), no rows written. Same arms and dev_v3.py as run_Kk.py."""
import subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
H = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONWARNINGS="ignore", OMP_NUM_THREADS="1")
jobs = [("syn", s, "pgnd_M0.02_K4_k1,pgnd_M0.02_K4_k2,pgnd_M0.02_K8_k2") for s in range(632, 640)]
def run(j):
    S, s, arms = j
    with open(H / f"log_Kk_{S}_{s}.txt", "w") as h:
        return subprocess.run([sys.executable, str(H / "dev_v3.py"), S, str(s), arms, "--out", str(H / "runs_Kk")], stdout=h, stderr=h, env=env).returncode
with ThreadPoolExecutor(8) as ex:
    print(list(ex.map(run, jobs)))
(H / "Kk_syn_rerun.done").write_text("ok")
