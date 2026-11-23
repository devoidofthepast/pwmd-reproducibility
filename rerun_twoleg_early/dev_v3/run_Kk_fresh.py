"""2026-11-22 (GPT review point 4): the K x k grid of run_Kk.py on Mk1/Edata7 development seeds 770-779, whose events are
not shared with the training seeds 600-619. Same dev_v3.py, arms and settings; output runs_Kk_fresh."""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

H = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONWARNINGS="ignore", OMP_NUM_THREADS="1")
arms = "la,pgnd_M0.02_K4_k1,pgnd_M0.02_K4_k2,pgnd_M0.02_K8_k1,pgnd_M0.02_K8_k2"
jobs = [(S, s) for S in ("mk1", "edata7") for s in range(770, 780)]


def run(j):
    S, s = j
    with open(H / f"log_Kkfresh_{S}_{s}.txt", "w") as h:
        return subprocess.run([sys.executable, str(H / "dev_v3.py"), S, str(s), arms, "--out", str(H / "runs_Kk_fresh")],
                              stdout=h, stderr=h, env=env).returncode


with ThreadPoolExecutor(int(sys.argv[1])) as ex:
    print(list(ex.map(run, jobs)))
(H / "Kk_fresh.done").write_text("ok")
