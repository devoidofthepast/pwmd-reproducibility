"""v3 amendment 3 launcher. Usage: python run_r2.py WORKERS PHASE   (PHASE main | heuL)
main: drl23 (S, C, L) and the M top-up and heu on S first, then mis on S, heu on C, rndL on L. heuL: after r2_heu_choice.txt.
Retries a job that exits with 0xC0000142; review_r2 / dev_v3 resume from their output."""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

H = Path(__file__).resolve().parent
DEV = H.parent / "dev_v3" / "dev_v3.py"
env = dict(os.environ, PYTHONWARNINGS="ignore", OMP_NUM_THREADS="1")
(H / "logs").mkdir(exist_ok=True)


def jobs(phase):
    if phase == "heuL":
        return [("r2", "L", s, "heuL") for s in range(830, 840)]
    j = [("r2", b, s, "drl23") for b, rng in (("S", range(800, 820)), ("C", range(820, 830)), ("L", range(830, 840))) for s in rng]
    j += [("dev", "syn", s, "la,pgnd_M0.05_K8_k1,pgnd_M0.1_K8_k1") for s in range(630, 640)]
    j += [("r2", "S", s, "heu") for s in range(800, 820)]
    j += [("r2", "L", s, "rndL") for s in range(830, 840)]  # long jobs early so they do not form the tail
    j += [("r2", "S", s, "mis") for s in range(800, 820)]
    j += [("r2", "C", s, "heu") for s in range(820, 830)]
    return j


def run(job):
    kind, b, s, g = job
    if kind == "r2":
        cmd = [sys.executable, str(H / "review_r2.py"), b, str(s), g]
        log = H / "logs" / f"r2_{b}_{g}_{s}.txt"
    else:
        cmd = [sys.executable, str(DEV), b, str(s), g, "--out", str(H.parent / "dev_v3" / "runs_M2")]
        log = H / "logs" / f"r2_M2_{s}.txt"
    rc = None
    for _ in range(5):
        with open(log, "a") as h:
            rc = subprocess.run(cmd, stdout=h, stderr=h, env=env).returncode
        if rc != 3221225794:
            break
        time.sleep(30)
    print(job, rc, time.strftime("%H:%M:%S"), flush=True)
    return job, rc


if __name__ == "__main__":
    workers, phase = int(sys.argv[1]), sys.argv[2]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(run, jobs(phase)))
    print(res, flush=True)
    (H / f"r2_{phase}.done").write_text(repr(res))
