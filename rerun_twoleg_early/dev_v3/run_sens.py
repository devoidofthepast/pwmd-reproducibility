"""v3 step 3(c) sensitivity (report only, not used for selection): at M=0.02, K=8, k=2 on syn 630-639, Mk1/Edata7 610-619.
Ensemble members 1 and 3 (first N of the 5 trained members); 5 training seeds (600-604, 5 members, trained here with the
same command as the 20-seed v3 models). la, and 5 members / 20 seeds, are reused from runs_Kk (and runs_M for syn la).
A job that exits with 0xC0000142 (process failed to start) is retried; dev_v3.py resumes from its output file."""
import subprocess, sys, os, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
H = Path(__file__).resolve().parent; R = H.parent
env = dict(os.environ, PYTHONWARNINGS="ignore", OMP_NUM_THREADS="1")
M5 = R / "models_physics_train_600_604_v3"
ARM = "pgnd_M0.02_K8_k2"
trained = threading.Event()
def train():
    e = dict(env, PYTHONPATH="src")
    with open(R / "train_physics_5seeds_v3.log", "w") as h:
        rc = subprocess.run([sys.executable, "-m", "su_modfjsp_ha_pc.run_pgnd_v4_train", "--context-dir", "labels_train_600_619_v3",
                             "--output-dir", str(M5), "--start-seed", "600", "--end-seed", "604",
                             "--objective-scales", "scales_train_600_619_v3/objective_scales.json"], stdout=h, stderr=h, cwd=R, env=e).returncode
    print("train rc", rc, flush=True); trained.set()
sets = [("syn", s) for s in range(630, 640)] + [(S, s) for S in ("mk1", "edata7") for s in range(610, 620)]
jobs = [(S, s, ["--members", str(n)], f"m{n}") for n in (1, 3) for S, s in sets] + [(S, s, ["--model-dir", str(M5)], "5seeds") for S, s in sets]
def run(j):
    S, s, extra, tag = j
    if tag == "5seeds":
        trained.wait()
    for attempt in range(5):
        with open(H / f"log_sens_{tag}_{S}_{s}.txt", "a") as h:
            rc = subprocess.run([sys.executable, str(H / "dev_v3.py"), S, str(s), ARM, *extra, "--out", str(H / "runs_sens")], stdout=h, stderr=h, env=env).returncode
        if rc != 3221225794:
            return rc
        time.sleep(30)
    return rc
threading.Thread(target=train).start()
with ThreadPoolExecutor(int(sys.argv[1])) as ex:
    print(list(ex.map(run, jobs)), flush=True)
(H / "sens.done").write_text("ok")
