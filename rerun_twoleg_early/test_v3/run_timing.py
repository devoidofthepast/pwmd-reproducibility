"""Run the timing jobs one after another (single process). Retries a job that fails to start (0xC0000142)."""
import os, subprocess, sys, time
from pathlib import Path
H = Path(__file__).resolve().parent
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONWARNINGS="ignore")
jobs = [(b, s, g) for b, seeds in (("S", (800, 801)), ("C", (820, 821)), ("L", (830, 831)))
        for g in (("rules", "drl", "pgnd", "reference") if b == "S" else ("rules", "drl", "pgnd")) for s in seeds]
(H / "logs").mkdir(exist_ok=True)
for b, s, g in jobs:
    for _ in range(5):
        with open(H / "logs" / f"timing_{b}_{s}_{g}.txt", "a") as h:
            rc = subprocess.run([sys.executable, str(H / "timing_v3.py"), b, str(s), g], stdout=h, stderr=h, env=env).returncode
        if rc != 3221225794:
            break
        time.sleep(30)
    print(b, s, g, rc, time.strftime("%H:%M:%S"), flush=True)
(H / "timing.done").write_text("ok")
