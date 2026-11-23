"""Amendment 3 section 7: when every rndL process has ended, start heuL seeds 836-839 on the freed cores (R2_FAST_VALUE=1),
one at a time while fewer than CAP experiment workers are active, so that the machine is not oversubscribed.
The queued copies in the 6-worker heuL launcher later skip them (per-case resume, per-job lock)."""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil

CAP = 16
os.environ["R2_FAST_VALUE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_r2 as R  # noqa: E402


def workers(tag=None):
    n = 0
    for p in psutil.process_iter(["cmdline"]):
        try:
            c = " ".join(p.info["cmdline"] or [])
        except Exception:
            continue
        if ("review_r2.py" in c or "dev_v3.py" in c) and (tag is None or tag in c):
            n += 1
    return n


while workers(" rndL"):
    time.sleep(120)
print("rndL finished", time.strftime("%H:%M:%S"), flush=True)
pending = [("r2", "L", s, "heuL") for s in range(836, 840)]
futures = []
with ThreadPoolExecutor(4) as ex:
    for job in pending:
        while workers() >= CAP:
            time.sleep(60)
        print("start", job, "active", workers(), time.strftime("%H:%M:%S"), flush=True)
        futures.append(ex.submit(R.run, job))
        time.sleep(20)
    res = [f.result() for f in futures]
print(res, time.strftime("%H:%M:%S"), flush=True)
(R.H / "r2_handoff.done").write_text(repr(res))
