"""Amendment 3 section 6: gate (S, C, L) and heuL (L, heuristic from r2_heu_choice.txt). Usage: python run_r2_extra.py WORKERS"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_r2 as R
jobs = [("r2", b, s, "gate") for b, rng in (("S", range(800, 820)), ("C", range(820, 830)), ("L", range(830, 840))) for s in rng]
jobs += [("r2", "L", s, "heuL") for s in range(830, 840)]
with ThreadPoolExecutor(int(sys.argv[1])) as ex:
    res = list(ex.map(R.run, jobs))
print(res, flush=True)
(R.H / "r2_extra.done").write_text(repr(res))
