"""v3 step 5 timing (全部重跑规则_冻结v3.md section 6; protocol in 执行记录.md "计时实验口径", 2026-11-13).
One process, one thread, CPU only, idle machine. The GPU stays visible to the process, because hiding it with
CUDA_VISIBLE_DEVICES=-1 crashes long PGND runs on this machine (执行记录.md, "CPU 崩溃排查"); the proposer inference is
forced to the CPU, and after every case torch.cuda.memory_allocated() must be 0.
First two seeds of each block: S 800-801, C 820-821 (Mk1 and Edata7), L 830-831.
Groups: rules, drl, pgnd (pgnd and pgnd-nocert) on every block; reference (rule, PGND and rollout over all candidates, events known) on S.
For the PGND arms the time between consecutive calls of the nomination selector is recorded as the duration of one decision.
Usage: python timing_v3.py BLOCK SEED GROUP   -> test_v3/timing/{BLOCK}_{SEED}_{GROUP}.jsonl (resumable)."""
import json
import sys
from pathlib import Path
from time import perf_counter

block, seed, group = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.argv = sys.argv[:1]
H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / "test_v3")]
import test_v3 as T  # noqa: E402
import torch  # noqa: E402

assert torch.get_num_threads() == 1
import su_modfjsp_ha_pc.pgnd_v4_model as PM  # noqa: E402

_pred = PM.predict_v4_details


def _cpu_predict(artifact, groups, *, groups_per_batch=128, device=None):
    return _pred(artifact, groups, groups_per_batch=groups_per_batch, device="cpu")


PM.predict_v4_details = _cpu_predict
import scenario_cert3 as S3  # noqa: E402

STAMPS = []
_orig = S3.neural_topk_selector


def timed_selector(artifacts, k):
    sel = _orig(artifacts, k)

    def select(*a, **kw):
        STAMPS.append(perf_counter())
        return sel(*a, **kw)
    return select


S3.neural_topk_selector = timed_selector
out = H / "test_v3" / "timing"
out.mkdir(exist_ok=True)
path = out / f"{block}_{seed}_{group}.jsonl"
done = {(r["case_id"], r["arm"]) for r in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
sc, prefs = T.C.scales(), T.C.preferences()
arm_list = T.arms(group, block, seed)
with path.open("a") as f:
    for set_name, case in T.cases_for(block, seed):
        for pi, pref in enumerate(prefs):
            cid = f"{set_name}_seed{seed}_{case['level']}_{case['mode']}_pref{pi:02d}"
            for arm, run in arm_list:
                if (cid, arm) in done:
                    continue
                STAMPS.clear()
                t0 = perf_counter()
                s, info = run(set_name, case, pi, pref)
                t1 = perf_counter()
                st = STAMPS + [t1] if STAMPS else []
                dec = [b - a for a, b in zip(st[:-1], st[1:])]
                rec = dict(case_id=cid, set=set_name, seed=seed, level=case["level"], mode=case["mode"], pref=pi, arm=arm,
                           value=T.C.partial_value(case["true"], s.operations, s.transports, pref, sc), wall_s=round(t1 - t0, 4),
                           **{k: v for k, v in (info or {}).items() if isinstance(v, (int, float))})
                if dec:
                    rec.update(decision_mean_s=round(sum(dec) / len(dec), 4), decision_max_s=round(max(dec), 4), timed_decisions=len(dec))
                gpu = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                assert gpu == 0, f"GPU memory used: {gpu}"
                rec["gpu_bytes"] = gpu
                f.write(json.dumps(rec) + "\n")
                f.flush()
