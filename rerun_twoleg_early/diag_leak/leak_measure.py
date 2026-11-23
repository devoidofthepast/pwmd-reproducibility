"""Diagnosis of the event-timing leak (2026-11-22, GPT review point 1). Not a result of the paper; decisions are unchanged.

At every commit, the committed operation and its transport are also timed in a shadow copy of the state on an instance that
keeps only the events already revealed (the event ids of `known`). A commit is "shifted" when its true timing differs from
the shadow timing, i.e. when an unrevealed event window moved the committed intervals. The real commit is done exactly as in
the test runs, and the case value is checked against the frozen test record.

Usage: python leak_measure.py BLOCK SEED   (BLOCK S or C; arms rule-lookahead and pgnd)
"""
import copy
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(H / "test_v3"), str(H / "src"), str(H / "baselines"), str(H.parent / "twoleg")]
import test_v3 as T  # noqa: E402
import event_blind as EB  # noqa: E402
import blind_arms as BA  # noqa: E402
import scenario_cert3 as S3  # noqa: E402

ORIG = EB._commit_strict
LOG = []


def wrapped(*, known, true, preference, true_job, true_op, machine, state, operations, transports):
    ids = {e.event_id for e in known.events}
    shadow_true = true.replace(events=tuple(e for e in true.events if e.event_id in ids))
    n_hidden = sum(1 for e in true.events if e.event_id not in ids)
    sh_state, sh_ops, sh_tr = copy.deepcopy(state), list(operations), list(transports)
    shadow = None
    try:
        ORIG(known=known, true=shadow_true, preference=preference, true_job=true_job, true_op=true_op, machine=machine,
             state=sh_state, operations=sh_ops, transports=sh_tr)
        so = sh_ops[-1]
        st = next((m for m in reversed(sh_tr) if m.operation_id == so.operation_id), None)
        shadow = (so.start, so.end, st.start if st else None, st.end if st else None)
    except Exception as e:  # noqa: BLE001  the shadow may be infeasible; recorded, the real commit is unaffected
        shadow = ("err", type(e).__name__)
    ORIG(known=known, true=true, preference=preference, true_job=true_job, true_op=true_op, machine=machine,
         state=state, operations=operations, transports=transports)
    ro = operations[-1]
    rt = next((m for m in reversed(transports) if m.operation_id == ro.operation_id), None)
    real = (ro.start, ro.end, rt.start if rt else None, rt.end if rt else None)
    LOG.append(dict(real=real, shadow=shadow, n_hidden=n_hidden))


EB._commit_strict = wrapped
BA._commit_strict = wrapped
S3._ns["_commit_strict"] = wrapped


def main():
    block, seed = sys.argv[1], int(sys.argv[2])
    frozen = {}
    for f in (H / "test_v3" / "runs" / block).glob(f"*_{seed}.jsonl"):
        for line in open(f):
            r = json.loads(line)
            frozen[(r["case_id"], r["arm"])] = r["value"]
    keep = {"rule-lookahead", "pgnd"}
    runners = [a for a in T.arms("rules", block, seed) + T.arms("pgnd", block, seed) if a[0] in keep]
    out = open(Path(__file__).parent / f"leak_{block}_{seed}.jsonl", "w")
    sc, prefs = T.C.scales(), T.C.preferences()
    for set_name, case in T.cases_for(block, seed):
        for pi, pref in enumerate(prefs):
            cid = f"{set_name}_seed{seed}_{case['level']}_{case['mode']}_pref{pi:02d}"
            for arm, run in runners:
                LOG.clear()
                sched, _ = run(set_name, case, pi, pref)
                val = T.C.partial_value(case["true"], sched.operations, sched.transports, pref, sc)
                n = len(LOG)
                sh = [x for x in LOG if x["shadow"][0] != "err" and x["real"] != x["shadow"]]
                err = sum(1 for x in LOG if x["shadow"][0] == "err")
                delay = [x["real"][0] - x["shadow"][0] for x in sh]
                out.write(json.dumps(dict(case_id=cid, set=set_name, arm=arm, commits=n, shifted=len(sh), shadow_err=err,
                                          delay_sum=sum(delay), delay_max=max(delay, default=0),
                                          hidden_at_commit=sum(x["n_hidden"] > 0 for x in LOG),
                                          value=val, frozen=frozen.get((cid, arm)))) + "\n")
                out.flush()


if __name__ == "__main__":
    main()
