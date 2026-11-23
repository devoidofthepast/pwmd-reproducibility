"""Parity of R2_FAST_VALUE=1 against stored outputs: heu-cost and pgnd on S seed 800 (first 4 cases each), exact equality."""
import json, os, sys, time
os.environ["R2_FAST_VALUE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["x", "S", "800", "heu"]
import review_r2 as R
T = R.T
old = {}
for f in ("runs/S/heu_800.jsonl", "runs/S/pgnd_800.jsonl"):
    for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)):
        r = json.loads(l); old[(r["case_id"], r["arm"])] = r
arms = dict(R.arms("heu", "S", 800)); arms.update(dict(T.arms("pgnd", "S", 800)))
ok = True
for set_name, case in T.cases_for("S", 800)[:2]:
    for pi in (0, 4):
        pref = T.C.preferences()[pi]
        cid = f"{set_name}_seed800_{case['level']}_{case['mode']}_pref{pi:02d}"
        for a in ("heu-cost", "pgnd"):
            t0 = time.perf_counter(); s, info = arms[a](set_name, case, pi, pref); w = time.perf_counter() - t0
            v = T.C.partial_value(case["true"], s.operations, s.transports, pref, T.C.scales())
            o = old[(cid, a)]
            same = v == o["value"] and info["accepted"] == o["accepted"] and info["rollouts"] == o["rollouts"]
            ok &= same
            print(cid, a, v, o["value"], info["accepted"], o["accepted"], round(w, 1), "s (stored", o["wall_s"], ")", "SAME" if same else "DIFF", flush=True)
print("PARITY OK" if ok else "PARITY FAILED")
