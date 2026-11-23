"""Parity of the fast value path on C (heu-ect, heu-cost; seed 820, 2 cases x 2 preferences): slow then fast in one
process, exact equality of value, accepted and rollouts."""
import os, sys
os.environ.pop("R2_FAST_VALUE", None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["x", "S", "820", "heu"]          # not C/heu, so the import keeps the slow path
import review_r2 as R
import su_modfjsp_ha_pc.pgnd_v5_exact_certificate as PC
T = R.T
slow = PC._weighted_value
arms = dict(R.arms("heu", "C", 820))
ok = True
for set_name, case in [T.cases_for("C", 820)[0], T.cases_for("C", 820)[-1]]:
    for pi in (0, 4):
        pref = T.C.preferences()[pi]
        for a in ("heu-ect", "heu-cost"):
            res = []
            for fn in (slow, R._fast_weighted_value):
                PC._weighted_value = fn
                s, info = arms[a](set_name, case, pi, pref)
                res.append((T.C.partial_value(case["true"], s.operations, s.transports, pref, T.C.scales()), info["accepted"], info["rollouts"]))
            same = res[0] == res[1]; ok &= same
            print(set_name, case["level"], case["mode"], pi, a, res, "SAME" if same else "DIFF", flush=True)
print("PARITY OK" if ok else "PARITY FAILED")
