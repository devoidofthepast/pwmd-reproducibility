"""v3 amendment 1 (全部重跑规则_v3修正案1_随机提名器_20261113.md): random top-k proposer under the same scenario certificate as pgnd.
Usage: python test_v3_rnd.py BLOCK SEED   (BLOCK S or C). Output: runs/{BLOCK}/random_{SEED}.jsonl. test_v3.py itself is not changed."""
import sys
sys.argv = sys.argv[:3] + ["random"]
import test_v3 as T  # noqa: E402

_arms = T.arms


def arms(group, block, seed):
    if group != "random":
        return _arms(group, block, seed)
    assert block in ("S", "C"), "amendment 1: P-S and P-C only"
    import scenario_cert2 as C2
    from scenario_cert3 import decode_scen3
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _explicit_preference_id
    sc = T.C.scales()
    M, K, k = T.CFG["M"], T.CFG["K"], T.CFG["k"]
    sel = C2.random_topk_selector(k)

    def run(set_name, case, pi, pref):
        return decode_scen3(true=case["true"], raw=case["raw"], preference=pref, level=case["level"], seed=case["seed"], mode=case["mode"],
                            preference_id=_explicit_preference_id(pi, pref), selector=sel, scales=sc, clock="strict", margin=M,
                            k=K, template=T.template_for(set_name, case["level"]))
    return [("rnd", run)]


T.arms = arms
if __name__ == "__main__":
    T.main()
