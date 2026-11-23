"""v3 implementation of the scenario certificate (全部重跑规则_冻结v3 section 6b: speed-ups change the implementation only).
decode_scen3 = scenario_cert2.decode_scen2 with identical scenarios evaluated once: scenarios whose event tuples are equal
are complete-rollout inputs that are equal, so their values are equal; each distinct scenario is evaluated once and the value
is reused. Returned values, accepted steps and the nominal rollout count are unchanged; unique_rollouts counts the rollouts
actually run. Parity with decode_scen2 is checked on smoke seeds (v3_step0/speed_parity.py)."""
import numpy as np

import event_blind as EB
import scenario_cert2 as C2
from scenario_cert import scenarios

_src = C2._src.replace("def decode_scen2(", "def decode_scen3(")


def _rep(old, new):
    global _src
    assert _src.count(old) == 1, old[:60]
    _src = _src.replace(old, new)


_rep("""                    def value(j, o, m, tag):
                        return [value1(inst, j, o, m, tag) for inst in scen]
""",
     """                    scen_sig = [tuple(inst.events) for inst in scen]

                    def value(j, o, m, tag):
                        nonlocal n_unique
                        memo = {}
                        out = []
                        for inst, sig in zip(scen, scen_sig):
                            if sig not in memo:
                                memo[sig] = value1(inst, j, o, m, tag)
                                n_unique += 1
                            out.append(memo[sig])
                        return out
""")
_rep("    while len(operations) < state.total_operations:\n", "    n_unique = 0\n    while len(operations) < state.total_operations:\n")
_rep("rollouts=n_rollouts,", "rollouts=n_rollouts, unique_rollouts=n_unique,")

_ns = dict(EB.__dict__, scenarios=scenarios, _accept=C2._accept, np=np)
exec(compile(_src, "scenario_cert3.decode_scen3", "exec"), _ns)
decode_scen3 = _ns["decode_scen3"]
neural_topk_selector = C2.neural_topk_selector
random_topk_selector = C2.random_topk_selector
