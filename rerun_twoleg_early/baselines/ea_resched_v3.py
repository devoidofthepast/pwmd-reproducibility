"""v3 (全部重跑规则_冻结v3.md section 4): GA, NSGA-II, NSGA-III and MOEA/D under hidden events by rescheduling, as Lei et al.
(ASOC 2026, section 6.3.4) run their metaheuristics on the dynamic problem: a schedule is optimised from the information
revealed at time 0, and whenever a new event is revealed the remaining operations are re-optimised from the current state,
with the previous best genome as one member of the initial population. Each (re)optimisation has the same wall-clock budget
for every algorithm.

Kept from the original baselines (ga_nsga2_harness.py, moea_extra.py): the random-key genome (one priority and one
machine key per operation), the decoding (highest-priority ready operation, machine by key, AGV and route by the shared
completion), the objectives (V for GA; the five normalised components for the multi-objective algorithms), population 100,
NSGA-III and MOEA/D reference directions (Das-Dennis, 5 objectives, 5 partitions; MOEA/D 15 neighbours, mating 0.7).
The executed genome of a multi-objective run is the member of the final population with the lowest V under the case
preference. Executed through blind_arms.policy_decode (strict clock)."""
from __future__ import annotations
import os as _pwmd_os
_PWMD_ROOT = _pwmd_os.path.abspath(_pwmd_os.path.join(_pwmd_os.path.dirname(__file__), *['..'] * 2))


def _pwmd_path(*parts):
    return _pwmd_os.path.join(_PWMD_ROOT, *parts)



import copy
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

H = Path(__file__).resolve().parents[1]
for p in (H / "src", H / "baselines", H.parent / "twoleg", Path(_pwmd_path('external_baselines', '_vendor_py'))):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

NAMES = ("makespan", "tardiness", "energy", "route_pressure", "event_exposure_pressure")
METHODS = ("ga", "nsga2", "nsga3", "moead")


def _components(instance, operations, transports, sc):
    from su_modfjsp_ha_pc.objective_contract import evaluate_objective_contract
    from su_modfjsp_ha_pc.schedule import ScheduleResult
    raw = evaluate_objective_contract(instance, ScheduleResult(algorithm="ea-v3", path_aware=True, operations=tuple(operations),
                                                               transports=tuple(transports), event_aware=True)).raw
    return np.array([raw[n] / sc.values[n] for n in NAMES], dtype=float)


def continue_with_genome(known, state, operations, transports, genome, op_index, preference):
    """Complete the schedule of the known instance from the current state with the genome as dispatching policy."""
    from su_modfjsp_ha_pc.hierarchical_transport_attention_policy import _append_fixed_joint_preference_transport_action
    from su_modfjsp_ha_pc.rules import _available_operations
    st = copy.deepcopy(state)
    ops, trs = list(operations), list(transports)
    n = len(op_index)
    mach = {m.machine_id: m for m in known.machines}
    total = sum(len(j.operations) for j in known.jobs)
    while len(ops) < total:
        available = _available_operations(known.jobs, st.next_op_index)
        if not available:
            break
        job, op = max(available, key=lambda it: (float(genome[op_index[it[1].operation_id]]), it[1].operation_id))
        el = op.eligible_machines
        mid = el[min(int(float(genome[n + op_index[op.operation_id]]) * len(el)), len(el) - 1)]
        _append_fixed_joint_preference_transport_action(
            instance=known, preference=preference, job=job, operation=op, machine=mach[mid],
            machine_available=st.machine_available, job_ready=st.job_ready, next_op_index=st.next_op_index,
            agv_available=st.agv_available, agv_node=st.agv_node, edge_reservations=st.edge_reservations,
            scheduled_operations=ops, transport_moves=trs)
    return ops, trs


def _algorithm(method, pop, init):
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.util.ref_dirs import get_reference_directions
    if method == "ga":
        return GA(pop_size=pop, sampling=init, eliminate_duplicates=True)
    if method == "nsga2":
        return NSGA2(pop_size=pop, sampling=init, eliminate_duplicates=True)
    dirs = get_reference_directions("das-dennis", 5, n_partitions=5)
    if method == "nsga3":
        return NSGA3(ref_dirs=dirs, pop_size=128, sampling=init, eliminate_duplicates=True)  # pop >= 126 reference directions (Deb and Jain 2014); user-confirmed 2026-11-12
    return MOEAD(ref_dirs=dirs, n_neighbors=15, prob_neighbor_mating=0.7, sampling=init)


class Rescheduler:
    def __init__(self, method, preference, sc, budget, seed, pop=100):
        self.method, self.pref, self.sc, self.budget, self.seed, self.pop = method, np.asarray(preference), sc, budget, seed, pop
        self.sig, self.genome, self.op_index = None, None, None
        self.n_opt, self.opt_s, self.evals = 0, 0.0, 0

    def _optimise(self, known, state, operations, transports):
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize
        from pymoo.termination.max_time import TimeBasedTermination
        ids = [o.operation_id for j in known.jobs for o in j.operations]
        op_index = {o: i for i, o in enumerate(ids)}
        n = len(ids)
        rng = np.random.default_rng(self.seed * 1000 + self.n_opt)
        rows = 126 if self.method == "moead" else 128 if self.method == "nsga3" else self.pop  # MOEA/D: one member per reference direction; NSGA-III: 128 >= 126 directions
        init = rng.random((rows, 2 * n))
        if self.genome is not None:  # warm start: the previous best genome, mapped by operation id
            g = rng.random(2 * n)
            for o, i in op_index.items():
                if o in self.op_index:
                    k = self.op_index[o]
                    g[i], g[n + i] = self.genome[k], self.genome[len(self.op_index) + k]
            init[0] = g
        outer = self
        mo = self.method != "ga"

        class P(ElementwiseProblem):
            def __init__(self):
                super().__init__(n_var=2 * n, n_obj=5 if mo else 1, xl=0.0, xu=1.0)

            def _evaluate(self, x, out, *a, **k):
                ops, trs = continue_with_genome(known, state, operations, transports, x, op_index, outer.pref)
                c = _components(known, ops, trs, outer.sc)
                outer.evals += 1
                out["F"] = c if mo else float(outer.pref @ c)

        t0 = perf_counter()
        res = minimize(P(), _algorithm(self.method, self.pop, init), TimeBasedTermination(self.budget), seed=self.seed + self.n_opt, verbose=False)
        X = np.atleast_2d(res.pop.get("X"))
        F = np.atleast_2d(res.pop.get("F"))
        v = F @ self.pref if mo else F[:, 0]
        self.genome, self.op_index = X[int(np.argmin(v))], op_index
        self.opt_s += perf_counter() - t0
        self.n_opt += 1

    def choose(self, known, state, operations, transports, preference):
        from su_modfjsp_ha_pc.rules import _available_operations
        sig = (tuple(known.events), tuple((j.job_id, tuple(o.operation_id for o in j.operations)) for j in known.jobs))
        if sig != self.sig:
            self._optimise(known, state, operations, transports)
            self.sig = sig
        n = len(self.op_index)
        available = _available_operations(known.jobs, state.next_op_index)
        job, op = max(available, key=lambda it: (float(self.genome[self.op_index[it[1].operation_id]]), it[1].operation_id))
        el = op.eligible_machines
        mid = el[min(int(float(self.genome[n + self.op_index[op.operation_id]]) * len(el)), len(el) - 1)]
        return job, op, next(m for m in known.machines if m.machine_id == mid)


def run_case(case, method, preference, sc, budget, seed):
    from blind_arms import policy_decode
    r = Rescheduler(method, preference, sc, budget, seed)
    t0 = perf_counter()
    s, _ = policy_decode(true=case["true"], raw=case["raw"], preference=preference, choose=r.choose, strict=True)
    return s, dict(reoptimisations=r.n_opt, optimise_s=round(r.opt_s, 2), evaluations=r.evals, wall_s=round(perf_counter() - t0, 2))
