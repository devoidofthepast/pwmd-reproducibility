"""v3 (全部重跑规则_冻结v3.md section 4): dispatching-rule baselines as choosers for blind_arms.policy_decode (strict clock).

Rules: the base rule of this paper (global earliest completion, written "lookahead" in the code), SPT, EDD, FIFO, MWKR and
the random rule with the operation keys of neural_operation_rollout.decode_rule_operation_rollout and the machine chosen by
the shared preference completion, as in the base action of event_blind.decode; and the two rules of Shen et al. (ASOC 2026,
section 5.1.2):
- G-EST: over all ready operation-machine pairs, the pair with the earliest start time, where the start is the latest of the
  job ready time, the machine available time and the AGV arrival at the machine (shared transport projection);
- MWKR-EAT: the job with the most remaining work, on the machine with the earliest completion time; the AGV and route are
  completed by the shared transport rule, as for every method.
"""
from __future__ import annotations

import random


def chooser(rule, seed=0):
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import (_best_preference_action, _best_transport_projection,
                                                               _operation_stage_lookahead)
    from su_modfjsp_ha_pc.rules import _available_operations
    rng = random.Random(seed)

    def res(state):
        return dict(machine_available=state.machine_available, job_ready=state.job_ready, agv_available=state.agv_available,
                    agv_node=state.agv_node, edge_reservations=state.edge_reservations)

    def remaining(item):
        job, op = item
        ops = job.operations
        return sum(min(o.processing_time.values()) for o in ops[ops.index(op):])

    def la_score(known, state, item):
        return _operation_stage_lookahead(instance=known, job=item[0], operation=item[1], machine_available=state.machine_available,
                                          job_ready_time=state.job_ready[item[0].job_id], agv_available=state.agv_available,
                                          agv_node=state.agv_node, edge_reservations=state.edge_reservations)

    def choose(known, state, operations, transports, preference):
        available = _available_operations(known.jobs, state.next_op_index)
        if rule == "gest":
            best = None
            mach = {m.machine_id: m for m in known.machines}
            for job, op in available:
                for mid in op.eligible_machines:
                    pr = _best_transport_projection(instance=known, job=job, machine=mach[mid], agv_available=state.agv_available,
                                                    agv_node=state.agv_node, edge_reservations=state.edge_reservations,
                                                    earliest_job_ready=state.job_ready[job.job_id])
                    start = max(state.job_ready[job.job_id], state.machine_available[mid], int(pr.get("route_end", state.job_ready[job.job_id])))
                    key = (start, start + op.processing_time[mid], op.operation_id, mid)
                    if best is None or key < best[0]:
                        best = (key, job, op, mach[mid])
            return best[1], best[2], best[3]
        if rule == "mwkr_eat":
            job, op = min(available, key=lambda it: (-remaining(it), it[1].operation_id))
            return job, op, la_score(known, state, (job, op))["machine"]
        if rule == "random":
            job, op = available[rng.randrange(len(available))]
        elif rule == "spt":
            job, op = min(available, key=lambda it: (min(it[1].processing_time.values()), it[1].operation_id))
        elif rule == "edd":
            job, op = min(available, key=lambda it: (float(getattr(it[0], "due_date", 0)), it[1].operation_id))
        elif rule == "fifo":
            job, op = min(available, key=lambda it: (float(getattr(it[0], "release_time", 0)), it[0].job_id, it[1].operation_id))
        elif rule == "mwkr":
            job, op = min(available, key=lambda it: (-remaining(it), it[1].operation_id))
        elif rule == "lookahead":
            job, op = min(available, key=lambda it: (la_score(known, state, it)["score"], it[1].operation_id))
        else:
            raise ValueError(rule)
        machine = _best_preference_action(instance=known, available=((job, op),), preference=preference, **res(state))["machine"]
        return job, op, machine

    return choose


RULES = ("lookahead", "gest", "mwkr_eat", "spt", "edd", "fifo", "mwkr", "random")
