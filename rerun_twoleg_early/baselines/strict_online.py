"""Strict online execution (2026-11-22, response to review point 1 on the event-timing leak).

In the frozen test runs, a committed action is timed against the true event windows at commit time, so that the state seen
by later decisions can reflect events that are not yet revealed. Here the dispatcher only ever sees a state timed against
the revealed events:

- every commit keeps its decision (operation, machine, AGV, route), planned under the known events as before;
- the committed schedule is re-timed by replaying the commits in commit order on an instance holding only the revealed
  events, whenever the set of revealed events grows (right-shift repair with fixed assignments and sequences);
- an interval that meets a newly revealed window starts after the window, so an operation in progress when a breakdown is
  revealed is restarted after the repair (preempt-repeat); idle time before the interruption is not recovered;
- after the last decision, all events are revealed and the final replay gives the executed schedule that is scored.

The decision logic, the certificate, the scenarios and the base rule are those of decode_scen3 and policy_decode; only the
state they read and the commit timing change. With every event revealed from the start, both functions reproduce the frozen
decoders exactly (checked in check_strict_online.py).
"""
import copy
import inspect

import blind_arms as BA
import event_blind as EB
import scenario_cert3 as S3

REVEAL_ALL = False  # test switch: reveal every event from the start (must reproduce the frozen decoders)


def revealed_instance(true, tt):
    if REVEAL_ALL:
        return true, frozenset(e.event_id for e in true.events)
    ids = frozenset(e.event_id for e in true.events if e.event_time <= tt)
    return true.replace(events=tuple(e for e in true.events if e.event_id in ids)), ids


def plan_of(*, known, true, preference, true_job, true_op, machine, state):
    """AGV and route of the commit, chosen under the known events (as in event_blind._commit_strict)."""
    import su_modfjsp_ha_pc.hierarchical_transport_attention_policy as hierarchical
    known_job = next(j for j in known.jobs if j.job_id == true_job.job_id)
    known_op = next((o for o in known_job.operations if o.operation_id == true_op.operation_id), None)
    if known_op is None or machine.machine_id not in known_op.eligible_machines:
        known_op = true_op
    return hierarchical._best_fixed_machine_preference_transport(
        instance=known.replace(jobs=tuple(true_job if j.job_id == true_job.job_id else j for j in known.jobs)),
        preference=preference, job=true_job, operation=true_op, machine=machine,
        machine_available=state.machine_available, job_ready=state.job_ready, agv_available=state.agv_available,
        agv_node=state.agv_node, edge_reservations=state.edge_reservations,
    )["transport"]


def apply_commit(inst, preference, true_job, true_op, machine, plan, state, operations, transports):
    """Timing of one commit with a fixed AGV and route on instance inst (the timing part of event_blind._commit_strict)."""
    import su_modfjsp_ha_pc.hierarchical_transport_attention_policy as hierarchical
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _legs_of, _route_length
    from su_modfjsp_ha_pc.rules import job_pickup_of, two_leg_early_schedule
    original = hierarchical._preference_transport_candidates

    def only_planned(*, instance, job, machine, agv_available, agv_node, edge_reservations, earliest_job_ready):
        if plan is None:
            return (None,)
        agv = next(a for a in instance.agvs if a.agv_id == plan["agv_id"])
        route = plan["route"]
        length = _route_length(instance, route)
        origin = agv_node[agv.agv_id]
        pickup = job_pickup_of(instance, agv_node, job.job_id)
        tl = two_leg_early_schedule(instance, agv, origin, pickup, route, agv_available[agv.agv_id], earliest_job_ready,
                                    edge_reservations)
        return ({
            "agv_id": agv.agv_id, "from_node": origin, "route": route, "start": tl["start"],
            "end": tl["end"], "wait": tl["delay"], "travel_time": tl["end"] - tl["start"], "route_length": length,
            "transport_energy": round(length * agv.consumption_rate, 6),
            "empty_end": tl["empty_end"], "loaded_start": tl["loaded_start"], "n_empty": tl["n_empty"],
            "route_legs": _legs_of(route, tl),
        },)

    hierarchical._preference_transport_candidates = only_planned
    try:
        hierarchical._append_fixed_joint_preference_transport_action(
            instance=inst, preference=preference, job=true_job, operation=true_op, machine=machine,
            machine_available=state.machine_available, job_ready=state.job_ready, next_op_index=state.next_op_index,
            agv_available=state.agv_available, agv_node=state.agv_node, edge_reservations=state.edge_reservations,
            scheduled_operations=operations, transport_moves=transports,
        )
    finally:
        hierarchical._preference_transport_candidates = original


FIELDS = ("machine_available", "job_ready", "next_op_index", "agv_available", "agv_node", "edge_reservations")


def snapshot(state):
    return copy.deepcopy({f: getattr(state, f) for f in FIELDS})


def replay(inst, preference, commits, state, operations, transports, init):
    for f in FIELDS:
        d = getattr(state, f)
        d.clear()
        d.update(copy.deepcopy(init[f]))
    operations.clear()
    transports.clear()
    for c in commits:
        apply_commit(inst, preference, *c, state, operations, transports)


def commit_online(commits, tt, *, known, true, preference, true_job, true_op, machine, state, operations, transports):
    plan = plan_of(known=known, true=true, preference=preference, true_job=true_job, true_op=true_op, machine=machine, state=state)
    commits.append((true_job, true_op, machine, plan))
    inst, _ = revealed_instance(true, tt)
    apply_commit(inst, preference, true_job, true_op, machine, plan, state, operations, transports)


SYNC_DEF = '''    operations, transports = [], []
    _commits = []
    _init = SO.snapshot(state)
    _rev = [None]

    def _sync(tt):
        inst, ids = SO.revealed_instance(true, tt)
        if ids == _rev[0]:
            return False
        _rev[0] = ids
        SO.replay(inst, preference, _commits, state, operations, transports, _init)
        return True
'''


def _patch(src, pairs):
    for old, new in pairs:
        assert src.count(old) == 1, old[:70]
        src = src.replace(old, new)
    return src


# ---- PWMD (decode_scen3) --------------------------------------------------------------------------------------------
_scen = _patch(S3._src.replace("def decode_scen3(", "def decode_scen3_online("), [
    ("    operations, transports = [], []\n", SYNC_DEF),
    ('''        if clock in ("min_ready", "strict"):
            visible = blind_instance(true, raw, t, state.next_op_index)
            ready = [
                state.job_ready[j.job_id] for j in visible.jobs
                if state.next_op_index[j.job_id] < len(j.operations)
            ]
            if ready:
                t = max(t, min(ready))
''', '''        while True:
            _sync(t)
            visible = blind_instance(true, raw, t, state.next_op_index)
            ready = [
                state.job_ready[j.job_id] for j in visible.jobs
                if state.next_op_index[j.job_id] < len(j.operations)
            ]
            _t_new = max(t, min(ready)) if ready else t
            if _t_new == t:
                break
            t = _t_new
'''),
    ("            t = later[0]\n", "            t = later[0]\n            _sync(t)\n"),
    ("_commit_strict(", "SO.commit_online(_commits, t, "),
    ("    schedule = ScheduleResult(", "    _sync(INF)\n    schedule = ScheduleResult("),
])
_ns = dict(S3._ns, SO=__import__(__name__))
exec(compile(_scen, "strict_online.decode_scen3_online", "exec"), _ns)
decode_scen3_online = _ns["decode_scen3_online"]

# ---- rules and policies (policy_decode) -------------------------------------------------------------------------------
_pol = _patch(inspect.getsource(BA.policy_decode).replace("def policy_decode(", "def policy_decode_online("), [
    ("    operations, transports = [], []\n", SYNC_DEF),
    ('''        if strict:
            visible = blind_instance(true, raw, t, state.next_op_index)
            ready = [state.job_ready[j.job_id] for j in visible.jobs if state.next_op_index[j.job_id] < len(j.operations)]
            if ready:
                t = max(t, min(ready))
''', '''        if strict:
            while True:
                _sync(t)
                visible = blind_instance(true, raw, t, state.next_op_index)
                ready = [state.job_ready[j.job_id] for j in visible.jobs if state.next_op_index[j.job_id] < len(j.operations)]
                _t_new = max(t, min(ready)) if ready else t
                if _t_new == t:
                    break
                t = _t_new
'''),
    ("                t = later[0]\n", "                t = later[0]\n                _sync(t)\n"),
    ("            _commit_strict(", "            SO.commit_online(_commits, t, "),
    ("    return ScheduleResult(", "    _sync(INF)\n    return ScheduleResult("),
])
_pns = dict(BA.__dict__, SO=__import__(__name__))
exec(compile(_pol, "strict_online.policy_decode_online", "exec"), _pns)
policy_decode_online = _pns["policy_decode_online"]
