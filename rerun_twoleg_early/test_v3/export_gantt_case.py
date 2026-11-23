"""Export the schedules of G-ECT and PGND on one test case for the Gantt figure (2026-11-19).
Case: synthetic seed 800, first case (level and mode of cases_for("S", 800)[0]), balanced preference (index 0)."""
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.argv = ["x", "S", "800", "pgnd"]
sys.path.insert(0, str(H))
import test_v3 as T  # noqa: E402

set_name, case = T.cases_for("S", 800)[0]
pi = 0
pref = T.C.preferences()[pi]
out = {"set": set_name, "seed": 800, "level": case["level"], "mode": case["mode"], "pref": pi, "schedules": {}}
arms = dict(T.arms("rules", "S", 800))
arms.update(dict(T.arms("pgnd", "S", 800)))
def legs(inst, sched, mv):
    """End of the empty leg and start of the loaded leg, reconstructed as in twoleg/run_early.check_schedule (rerun variant):
    the AGV travels the empty leg from mv.start, may wait at the pickup node, and travels the loaded leg until mv.end."""
    import math
    edges = {e.edge_id: e for e in inst.path_edges}
    speed = {a.agv_id: a.speed for a in inst.agvs}
    mnode = {m.machine_id: m.node_id for m in inst.machines}
    ops = {o.operation_id: o for o in sched.operations}
    info = {op.operation_id: (job, i) for job in inst.jobs for i, op in enumerate(job.operations)}
    job, i = info[mv.operation_id]
    pickup = mnode[ops[job.operations[i - 1].operation_id].machine_id] if i else inst.agvs[0].start_node_id
    nodes = [mv.from_node]
    for e in mv.edge_ids:
        ed = edges[e]
        nodes.append(ed.to_node if nodes[-1] == ed.from_node else ed.from_node)
    k = nodes.index(pickup)
    le = sum(edges[e].length for e in mv.edge_ids[:k])
    ll = sum(edges[e].length for e in mv.edge_ids[k:])
    te = math.ceil(le / speed[mv.agv_id]) if k else 0
    tl = max(1, math.ceil(ll / speed[mv.agv_id]))
    return dict(empty_end=mv.start + te, loaded_start=mv.end - tl)


for a in ("rule-lookahead", "pgnd"):
    s, info = arms[a](set_name, case, pi, pref)
    v = T.C.partial_value(case["true"], s.operations, s.transports, pref, T.C.scales())
    out["schedules"][a] = {
        "value": v,
        "operations": [dict(job=o.job_id, op=o.op_index, machine=o.machine_id, start=o.start, end=o.end) for o in s.operations],
        "transports": [dict(move=t.move_id, job=t.job_id, agv=t.agv_id, start=t.start, end=t.end, frm=t.from_node, to=t.to_node,
                            edges=list(t.edge_ids), **legs(case["true"], s, t)) for t in s.transports],
        "checks": T.checks(case["true"], s),
    }
    print(a, v, out["schedules"][a]["checks"], flush=True)
ev = case["true"].events
out["events"] = [dict(id=e.event_id, type=e.event_type, entity=e.affected_entity, time=e.event_time, duration=e.duration) for e in ev]
(H / "gantt_case_s800.json").write_text(json.dumps(out, indent=1))
print("saved", case["level"], case["mode"])
