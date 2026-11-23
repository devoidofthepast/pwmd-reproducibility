"""Early-departure test (提前出发试验_规则与记录.md): lookahead and PGND (k=1, frozen models) on seeds 510-519 under
three simulators: oneleg (original), twoleg (wait until job ready), early (depart when free).

Usage: python run_early.py --variant {oneleg,twoleg,early} --seeds 510 511 ... --out DIR
Writes one JSON line per case to DIR/progress_<variant>_<firstseed>.jsonl (resumable).
"""
import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
SRC = {"oneleg": "D:/codex_work/urgent_order_dual_agent_20260821/src", "twoleg": str(HERE / "src"),
       "early": str(HERE / "early" / "src"), "early_fix": str(HERE / "early_fix" / "src"), "spfix": str(HERE / "spfix" / "src"), "rerun": str(HERE.parent / "rerun_twoleg_early" / "src")}


def setup(variant):
    sys.path.insert(0, "E:/pgnd_revision/v5_rebuild")
    sys.path.insert(0, SRC[variant])


def _walk(edges, start, route):
    node, nodes = start, [start]
    for eid in route:
        e = edges[eid]
        if e.from_node == node:
            node = e.to_node
        elif e.to_node == node:
            node = e.from_node
        else:
            return None
        nodes.append(node)
    return nodes


def check_schedule(instance, schedule, variant):
    """Independent route and timing check. Returns (route_bad, timing_bad, msgs, stats)."""
    edges = {e.edge_id: e for e in instance.path_edges}
    speed = {a.agv_id: a.speed for a in instance.agvs}
    machine_node = {m.machine_id: m.node_id for m in instance.machines}
    ops = {o.operation_id: o for o in schedule.operations}
    info = {}
    for job in instance.jobs:
        for i, op in enumerate(job.operations):
            info[op.operation_id] = (job, i)
    agv_pos = {a.agv_id: a.start_node_id for a in instance.agvs}
    depot = instance.agvs[0].start_node_id if instance.agvs else None
    route_bad = timing_bad = 0
    msgs, empty_len, pickup_wait, move_time = [], [], [], []
    for mv in sorted(schedule.transports, key=lambda m: (m.start, m.move_id)):
        job, i = info[mv.operation_id]
        prev = ops[job.operations[i - 1].operation_id] if i else None
        ready = prev.end if prev else job.release_time
        pickup = machine_node[prev.machine_id] if prev else depot
        if mv.from_node != agv_pos[mv.agv_id]:
            route_bad += 1; msgs.append(f"{mv.move_id} from {mv.from_node} but AGV at {agv_pos[mv.agv_id]}")
        nodes = _walk(edges, mv.from_node, mv.edge_ids)
        if nodes is None or nodes[-1] != mv.to_node:
            route_bad += 1; msgs.append(f"{mv.move_id}: route does not connect {mv.from_node}->{mv.to_node}")
            agv_pos[mv.agv_id] = mv.to_node
            continue
        if variant != "oneleg":
            if pickup not in nodes:
                route_bad += 1; msgs.append(f"{mv.move_id}: pickup {pickup} not visited")
                agv_pos[mv.agv_id] = mv.to_node
                continue
            k = nodes.index(pickup)  # first arrival at pickup = end of empty leg
            le = sum(edges[e].length for e in mv.edge_ids[:k]); ll = sum(edges[e].length for e in mv.edge_ids[k:])
            empty_len.append(le)
        if variant in ("early", "early_fix", "spfix", "rerun"):
            te = math.ceil(le / speed[mv.agv_id]) if k else 0
            tl = max(1, math.ceil(ll / speed[mv.agv_id]))
            ete, tlb = mv.start + te, mv.end - tl
            if tlb < ete:
                timing_bad += 1; msgs.append(f"{mv.move_id}: TLB {tlb} < ETE {ete}")
            if tlb < ready:
                timing_bad += 1; msgs.append(f"{mv.move_id}: loaded leg starts {tlb} before job ready {ready}")
            pickup_wait.append(tlb - ete)
            move_time.append(te + tl)
        if variant not in ("early", "early_fix", "spfix", "rerun"):
            move_time.append(mv.end - mv.start)
        if variant == "twoleg":
            if mv.start < ready:
                timing_bad += 1; msgs.append(f"{mv.move_id}: starts {mv.start} before job ready {ready}")
        if ops[mv.operation_id].start < mv.end:
            timing_bad += 1; msgs.append(f"{mv.move_id}: operation starts before delivery")
        if ops[mv.operation_id].machine_id and machine_node[ops[mv.operation_id].machine_id] != mv.to_node:
            route_bad += 1; msgs.append(f"{mv.move_id}: delivered to wrong node")
        agv_pos[mv.agv_id] = mv.to_node
    stats = {"empty_len": sum(empty_len) / len(empty_len) if empty_len else 0.0,
             "pickup_wait": sum(pickup_wait) / len(pickup_wait) if pickup_wait else 0.0,
             "move_time": sum(move_time)}
    return route_bad, timing_bad, msgs[:5], stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=tuple(SRC), required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if not set(a.seeds) <= set(range(510, 540)) and not a.smoke:
        raise SystemExit("seeds must be in 510-539")
    if a.smoke and set(a.seeds) & set(range(400, 520)):
        raise SystemExit("smoke seeds must be outside 400-519")
    setup(a.variant)
    import su_modfjsp_ha_pc as pkg
    assert Path(pkg.__file__).resolve().parent.parent == Path(SRC[a.variant]).resolve(), pkg.__file__
    from run_public_core_v5_block import load_frozen_stack
    from su_modfjsp_ha_pc.generator import generate_instance
    from su_modfjsp_ha_pc.structural import realize_structural_events
    from su_modfjsp_ha_pc.neural_operation_rollout import _detail_row, decode_rule_operation_rollout
    from su_modfjsp_ha_pc.pgnd_v5_exact_certificate import decode_pgnd_v5_exact_certificate, neural_ensemble_proposal_selector
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _explicit_preference_id, _explicit_teacher_preferences
    from su_modfjsp_ha_pc.metrics import evaluate_feasibility

    scales, artifacts = load_frozen_stack()
    selector = neural_ensemble_proposal_selector(artifacts)
    a.out.mkdir(parents=True, exist_ok=True)
    prog = a.out / f"progress_{a.variant}_{a.seeds[0]}.jsonl"
    done = set()
    if prog.exists():
        done = {json.loads(l)["case_id"] for l in prog.read_text().splitlines() if l.strip()}
    for seed in a.seeds:
        for level in ("L4", "L5"):
            raw = generate_instance(level, "small", seed)
            for mode in ("raw", "realized"):
                inst = realize_structural_events(raw).instance if mode == "realized" else raw
                prefs = _explicit_teacher_preferences(len(inst.objectives), max_preferences=6)
                for pi, pref in enumerate(prefs):
                    if a.smoke and (pi > 1 or level == "L5"):
                        continue
                    cid = f"seed{seed}_{level}_{mode}_pref{pi}"
                    if cid in done:
                        continue
                    pid = _explicit_preference_id(pi, pref)
                    rec = {"case_id": cid, "seed": seed, "level": level, "mode": mode, "pref": pi, "variant": a.variant}
                    trace = []
                    t0 = perf_counter()
                    v5 = decode_pgnd_v5_exact_certificate(instance=inst, preference=pref, level=level, seed=seed, mode=mode,
                                                          preference_id=pid, proposal_selector=selector, trace_rows=trace,
                                                          objective_scales=scales)
                    rec["pgnd_ms"] = (perf_counter() - t0) * 1000
                    rec["accepted"] = sum(1 for r in trace if str(r.get("accepted")) in ("1", "True"))
                    rec["invariant_failures"] = sum(int(r.get("invariant_failure", 0)) for r in trace)
                    look = decode_rule_operation_rollout(instance=inst, preference=pref, rule="lookahead", random_seed=seed)
                    for name, sch in (("pgnd", v5), ("lookahead", look)):
                        row = _detail_row(instance=inst, schedule=sch, teacher_operations=(), level=level, seed=seed, mode=mode,
                                          preference_id=pid, preference=pref, method=name, objective_scales=scales)
                        rec[f"V_{name}"] = row["target_weighted_value"]
                        for k in ("makespan", "total_tardiness", "total_energy"):
                            if k in row:
                                rec[f"{k}_{name}"] = row[k]
                        rb, tb, msgs, st = check_schedule(inst, sch, a.variant)
                        rec[f"route_bad_{name}"] = rb
                        rec[f"timing_bad_{name}"] = tb
                        rep = evaluate_feasibility(inst, sch)
                        rec[f"hard_viol_{name}"] = (rep.precedence_violations + rep.machine_overlap_violations
                                                    + rep.agv_overlap_violations + rep.edge_capacity_violations)
                        rec[f"event_viol_{name}"] = rep.dynamic_event_violations
                        if msgs:
                            rec[f"msgs_{name}"] = msgs
                        rec[f"empty_len_{name}"] = st["empty_len"]
                        rec[f"pickup_wait_{name}"] = st["pickup_wait"]
                        rec[f"move_time_{name}"] = st["move_time"]
                        rec[f"route_len_{name}"] = sum(sum(next(e.length for e in inst.path_edges if e.edge_id == x) for x in m.edge_ids) for m in sch.transports) / max(1, len(sch.transports))
                        rec[f"transport_time_{name}"] = sum(m.end - m.start for m in sch.transports)
                    with prog.open("a", encoding="utf-8") as h:
                        h.write(json.dumps(rec) + "\n")
                    print(cid, a.variant, f"pgnd={rec['V_pgnd']:.4f} look={rec['V_lookahead']:.4f} route={rec['route_bad_pgnd']}/{rec['route_bad_lookahead']} timing={rec['timing_bad_pgnd']}/{rec['timing_bad_lookahead']} {rec['pgnd_ms']:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
