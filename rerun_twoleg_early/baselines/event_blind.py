"""Event-blind evaluation of the frozen PGND-V5 stack. See PROTOCOL_event_blind.md.

Run with PYTHONPATH pointing at the project's src directory. Writes only to --out.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
from pathlib import Path
from time import perf_counter

from rerun_paths import MODEL_DIR, SCALES  # rerun v1
RW = re.compile(r"_RW_(EV\d+)$")
INF = 10**12


def _reindexed(job, operations):
    from su_modfjsp_ha_pc.structural import _reindex_job
    return _reindex_job(job, list(operations))


def blind_instance(true, raw, t, next_op_index):
    """Instance holding only what is known at clock t. Committed operations always stay."""
    revealed = {e.event_id for e in true.events if e.event_time <= t}
    raw_jobs = {j.job_id: j for j in raw.jobs}
    raw_ops = {o.operation_id: o for j in raw.jobs for o in j.operations}
    unrevealed_branch = {
        e.affected_entity for e in true.events
        if e.event_type == "route_branch" and e.event_id not in revealed
    }
    jobs = []
    for job in true.jobs:
        done = next_op_index[job.job_id]
        if job.job_id not in raw_jobs and done == 0 and job.release_time > t:
            continue
        ops = list(job.operations[:done])
        for op in job.operations[done:]:
            match = RW.search(op.operation_id)
            if match and match.group(1) not in revealed:
                continue
            if op.operation_id in unrevealed_branch and op.operation_id in raw_ops:
                op = raw_ops[op.operation_id]
            ops.append(op)
        if not ops:
            continue
        jobs.append(_reindexed(job, ops) if tuple(ops) != job.operations else job)
    return true.replace(
        jobs=tuple(jobs),
        events=tuple(e for e in true.events if e.event_id in revealed),
    )


def _commit_strict(*, known, true, preference, true_job, true_op, machine, state, operations, transports):
    """Commit with the AGV and route chosen under known events only; timing still follows true events."""
    import math
    import su_modfjsp_ha_pc.hierarchical_transport_attention_policy as hierarchical
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _route_length
    from su_modfjsp_ha_pc.rules import _first_transport_feasible_start

    known_job = next(j for j in known.jobs if j.job_id == true_job.job_id)
    known_op = next((o for o in known_job.operations if o.operation_id == true_op.operation_id), None)
    if known_op is None or machine.machine_id not in known_op.eligible_machines:
        known_op = true_op  # a change revealed at execution: its route is planned from the executed operation
    plan = hierarchical._best_fixed_machine_preference_transport(
        instance=known.replace(jobs=tuple(true_job if j.job_id == true_job.job_id else j for j in known.jobs)),
        preference=preference, job=true_job, operation=true_op, machine=machine,
        machine_available=state.machine_available, job_ready=state.job_ready, agv_available=state.agv_available,
        agv_node=state.agv_node, edge_reservations=state.edge_reservations,
    )["transport"]
    original = hierarchical._preference_transport_candidates

    def only_planned(*, instance, job, machine, agv_available, agv_node, edge_reservations, earliest_job_ready):
        if plan is None:
            return (None,)
        from su_modfjsp_ha_pc.path_conflict_neural_dataset import _legs_of
        from su_modfjsp_ha_pc.rules import job_pickup_of, two_leg_early_schedule

        agv = next(a for a in instance.agvs if a.agv_id == plan["agv_id"])
        route = plan["route"]
        length = _route_length(instance, route)
        origin = agv_node[agv.agv_id]
        pickup = job_pickup_of(instance, agv_node, job.job_id)
        tl = two_leg_early_schedule(instance, agv, origin, pickup, route, agv_available[agv.agv_id], earliest_job_ready,
                                    edge_reservations)  # rerun v1: empty leg + loaded leg, timed against the true events
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
            instance=true, preference=preference, job=true_job, operation=true_op, machine=machine,
            machine_available=state.machine_available, job_ready=state.job_ready, next_op_index=state.next_op_index,
            agv_available=state.agv_available, agv_node=state.agv_node, edge_reservations=state.edge_reservations,
            scheduled_operations=operations, transport_moves=transports,
        )
    finally:
        hierarchical._preference_transport_candidates = original


def decode(*, true, raw, preference, level, seed, mode, preference_id, selector, scales, clock, trace=None, margin=0.0, nogate=False):
    """clock is 'oracle', 'min_ready', 'max_start' or 'strict'. selector None means rule only.

    'strict' is min_ready plus AGV and route planned under known events. margin is a relative acceptance margin.
    """
    from su_modfjsp_ha_pc.deterministic_dispatch_rules import select_operation_by_rule
    from su_modfjsp_ha_pc.exact_advantage_learnability import build_inference_rows_from_candidates
    from su_modfjsp_ha_pc.hierarchical_transport_attention_policy import _append_fixed_joint_preference_transport_action
    from su_modfjsp_ha_pc.joint_operation_machine_value_policy import (
        _append_selected_preference_action,
        _resolve_joint_candidate,
    )
    from su_modfjsp_ha_pc.neural_operation_rollout import _RolloutState
    from su_modfjsp_ha_pc.path_aware_selection_attention_policy import (
        _complete_schedule_after_forced_path_aware_joint_action,
        _path_aware_joint_group_rows,
    )
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _best_preference_action
    from su_modfjsp_ha_pc.pgnd_v4_model import build_v4_group
    from su_modfjsp_ha_pc.pgnd_v5_exact_certificate import _weighted_value
    from su_modfjsp_ha_pc.rules import _available_operations
    from su_modfjsp_ha_pc.schedule import ScheduleResult

    state = _RolloutState(
        machine_available={m.machine_id: 0 for m in true.machines},
        job_ready={j.job_id: j.release_time for j in true.jobs},
        next_op_index={j.job_id: 0 for j in true.jobs},
        agv_available={a.agv_id: 0 for a in true.agvs},
        agv_node={a.agv_id: a.start_node_id for a in true.agvs},
        edge_reservations={e.edge_id: [] for e in true.path_edges},
        total_operations=sum(len(j.operations) for j in true.jobs),
    )
    operations, transports = [], []
    machine_by_id = {m.machine_id: m for m in true.machines}
    true_jobs = {j.job_id: j for j in true.jobs}
    event_times = sorted({e.event_time for e in true.events})
    t = INF if clock == "oracle" else 0
    step = 0
    n_failures = n_accepted = n_rollouts = n_proposals = 0

    def resources():
        return dict(
            machine_available=state.machine_available,
            job_ready=state.job_ready,
            agv_available=state.agv_available,
            agv_node=state.agv_node,
            edge_reservations=state.edge_reservations,
        )

    while len(operations) < state.total_operations:
        if clock in ("min_ready", "strict"):
            visible = blind_instance(true, raw, t, state.next_op_index)
            ready = [
                state.job_ready[j.job_id] for j in visible.jobs
                if state.next_op_index[j.job_id] < len(j.operations)
            ]
            if ready:
                t = max(t, min(ready))
        known = true if clock == "oracle" else blind_instance(true, raw, t, state.next_op_index)
        available = _available_operations(known.jobs, state.next_op_index)
        while not available:
            later = [x for x in event_times if x > t]
            if not later:
                raise ValueError("no schedulable operations remain")
            t = later[0]
            known = blind_instance(true, raw, t, state.next_op_index)
            available = _available_operations(known.jobs, state.next_op_index)

        base_job, base_op = select_operation_by_rule(rule="lookahead", instance=known, available=available, **resources())
        base_action = _best_preference_action(instance=known, available=((base_job, base_op),), preference=preference, **resources())
        chosen = (base_job, base_op, base_action["machine"])
        accepted = False
        if selector is not None:
            try:
                group_id = f"blind-{level}-{seed:03d}-{mode}-{preference_id}-{step:03d}"
                rows = _path_aware_joint_group_rows(
                    instance=known, available=available, preference=preference, preference_id=preference_id,
                    level=level, seed=seed, mode=mode, group_id=group_id, candidate_uid_start=step * 10000,
                    step_index=step, selected_operation_id="", selected_machine_id="",
                    total_operations=sum(len(j.operations) for j in known.jobs), **resources(),
                )
                inference = build_inference_rows_from_candidates(
                    rows, base_operation_id=base_op.operation_id, base_machine_id=base_action["machine"].machine_id,
                )
                group = build_v4_group(inference)
                ordered = tuple(sorted(rows, key=lambda r: (str(r.get("candidate_operation_id", "")), str(r.get("candidate_machine_id", "")))))
                index = selector(ordered, group, group.base_index)
                if index is not None:
                    n_proposals += 1
                    job, op, machine = _resolve_joint_candidate(known, available, ordered[index])

                    if nogate:
                        chosen = (job, op, machine)
                        accepted = True
                        n_accepted += 1
                        raise StopIteration

                    def value(j, o, m, tag):
                        schedule = _complete_schedule_after_forced_path_aware_joint_action(
                            instance=known, preference=preference, forced_job=j, forced_operation=o, forced_machine=m,
                            machine_available=state.machine_available, job_ready=state.job_ready,
                            next_op_index=state.next_op_index, agv_available=state.agv_available,
                            agv_node=state.agv_node, edge_reservations=state.edge_reservations,
                            scheduled_prefix=tuple(operations), transport_prefix=tuple(transports),
                            continuation_rule="lookahead", continuation_level=level, continuation_seed=seed,
                            continuation_mode=mode, continuation_preference_id=preference_id,
                        )
                        return _weighted_value(known, schedule, preference, level, seed, mode, preference_id, tag, scales)

                    base_value = value(base_job, base_op, base_action["machine"], "base")
                    proposal_value = value(job, op, machine, "proposal")
                    n_rollouts += 2
                    if proposal_value < base_value * (1.0 - margin) - 1.0e-12:
                        chosen = (job, op, machine)
                        accepted = True
                        n_accepted += 1
            except StopIteration:  # uncertified arm committed its nomination, no rollout
                pass
            except Exception as error:  # the certificate falls back to the base action, as in V5
                n_failures += 1
                accepted = False
                chosen = (base_job, base_op, base_action["machine"])
                if trace is not None:
                    trace.append(f"step {step}: {type(error).__name__}: {error}")

        job_id = chosen[0].job_id
        true_job = true_jobs[job_id]
        true_op = true_job.operations[state.next_op_index[job_id]]
        machine = chosen[2]
        if clock == "oracle" and not accepted:
            _append_selected_preference_action(
                instance=true, selected=base_action, next_op_index=state.next_op_index,
                scheduled_operations=operations, transport_moves=transports, **resources(),
            )
        else:
            if machine.machine_id not in true_op.eligible_machines:
                machine = _best_preference_action(
                    instance=true, available=((true_job, true_op),), preference=preference, **resources()
                )["machine"]
            if clock == "strict":
                _commit_strict(
                    known=known, true=true, preference=preference, true_job=true_job, true_op=true_op,
                    machine=machine_by_id[machine.machine_id], state=state, operations=operations, transports=transports,
                )
            else:
                _append_fixed_joint_preference_transport_action(
                    instance=true, preference=preference, job=true_job, operation=true_op,
                    machine=machine_by_id[machine.machine_id], next_op_index=state.next_op_index,
                    scheduled_operations=operations, transport_moves=transports, **resources(),
                )
        if clock == "max_start":
            committed = operations[-1]
            move = next((m for m in reversed(transports) if m.operation_id == committed.operation_id), None)
            t = max(t, move.start if move is not None else committed.start)
        step += 1
    schedule = ScheduleResult(
        algorithm="pgnd-v5-event-blind", path_aware=bool(true.path_edges), operations=tuple(operations),
        transports=tuple(transports), event_aware=bool(true.events),
    )
    return schedule, dict(decisions=step, proposals=n_proposals, accepted=n_accepted, rollouts=n_rollouts, failures=n_failures)


def randomize_events(raw, seed, level):
    """Exploratory block: draw event times and targets per seed instead of the fixed generator script."""
    import random
    rng = random.Random(seed * 1000 + (5 if level == "L5" else 4))
    machines = [m.machine_id for m in raw.machines]
    edges = [e.edge_id for e in raw.path_edges]
    agvs = [a.agv_id for a in raw.agvs]
    ops = [o.operation_id for j in raw.jobs for o in j.operations]
    events = list(raw.events)
    result = []
    pending_breakdown = {}
    for event in events:
        kind = event.event_type
        time = rng.randint(5, 120)
        entity = event.affected_entity
        if kind == "machine_breakdown":
            entity = rng.choice(machines)
            pending_breakdown[event.affected_entity] = (time, entity, event.event_time)
        elif kind == "machine_recovery":
            start, entity, old = pending_breakdown[event.affected_entity]
            time = start + (event.event_time - old)
        elif kind in ("path_blockage", "congestion_spike"):
            entity = rng.choice(edges)
        elif kind == "agv_failure":
            entity = rng.choice(agvs)
        elif kind in ("route_branch", "quality_rework"):
            entity = rng.choice(ops)
        result.append(type(event)(event.event_id, time, kind, entity, event.duration, event.severity, event.scenario_id))
    return raw.replace(events=tuple(result))


def run_seed(job):
    seed, out, arms, limit, size, randomize, core, only = job
    import torch
    torch.set_num_threads(1)
    from su_modfjsp_ha_pc.generator import generate_instance
    from su_modfjsp_ha_pc.neural_operation_rollout import _detail_row
    from su_modfjsp_ha_pc.objective_contract import read_objective_scales
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _explicit_preference_id, _explicit_teacher_preferences
    from su_modfjsp_ha_pc.pgnd_v5_exact_certificate import neural_ensemble_proposal_selector
    from su_modfjsp_ha_pc.run_pgnd_v4_learnability_gate import _load_artifact
    from su_modfjsp_ha_pc.run_pgnd_v5_smoke import MEMBER_STATES
    from su_modfjsp_ha_pc.structural import realize_structural_events

    scales = read_objective_scales(SCALES)
    selector = neural_ensemble_proposal_selector(tuple(_load_artifact(MODEL_DIR / f"pgnd_v4_member_{s}.pt") for s in MEMBER_STATES))
    from su_modfjsp_ha_pc.pgnd_v5_proposer_ablation import deterministic_random_proposal_selector
    random_selector = deterministic_random_proposal_selector()
    path = Path(out) / (f"seed_{seed}.csv" if only is None else f"seed_{seed}_{only[0]}_{only[1]}.csv")
    done = set()
    if path.exists():
        with path.open(newline="") as handle:
            done = {(r["case_id"], r["arm"]) for r in csv.DictReader(handle)}
    fields = ["case_id", "seed", "level", "mode", "preference_id", "arm", "value", "decisions", "proposals", "accepted", "rollouts", "failures", "wall_s", "feasible", "route_bad", "timing_bad", "hard_viol"]
    count = 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not done:
            writer.writeheader()
        for level in ("L4", "L5"):
            if core:
                from su_modfjsp_ha_pc.public_fjsp import load_and_extend_public_fjsp
                raw = load_and_extend_public_fjsp(Path(core), level=level, seed=seed, n_agvs=3)
            else:
                raw = generate_instance(level, size, seed)
            if randomize:
                raw = randomize_events(raw, seed, level)
            for mode in ("raw", "realized"):
                if only is not None and (level, mode) != only:
                    continue
                true = realize_structural_events(raw).instance if mode == "realized" else raw
                for index, preference in enumerate(_explicit_teacher_preferences(len(true.objectives), max_preferences=6)):
                    if limit is not None and count >= limit:
                        return seed
                    count += 1
                    preference_id = _explicit_preference_id(index, preference)
                    case_id = f"seed{seed:03d}_{level}_{mode}_pref{index:02d}"
                    for arm in arms:
                        if (case_id, arm) in done:
                            continue
                        method, clock = arm.split("@")
                        trace = []
                        started = perf_counter()
                        schedule, info = decode(
                            true=true, raw=raw, preference=preference, level=level, seed=seed, mode=mode,
                            preference_id=preference_id, selector=selector if method.startswith("pgnd") else (random_selector if method.startswith("rnd") else None),
                            scales=scales, clock=clock, trace=trace,
                            margin=(float(method[4:]) if method.startswith("pgnd") and len(method) > 4 else (float(method[3:]) if method.startswith("rnd") and len(method) > 3 else 0.0)),
                        )
                        wall = perf_counter() - started
                        row = _detail_row(
                            instance=true, schedule=schedule, teacher_operations=(), level=level, seed=seed, mode=mode,
                            preference_id=preference_id, preference=preference, method=arm, objective_scales=scales,
                        )
                        import sys as _sys
                        _tw = str(Path(__file__).resolve().parents[2] / "twoleg")
                        if _tw not in _sys.path:
                            _sys.path.append(_tw)
                        from run_early import check_schedule
                        from su_modfjsp_ha_pc.metrics import evaluate_feasibility
                        _rb, _tb, _msgs, _ = check_schedule(true, schedule, "rerun")
                        _rep = evaluate_feasibility(true, schedule)
                        _hv = (_rep.precedence_violations + _rep.machine_overlap_violations + _rep.agv_overlap_violations
                               + _rep.edge_capacity_violations)
                        writer.writerow(dict(
                            case_id=case_id, seed=seed, level=level, mode=mode, preference_id=preference_id, arm=arm,
                            value=row["target_weighted_value"], wall_s=round(wall, 2), feasible=row.get("feasible", ""), **info,
                            route_bad=_rb, timing_bad=_tb, hard_viol=_hv,
                        ))
                        if _rb or _tb or _hv:
                            with (Path(out) / "check_failures.log").open("a") as log:
                                log.write(f"{case_id} {arm}: {_rb} {_tb} {_hv} {_msgs[:3]}\n")
                        handle.flush()
                        if trace:
                            with (Path(out) / "failures.log").open("a") as log:
                                log.write(f"{case_id} {arm}: " + " | ".join(trace[:3]) + "\n")
    return seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", required=True, help="first-last, e.g. 300-309")
    parser.add_argument("--arms", default="la@oracle,pgnd@oracle,la@min_ready,pgnd@min_ready,la@max_start,pgnd@max_start")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="cases per seed, smoke only")
    parser.add_argument("--size", default="small")
    parser.add_argument("--randomize-events", action="store_true")
    parser.add_argument("--split", action="store_true", help="one worker task per seed, level and mode")
    parser.add_argument("--core", default=None, help="path to a public FJSP core")
    args = parser.parse_args()
    first, last = map(int, args.seeds.split("-"))
    os.makedirs(args.out, exist_ok=True)
    arms = tuple(args.arms.split(","))
    Path(args.out, "run_config.json").write_text(json.dumps(vars(args), indent=1))
    only_values = [None]
    if args.split:
        only_values = [(level, mode) for level in ("L4", "L5") for mode in ("raw", "realized")]
    jobs = [
        (seed, args.out, arms, args.limit, args.size, args.randomize_events, args.core, only)
        for seed in range(first, last + 1) for only in only_values
    ]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for seed in pool.imap_unordered(run_seed, jobs, chunksize=1):
            print("done", seed, flush=True)


if __name__ == "__main__":
    main()
