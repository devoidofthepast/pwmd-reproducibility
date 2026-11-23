"""D12: strict blind evaluation of native PPO and an uncertified proposer, with raw objectives logged.

Arms: la, pgnd (margin 0.02), rnd (margin 0.02), nogate (commit the nomination with no rollout),
ppoA and ppoB under the strict loop, and ppoA_orig and ppoB_orig, the original evaluation that sees future event counts and true routes.
Writes only to --out. Lookahead, PGND and random values are checked against the frozen confirmation files by analyze_blind_arms.py.
"""
from __future__ import annotations
import os as _pwmd_os
_PWMD_ROOT = _pwmd_os.path.abspath(_pwmd_os.path.join(_pwmd_os.path.dirname(__file__), *['..'] * 2))


def _pwmd_path(*parts):
    return _pwmd_os.path.join(_PWMD_ROOT, *parts)



import argparse
import csv
import multiprocessing as mp
import os
import sys
from pathlib import Path
from time import perf_counter

from event_blind import INF, MODEL_DIR, SCALES, _commit_strict, blind_instance, decode, randomize_events

BASE = _pwmd_path('rerun_twoleg_early', 'baselines')  # rerun v1
OBJECTIVES = ("makespan", "tardiness", "energy", "teacher_path_pressure", "teacher_stability_pressure")
DQN_DIR = _pwmd_path('rerun_twoleg_early', 'drl', 'cheng', 'final', 'selected')  # rerun v1
CHECKPOINTS = {"ppoA": _pwmd_path('rerun_twoleg_early', 'drl', 'ppo_a', 'SELECTED', 'checkpoint_latest.pt'), "ppoB": _pwmd_path('rerun_twoleg_early', 'drl', 'ppo_b', 'SELECTED', 'checkpoint_latest.pt')}  # rerun v1: set after selection


def load_policies():
    import torch
    sys.path.insert(0, BASE)
    from ppo_env import PPOScheduleEnv, load_frozen_objective_scales
    from ppo_train import CandidatePolicy
    from su_modfjsp_ha_pc.native_ppo_baseline import NativePPOActorCritic, encode_observation

    payload = torch.load(CHECKPOINTS["ppoA"], map_location="cpu", weights_only=False)
    policy_a = CandidatePolicy(hidden=int(payload.get("hidden", 128)))
    policy_a.load_state_dict(payload["model_state"])
    policy_a.eval()
    env = PPOScheduleEnv(load_frozen_objective_scales())

    def choose_a(known, state, operations, transports, preference):
        import numpy as np
        env.instance = known
        env.preference = tuple(float(v) for v in preference)
        env.state = state
        env.operations = operations
        env.transports = transports
        env._time_scale = max(1.0, float(sum(min(o.processing_time.values()) for j in known.jobs for o in j.operations)))
        env._energy_scale = max(1.0, float(sum(min(o.processing_energy.values()) for j in known.jobs for o in j.operations)))
        env._max_ops_per_job = max(len(j.operations) for j in known.jobs)
        env._build_candidates()
        obs = env._observation()
        with torch.no_grad():
            logits, _ = policy_a(torch.as_tensor(obs["candidates"], dtype=torch.float32).unsqueeze(0),
                                 torch.as_tensor(obs["context"], dtype=torch.float32).unsqueeze(0),
                                 torch.ones(1, obs["candidates"].shape[0], dtype=torch.bool))
        c = env.candidates[int(torch.argmax(logits[0]).item())]
        return c.job, c.operation, c.machine

    payload = torch.load(CHECKPOINTS["ppoB"], map_location="cpu", weights_only=False)
    policy_b = NativePPOActorCritic(candidate_dim=payload["candidate_dim"], context_dim=payload["context_dim"], hidden_size=payload["hidden"])
    policy_b.load_state_dict(payload["model_state"])
    policy_b.eval()

    def choose_b(known, state, operations, transports, preference):
        now = max([0] + [o.end for o in operations] + [m.end for m in transports])
        x, g, actions = encode_observation(
            instance=known, preference=preference, machine_available=state.machine_available, job_ready=state.job_ready,
            next_op_index=state.next_op_index, agv_available=state.agv_available, agv_node=state.agv_node,
            edge_reservations=state.edge_reservations, scheduled_operation_count=len(operations), decision_time=now,
        )
        with torch.no_grad():
            logits, _ = policy_b(torch.from_numpy(x).unsqueeze(0), torch.from_numpy(g).unsqueeze(0), torch.ones((1, len(actions)), dtype=torch.bool))
        return actions[int(logits.argmax(dim=1).item())]

    from cheng_dqn import chooser as dqn_chooser, network as dqn_network  # rerun v1: Cheng DQN arms
    dqn = []
    for i in range(6):
        net = dqn_network()
        net.load_state_dict(torch.load(f"{DQN_DIR}/dqn_pref{i:02d}.pt"))
        net.eval()
        dqn.append(dqn_chooser(net))
    return {"ppoA": choose_a, "ppoB": choose_b, "dqn": dqn}


def policy_decode(*, true, raw, preference, choose, strict):
    """The strict blind clock of event_blind.decode with a policy choosing the action. strict=False sees the true instance."""
    from su_modfjsp_ha_pc.hierarchical_transport_attention_policy import _append_fixed_joint_preference_transport_action
    from su_modfjsp_ha_pc.neural_operation_rollout import _RolloutState
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _best_preference_action
    from su_modfjsp_ha_pc.rules import _available_operations
    from su_modfjsp_ha_pc.schedule import ScheduleResult

    state = _RolloutState(
        machine_available={m.machine_id: 0 for m in true.machines}, job_ready={j.job_id: j.release_time for j in true.jobs},
        next_op_index={j.job_id: 0 for j in true.jobs}, agv_available={a.agv_id: 0 for a in true.agvs},
        agv_node={a.agv_id: a.start_node_id for a in true.agvs}, edge_reservations={e.edge_id: [] for e in true.path_edges},
        total_operations=sum(len(j.operations) for j in true.jobs),
    )
    operations, transports = [], []
    machine_by_id = {m.machine_id: m for m in true.machines}
    true_jobs = {j.job_id: j for j in true.jobs}
    event_times = sorted({e.event_time for e in true.events})
    t = 0 if strict else INF
    steps = 0
    while len(operations) < state.total_operations:
        if strict:
            visible = blind_instance(true, raw, t, state.next_op_index)
            ready = [state.job_ready[j.job_id] for j in visible.jobs if state.next_op_index[j.job_id] < len(j.operations)]
            if ready:
                t = max(t, min(ready))
            known = blind_instance(true, raw, t, state.next_op_index)
            while not _available_operations(known.jobs, state.next_op_index):
                later = [x for x in event_times if x > t]
                if not later:
                    raise ValueError("no schedulable operations remain")
                t = later[0]
                known = blind_instance(true, raw, t, state.next_op_index)
        else:
            known = true
        picked = choose(known, state, operations, transports, preference)
        job, machine = picked[0], picked[2]
        context = picked[3] if len(picked) > 3 else None
        true_job = true_jobs[job.job_id]
        true_op = true_job.operations[state.next_op_index[job.job_id]]
        if machine.machine_id not in true_op.eligible_machines:
            machine = _best_preference_action(
                instance=true, available=((true_job, true_op),), preference=preference,
                machine_available=state.machine_available, job_ready=state.job_ready, agv_available=state.agv_available,
                agv_node=state.agv_node, edge_reservations=state.edge_reservations,
            )["machine"]
        if context is not None:
            context.__enter__()
        try:
          if strict:
            _commit_strict(known=known, true=true, preference=preference, true_job=true_job, true_op=true_op,
                           machine=machine_by_id[machine.machine_id], state=state, operations=operations, transports=transports)
          else:
            _append_fixed_joint_preference_transport_action(
                instance=true, preference=preference, job=true_job, operation=true_op, machine=machine_by_id[machine.machine_id],
                machine_available=state.machine_available, job_ready=state.job_ready, next_op_index=state.next_op_index,
                agv_available=state.agv_available, agv_node=state.agv_node, edge_reservations=state.edge_reservations,
                scheduled_operations=operations, transport_moves=transports,
            )
        finally:
            if context is not None:
                context.__exit__(None, None, None)
        steps += 1
    return ScheduleResult(algorithm="policy", path_aware=bool(true.path_edges), operations=tuple(operations), transports=tuple(transports), event_aware=bool(true.events)), steps


def _checks(true, schedule, out, seed, level, mode, index, arm, info):
    """rerun v1: independent route/two-leg timing check and hard-constraint audit."""
    tw = str(Path(__file__).resolve().parents[2] / "twoleg")
    if tw not in sys.path:
        sys.path.append(tw)
    from run_early import check_schedule
    from su_modfjsp_ha_pc.metrics import evaluate_feasibility
    rb, tb, msgs, _ = check_schedule(true, schedule, "rerun")
    rep = evaluate_feasibility(true, schedule)
    hv = rep.precedence_violations + rep.machine_overlap_violations + rep.agv_overlap_violations + rep.edge_capacity_violations
    if rb or tb or hv or info.get("failures"):
        with (Path(out) / "check_failures.log").open("a") as log:
            log.write(f"seed{seed}_{level}_{mode}_pref{index} {arm}: {rb} {tb} {hv} {info.get('failures')} {msgs[:3]}\n")
    return dict(route_bad=rb, timing_bad=tb, hard_viol=hv)


def run_seed(job):
    seed, out, core, arms, fixed = job
    import torch
    torch.set_num_threads(1)
    from su_modfjsp_ha_pc.generator import generate_instance
    from su_modfjsp_ha_pc.neural_operation_rollout import _detail_row
    from su_modfjsp_ha_pc.objective_contract import read_objective_scales
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _explicit_preference_id, _explicit_teacher_preferences
    from su_modfjsp_ha_pc.pgnd_v5_exact_certificate import neural_ensemble_proposal_selector
    from su_modfjsp_ha_pc.pgnd_v5_proposer_ablation import deterministic_random_proposal_selector
    from su_modfjsp_ha_pc.run_pgnd_v4_learnability_gate import _load_artifact
    from su_modfjsp_ha_pc.run_pgnd_v5_smoke import MEMBER_STATES
    from su_modfjsp_ha_pc.structural import realize_structural_events

    scales = read_objective_scales(SCALES)
    neural = neural_ensemble_proposal_selector(tuple(_load_artifact(MODEL_DIR / f"pgnd_v4_member_{s}.pt") for s in MEMBER_STATES))
    random_selector = deterministic_random_proposal_selector()
    policies = load_policies()
    path = Path(out) / f"seed_{seed}.csv"
    fields = ["case_id", "seed", "level", "mode", "pref", "arm", "value", "wall_s", "accepted", "failures", "feasible", *OBJECTIVES, "route_bad", "timing_bad", "hard_viol"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for level in ("L4", "L5"):
            if core:
                from su_modfjsp_ha_pc.public_fjsp import load_and_extend_public_fjsp
                raw = load_and_extend_public_fjsp(Path(core), level=level, seed=seed, n_agvs=3)
            else:
                raw = generate_instance(level, "small", seed)
            if not fixed:  # rerun v1
                raw = randomize_events(raw, seed, level)
            for mode in ("raw", "realized"):
                true = realize_structural_events(raw).instance if mode == "realized" else raw
                for index, preference in enumerate(_explicit_teacher_preferences(len(true.objectives), max_preferences=6)):
                    preference_id = _explicit_preference_id(index, preference)
                    for arm in arms:
                        started = perf_counter()
                        info = dict(accepted=0, failures=0)
                        if arm in ("la", "pgnd", "rnd", "nogate"):
                            selector = {"la": None, "pgnd": neural, "rnd": random_selector, "nogate": neural}[arm]
                            schedule, info = decode(true=true, raw=raw, preference=preference, level=level, seed=seed, mode=mode, preference_id=preference_id,
                                                    selector=selector, scales=scales, clock="strict", margin=0.005 if arm in ("pgnd", "rnd") else 0.0, nogate=arm == "nogate")
                        else:
                            name = arm.replace("_orig", "")
                            choose = policies[name][index] if name == "dqn" else policies[name]
                            schedule, _ = policy_decode(true=true, raw=raw, preference=preference, choose=choose, strict=not arm.endswith("_orig"))
                        row = _detail_row(instance=true, schedule=schedule, teacher_operations=(), level=level, seed=seed, mode=mode, preference_id=preference_id, preference=preference, method=arm, objective_scales=scales)
                        writer.writerow(dict(case_id=f"seed{seed:03d}_{level}_{mode}_pref{index:02d}", seed=seed, level=level, mode=mode, pref=index, arm=arm,
                                             value=row["target_weighted_value"], wall_s=round(perf_counter() - started, 3), accepted=info["accepted"], failures=info["failures"],
                                             feasible=row.get("feasible", ""), **{k: row[k] for k in OBJECTIVES}, **_checks(true, schedule, out, seed, level, mode, index, arm, info)))
                        handle.flush()
    return seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="320-329")
    parser.add_argument("--core", default=None)
    parser.add_argument("--arms", default="la,pgnd,rnd,nogate,ppoA,ppoB,ppoA_orig,ppoB_orig")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--fixed-events", action="store_true")  # rerun v1
    args = parser.parse_args()
    first, last = map(int, args.seeds.split("-"))
    os.makedirs(args.out, exist_ok=True)
    jobs = [(s, args.out, args.core, tuple(args.arms.split(",")), args.fixed_events) for s in range(first, last + 1)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for seed in pool.imap_unordered(run_seed, jobs, chunksize=1):
            print("done", seed, flush=True)
