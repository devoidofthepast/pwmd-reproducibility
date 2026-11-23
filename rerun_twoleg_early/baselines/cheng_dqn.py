"""Reimplementation of Cheng et al. (2025, Computers & Operations Research 181, 107087) in the shared environment.

DQN that selects a composite dispatching rule at each decision, from twelve shop-floor state features.
Taken from the paper: the twelve features, the four operation rules (MOPNR, FIFO, LWKR, MWKR) and two machine rules (EET, SPT),
the reward as the decrease of a weighted objective between consecutive states, one model per weight vector, a five-layer ReLU network
(64-128-64-32), Adam, epsilon 1.0 to 0.1, discount 0.99, learning rate 1e-4, batch 32, replay buffer 1000, target copy every 100 steps,
and 3000 training episodes.
Adapted: the two AGV rules are replaced by the shared transport completion used by every method, so there are 8 actions instead of 16;
the weighted reward uses the five normalized components of this study; time features are divided by the instance's total minimum
processing time. Writes only to cheng_dqn/.
"""
from __future__ import annotations
import os as _pwmd_os
_PWMD_ROOT = _pwmd_os.path.abspath(_pwmd_os.path.join(_pwmd_os.path.dirname(__file__), *['..'] * 2))


def _pwmd_path(*parts):
    return _pwmd_os.path.join(_PWMD_ROOT, *parts)



import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import statistics as st
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, _pwmd_path('rerun_twoleg_early', 'src'))  # rerun v1
from event_blind import SCALES  # noqa: E402

OP_RULES = ("MOPNR", "FIFO", "LWKR", "MWKR")
MACHINE_RULES = ("EET", "SPT")
ACTIONS = tuple((o, m) for o in OP_RULES for m in MACHINE_RULES)
NAMES = ("makespan", "tardiness", "energy", "route_pressure", "event_exposure_pressure")


def remaining_work(job, index):
    return sum(min(op.processing_time.values()) for op in job.operations[index:])


def apply_rule(action, instance, state):
    from su_modfjsp_ha_pc.rules import _available_operations
    op_rule, machine_rule = ACTIONS[action]
    available = _available_operations(instance.jobs, state.next_op_index)
    key = {
        "MOPNR": lambda jo: (-(len(jo[0].operations) - state.next_op_index[jo[0].job_id]), jo[0].job_id),
        "FIFO": lambda jo: (state.job_ready[jo[0].job_id], jo[0].job_id),
        "LWKR": lambda jo: (remaining_work(jo[0], state.next_op_index[jo[0].job_id]), jo[0].job_id),
        "MWKR": lambda jo: (-remaining_work(jo[0], state.next_op_index[jo[0].job_id]), jo[0].job_id),
    }[op_rule]
    job, op = min(available, key=key)
    machines = {m.machine_id: m for m in instance.machines}
    if machine_rule == "EET":
        mid = min(op.eligible_machines, key=lambda m: (state.machine_available[m], op.processing_time[m], m))
    else:
        mid = min(op.eligible_machines, key=lambda m: (op.processing_time[m], state.machine_available[m], m))
    return job, op, machines[mid]


def features(instance, state, operations, transports):
    scale = max(1.0, float(sum(min(o.processing_time.values()) for j in instance.jobs for o in j.operations)))
    busy, finish = {m.machine_id: 0.0 for m in instance.machines}, {m.machine_id: 0.0 for m in instance.machines}
    jend = {j.job_id: 0.0 for j in instance.jobs}
    for o in operations:
        busy[o.machine_id] += o.end - o.start
        finish[o.machine_id] = max(finish[o.machine_id], o.end)
        jend[o.job_id] = max(jend[o.job_id], o.end)
    vbusy, vend = {a.agv_id: 0.0 for a in instance.agvs}, {a.agv_id: 0.0 for a in instance.agvs}
    for m in transports:
        vbusy[m.agv_id] += m.end - m.start
        vend[m.agv_id] = max(vend[m.agv_id], m.end)
    mu = [busy[k] / finish[k] if finish[k] > 0 else 0.0 for k in busy]
    mct = [finish[k] / scale for k in finish]
    cr = [state.next_op_index[j.job_id] / len(j.operations) for j in instance.jobs]
    jct = [jend[j.job_id] / scale for j in instance.jobs]
    vu = [vbusy[k] / vend[k] if vend[k] > 0 else 0.0 for k in vbusy]
    vct = [vend[k] / scale for k in vend]
    out = []
    for values in (mu, mct, cr, jct, vu, vct):
        out += [float(np.mean(values)), float(np.std(values))]
    return np.asarray(out, dtype=np.float32)


def network():
    import torch.nn as nn
    return nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, len(ACTIONS)))


def partial_value(instance, operations, transports, preference, scales):
    from su_modfjsp_ha_pc.objective_contract import evaluate_objective_contract
    from su_modfjsp_ha_pc.schedule import ScheduleResult
    raw = evaluate_objective_contract(instance, ScheduleResult(algorithm="dqn", path_aware=True, operations=tuple(operations), transports=tuple(transports), event_aware=True)).raw
    return sum(w * raw[n] / scales.values[n] for w, n in zip(preference, NAMES))


def train_one(job):
    pref_index, episodes, out, run, lr = job
    import torch
    torch.set_num_threads(1)
    from su_modfjsp_ha_pc.generator import generate_instance
    from su_modfjsp_ha_pc.hierarchical_transport_attention_policy import _append_fixed_joint_preference_transport_action
    from su_modfjsp_ha_pc.neural_operation_rollout import _RolloutState
    from su_modfjsp_ha_pc.objective_contract import read_objective_scales
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _explicit_teacher_preferences
    from su_modfjsp_ha_pc.structural import realize_structural_events

    scales = read_objective_scales(SCALES)
    rng = random.Random(1000 + 100 * run + pref_index)
    torch.manual_seed(1000 + 100 * run + pref_index)
    instances = []
    for seed in range(600, 620):  # rerun v1 training seeds
        for level in ("L4", "L5"):
            raw = generate_instance(level, "small", seed)
            instances += [raw, realize_structural_events(raw).instance]
    preference = _explicit_teacher_preferences(len(instances[0].objectives), max_preferences=6)[pref_index]
    q, target = network(), network()
    target.load_state_dict(q.state_dict())
    optimizer = torch.optim.Adam(q.parameters(), lr=lr)
    buffer, steps, log = [], 0, []
    started = perf_counter()
    for episode in range(episodes):
        epsilon = max(0.1, 1.0 - 0.9 * episode / max(1, episodes - 1))
        instance = rng.choice(instances)
        state = _RolloutState(
            machine_available={m.machine_id: 0 for m in instance.machines}, job_ready={j.job_id: j.release_time for j in instance.jobs},
            next_op_index={j.job_id: 0 for j in instance.jobs}, agv_available={a.agv_id: 0 for a in instance.agvs},
            agv_node={a.agv_id: a.start_node_id for a in instance.agvs}, edge_reservations={e.edge_id: [] for e in instance.path_edges},
            total_operations=sum(len(j.operations) for j in instance.jobs),
        )
        operations, transports = [], []
        s = features(instance, state, operations, transports)
        value = 0.0
        while len(operations) < state.total_operations:
            if rng.random() < epsilon:
                a = rng.randrange(len(ACTIONS))
            else:
                with torch.no_grad():
                    a = int(q(torch.from_numpy(s)).argmax().item())
            job, op, machine = apply_rule(a, instance, state)
            _append_fixed_joint_preference_transport_action(
                instance=instance, preference=preference, job=job, operation=op, machine=machine,
                machine_available=state.machine_available, job_ready=state.job_ready, next_op_index=state.next_op_index,
                agv_available=state.agv_available, agv_node=state.agv_node, edge_reservations=state.edge_reservations,
                scheduled_operations=operations, transport_moves=transports,
            )
            new_value = partial_value(instance, operations, transports, preference, scales)
            reward = value - new_value
            value = new_value
            done = len(operations) >= state.total_operations
            s2 = features(instance, state, operations, transports)
            buffer.append((s, a, reward, s2, done))
            if len(buffer) > 1000:
                buffer.pop(0)
            s = s2
            steps += 1
            if len(buffer) >= 32:
                batch = rng.sample(buffer, 32)
                bs = torch.from_numpy(np.stack([b[0] for b in batch]))
                ba = torch.tensor([b[1] for b in batch])
                br = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                bs2 = torch.from_numpy(np.stack([b[3] for b in batch]))
                bd = torch.tensor([b[4] for b in batch], dtype=torch.float32)
                with torch.no_grad():
                    y = br + 0.99 * (1 - bd) * target(bs2).max(dim=1).values
                loss = ((q(bs).gather(1, ba[:, None]).squeeze(1) - y) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if steps % 100 == 0:
                target.load_state_dict(q.state_dict())
        log.append(dict(episode=episode, value=value, epsilon=epsilon, wall_s=perf_counter() - started))
    Path(out).mkdir(parents=True, exist_ok=True)
    torch.save(q.state_dict(), f"{out}/dqn_pref{pref_index:02d}.pt")
    with open(f"{out}/train_log_pref{pref_index:02d}.jsonl", "w") as handle:
        for row in log:
            handle.write(json.dumps(row) + "\n")
    return run, pref_index, perf_counter() - started, st.mean(r["value"] for r in log[-100:])


def chooser(model):
    import torch

    def choose(known, state, operations, transports, preference):
        with torch.no_grad():
            a = int(model(torch.from_numpy(features(known, state, operations, transports))).argmax().item())
        return apply_rule(a, known, state)
    return choose


def eval_seed(job):
    seed, out, models, core, randomize = job
    import torch
    torch.set_num_threads(1)
    from blind_arms import policy_decode
    from event_blind import randomize_events
    from su_modfjsp_ha_pc.generator import generate_instance
    from su_modfjsp_ha_pc.neural_operation_rollout import _detail_row
    from su_modfjsp_ha_pc.objective_contract import read_objective_scales
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _explicit_preference_id, _explicit_teacher_preferences
    from su_modfjsp_ha_pc.structural import realize_structural_events

    scales = read_objective_scales(SCALES)
    nets = []
    for i in range(6):
        net = network()
        net.load_state_dict(torch.load(f"{models}/dqn_pref{i:02d}.pt"))
        net.eval()
        nets.append(net)
    path = Path(out) / f"seed_{seed}.csv"
    fields = ["case_id", "seed", "level", "mode", "pref", "arm", "value", "wall_s", "makespan", "tardiness", "energy", "teacher_path_pressure", "teacher_stability_pressure"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for level in ("L4", "L5"):
            if core:
                from su_modfjsp_ha_pc.public_fjsp import load_and_extend_public_fjsp
                raw = load_and_extend_public_fjsp(Path(core), level=level, seed=seed, n_agvs=3)
            else:
                raw = generate_instance(level, "small", seed)
            if randomize:
                raw = randomize_events(raw, seed, level)
            for mode in ("raw", "realized"):
                true = realize_structural_events(raw).instance if mode == "realized" else raw
                for index, preference in enumerate(_explicit_teacher_preferences(len(true.objectives), max_preferences=6)):
                    for arm, strict in (("dqn_strict", True), ("dqn_oracle", False)):
                        started = perf_counter()
                        schedule, _ = policy_decode(true=true, raw=raw, preference=preference, choose=chooser(nets[index]), strict=strict)
                        row = _detail_row(instance=true, schedule=schedule, teacher_operations=(), level=level, seed=seed, mode=mode,
                                          preference_id=_explicit_preference_id(index, preference), preference=preference, method=arm, objective_scales=scales)
                        writer.writerow(dict(case_id=f"seed{seed:03d}_{level}_{mode}_pref{index:02d}", seed=seed, level=level, mode=mode, pref=index, arm=arm,
                                             value=row["target_weighted_value"], wall_s=round(perf_counter() - started, 3),
                                             **{k: row[k] for k in ("makespan", "tardiness", "energy", "teacher_path_pressure", "teacher_stability_pressure")}))
                        handle.flush()
    return seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "eval"))
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--models", default="cheng_dqn/models")
    parser.add_argument("--out")
    parser.add_argument("--seeds", default="320-329")
    parser.add_argument("--core", default=None)
    parser.add_argument("--fixed-events", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    if args.command == "train":
        with mp.get_context("spawn").Pool(18) as pool:
            for result in pool.imap_unordered(train_one, [(i, args.episodes, f"{args.models}/run{r}", r, args.lr) for r in range(args.runs) for i in range(6)]):
                print("trained", result, flush=True)
    else:
        first, last = map(int, args.seeds.split("-"))
        os.makedirs(args.out, exist_ok=True)
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for seed in pool.imap_unordered(eval_seed, [(s, args.out, args.models, args.core, not args.fixed_events) for s in range(first, last + 1)]):
                print("done", seed, flush=True)
