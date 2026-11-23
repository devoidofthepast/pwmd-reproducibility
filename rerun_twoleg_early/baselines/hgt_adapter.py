"""v3: HGT (Liu et al., ESWA 2026, official code external_baselines/MDFJSSP-HGT) in the shared environment (drl_v3_common.py).

Kept from the official code (env/base_class.py, env/mdfjssp.py, policy/hgan.py, agent/ppo.py) and the paper (section 5.1):
the heterogeneous graph with a dummy finished operation linked to every first operation, the seven operation features,
four machine features and two arc features with their definitions, the machine-operation and operation encoder masks,
the action mask (idle, not broken-down machine; unfinished, not processing operation whose predecessor has finished),
the HGAN encoder (4 layers, d_model 128, d_hidden 256, 16 heads, d_kv 16, dropout 0.2) with its batch-norm feature
embedding and bilinear machine-operation scores, sampling in training mode, and the training settings of the paper
(100 iterations of 10 episodes, 3 epochs, minibatch 64, Adam with weight decay 1e-2, learning-rate warm-up to 5e-5
followed by cosine annealing, gradient clipping at 1, discount 0.9, clip 0.15, value 1, entropy 0.01).
Adapted (drl_v3_common): the decision time is the earliest time at which a ready operation and an idle machine meet in the
shared two-leg simulator, and every time-based feature is read from the simulator at that time (processing and finished
status from the committed schedule, remaining processing time, known breakdowns); the AGV and route are completed by the
shared transport rule; the preference vector enters through a linear embedding added to the operation and machine
embeddings before the scores and the value head; the reward is the decrease of the weighted objective.
"""
from __future__ import annotations
import os as _pwmd_os
_PWMD_ROOT = _pwmd_os.path.abspath(_pwmd_os.path.join(_pwmd_os.path.dirname(__file__), *['..'] * 2))


def _pwmd_path(*parts):
    return _pwmd_os.path.join(_PWMD_ROOT, *parts)



import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(_pwmd_path('external_baselines', 'MDFJSSP-HGT'))
ARGS = dict(d_machine_raw=4, d_operation_raw=7, d_arc_raw=2, d_model=128, d_hidden=256, num_layers=4, num_head=16, d_kv=16, dropout=0.2)
PPO = dict(gamma=0.9, lam=1.0, clip=0.15, epochs=3, minibatch=64, vf_coef=1.0, ent_coef=0.01, max_grad_norm=1.0)
ITERATIONS, EPISODES = 100, 10
N_PREF = 5


def _import():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from env.base_class import State  # noqa: E402
    from policy.hgan import HGAN  # noqa: E402
    return HGAN, State


def make_model():
    import torch.nn as nn
    HGAN, _ = _import()

    class HGTPref(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = HGAN(**ARGS, device="cpu")
            self.pref_o = nn.Linear(N_PREF, ARGS["d_model"])
            self.pref_m = nn.Linear(N_PREF, ARGS["d_model"])

    m = HGTPref()
    m.eval()
    return m


def make_optimizer(model):
    import torch
    return torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-2)


def lr_schedule(it, warm=14, total=ITERATIONS, peak=5e-5):
    """Warm-up then cosine annealing (agent/ppo.py WarmupCosineAnnealingLR), stepped per iteration; the paper warms up
    over the first 300 minibatches, about 14 iterations here."""
    if it < warm:
        return peak * (it + 1) / warm
    return 0.5 * peak * (1 + math.cos(math.pi * (it - warm) / max(1, total - warm)))


def _overlap(a, b, c, d):
    return max(0, min(b, d) - max(a, c))


def build(known, state, t, preference, operations):
    import torch
    machines = [m.machine_id for m in known.machines]
    M, col = len(machines), {m: i for i, m in enumerate(machines)}
    jobs = list(known.jobs)
    by_op = {o.operation_id: o for o in operations}
    ops = [None]  # index 0: dummy finished operation
    first = []
    for job in jobs:
        first.append(len(ops))
        ops += [(job, k, op) for k, op in enumerate(job.operations)]
    N0 = len(ops)
    # decision time: earliest start of a ready operation on a free machine
    pairs = []
    for job in jobs:
        nk = state.next_op_index[job.job_id]
        if nk < len(job.operations):
            for mid in job.operations[nk].eligible_machines:
                pairs.append(max(state.job_ready[job.job_id], state.machine_available[mid]))
    tn = min(pairs) if pairs else t
    op_time = np.zeros((M, N0))
    timer = np.zeros((M, N0))
    present = np.zeros(N0, dtype=bool)
    finished = np.zeros(N0, dtype=bool)
    processing = np.zeros(N0, dtype=bool)
    finished[0] = True
    adj = np.zeros((N0, N0), dtype=bool)
    feat_o = np.zeros((N0, 7))
    for j, job in enumerate(jobs):
        nk = state.next_op_index[job.job_id]
        n = len(job.operations)
        for k, op in enumerate(job.operations):
            i = first[j] + k
            present[i] = True
            adj[0 if k == 0 else i - 1, i] = True
            feat_o[i, 2] = (k + 1) / n
            feat_o[i, 5] = len(op.eligible_machines)
            feat_o[i, 6] = n - k - 1
            for mid in op.eligible_machines:  # the official state keeps every eligible arc, also after assignment
                op_time[col[mid], i] = op.processing_time[mid]
            rec = by_op.get(op.operation_id) if k < nk else None
            if rec is not None:
                c = col[rec.machine_id]
                if rec.end <= tn:
                    finished[i] = True
                else:
                    processing[i] = True
                    timer[c, i] = min(op.processing_time[rec.machine_id], rec.end - tn)
    mo = op_time > 0
    feat_o[:, 0] = processing.astype(float)
    feat_o[:, 1] = op_time.sum(0) / (mo.sum(0) + 1e-5)
    feat_o[:, 3] = timer.sum(0)
    for j, job in enumerate(jobs):
        nk = state.next_op_index[job.job_id]
        rest = [min(op.processing_time[m] for m in op.eligible_machines) for op in job.operations[nk:]]
        start = max(state.job_ready[job.job_id], tn) if rest else state.job_ready[job.job_id]
        feat_o[first[j]:first[j] + len(job.operations), 4] = start + sum(rest)
    idle = np.array([state.machine_available[m] <= tn for m in machines])
    broken = np.zeros(M, dtype=bool)
    broken_rec = np.zeros(M)
    for e in known.events:
        if e.event_type == "machine_breakdown" and e.affected_entity in col and e.duration > 0:
            c = col[e.affected_entity]
            if e.event_time <= tn < e.event_time + e.duration:
                broken[c] = True
            broken_rec[c] += _overlap(0, tn, e.event_time, e.event_time + e.duration)
    feat_m = np.zeros((M, 4))
    feat_m[:, 0] = np.where(broken, -1.0, np.where(idle, 0.0, 1.0))
    feat_m[:, 1] = (mo & ~finished[None, :] & present[None, :]).sum(1)
    feat_m[:, 2] = np.where(timer > 0, op_time - timer, 0).sum(1)
    feat_m[:, 3] = broken_rec / (tn + 1e-5)
    arc = np.zeros((M, N0, 2))
    arc[:, :, 0] = np.where(op_time == 0, op_time.max(), op_time)
    arc[:, :, 1] = (op_time - timer) * np.sign(timer)
    pred_done = (finished.astype(float) @ adj.astype(float)) > 0
    base = idle[:, None] & mo & present[None, :] & ~finished[None, :] & pred_done[None, :] & ~processing[None, :]
    amask = base & ~broken[:, None]
    if not amask.any():  # every free machine is broken down at the decision time: offer the pairs on them
        amask = base
    cands, flat = [], []
    for c in range(M):
        for i in np.nonzero(amask[c])[0]:
            job, k, op = ops[i]
            if k != state.next_op_index[job.job_id]:
                continue
            flat.append(c * N0 + int(i))
            cands.append((job, op, known.machines[c]))
    if not cands:  # no pair meets the action mask: offer every ready pair (never reached in the smoke runs)
        for j, job in enumerate(jobs):
            nk = state.next_op_index[job.job_id]
            if nk < len(job.operations):
                for mid in job.operations[nk].eligible_machines:
                    c, i = col[mid], first[j] + nk
                    amask[c, i] = True
                    flat.append(c * N0 + i)
                    cands.append((job, job.operations[nk], known.machines[c]))
    T = lambda a, dt=torch.float: torch.tensor(np.asarray(a)[None], dtype=dt)
    obs = dict(op=T(feat_o), m=T(feat_m), arc=T(arc), broken=T(broken[:, None], torch.bool), idle=T(idle[:, None], torch.bool),
               mo=T(mo, torch.bool), present=T(present[None, :], torch.bool), proc=T(processing[None, :], torch.bool),
               fin=T(finished[None, :], torch.bool), adj=T(adj, torch.bool), amask=T(amask, torch.bool),
               pref=torch.tensor([list(preference)], dtype=torch.float), flat=torch.tensor(flat))
    return obs, cands


def _state(o):
    _, State = _import()
    return State(operation_raw_feature=o["op"], machine_raw_feature=o["m"], arc_raw_feature=o["arc"], mask_broken_down=o["broken"],
                 mask_machine=o["idle"], mask_machine_operation=o["mo"], mask_operation=o["present"], mask_operation_processing=o["proc"],
                 mask_operation_finished=o["fin"], operation_adj_matrix=o["adj"], operation_time=None, accu_matrix=None,
                 machine_broken_record=None, operation_finished_time_earliest=None, last_operation_per_job=None)


def _forward(model, o):
    import torch
    import torch.nn.functional as F
    net = model.net
    st = _state(o)
    h_m, h_o = net.encode(st)
    h_m = h_m + model.pref_m(o["pref"])[:, None, :]
    h_o = h_o + model.pref_o(o["pref"])[:, None, :]
    x = net.w_m(h_m).bmm(net.w_o(h_o).transpose(-1, -2))
    x = x - x.max(dim=-1, keepdim=True)[0]
    x = x.masked_fill(~o["amask"], float("-inf"))
    scores = net.flat(x)
    om = (~o["fin"] * o["present"]).transpose(-1, -2)
    ho = torch.where(om, h_o, torch.zeros_like(h_o))
    pooled = torch.cat([ho.sum(dim=1) / om.sum(dim=1).float().clamp_min(1), h_m.mean(dim=1)], dim=-1)
    v = net.evaluate_mlp(pooled).reshape(-1)
    return scores, v


def forward(model, obs):
    s, v = _forward(model, obs)
    return s[0][obs["flat"]], v[0]


def shape_key(obs):
    return tuple(obs["arc"].shape)


def forward_batch(model, obs_list):
    import torch
    keys = ("op", "m", "arc", "broken", "idle", "mo", "present", "proc", "fin", "adj", "amask", "pref")
    o = {k: torch.cat([x[k] for x in obs_list]) for k in keys}
    s, v = _forward(model, o)
    return [s[b][x["flat"]] for b, x in enumerate(obs_list)], v


class Adapter:
    reward = "estimate"  # dense estimated-objective reward, as the original code (author 2026-11-11)
    build = staticmethod(build)
    forward = staticmethod(forward)
    forward_batch = staticmethod(forward_batch)
    shape_key = staticmethod(shape_key)
    rollout_train_mode = True  # agent/ppo.py take_action(training=True)


def chooser(model):
    import torch
    from drl_v3_common import current_time

    def choose(known, state, operations, transports, preference):
        model.eval()
        obs, cands = build(known, state, current_time(known, state), preference, operations)
        with torch.no_grad():
            logits, _ = forward(model, obs)
        return cands[int(torch.argmax(logits).item())]
    return choose
