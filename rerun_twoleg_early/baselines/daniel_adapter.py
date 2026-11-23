"""v3: DANIEL (Wang et al., IEEE TNNLS 2023, official code external_baselines/FJSP-DRL-DANIEL) in the shared environment
(drl_v3_common.py).

Kept from the official code (fjsp_env_various_op_nums.py, model/main_model.py, params.py): the ten operation features, the
eight machine features and the eight pair features with their definitions and normalisation, the operation and machine
masks, the competition tensor, the dynamic pair mask (a pair is offered only if it can start at the earliest pair start
time), the dual attention network and the actor/critic with the default sizes (OAB/MAB heads [4, 4], output dims [32, 8],
actor and critic 3 layers of 64), and the PPO settings (lr 3e-4, gamma 1, GAE lambda 0.98, 4 epochs, clip 0.2, value 0.5,
entropy 0.01, minibatch 1024, 20 environments per update, 1000 updates).
Adapted (drl_v3_common): every time is taken from the shared two-leg simulator (job ready times, machine available times,
completion times of scheduled operations), divided by the largest processing time as in the official normalisation; the
AGV and route are completed by the shared transport rule; the preference vector is appended to the pair features of the
actor and to the critic input; the reward is the decrease of the weighted objective.
"""
from __future__ import annotations
import os as _pwmd_os
_PWMD_ROOT = _pwmd_os.path.abspath(_pwmd_os.path.join(_pwmd_os.path.dirname(__file__), *['..'] * 2))


def _pwmd_path(*parts):
    return _pwmd_os.path.join(_PWMD_ROOT, *parts)



import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(_pwmd_path('external_baselines', 'FJSP-DRL-DANIEL'))
CONFIG = SimpleNamespace(device="cpu", fea_j_input_dim=10, fea_m_input_dim=8, dropout_prob=0.0, num_heads_OAB=[4, 4],
                         num_heads_MAB=[4, 4], layer_fea_output_dim=[32, 8], num_mlp_layers_actor=3, hidden_dim_actor=64,
                         num_mlp_layers_critic=3, hidden_dim_critic=64)
PPO = dict(gamma=1.0, lam=0.98, clip=0.2, epochs=4, minibatch=1024, vf_coef=0.5, ent_coef=0.01)
N_PREF = 5


def _import():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import model.main_model as mm  # noqa: E402
    from model.sub_layers import Actor, Critic  # noqa: E402
    return mm, Actor, Critic


def make_model():
    import torch.nn as nn
    mm, Actor, Critic = _import()

    class DanielPref(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = mm.DANIEL(CONFIG)
            d = self.net.embedding_output_dim
            self.net.actor = Actor(CONFIG.num_mlp_layers_actor, 4 * d + 8 + N_PREF, CONFIG.hidden_dim_actor, 1)
            self.net.critic = Critic(CONFIG.num_mlp_layers_critic, 2 * d + N_PREF, CONFIG.hidden_dim_critic, 1)

    m = DanielPref()
    m.eval()
    return m


def make_optimizer(model):
    import torch
    return torch.optim.Adam(model.parameters(), lr=3e-4)


def _logic(x):
    """logic_operator of the official env for one instance: x [A, B] -> [B, B, A] (transposed pairwise AND)."""
    x = x.T
    return np.logical_and(x[:, None, :], x[None, :, :]).astype(np.float32)


def build(known, state, t, preference, operations):
    import torch
    machines = [m.machine_id for m in known.machines]
    M, col = len(machines), {m: i for i, m in enumerate(machines)}
    jobs = list(known.jobs)
    J = len(jobs)
    end_of = {o.operation_id: (o.machine_id, o.end) for o in operations}
    ops, first, last = [], [], []
    for job in jobs:
        first.append(len(ops))
        ops += [(job, k, op) for k, op in enumerate(job.operations)]
        last.append(len(ops) - 1)
    N = len(ops)
    ub = max(op.processing_time[m] for _, _, op in ops for m in op.eligible_machines)
    pt = np.zeros((N, M))
    for i, (_, _, op) in enumerate(ops):
        for mid in op.eligible_machines:
            pt[i, col[mid]] = op.processing_time[mid] / ub
    rel = pt != 0
    masked = np.ma.array(pt, mask=~rel)
    op_mean = masked.mean(axis=1).data
    op_min = masked.min(axis=1).data
    op_max = masked.max(axis=1).data
    span = op_max - op_min
    mch_min_pt = masked.max(axis=0).filled(0)  # as in the official code (np.max)
    mch_mean_pt = masked.mean(axis=0).filled(0)
    sched = np.zeros(N, dtype=bool)
    op_ct = np.zeros(N)
    for i, (job, k, op) in enumerate(ops):
        if k < state.next_op_index[job.job_id]:
            sched[i] = True
            op_ct[i] = end_of[op.operation_id][1] / ub if op.operation_id in end_of else state.job_ready[job.job_id] / ub
    remain = rel & ~sched[:, None]
    cand = np.array([first[j] + min(state.next_op_index[job.job_id], len(job.operations) - 1) for j, job in enumerate(jobs)])
    finished = np.array([state.next_op_index[job.job_id] >= len(job.operations) for job in jobs])
    cft = np.array([state.job_ready[job.job_id] / ub for job in jobs], dtype=float)
    mft = np.array([state.machine_available[m] / ub for m in machines], dtype=float)
    cand_rel = np.where(finished[:, None], True, ~rel[cand])  # candidate_process_relation (True = not processable)
    pair_ft = np.maximum(cft[:, None], mft[None, :])
    nst = np.ma.array(pair_ft, mask=cand_rel).min()
    nst = float(nst) if nst is not np.ma.masked else float(pair_ft.min())
    deleted = sched & (op_ct <= nst)
    # operation features
    op_ct_lb = np.zeros(N)
    left = np.zeros(N)
    remain_work_job = np.zeros(N)
    for j, job in enumerate(jobs):
        nk = state.next_op_index[job.job_id]
        acc = cft[j]
        for k in range(len(job.operations)):
            i = first[j] + k
            if k < nk:
                op_ct_lb[i] = op_ct[i]
            else:
                acc += op_min[i]
                op_ct_lb[i] = acc
        left[first[j]:last[j] + 1] = len(job.operations) - nk
        remain_work_job[first[j]:last[j] + 1] = op_mean[first[j] + nk:last[j] + 1].sum()
    waiting = np.zeros(N)
    for j in range(J):
        if not finished[j]:
            waiting[cand[j]] = max(nst - cft[j], 0.0)
    remain_work_op = np.maximum(op_ct - nst, 0) * sched
    avail_mch = rel.sum(1) / M
    fea_j = np.stack((sched.astype(float), op_ct_lb, op_min, span, op_mean, waiting, remain_work_op, left, remain_work_job, avail_mch), axis=1)
    fea_j[deleted] = 0
    n_left = max(N - deleted.sum(), 1e-8)
    mean = fea_j.sum(0) / n_left
    temp = np.where(deleted[:, None], mean[None, :], fea_j)
    std = np.sqrt(np.var(temp, axis=0) * N / n_left)
    fea_j = (temp - mean[None, :]) / (std[None, :] + 1e-8)
    # op mask: [no predecessor, -, no successor], plus deleted neighbours (update_op_mask)
    op_mask = np.zeros((N, 3), dtype=np.float32)
    op_mask[first, 0] = 1
    op_mask[last, 2] = 1
    obj = np.zeros_like(op_mask)
    obj[:, 2] = deleted
    obj[1:, 0] = deleted[:-1]
    op_mask = np.logical_or(obj, op_mask).astype(np.float32)
    # pair masks and machine features
    dyn = cand_rel | (pair_ft > nst + 1e-12)
    comp = _logic(~dyn)  # [M, M, J]
    mch_mask = _logic(remain).sum(-1).astype(bool)  # [M, M]
    del_m = ~mch_mask.sum(-1).astype(bool)
    np.fill_diagonal(mch_mask, True)
    jc = (~dyn).sum(0)
    op_nums = remain.sum(0)
    free_dur = nst - mft
    working = (free_dur < 0).astype(float)
    m_wait = (1 - working) * free_dur
    m_remain = np.maximum(-free_dur, 0)
    fea_m = np.stack((jc, op_nums, mch_min_pt, mch_mean_pt, m_wait, m_remain, mft, working), axis=1).astype(float)
    fea_m[del_m] = 0
    n_left_m = max(M - del_m.sum(), 1e-8)
    mean_m = fea_m.sum(0) / n_left_m
    temp_m = np.where(del_m[:, None], mean_m[None, :], fea_m)
    std_m = np.sqrt(np.var(temp_m, axis=0) * M / n_left_m)
    fea_m = (temp_m - mean_m[None, :]) / (std_m[None, :] + 1e-8)
    # pair features
    cand_pt = pt[cand]  # official candidate_pt: a finished job keeps the processing times of its last operation
    remain_pt = np.ma.array(pt, mask=~remain)
    max_remain = float(remain_pt.max()) if remain.any() else 1e-8
    mch_max_remain = remain_pt.max(axis=0).filled(1e-8)
    pair_max = cand_pt.max() + 1e-8
    mch_max_cand = cand_pt.max(axis=0) + 1e-8
    pair_wait = waiting[cand][:, None] + m_wait[None, :]
    job_rw = remain_work_job[cand][:, None] + 1e-8
    fea_pairs = np.stack((cand_pt, cand_pt / op_max[cand][:, None], cand_pt / mch_max_cand[None, :], cand_pt / max_remain,
                          cand_pt / mch_max_remain[None, :], cand_pt / pair_max, cand_pt / job_rw, pair_wait), axis=-1)
    valid = [(j * M + c, (jobs[j], jobs[j].operations[state.next_op_index[jobs[j].job_id]],
                          known.machines[c])) for j in range(J) for c in range(M) if not dyn[j, c]]
    T = lambda a, dt=torch.float: torch.tensor(np.asarray(a)[None], dtype=dt)
    obs = dict(fea_j=T(fea_j), op_mask=T(op_mask), cand=T(cand, torch.long), fea_m=T(fea_m), mch_mask=T(mch_mask, torch.bool),
               comp=T(comp), dyn=T(dyn, torch.bool), pairs=T(fea_pairs), pref=torch.tensor([list(preference)], dtype=torch.float),
               flat=torch.tensor([v[0] for v in valid]))
    return obs, [v[1] for v in valid]


def _forward(model, fea_j, op_mask, cand, fea_m, mch_mask, comp, dyn, pairs, pref):
    """DANIEL.forward with the preference appended to the pair features and to the critic input; returns scores."""
    import torch
    net = model.net
    fea_j, fea_m, gj, gm = net.feature_exact(fea_j, op_mask, cand, fea_m, mch_mask, comp)
    sz_b, M, _, J = comp.size()
    d = fea_j.size(-1)
    idx = cand.unsqueeze(-1).repeat(1, 1, d).long()
    fjc = torch.gather(fea_j, 1, idx).unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
    fm = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)
    p = pairs.reshape(sz_b, -1, 8)
    pr = pref[:, None, :].expand(sz_b, M * J, pref.size(-1))
    x = torch.cat((fjc, fm, gj.unsqueeze(1).expand_as(fjc), gm.unsqueeze(1).expand_as(fjc), p, pr), dim=-1)
    scores = net.actor(x).squeeze(-1)
    scores = scores.masked_fill(dyn.reshape(sz_b, -1), float("-inf"))
    v = net.critic(torch.cat((gj, gm, pref), dim=-1)).reshape(sz_b)
    return scores, v


def forward(model, obs):
    s, v = _forward(model, obs["fea_j"], obs["op_mask"], obs["cand"], obs["fea_m"], obs["mch_mask"], obs["comp"], obs["dyn"],
                    obs["pairs"], obs["pref"])
    return s[0][obs["flat"]], v[0]


def shape_key(obs):
    return tuple(obs["fea_j"].shape) + tuple(obs["comp"].shape)


def forward_batch(model, obs_list):
    import torch
    cat = lambda k: torch.cat([o[k] for o in obs_list])
    s, v = _forward(model, cat("fea_j"), cat("op_mask"), cat("cand"), cat("fea_m"), cat("mch_mask"), cat("comp"), cat("dyn"),
                    cat("pairs"), cat("pref"))
    return [s[b][o["flat"]] for b, o in enumerate(obs_list)], v


class Adapter:
    reward = "estimate"  # dense estimated-objective reward, as the original code (author 2026-11-11)
    build = staticmethod(build)
    forward = staticmethod(forward)
    forward_batch = staticmethod(forward_batch)
    shape_key = staticmethod(shape_key)


def chooser(model):
    import torch
    from drl_v3_common import current_time

    def choose(known, state, operations, transports, preference):
        obs, cands = build(known, state, current_time(known, state), preference, operations)
        with torch.no_grad():
            logits, _ = forward(model, obs)
        return cands[int(torch.argmax(logits).item())]
    return choose
