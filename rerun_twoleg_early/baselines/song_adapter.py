"""v3: DRL-S (Song et al., IEEE TII 2023, official code external_baselines/fjsp-drl-song, commit d7637b7) in the shared
environment (drl_v3_common.py).

Kept from the official code: the heterogeneous graph (operation-machine arcs with processing times, predecessor and
successor arcs), the six raw operation features and three raw machine features with their definitions, the feature
normalisation, the GATedge machine embedding and MLPs operation embedding (two layers), the actor/critic MLPs and their
sizes (config.json), and the PPO settings (lr 2e-4, betas (0.9, 0.999), gamma 1, K_epochs 3, clip 0.2, vf 0.5,
entropy 0.01, minibatch 512, 20 episodes per update, 1000 updates).
Adapted (drl_v3_common): times come from the shared two-leg simulator (start and completion times of scheduled
operations, the job ready time of the next operation, machine available times); the candidate pairs are the next
operation of every unfinished job with each eligible machine, as for every method; the AGV and route are completed by the
shared transport rule; the preference vector is concatenated to the actor and critic inputs; the reward is the decrease of
the weighted objective.
"""
from __future__ import annotations
import os as _pwmd_os
_PWMD_ROOT = _pwmd_os.path.abspath(_pwmd_os.path.join(_pwmd_os.path.dirname(__file__), *['..'] * 2))


def _pwmd_path(*parts):
    return _pwmd_os.path.join(_PWMD_ROOT, *parts)



import sys
from pathlib import Path

import numpy as np

REPO = Path(_pwmd_path('external_baselines', 'fjsp-drl-song'))
CONFIG = dict(in_size_ma=3, out_size_ma=8, in_size_ope=6, out_size_ope=8, hidden_size_ope=128, num_heads=[1, 1], dropout=0.0,
              n_latent_actor=64, n_latent_critic=64, n_hidden_actor=3, n_hidden_critic=3, action_dim=1)
PPO = dict(gamma=1.0, lam=1.0, clip=0.2, epochs=3, minibatch=512, vf_coef=0.5, ent_coef=0.01)
N_PREF = 5


def _import():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import PPO_model  # noqa: E402
    return PPO_model


def make_model():
    import torch
    m = _import()
    paras = dict(CONFIG, device=torch.device("cpu"))
    paras["actor_in_dim"] = paras["out_size_ma"] * 2 + paras["out_size_ope"] * 2 + N_PREF  # + preference (v3)
    paras["critic_in_dim"] = paras["out_size_ma"] + paras["out_size_ope"] + N_PREF
    model = m.HGNNScheduler(paras)
    model.eval()
    return model


def make_optimizer(model):
    import torch
    return torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))


def build(known, state, t, preference, operations):
    """Graph of the operations of the jobs known now (scheduled ones keep only their assigned machine arc)."""
    import torch
    machines = [m.machine_id for m in known.machines]
    col = {m: i for i, m in enumerate(machines)}
    done_ops, busy = {}, {}
    for o in operations:
        done_ops[o.operation_id] = (o.machine_id, o.start)
        busy[o.machine_id] = busy.get(o.machine_id, 0.0) + (o.end - o.start)
    ops, job_first, job_last, next_idx, cands = [], [], [], [], []
    for job in known.jobs:
        first = len(ops)
        for k, op in enumerate(job.operations):
            ops.append((job, k, op))
        job_first.append(first)
        job_last.append(len(ops) - 1)
        nk = state.next_op_index[job.job_id]
        next_idx.append(first + min(nk, len(job.operations) - 1))
    n, m = len(ops), len(machines)
    pt = np.zeros((n, m))
    adj = np.zeros((n, m), dtype=np.int64)
    pre = np.zeros((n, n), dtype=bool)
    feat_o = np.zeros((n, 6))
    for i, (job, k, op) in enumerate(ops):
        sched = k < state.next_op_index[job.job_id]
        rec = done_ops.get(op.operation_id)
        if sched and rec is not None:
            adj[i, col[rec[0]]] = 1
            pt[i, col[rec[0]]] = op.processing_time[rec[0]]
        else:
            for mid in op.eligible_machines:
                adj[i, col[mid]] = 1
                pt[i, col[mid]] = op.processing_time[mid]
        if k + 1 < len(job.operations):
            pre[i, i + 1] = True
    # raw features (Song et al., Table I): status, #neighbour machines, processing time, #unscheduled ops in job,
    # job completion time (estimate), start time (actual or estimate)
    for ji, job in enumerate(known.jobs):
        nk = state.next_op_index[job.job_id]
        est = float(state.job_ready[job.job_id])
        for k, op in enumerate(job.operations):
            i = job_first[ji] + k
            rec = done_ops.get(op.operation_id)
            feat_o[i, 0] = 1.0 if k < nk else 0.0
            feat_o[i, 1] = adj[i].sum()
            feat_o[i, 2] = pt[i][adj[i] == 1].mean() if adj[i].any() else 0.0
            feat_o[i, 3] = len(job.operations) - nk
            if k < nk and rec is not None:
                feat_o[i, 5] = rec[1]
            else:
                feat_o[i, 5] = est
                est += feat_o[i, 2]
        end = feat_o[job_last[ji], 5] + feat_o[job_last[ji], 2]
        feat_o[job_first[ji]:job_last[ji] + 1, 4] = end
    # environment time of the official code: the earliest time at which an idle machine meets a job that is not being
    # processed; only those pairs are eligible (non-delay action set of fjsp_env.if_no_eligible / next_time)
    pair_start = {}
    for ji, job in enumerate(known.jobs):
        nk = state.next_op_index[job.job_id]
        if nk < len(job.operations):
            for mid in job.operations[nk].eligible_machines:
                pair_start[(ji, col[mid])] = max(state.job_ready[job.job_id], state.machine_available[mid])
    t_env = min(pair_start.values()) if pair_start else t
    feat_m = np.zeros((m, 3))
    for c, mid in enumerate(machines):
        feat_m[c, 0] = adj[:, c].sum()
        feat_m[c, 1] = state.machine_available[mid]
        feat_m[c, 2] = min(busy.get(mid, 0.0), t_env) / (t_env + 1e-9)
    # candidates: next operation of each unfinished job x eligible machine
    elig = np.zeros((len(known.jobs), m), dtype=bool)
    for ji, job in enumerate(known.jobs):
        nk = state.next_op_index[job.job_id]
        if nk < len(job.operations):
            op = job.operations[nk]
            for mid in op.eligible_machines:
                elig[ji, col[mid]] = pair_start[(ji, col[mid])] <= t_env
    # action order of the official code: index = machine * n_jobs + job
    order = []
    for c in range(m):
        for ji, job in enumerate(known.jobs):
            if elig[ji, c]:
                op = job.operations[state.next_op_index[job.job_id]]
                order.append((c * len(known.jobs) + ji, (job, op, next(x for x in known.machines if x.machine_id == machines[c]))))
    obs = dict(
        feat_o=torch.tensor(feat_o[None], dtype=torch.float), feat_m=torch.tensor(feat_m[None], dtype=torch.float),
        pt=torch.tensor(pt[None], dtype=torch.float), adj=torch.tensor(adj[None]), pre=torch.tensor(pre[None]).long(),
        next_idx=torch.tensor([next_idx]), elig=torch.tensor(elig[None]), flat=torch.tensor([o[0] for o in order]),
        pref=torch.tensor([list(preference)], dtype=torch.float), n=n,
    )
    return obs, [o[1] for o in order]


def forward(model, obs):
    """get_action_prob/evaluate of the official HGNNScheduler on one instance, with the preference appended to the actor
    and critic inputs. Returns logits over the candidate list and the state value."""
    import torch
    nf = _normalize_one(model, obs["feat_o"][0], obs["feat_m"][0], obs["pt"][0])
    feats = (nf[0][None], nf[1][None], nf[2][None])
    adj, pre = obs["adj"], obs["pre"]
    sub = pre.transpose(1, 2)
    bidx = torch.arange(1)
    for i in range(len(model.num_heads)):
        h_mas = model.get_machines[i](adj, bidx, feats)
        feats = (feats[0], h_mas, feats[2])
        h_opes = model.get_operations[i](adj, pre, sub, bidx, feats)
        feats = (h_opes, feats[1], feats[2])
    h_mas_pooled = h_mas.mean(dim=-2)
    h_opes_pooled = h_opes.mean(dim=-2)
    jobs_gather = obs["next_idx"][..., :, None].expand(-1, -1, h_opes.size(-1))
    h_jobs = h_opes.gather(1, jobs_gather)
    h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, obs["pt"].size(-1), -1)
    h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
    h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
    h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)
    pref_padding = obs["pref"][:, None, None, :].expand(*h_jobs_padding.shape[:3], obs["pref"].size(-1))
    h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding, pref_padding), dim=-1).transpose(1, 2)
    scores = model.actor(h_actions).flatten(1)[0]
    value = model.critic(torch.cat((h_opes_pooled, h_mas_pooled, obs["pref"]), dim=-1)).reshape(())
    return scores[obs["flat"]], value


def _normalize_one(model, feat_o, feat_m, pt):
    """The per-instance normalisation of HGNNScheduler.get_normalized (evaluation branch), for one sample."""
    import torch
    mo, so = feat_o.mean(dim=-2, keepdim=True), feat_o.std(dim=-2, keepdim=True)
    mm, sm = feat_m.mean(dim=-2, keepdim=True), feat_m.std(dim=-2, keepdim=True)
    pt = pt.clone()
    idx = torch.nonzero(pt)
    vals = pt[idx[:, 0], idx[:, 1]]
    pt[idx[:, 0], idx[:, 1]] = model.feature_normalize(vals)
    return (feat_o - mo) / (so + 1e-5), (feat_m - mm) / (sm + 1e-5), pt


def shape_key(obs):
    return tuple(obs["pt"].shape) + tuple(obs["next_idx"].shape)


def forward_batch(model, obs_list):
    """Same computation as forward() for a list of observations with equal shapes, in one pass."""
    import torch
    norm = [_normalize_one(model, o["feat_o"][0], o["feat_m"][0], o["pt"][0]) for o in obs_list]
    feats = (torch.stack([x[0] for x in norm]), torch.stack([x[1] for x in norm]), torch.stack([x[2] for x in norm]))
    adj = torch.cat([o["adj"] for o in obs_list]); pre = torch.cat([o["pre"] for o in obs_list]); sub = pre.transpose(1, 2)
    B = len(obs_list); bidx = torch.arange(B)
    for i in range(len(model.num_heads)):
        h_mas = model.get_machines[i](adj, bidx, feats)
        feats = (feats[0], h_mas, feats[2])
        h_opes = model.get_operations[i](adj, pre, sub, bidx, feats)
        feats = (h_opes, feats[1], feats[2])
    h_mas_pooled = h_mas.mean(dim=-2)
    h_opes_pooled = h_opes.mean(dim=-2)
    next_idx = torch.cat([o["next_idx"] for o in obs_list])
    h_jobs = h_opes.gather(1, next_idx[..., :, None].expand(-1, -1, h_opes.size(-1)))
    m = obs_list[0]["pt"].size(-1)
    h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, m, -1)
    h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
    pref = torch.cat([o["pref"] for o in obs_list])
    h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding),
                           h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding),
                           pref[:, None, None, :].expand(*h_jobs_padding.shape[:3], pref.size(-1))), dim=-1).transpose(1, 2)
    scores = model.actor(h_actions).flatten(1)
    values = model.critic(torch.cat((h_opes_pooled, h_mas_pooled, pref), dim=-1)).reshape(B)
    return [scores[b][o["flat"]] for b, o in enumerate(obs_list)], values


class Adapter:
    reward = "estimate"  # dense estimated-objective reward, as the original code (author 2026-11-11)
    build = staticmethod(build)
    forward = staticmethod(forward)
    forward_batch = staticmethod(forward_batch)
    shape_key = staticmethod(shape_key)


def chooser(model):
    """Greedy policy for evaluation (blind_arms.policy_decode)."""
    import torch
    from drl_v3_common import current_time

    def choose(known, state, operations, transports, preference):
        obs, cands = build(known, state, current_time(known, state), preference, operations)
        with torch.no_grad():
            logits, _ = forward(model, obs)
        return cands[int(torch.argmax(logits).item())]
    return choose

