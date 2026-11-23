"""v3 (全部重跑规则_冻结v3.md section 2): shared harness for the published DRL baselines (DANIEL, DRL-S, HGT).

What is shared by every baseline (the adaptations listed in the manuscript):
- the instance: the two-leg AGV model; events drawn per seed (event_blind.randomize_events), as in the test sets;
- the decision: the method picks a ready operation and an eligible machine; the AGV and the route are completed by the shared
  transport rule (as Xiao et al., ASOC 2026, add a nearest-vehicle rule to DRL-S and DANIEL);
- the clock: strict, the policy sees only the events revealed so far (blind_arms.policy_decode);
- the objective: one model covers the six preference vectors; the preference vector enters the policy input and the reward
  is the decrease of the weighted objective V (Chen ESWA 2026; Liu AEI 2027);
- model selection: mean V on validation seeds 640-649 (random events, strict clock).
Everything else (state features, network, PPO settings) follows each method's original code, in its own adapter.

An adapter provides:
  build(known, state, t, preference, operations) -> (obs, candidates)   candidates: list of (job, operation, machine)
  forward(model, obs) -> (logits over candidates, value)    torch tensors, logits 1-D
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

H = Path(__file__).resolve().parents[1]
for p in (H / "src", H / "baselines", H.parent / "twoleg"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

NAMES = ("makespan", "tardiness", "energy", "route_pressure", "event_exposure_pressure")
TRAIN_SEEDS = range(600, 620)
VALID_SEEDS = range(640, 650)


def scales():
    from rerun_paths import SCALES
    from su_modfjsp_ha_pc.objective_contract import read_objective_scales
    return read_objective_scales(SCALES)


def preferences():
    from su_modfjsp_ha_pc.path_conflict_neural_dataset import _explicit_teacher_preferences
    return tuple(tuple(p) for p in _explicit_teacher_preferences(5, max_preferences=6))


def instances(seeds, core=None):
    """(raw, true) pairs with per-seed random events, L4/L5 x raw/realized."""
    from event_blind import randomize_events
    from su_modfjsp_ha_pc.generator import generate_instance
    from su_modfjsp_ha_pc.structural import realize_structural_events
    out = []
    for seed in seeds:
        for level in ("L4", "L5"):
            if core:
                from su_modfjsp_ha_pc.public_fjsp import load_and_extend_public_fjsp
                base = load_and_extend_public_fjsp(Path(core), level=level, seed=seed, n_agvs=3)
            else:
                base = generate_instance(level, "small", seed)
            raw = randomize_events(base, seed, level)
            for mode in ("raw", "realized"):
                true = realize_structural_events(raw).instance if mode == "realized" else raw
                out.append(dict(seed=seed, level=level, mode=mode, raw=raw, true=true))
    return out


def partial_value(instance, operations, transports, preference, sc):
    from su_modfjsp_ha_pc.objective_contract import evaluate_objective_contract
    from su_modfjsp_ha_pc.schedule import ScheduleResult
    if not operations:
        return 0.0
    raw = evaluate_objective_contract(instance, ScheduleResult(algorithm="drl-v3", path_aware=True, operations=tuple(operations),
                                                               transports=tuple(transports), event_aware=True)).raw
    return sum(w * raw[n] / sc.values[n] for w, n in zip(preference, NAMES))


def estimated_value(instance, state, operations, transports, preference, sc):
    """Dense reward of the original DRL-S, DANIEL and HGT codes, extended to the weighted objective (author 2026-11-11):
    unscheduled operations are completed by their lower bounds, as the originals estimate completion times. Each job ends no
    earlier than its ready time plus the minimum processing times of its remaining operations; the makespan and the
    tardiness use these estimated job completions; the energy adds the minimum processing energy of the remaining
    operations; route and event-exposure pressure count the committed part only. Equal to V once every operation is
    committed."""
    from su_modfjsp_ha_pc.objective_contract import evaluate_objective_contract
    from su_modfjsp_ha_pc.schedule import ScheduleResult
    if operations:
        raw = dict(evaluate_objective_contract(instance, ScheduleResult(algorithm="drl-v3", path_aware=True, operations=tuple(operations),
                                                                    transports=tuple(transports), event_aware=True)).raw)
    else:
        raw = {n: 0.0 for n in NAMES}
    end = {}
    for o in operations:
        end[o.job_id] = max(end.get(o.job_id, 0), o.end)
    mk, tard, extra_e = raw["makespan"], 0.0, 0.0
    for job in instance.jobs:
        k = state.next_op_index[job.job_id]
        rest = job.operations[k:]
        if rest:
            c = state.job_ready[job.job_id] + sum(min(op.processing_time.values()) for op in rest)
            extra_e += sum(min(op.processing_energy.values()) for op in rest)
        else:
            c = end.get(job.job_id, 0)
        mk = max(mk, c)
        tard += max(0, c - job.due_date)
    raw["makespan"], raw["tardiness"], raw["energy"] = mk, tard, raw["energy"] + extra_e
    return sum(w * raw[n] / sc.values[n] for w, n in zip(preference, NAMES))


def current_time(known, state):
    ready = [state.job_ready[j.job_id] for j in known.jobs if state.next_op_index[j.job_id] < len(j.operations)]
    return min(ready) if ready else 0


def run_episode(adapter, model, case, preference, sc, sample):
    """One strict-clock episode. Returns (records, final V). records: obs, action, logp, value, reward."""
    import torch
    from blind_arms import policy_decode
    records = []
    prev = [0.0]

    dense = getattr(adapter, "reward", "partial") == "estimate"

    def choose(known, state, operations, transports, pref):
        v = estimated_value(case["true"], state, operations, transports, preference, sc) if dense else partial_value(case["true"], operations, transports, preference, sc)
        if records:
            records[-1]["reward"] = prev[0] - v
        prev[0] = v
        obs, cands = adapter.build(known, state, current_time(known, state), preference, operations)
        if sample and getattr(adapter, "rollout_train_mode", False):
            model.train()
        else:
            model.eval()
        with torch.no_grad():
            logits, value = adapter.forward(model, obs)
        if sample:
            dist = torch.distributions.Categorical(logits=logits)
            a = int(dist.sample().item())
            logp = float(dist.log_prob(torch.tensor(a)).item())
        else:
            a, logp = int(torch.argmax(logits).item()), 0.0
        records.append(dict(obs=obs, action=a, logp=logp, value=float(value), reward=0.0))
        return cands[a]

    schedule, _ = policy_decode(true=case["true"], raw=case["raw"], preference=preference, choose=choose, strict=True)
    final = partial_value(case["true"], schedule.operations, schedule.transports, preference, sc)
    records[-1]["reward"] = prev[0] - final
    return records, final, schedule


def ppo_update(adapter, model, optimizer, records, *, gamma, lam, clip, epochs, minibatch, vf_coef, ent_coef, rng, max_grad_norm=None):
    """Clipped PPO with GAE over complete episodes (records grouped by episode with key 'ep')."""
    import torch
    by_ep = {}
    for r in records:
        by_ep.setdefault(r["ep"], []).append(r)
    for steps in by_ep.values():
        gae, nxt = 0.0, 0.0
        for s in reversed(steps):
            delta = s["reward"] + gamma * nxt - s["value"]
            gae = delta + gamma * lam * gae
            s["adv"], s["ret"] = gae, gae + s["value"]
            nxt = s["value"]
    adv = np.array([r["adv"] for r in records])
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for i, r in enumerate(records):
        r["nadv"] = float(adv[i])
    model.train()
    for _ in range(epochs):
        order = list(range(len(records)))
        rng.shuffle(order)
        for start in range(0, len(order), minibatch):
            losses = []
            idx = order[start:start + minibatch]
            if hasattr(adapter, "forward_batch"):  # one pass per group of equal-shaped observations
                groups = {}
                for i in idx:
                    groups.setdefault(adapter.shape_key(records[i]["obs"]), []).append(i)
                outs = []
                for g in groups.values():
                    lg, vs = adapter.forward_batch(model, [records[i]["obs"] for i in g])
                    outs += [(i, lg[k], vs[k]) for k, i in enumerate(g)]
            else:
                outs = [(i, *adapter.forward(model, records[i]["obs"])) for i in idx]
            for i, logits, value in outs:
                r = records[i]
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(torch.tensor(r["action"]))
                ratio = torch.exp(logp - r["logp"])
                pg = -torch.min(ratio * r["nadv"], ratio.clamp(1 - clip, 1 + clip) * r["nadv"])
                vf = (value - r["ret"]) ** 2
                losses.append(pg + vf_coef * vf - ent_coef * dist.entropy())
            optimizer.zero_grad()
            torch.stack([x.reshape(()) for x in losses]).mean().backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
    model.eval()


def evaluate(adapter, model, cases, prefs, sc):
    vals = []
    for case in cases:
        for pref in prefs:
            _, v, _ = run_episode(adapter, model, case, pref, sc, sample=False)
            vals.append(v)
    return float(np.mean(vals))


def train(adapter, make_model, make_optimizer, *, name, out, iterations, episodes_per_iter, ppo, seed=0, valid_every=10,
          log_every=1):
    """Generic loop: each iteration samples `episodes_per_iter` (instance, preference) pairs from the training set,
    runs them with the sampling policy, and applies one PPO update. Checkpoints are scored on validation seeds 640-649."""
    import torch
    torch.set_num_threads(1)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    sc, prefs = scales(), preferences()
    train_cases = instances(TRAIN_SEEDS)
    valid_cases = instances(VALID_SEEDS)
    model = make_model()
    optimizer = make_optimizer(model)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{name}_train_log.jsonl"
    best = math.inf
    t0 = perf_counter()
    for it in range(iterations):
        records, finals = [], []
        for ep in range(episodes_per_iter):
            case, pref = rng.choice(train_cases), rng.choice(prefs)
            recs, final, _ = run_episode(adapter, model, case, pref, sc, sample=True)
            for r in recs:
                r["ep"] = ep
            records += recs
            finals.append(final)
        ppo_update(adapter, model, optimizer, records, rng=rng, **ppo)
        row = dict(it=it, train_V=float(np.mean(finals)), wall_s=round(perf_counter() - t0, 1))
        if (it + 1) % valid_every == 0 or it == iterations - 1:
            row["valid_V"] = evaluate(adapter, model, valid_cases, prefs, sc)
            torch.save(model.state_dict(), out / f"{name}_it{it + 1:04d}.pt")
            if row["valid_V"] < best:
                best = row["valid_V"]
                torch.save(model.state_dict(), out / f"{name}_selected.pt")
                row["selected"] = True
        with log.open("a") as h:
            h.write(json.dumps(row) + "\n")
        if it % log_every == 0:
            print(name, row, flush=True)
    return best


# ---- parallel episode collection (v3 6b: implementation only; same episodes, same update) ----
_W = {}


def _worker_init(adapter_module):
    import importlib
    import torch
    torch.set_num_threads(1)
    mod = importlib.import_module(adapter_module)
    _W["mod"] = mod
    _W["model"] = mod.make_model()
    _W["sc"] = scales()
    _W["prefs"] = preferences()
    _W["cases"] = {}


def _worker_episode(task):
    import torch
    seeds_key, case_index, pref_index, state_dict, torch_seed, sample = task
    if seeds_key not in _W["cases"]:
        _W["cases"][seeds_key] = instances(TRAIN_SEEDS if seeds_key == "train" else VALID_SEEDS)
    model = _W["model"]
    model.load_state_dict(state_dict)
    model.eval()
    torch.manual_seed(torch_seed)
    recs, final, _ = run_episode(_W["mod"].Adapter, model, _W["cases"][seeds_key][case_index], _W["prefs"][pref_index], _W["sc"], sample)
    return recs, final


def train_parallel(adapter_module, *, name, out, iterations, episodes_per_iter, ppo, workers, seed=0, valid_every=50, lr_schedule=None):
    """train() with the episodes of each iteration (and the validation runs) spread over worker processes. The episode
    draw (case, preference, torch seed) is made in the main process from one RNG, so the run is reproducible for a fixed
    worker count."""
    import importlib
    import multiprocessing as mp
    import torch
    torch.set_num_threads(1)
    mod = importlib.import_module(adapter_module)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    n_train = len(instances(TRAIN_SEEDS))
    n_valid = len(instances(VALID_SEEDS))
    n_pref = len(preferences())
    model = mod.make_model()
    optimizer = mod.make_optimizer(model)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{name}_train_log.jsonl"
    best, t0, first = math.inf, perf_counter(), 0
    ck = out / f"{name}_last.pt"
    if ck.exists():  # resume
        st = torch.load(ck)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["opt"]); first = st["it"] + 1
        rng.setstate(st["rng"]); best = st["best"]; t0 -= st["wall"]
    with mp.get_context("spawn").Pool(workers, initializer=_worker_init, initargs=(adapter_module,)) as pool:
        for it in range(first, iterations):
            if lr_schedule:
                for g in optimizer.param_groups:
                    g["lr"] = lr_schedule(it)
            sd = {k: v.clone() for k, v in model.state_dict().items()}
            tasks = [("train", rng.randrange(n_train), rng.randrange(n_pref), sd, rng.randrange(2 ** 31), True) for _ in range(episodes_per_iter)]
            records, finals = [], []
            for ep, (recs, final) in enumerate(pool.imap(_worker_episode, tasks)):
                for r in recs:
                    r["ep"] = ep
                records += recs
                finals.append(final)
            ppo_update(mod.Adapter, model, optimizer, records, rng=rng, **ppo)
            row = dict(it=it, train_V=float(np.mean(finals)), wall_s=round(perf_counter() - t0, 1))
            if (it + 1) % valid_every == 0 or it == iterations - 1:
                sd = {k: v.clone() for k, v in model.state_dict().items()}
                vt = [("valid", c, p, sd, 0, False) for c in range(n_valid) for p in range(n_pref)]
                row["valid_V"] = float(np.mean([f for _, f in pool.imap(_worker_episode, vt, chunksize=4)]))
                torch.save(model.state_dict(), out / f"{name}_it{it + 1:04d}.pt")
                if row["valid_V"] < best:
                    best = row["valid_V"]
                    torch.save(model.state_dict(), out / f"{name}_selected.pt")
                    row["selected"] = True
            torch.save(dict(model=model.state_dict(), opt=optimizer.state_dict(), it=it, rng=rng.getstate(), best=best,
                            wall=perf_counter() - t0), ck)
            with log.open("a") as h:
                h.write(json.dumps(row) + "\n")
            print(name, row, flush=True)
    return best
