"""v3 (全部重跑规则_冻结v3.md section 2): Cheng et al. (COR 2025) composite-rule DQN, retrained on per-seed random events
with the strict clock (the policy sees only revealed events, blind_arms.policy_decode). Network, features, actions, reward
(decrease of the weighted objective, computed on the executed true schedule) and DQN settings are those of cheng_dqn.py;
one model per preference as in the original. Model selection on validation seeds 640-649 (random events, strict clock)."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import statistics as st
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cheng_dqn as B  # noqa: E402


def train_one(job):
    pref_index, episodes, out, run, lr = job
    import torch
    torch.set_num_threads(1)
    from blind_arms import policy_decode
    import drl_v3_common as C
    sc = C.scales()
    rng = random.Random(1000 + 100 * run + pref_index)
    torch.manual_seed(1000 + 100 * run + pref_index)
    cases = C.instances(C.TRAIN_SEEDS)
    preference = C.preferences()[pref_index]
    q, target = B.network(), B.network()
    target.load_state_dict(q.state_dict())
    optimizer = torch.optim.Adam(q.parameters(), lr=lr)
    buffer, steps, log = [], [0], []
    started = perf_counter()

    def learn():
        if len(buffer) < 32:
            return
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

    def push(item):
        buffer.append(item)
        if len(buffer) > 1000:
            buffer.pop(0)
        steps[0] += 1
        learn()
        if steps[0] % 100 == 0:
            target.load_state_dict(q.state_dict())

    for episode in range(episodes):
        epsilon = max(0.1, 1.0 - 0.9 * episode / max(1, episodes - 1))
        case = rng.choice(cases)
        prev = {}

        def choose(known, state, operations, transports, pref):
            s = B.features(known, state, operations, transports)
            v = C.partial_value(case["true"], operations, transports, preference, sc)
            if prev:
                push((prev["s"], prev["a"], prev["v"] - v, s, False))
            if rng.random() < epsilon:
                a = rng.randrange(len(B.ACTIONS))
            else:
                with torch.no_grad():
                    a = int(q(torch.from_numpy(s)).argmax().item())
            prev.update(s=s, a=a, v=v)
            return B.apply_rule(a, known, state)

        schedule, _ = policy_decode(true=case["true"], raw=case["raw"], preference=preference, choose=choose, strict=True)
        final = C.partial_value(case["true"], schedule.operations, schedule.transports, preference, sc)
        push((prev["s"], prev["a"], prev["v"] - final, prev["s"], True))
        log.append(dict(episode=episode, value=final, epsilon=epsilon, wall_s=perf_counter() - started))
    Path(out).mkdir(parents=True, exist_ok=True)
    torch.save(q.state_dict(), f"{out}/dqn_pref{pref_index:02d}.pt")
    with open(f"{out}/train_log_pref{pref_index:02d}.jsonl", "w") as handle:
        for row in log:
            handle.write(json.dumps(row) + "\n")
    return run, pref_index, perf_counter() - started, st.mean(r["value"] for r in log[-100:])


def valid_one(job):
    """Mean V over validation seeds 640-649 (random events, strict clock) of one preference model."""
    models, pref_index = job
    import torch
    torch.set_num_threads(1)
    from blind_arms import policy_decode
    import drl_v3_common as C
    sc, pref = C.scales(), C.preferences()[pref_index]
    net = B.network()
    net.load_state_dict(torch.load(f"{models}/dqn_pref{pref_index:02d}.pt"))
    net.eval()
    vals = []
    for case in C.instances(C.VALID_SEEDS):
        s, _ = policy_decode(true=case["true"], raw=case["raw"], preference=pref, choose=B.chooser(net), strict=True)
        vals.append(C.partial_value(case["true"], s.operations, s.transports, pref, sc))
    return models, pref_index, float(np.mean(vals))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("train", "valid"))
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--models", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        if a.command == "train":
            for r in pool.imap_unordered(train_one, [(i, a.episodes, f"{a.models}/run{r}", r, a.lr) for r in range(a.runs) for i in range(6)]):
                print("trained", r, flush=True)
        else:
            res = list(pool.imap(valid_one, [(str(p), i) for p in sorted(Path(a.models).glob("run*")) for i in range(6)]))
            json.dump(res, open(Path(a.models) / "valid.json", "w"), indent=1)
            print(res)
