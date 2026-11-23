"""PPO training for the native-DRL baseline (Block N2 of PREREG_BASELINE_EXPANSION_20260727).

Frozen hyperparameters per preregistration: clip 0.2, GAE lambda 0.95, lr 3e-4,
no hyperparameter search. gamma uses the literature default 0.99. Training
distribution = seeds 100-119 x {L4, L5} x {raw, realized} x 6 preferences,
size "small". The only permitted selection is 3 random inits -> pick 1 by
training return (run with --init-seed 0/1/2), frozen after the seed-224 smoke.

Reward is sparse: terminal -V(sigma, w) under the frozen objective-scale
contract; the preference vector is part of the state (see ppo_env).

Smoke mode (--smoke): seed 100 only, L4 raw, 2 updates, tiny batch.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppo_env import (  # noqa: E402
    CANDIDATE_DIM,
    CONTEXT_DIM,
    FIREWALL_SEEDS,
    PPOScheduleEnv,
    load_frozen_objective_scales,
)
from su_modfjsp_ha_pc.generator import generate_instance  # noqa: E402
from su_modfjsp_ha_pc.path_conflict_neural_dataset import (  # noqa: E402
    _explicit_preference_id,
    _explicit_teacher_preferences,
)
from su_modfjsp_ha_pc.structural import realize_structural_events  # noqa: E402

TRAIN_SEEDS = tuple(range(600, 620))  # rerun v1
TRAIN_LEVELS = ("L4", "L5")
TRAIN_MODES = ("raw", "realized")
N_PREFERENCES = 6
SIZE = "small"


class CandidatePolicy(nn.Module):
    """Shared MLP scoring each (o,m) candidate + a pooled value head."""

    def __init__(self, cand_dim: int = CANDIDATE_DIM, ctx_dim: int = CONTEXT_DIM, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cand_dim + ctx_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.score_head = nn.Linear(hidden, 1)
        self.value_head = nn.Sequential(
            nn.Linear(ctx_dim + 2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, candidates: torch.Tensor, context: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """candidates (B,N,F), context (B,C), mask (B,N) bool -> logits (B,N), value (B,)."""
        batch, n_cand, _ = candidates.shape
        ctx_expanded = context.unsqueeze(1).expand(batch, n_cand, context.shape[-1])
        emb = self.encoder(torch.cat([candidates, ctx_expanded], dim=-1))
        logits = self.score_head(emb).squeeze(-1)
        logits = logits.masked_fill(~mask, float("-inf"))
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        mean_pool = (emb * mask_f).sum(dim=1) / denom
        max_pool = emb.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        value = self.value_head(torch.cat([context, mean_pool, max_pool], dim=-1)).squeeze(-1)
        return logits, value


class InstanceCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[int, str, str], object] = {}

    def get(self, seed: int, level: str, mode: str) -> object:
        key = (seed, level, mode)
        if key not in self._cache:
            raw = generate_instance(level, SIZE, seed)
            self._cache[key] = realize_structural_events(raw).instance if mode == "realized" else raw
        return self._cache[key]


def sample_case(rng: random.Random, seeds, levels, modes, cache: InstanceCache):
    seed = rng.choice(seeds)
    level = rng.choice(levels)
    mode = rng.choice(modes)
    pref_index = rng.randrange(N_PREFERENCES)
    instance = cache.get(seed, level, mode)
    preferences = _explicit_teacher_preferences(len(instance.objectives), max_preferences=N_PREFERENCES)
    preference = preferences[pref_index]
    preference_id = _explicit_preference_id(pref_index, preference)
    return instance, preference, level, seed, mode, preference_id


@torch.no_grad()
def run_episode(env: PPOScheduleEnv, policy: CandidatePolicy, device, case, greedy: bool = False):
    instance, preference, level, seed, mode, preference_id = case
    obs = env.reset(
        instance=instance, preference=preference, level=level, seed=seed, mode=mode,
        preference_id=preference_id,
    )
    steps = []
    total_reward = 0.0
    while True:
        cand = torch.as_tensor(obs["candidates"], dtype=torch.float32, device=device).unsqueeze(0)
        ctx = torch.as_tensor(obs["context"], dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.ones(1, cand.shape[1], dtype=torch.bool, device=device)
        logits, value = policy(cand, ctx, mask)
        dist = torch.distributions.Categorical(logits=logits[0])
        action = torch.argmax(logits[0]) if greedy else dist.sample()
        logp = dist.log_prob(action)
        next_obs, reward, done, _info = env.step(int(action.item()))
        total_reward += reward
        steps.append(
            {
                "candidates": obs["candidates"],
                "context": obs["context"],
                "action": int(action.item()),
                "logp": float(logp.item()),
                "value": float(value.item()),
                "reward": float(reward),
            }
        )
        if done:
            break
        obs = next_obs
    return steps, total_reward


def compute_gae(steps, gamma: float, lam: float) -> None:
    advantage = 0.0
    next_value = 0.0
    for step in reversed(steps):
        delta = step["reward"] + gamma * next_value - step["value"]
        advantage = delta + gamma * lam * advantage
        step["advantage"] = advantage
        step["return"] = advantage + step["value"]
        next_value = step["value"]


def pad_batch(steps, device):
    max_cand = max(step["candidates"].shape[0] for step in steps)
    batch = len(steps)
    cand = np.zeros((batch, max_cand, CANDIDATE_DIM), dtype=np.float32)
    mask = np.zeros((batch, max_cand), dtype=bool)
    ctx = np.zeros((batch, CONTEXT_DIM), dtype=np.float32)
    for i, step in enumerate(steps):
        n = step["candidates"].shape[0]
        cand[i, :n] = step["candidates"]
        mask[i, :n] = True
        ctx[i] = step["context"]
    tensors = {
        "candidates": torch.as_tensor(cand, device=device),
        "mask": torch.as_tensor(mask, device=device),
        "context": torch.as_tensor(ctx, device=device),
        "actions": torch.as_tensor([s["action"] for s in steps], dtype=torch.long, device=device),
        "old_logp": torch.as_tensor([s["logp"] for s in steps], dtype=torch.float32, device=device),
        "advantages": torch.as_tensor([s["advantage"] for s in steps], dtype=torch.float32, device=device),
        "returns": torch.as_tensor([s["return"] for s in steps], dtype=torch.float32, device=device),
    }
    return tensors


def ppo_update(policy, optimizer, steps, device, args):
    advantages = np.asarray([s["advantage"] for s in steps], dtype=np.float64)
    adv_mean, adv_std = float(advantages.mean()), float(advantages.std() + 1e-8)
    for step in steps:
        step["advantage"] = (step["advantage"] - adv_mean) / adv_std
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    n_minibatches = 0
    indices = np.arange(len(steps))
    for _epoch in range(args.epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), args.minibatch):
            chunk = [steps[i] for i in indices[start : start + args.minibatch]]
            batch = pad_batch(chunk, device)
            logits, values = policy(batch["candidates"], batch["context"], batch["mask"])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(batch["actions"])
            ratio = torch.exp(logp - batch["old_logp"])
            surr1 = ratio * batch["advantages"]
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * batch["advantages"]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (values - batch["returns"]).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite PPO loss")
            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(entropy.item())
            stats["approx_kl"] += float((batch["old_logp"] - logp).mean().item())
            n_minibatches += 1
    return {key: value / max(1, n_minibatches) for key, value in stats.items()}


def save_checkpoint(path: Path, policy, optimizer, update_index: int, args, py_rng: random.Random):
    payload = {
        "protocol": "pgnd-v5-native-ppo-baseline-v1",
        "model_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "update_index": update_index,
        "config": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
        "candidate_dim": CANDIDATE_DIM,
        "context_dim": CONTEXT_DIM,
        "hidden": args.hidden,
        "rng": {
            "python": py_rng.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, policy, optimizer, py_rng: random.Random) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    rng = payload.get("rng", {})
    if rng.get("python") is not None:
        py_rng.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.set_rng_state(torch.as_tensor(rng["torch"], dtype=torch.uint8))
    if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([torch.as_tensor(s, dtype=torch.uint8) for s in rng["torch_cuda"]])
    return int(payload["update_index"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)  # frozen (prereg)
    parser.add_argument("--clip", type=float, default=0.2)  # frozen (prereg)
    parser.add_argument("--gae-lambda", type=float, default=0.95)  # frozen (prereg)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--init-seed", type=int, default=0, help="random init id (0/1/2 per prereg)")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="seed 100 / L4 raw, 2 updates, tiny batch")
    args = parser.parse_args()

    if args.smoke:
        args.updates = 2
        args.episodes_per_update = 2
        args.epochs = 2
        args.minibatch = 64

    seeds = (390,) if args.smoke else TRAIN_SEEDS  # rerun v1 smoke seed
    levels = ("L4",) if args.smoke else TRAIN_LEVELS
    modes = ("raw",) if args.smoke else TRAIN_MODES
    if any(seed in FIREWALL_SEEDS for seed in seeds):
        raise RuntimeError("training seeds intersect the evaluation firewall")

    device = torch.device(args.device)
    torch.manual_seed(1000 + args.init_seed)
    np.random.seed(1000 + args.init_seed)
    py_rng = random.Random(1000 + args.init_seed)

    objective_scales = load_frozen_objective_scales()
    env = PPOScheduleEnv(objective_scales)
    cache = InstanceCache()
    policy = CandidatePolicy(hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    latest = args.out_dir / "checkpoint_latest.pt"
    log_path = args.out_dir / "train_log.jsonl"
    start_update = 0
    if args.resume and latest.exists():
        start_update = load_checkpoint(latest, policy, optimizer, py_rng)
        print(f"[ppo-train] resumed from update {start_update}", flush=True)

    for update in range(start_update, args.updates):
        t0 = perf_counter()
        all_steps = []
        episode_returns = []
        for _ in range(args.episodes_per_update):
            case = sample_case(py_rng, seeds, levels, modes, cache)
            steps, episode_return = run_episode(env, policy, device, case)
            compute_gae(steps, args.gamma, args.gae_lambda)
            all_steps.extend(steps)
            episode_returns.append(episode_return)
        stats = ppo_update(policy, optimizer, all_steps, device, args)
        wall_s = perf_counter() - t0
        record = {
            "update": update + 1,
            "mean_return": float(np.mean(episode_returns)),
            "mean_weighted_value": float(-np.mean(episode_returns)),
            "n_steps": len(all_steps),
            "wall_s": round(wall_s, 3),
            **{key: round(value, 6) for key, value in stats.items()},
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"[ppo-train] {json.dumps(record)}", flush=True)
        if (update + 1) % args.checkpoint_every == 0 or (update + 1) == args.updates:
            save_checkpoint(latest, policy, optimizer, update + 1, args, py_rng)
            save_checkpoint(
                args.out_dir / f"checkpoint_update{update + 1:05d}.pt",
                policy, optimizer, update + 1, args, py_rng,
            )
            print(f"[ppo-train] checkpoint @ update {update + 1}", flush=True)

    print("[ppo-train] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
