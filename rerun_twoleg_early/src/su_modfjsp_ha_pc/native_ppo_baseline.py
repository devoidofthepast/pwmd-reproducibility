"""Block N native PPO baseline for PGND-V5.

This module implements the pre-registered Block N baseline from
``PREREG_BASELINE_EXPANSION_20260727.md``:

* action space: feasible ``(operation, machine)`` pairs, identical candidate set
  to the PGND framework;
* transport / AGV / route decisions are completed by the shared deterministic
  transport rule used by all methods;
* reward: sparse terminal ``-V(sigma, w)`` under the frozen objective-scale
  contract;
* state features: generic scheduling features only (no PGND proposer feature
  engineering, no per-candidate transport timing; AGV information enters only
  through context aggregates);
* PPO defaults: gamma 0.99, GAE 0.95, clip 0.2, lr 3e-4, vf 0.5, ent 0.01;
* training: seeds 100-119, L4/L5, raw/realized, balanced + five pure
  preferences, 3 random initializations, 3000 updates x 16 episodes;
* selection: init with best mean episode return over the last 100 updates;
* evaluation: seeds 220-229, L4/L5, raw/realized, six preferences (240 cases).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from su_modfjsp_ha_pc.generator import generate_instance
from su_modfjsp_ha_pc.hierarchical_transport_attention_policy import (
    _append_fixed_joint_preference_transport_action,
)
from su_modfjsp_ha_pc.objective_contract import (
    ObjectiveScales,
    evaluate_objective_contract,
    objective_breakdown,
    read_objective_scales,
)
from su_modfjsp_ha_pc.rules import _available_operations
from su_modfjsp_ha_pc.schedule import ScheduleResult
from su_modfjsp_ha_pc.structural import realize_structural_events

PROTOCOL = "pgnd-v5-native-ppo-baseline-v2"
CANDIDATE_FEATURE_NAMES = (
    "job_ready_delay",
    "due_slack",
    "remaining_ops_fraction",
    "op_index_fraction",
    "machine_available_delay",
    "processing_time_fraction",
    "processing_energy_fraction",
    "machine_energy_rate_fraction",
    "eligible_machine_count_fraction",
    "machine_busy_until_fraction",
    "job_remaining_processing_fraction",
    "is_first_operation",
)

CONTEXT_FEATURE_NAMES = (
    "pref_makespan",
    "pref_tardiness",
    "pref_energy",
    "pref_route_pressure",
    "pref_event_exposure",
    "decision_time_fraction",
    "progress_fraction",
    "log1p_job_count",
    "log1p_machine_count",
    "log1p_agv_count",
    "log1p_candidate_count",
    "min_machine_available_delay",
    "mean_machine_available_delay",
    "max_machine_available_delay",
    "min_agv_available_delay",
    "mean_agv_available_delay",
    "max_agv_available_delay",
    "log1p_event_count",
)


def six_preferences(n_objectives: int) -> tuple[tuple[float, ...], ...]:
    """Balanced plus pure preference vectors used by Block N."""
    if n_objectives <= 0:
        raise ValueError("n_objectives must be positive")
    balanced = tuple(1.0 / n_objectives for _ in range(n_objectives))
    preferences = [balanced]
    for index in range(n_objectives):
        vector = [0.0] * n_objectives
        vector[index] = 1.0
        preferences.append(tuple(vector))
    return tuple(preferences)


def instance_for_context(level: str, mode: str, seed: int, size: str = "small"):
    """Return the raw or structurally realized instance used by the SU protocol."""
    if mode not in {"raw", "realized"}:
        raise ValueError("mode must be raw or realized")
    raw = generate_instance(level, size, seed)
    if mode == "realized":
        return realize_structural_events(raw).instance
    return raw


@dataclass(frozen=True)
class NativePPOConfig:
    updates: int = 3000
    episodes_per_update: int = 16
    epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3.0e-4
    clip_ratio: float = 0.2
    gae_lambda: float = 0.95
    gamma: float = 0.99
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    maximum_gradient_norm: float = 0.5
    hidden_size: int = 128
    init_seed: int = 0
    checkpoint_every: int = 10

    def __post_init__(self) -> None:
        if self.updates <= 0 or self.episodes_per_update <= 0:
            raise ValueError("updates and episodes_per_update must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.clip_ratio <= 0.0 or self.learning_rate <= 0.0:
            raise ValueError("clip_ratio and learning_rate must be positive")


@dataclass(frozen=True)
class NativePPOTransition:
    candidate_x: torch.Tensor
    global_x: torch.Tensor
    action_index: int
    old_log_probability: float
    old_value: float
    reward: float
    done: bool


@dataclass(frozen=True)
class NativePPOEpisode:
    steps: tuple[NativePPOTransition, ...]
    weighted_value: float
    schedule: ScheduleResult | None = None


@dataclass(frozen=True)
class NativePPOTrainingContext:
    level: str
    mode: str
    seed: int
    preference: tuple[float, ...]


class NativePPOActorCritic(nn.Module):
    """Masked MLP actor-critic over the current operation-machine candidate set.

    The architecture is intentionally generic: candidate features are
    concatenated with the global context for each action, a shared MLP produces
    per-action logits, and a scalar critic pools the action embeddings.
    """

    def __init__(
        self,
        *,
        candidate_dim: int,
        context_dim: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        if candidate_dim <= 0 or context_dim <= 0 or hidden_size <= 0:
            raise ValueError("input dimensions must be positive")
        self.candidate_dim = candidate_dim
        self.context_dim = context_dim
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(candidate_dim + context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.score_head = nn.Linear(hidden_size, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size + context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        candidate_x: torch.Tensor,
        global_x: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_x.ndim != 3 or global_x.ndim != 2 or action_mask.ndim != 2:
            raise ValueError("unexpected tensor ranks")
        if candidate_x.shape[:2] != action_mask.shape:
            raise ValueError("candidate and mask shapes must agree")
        if candidate_x.shape[0] != global_x.shape[0]:
            raise ValueError("candidate and global batch sizes must agree")
        batch, actions, _ = candidate_x.shape
        context = global_x.unsqueeze(1).expand(batch, actions, -1)
        hidden = self.encoder(torch.cat((candidate_x, context), dim=-1))
        logits = self.score_head(hidden).squeeze(-1)
        logits = logits.masked_fill(~action_mask, -torch.inf)
        weights = action_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        value = self.value_head(torch.cat((pooled, global_x), dim=-1)).squeeze(-1)
        return logits, value

    def distribution(
        self,
        candidate_x: torch.Tensor,
        global_x: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        logits, value = self(candidate_x, global_x, action_mask)
        return Categorical(logits=logits), value


def build_train_contexts(
    *,
    levels: Sequence[str] = ("L4", "L5"),
    modes: Sequence[str] = ("raw", "realized"),
    seeds: Sequence[int],
    preferences: Sequence[tuple[float, ...]],
) -> tuple[NativePPOTrainingContext, ...]:
    contexts = []
    for level in levels:
        for mode in modes:
            for seed in seeds:
                for preference in preferences:
                    contexts.append(
                        NativePPOTrainingContext(
                            level=level,
                            mode=mode,
                            seed=seed,
                            preference=preference,
                        )
                    )
    return tuple(contexts)


def _time_scale(instance) -> float:
    return max(
        1.0,
        max((job.due_date for job in instance.jobs), default=1.0),
        max(
            (op.processing_time[machine_id] for job in instance.jobs for op in job.operations for machine_id in op.eligible_machines),
            default=1.0,
        ),
    )


def _energy_scale(instance) -> float:
    values = [
        op.processing_energy[machine_id]
        for job in instance.jobs
        for op in job.operations
        for machine_id in op.eligible_machines
    ]
    return max(1.0, *values)


def _max_machine_energy_rate(instance) -> float:
    return max(1.0, *(machine.energy_rate for machine in instance.machines))


def encode_observation(
    *,
    instance,
    preference: tuple[float, ...],
    machine_available: dict[str, int],
    job_ready: dict[str, int],
    next_op_index: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
    scheduled_operation_count: int,
    decision_time: int,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[object, object, object], ...]]:
    """Encode the current state into candidate/context arrays.

    Returns ``(candidate_x, global_x, action_keys)`` where each action key is
    ``(job, operation, machine)``.
    """
    available = _available_operations(instance.jobs, next_op_index)
    machine_by_id = {machine.machine_id: machine for machine in instance.machines}
    actions: list[tuple[object, object, object]] = []
    for job, operation in available:
        for machine_id in operation.eligible_machines:
            actions.append((job, operation, machine_by_id[machine_id]))
    if not actions:
        raise ValueError("no feasible operation-machine actions")

    now = float(decision_time)
    time_scale = _time_scale(instance)
    energy_scale = _energy_scale(instance)
    max_energy_rate = _max_machine_energy_rate(instance)
    max_ops = max(len(job.operations) for job in instance.jobs)
    candidate_rows = []
    for job, operation, machine in actions:
        op_index = operation.op_index
        remaining_ops = len(job.operations) - op_index - 1
        job_remaining_processing = sum(
            min(candidate.processing_time.values())
            for candidate in job.operations[op_index:]
        )
        candidate_rows.append(
            (
                (job_ready[job.job_id] - now) / time_scale,
                (job.due_date - now) / time_scale,
                remaining_ops / max_ops,
                op_index / max_ops,
                (machine_available[machine.machine_id] - now) / time_scale,
                operation.processing_time[machine.machine_id] / time_scale,
                operation.processing_energy[machine.machine_id] / energy_scale,
                machine.energy_rate / max_energy_rate,
                len(operation.eligible_machines) / len(instance.machines),
                machine_available[machine.machine_id] / time_scale,
                job_remaining_processing / (time_scale * max_ops),
                1.0 if op_index == 0 else 0.0,
            )
        )
    candidate_x = np.asarray(candidate_rows, dtype=np.float32)

    machine_delays = [float(machine_available[m.machine_id] - now) / time_scale for m in instance.machines]
    agv_delays = [float(agv_available[a.agv_id] - now) / time_scale for a in instance.agvs]
    event_count = sum(
        1 for event in instance.events if float(getattr(event, "event_time", math.inf)) >= now
    )
    global_x = np.asarray(
        (
            *preference,
            now / time_scale,
            scheduled_operation_count / max(1, sum(len(job.operations) for job in instance.jobs)),
            math.log1p(len(instance.jobs)),
            math.log1p(len(instance.machines)),
            math.log1p(len(instance.agvs)),
            math.log1p(len(actions)),
            min(machine_delays, default=0.0),
            sum(machine_delays) / max(1, len(machine_delays)),
            max(machine_delays, default=0.0),
            min(agv_delays, default=0.0),
            sum(agv_delays) / max(1, len(agv_delays)),
            max(agv_delays, default=0.0),
            math.log1p(event_count),
        ),
        dtype=np.float32,
    )
    if candidate_x.shape[1] != len(CANDIDATE_FEATURE_NAMES):
        raise RuntimeError("candidate feature width changed unexpectedly")
    if global_x.shape != (len(CONTEXT_FEATURE_NAMES),):
        raise RuntimeError("context feature width changed unexpectedly")
    if not np.isfinite(candidate_x).all() or not np.isfinite(global_x).all():
        raise RuntimeError("native PPO features must be finite")
    return candidate_x, global_x, tuple(actions)


def collect_episode(
    *,
    instance,
    preference: tuple[float, ...],
    model: nn.Module,
    scales: ObjectiveScales,
    generator: random.Random,
    deterministic: bool,
    device: torch.device,
) -> NativePPOEpisode:
    """Run one episode with a native PPO policy and shared transport completion."""
    machine_available = {machine.machine_id: 0 for machine in instance.machines}
    job_ready = {job.job_id: job.release_time for job in instance.jobs}
    next_op_index = {job.job_id: 0 for job in instance.jobs}
    agv_available = {agv.agv_id: 0 for agv in instance.agvs}
    agv_node = {agv.agv_id: agv.start_node_id for agv in instance.agvs}
    edge_reservations = {edge.edge_id: [] for edge in instance.path_edges}
    scheduled_operations = []
    transport_moves = []
    total_operations = sum(len(job.operations) for job in instance.jobs)

    transitions: list[NativePPOTransition] = []
    was_training = model.training
    model.eval()
    step_index = 0
    torch_generator = None
    if not deterministic:
        torch_generator = torch.Generator(device="cpu")
        torch_generator.manual_seed(generator.randrange(1 << 30))
    try:
        while len(scheduled_operations) < total_operations:
            decision_time = max(
                [0]
                + [operation.end for operation in scheduled_operations]
                + [move.end for move in transport_moves]
            )
            candidate_x, global_x, actions = encode_observation(
                instance=instance,
                preference=preference,
                machine_available=machine_available,
                job_ready=job_ready,
                next_op_index=next_op_index,
                agv_available=agv_available,
                agv_node=agv_node,
                edge_reservations=edge_reservations,
                scheduled_operation_count=len(scheduled_operations),
                decision_time=decision_time,
            )
            candidate_t = torch.from_numpy(candidate_x).unsqueeze(0).to(device)
            global_t = torch.from_numpy(global_x).unsqueeze(0).to(device)
            action_mask = torch.ones((1, len(actions)), dtype=torch.bool, device=device)
            with torch.no_grad():
                distribution, value = model.distribution(candidate_t, global_t, action_mask)
                if deterministic:
                    selected = distribution.logits.argmax(dim=1)
                else:
                    probs = distribution.probs.cpu()
                    # Avoid NaN with the categorical sampler by renormalizing after mask.
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    sampled = torch.multinomial(probs, num_samples=1, generator=torch_generator).squeeze(1)
                    selected = sampled.to(device)
                log_probability = distribution.log_prob(selected)
            action_index = int(selected.item())
            job, operation, machine = actions[action_index]
            _append_fixed_joint_preference_transport_action(
                instance=instance,
                preference=preference,
                job=job,
                operation=operation,
                machine=machine,
                machine_available=machine_available,
                job_ready=job_ready,
                next_op_index=next_op_index,
                agv_available=agv_available,
                agv_node=agv_node,
                edge_reservations=edge_reservations,
                scheduled_operations=scheduled_operations,
                transport_moves=transport_moves,
            )
            done = len(scheduled_operations) >= total_operations
            transitions.append(
                NativePPOTransition(
                    candidate_x=torch.from_numpy(candidate_x).to(device),
                    global_x=torch.from_numpy(global_x).to(device),
                    action_index=action_index,
                    old_log_probability=float(log_probability.item()),
                    old_value=float(value.item()),
                    reward=0.0,
                    done=done,
                )
            )
            step_index += 1
            if step_index > total_operations * 20:
                raise RuntimeError("native PPO episode exceeded its decision limit")
    finally:
        model.train(was_training)

    schedule = ScheduleResult(
        algorithm="native-ppo-block-n-v2",
        path_aware=bool(instance.path_edges),
        operations=tuple(scheduled_operations),
        transports=tuple(transport_moves),
        event_aware=bool(instance.events),
    )
    evaluation = evaluate_objective_contract(instance, schedule)
    weighted_value = objective_breakdown(evaluation.raw, preference, scales).weighted_value
    terminal = transitions[-1]
    transitions[-1] = NativePPOTransition(
        candidate_x=terminal.candidate_x,
        global_x=terminal.global_x,
        action_index=terminal.action_index,
        old_log_probability=terminal.old_log_probability,
        old_value=terminal.old_value,
        reward=-weighted_value,
        done=True,
    )
    return NativePPOEpisode(
        steps=tuple(transitions),
        weighted_value=weighted_value,
        schedule=schedule,
    )


def _pad_episodes(
    episodes: Sequence[NativePPOEpisode],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    steps = [step for episode in episodes for step in episode.steps]
    if not steps:
        raise ValueError("cannot pad an empty training batch")
    max_actions = max(step.candidate_x.shape[0] for step in steps)
    candidate_width = steps[0].candidate_x.shape[1]
    global_width = steps[0].global_x.shape[0]
    count = len(steps)
    candidate_x = torch.zeros((count, max_actions, candidate_width), dtype=torch.float32)
    global_x = torch.zeros((count, global_width), dtype=torch.float32)
    action_mask = torch.zeros((count, max_actions), dtype=torch.bool)
    action_index = torch.zeros(count, dtype=torch.long)
    old_log_probability = torch.zeros(count, dtype=torch.float32)
    rewards = torch.zeros(count, dtype=torch.float32)
    dones = torch.zeros(count, dtype=torch.bool)
    for row, step in enumerate(steps):
        action_count = step.candidate_x.shape[0]
        candidate_x[row, :action_count] = step.candidate_x
        global_x[row] = step.global_x
        action_mask[row, :action_count] = True
        action_index[row] = step.action_index
        old_log_probability[row] = step.old_log_probability
        rewards[row] = step.reward
        dones[row] = step.done
    return (
        candidate_x.to(device),
        global_x.to(device),
        action_mask.to(device),
        action_index.to(device),
        old_log_probability.to(device),
        rewards.to(device),
        dones.to(device),
    )


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.ndim != 1 or values.shape != rewards.shape or dones.shape != rewards.shape:
        raise ValueError("reward/value/done tensors must have identical one-dimensional shapes")
    if rewards.numel() == 0:
        raise ValueError("cannot compute GAE for an empty batch")
    advantages = torch.zeros_like(rewards)
    next_value = torch.zeros((), dtype=values.dtype, device=values.device)
    next_advantage = torch.zeros((), dtype=values.dtype, device=values.device)
    for index in range(rewards.numel() - 1, -1, -1):
        continuation = (~dones[index]).to(values.dtype)
        delta = rewards[index] + gamma * next_value * continuation - values[index]
        next_advantage = delta + gamma * gae_lambda * continuation * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    return advantages + values, advantages


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    episodes: Sequence[NativePPOEpisode],
    *,
    config: NativePPOConfig,
    generator: torch.Generator,
) -> dict[str, float]:
    """Apply one PPO update over a mini-batch of trajectories."""
    (
        candidate_x,
        global_x,
        action_mask,
        action_index,
        old_log_probability,
        rewards,
        dones,
    ) = _pad_episodes(episodes, device=next(model.parameters()).device)
    with torch.no_grad():
        _, values = model.distribution(candidate_x, global_x, action_mask)
    returns, advantages = compute_gae(
        rewards,
        values,
        dones,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

    model.train()
    metrics = []
    for _epoch in range(config.epochs):
        permutation = torch.randperm(action_index.numel(), generator=generator)
        for start in range(0, action_index.numel(), config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size].to(action_index.device)
            distribution, value = model.distribution(candidate_x[indices], global_x[indices], action_mask[indices])
            new_log_probability = distribution.log_prob(action_index[indices])
            ratio = (new_log_probability - old_log_probability[indices]).exp()
            advantage = advantages[indices]
            unclipped = ratio * advantage
            clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantage
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            critic_loss = torch.mean((value - returns[indices]) ** 2)
            entropy = distribution.entropy().mean()
            loss = (
                actor_loss
                + config.value_coefficient * critic_loss
                - config.entropy_coefficient * entropy
            )
            if not torch.isfinite(loss):
                raise RuntimeError("native PPO produced a nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), config.maximum_gradient_norm)
            optimizer.step()
            metrics.append(
                (
                    float(loss.item()),
                    float(actor_loss.item()),
                    float(critic_loss.item()),
                    float(entropy.item()),
                    float(gradient_norm.item()),
                )
            )
    means = tuple(sum(row[i] for row in metrics) / len(metrics) for i in range(5))
    return {
        "loss": means[0],
        "policy_loss": means[1],
        "value_loss": means[2],
        "entropy": means[3],
        "gradient_norm": means[4],
        "transition_count": action_index.numel(),
        "minibatch_count": len(metrics),
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    update_index: int,
    config: NativePPOConfig,
    candidate_dim: int,
    context_dim: int,
    rng_python: random.Random,
    rng_torch: torch.Generator,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "protocol": PROTOCOL,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "update_index": update_index,
        "config": {
            "updates": config.updates,
            "episodes_per_update": config.episodes_per_update,
            "epochs": config.epochs,
            "minibatch": config.minibatch_size,
            "lr": config.learning_rate,
            "clip": config.clip_ratio,
            "gae_lambda": config.gae_lambda,
            "gamma": config.gamma,
            "vf_coef": config.value_coefficient,
            "ent_coef": config.entropy_coefficient,
            "max_grad_norm": config.maximum_gradient_norm,
            "hidden": config.hidden_size,
            "init_seed": config.init_seed,
            "checkpoint_every": config.checkpoint_every,
        },
        "candidate_dim": candidate_dim,
        "context_dim": context_dim,
        "hidden": config.hidden_size,
        "rng": {
            "python": rng_python.getstate(),
            "torch": rng_torch.get_state(),
        },
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, *, device: torch.device):
    """Load a checkpoint and return model, optimizer, config, update_index, rngs."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("protocol") != PROTOCOL:
        raise ValueError(f"checkpoint protocol mismatch: {checkpoint.get('protocol')}")
    config = NativePPOConfig(
        updates=int(checkpoint["config"]["updates"]),
        episodes_per_update=int(checkpoint["config"]["episodes_per_update"]),
        epochs=int(checkpoint["config"]["epochs"]),
        minibatch_size=int(checkpoint["config"]["minibatch"]),
        learning_rate=float(checkpoint["config"]["lr"]),
        clip_ratio=float(checkpoint["config"]["clip"]),
        gae_lambda=float(checkpoint["config"]["gae_lambda"]),
        gamma=float(checkpoint["config"]["gamma"]),
        value_coefficient=float(checkpoint["config"]["vf_coef"]),
        entropy_coefficient=float(checkpoint["config"]["ent_coef"]),
        maximum_gradient_norm=float(checkpoint["config"]["max_grad_norm"]),
        hidden_size=int(checkpoint["config"]["hidden"]),
        init_seed=int(checkpoint["config"]["init_seed"]),
        checkpoint_every=int(checkpoint["config"]["checkpoint_every"]),
    )
    model = NativePPOActorCritic(
        candidate_dim=int(checkpoint["candidate_dim"]),
        context_dim=int(checkpoint["context_dim"]),
        hidden_size=config.hidden_size,
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return model, optimizer, config, int(checkpoint["update_index"]), checkpoint["rng"]


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
