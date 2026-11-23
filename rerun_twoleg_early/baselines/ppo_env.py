"""Native-PPO environment wrapper for the AGV-route-event scheduling game (Block N2).

Preregistration: PREREG_BASELINE_EXPANSION_20260727.md (frozen), Block N / N2.

The agent plays the IDENTICAL game as every other method:
- action space per step = all (operation, machine) candidates, i.e. each
  available job's next operation crossed with its eligible machines (the same
  joint candidate space enumerated by the PGND V5 certificate loop);
- committing an action uses the SHARED deterministic transport completion
  (`_append_fixed_joint_preference_transport_action`), exactly the machinery
  the certificate uses for an accepted proposal;
- terminal reward = -V(sigma, w) under the frozen objective-scale contract
  (same `_detail_row` evaluator used by every replay/confirmation block).

State features are GENERIC scheduling features only (operation readiness,
machine availability/load, AGV availability aggregates, event flags,
remaining-operation counts, preference vector). The PGND proposer's
engineered candidate features (`_candidate_row` / regret machinery) are
deliberately NOT used, per preregistration.
"""

from __future__ import annotations
import os as _pwmd_os
_PWMD_ROOT = _pwmd_os.path.abspath(_pwmd_os.path.join(_pwmd_os.path.dirname(__file__), *['..'] * 2))


def _pwmd_path(*parts):
    return _pwmd_os.path.join(_PWMD_ROOT, *parts)



import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PLAYGROUND_SRC = Path(_pwmd_path('rerun_twoleg_early', 'src'))  # rerun v1
_V5_REBUILD = Path(__file__).resolve().parent.parent
for _p in (str(_PLAYGROUND_SRC), str(_V5_REBUILD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from su_modfjsp_ha_pc.hierarchical_transport_attention_policy import (  # noqa: E402
    _append_fixed_joint_preference_transport_action,
)
from su_modfjsp_ha_pc.neural_operation_rollout import _detail_row, _RolloutState  # noqa: E402
from su_modfjsp_ha_pc.objective_contract import ObjectiveScales, read_objective_scales  # noqa: E402
from su_modfjsp_ha_pc.pgnd_v4_data import sha256_file  # noqa: E402
from su_modfjsp_ha_pc.rules import _available_operations  # noqa: E402
from su_modfjsp_ha_pc.schedule import ScheduleResult  # noqa: E402

CANDIDATE_DIM = 12
N_OBJECTIVES_MAX = 5
CONTEXT_DIM = 13 + N_OBJECTIVES_MAX  # 18

# Firewalled evaluation seed ranges (preregistered; never used for training/tuning).
FIREWALL_SEEDS = frozenset(range(630, 770))  # rerun v1: every dev/test seed block

_EVENT_GROUPS = {
    "machine_breakdown": 0,
    "machine_recovery": 0,
    "path_blockage": 1,
    "congestion_spike": 1,
    "agv_failure": 2,
    "route_branch": 3,
    "quality_rework": 3,
    "new_job": 3,
}


def load_frozen_objective_scales() -> ObjectiveScales:
    """Load the frozen objective-scale contract exactly as run_public_core_v5_block does."""
    from rerun_paths import EXPECTED_SCALES_SHA, SCALES  # rerun v1

    if sha256_file(SCALES) != EXPECTED_SCALES_SHA:
        raise RuntimeError("objective scales hash mismatch vs preregistration")
    objective_scales = read_objective_scales(SCALES)
    scale_source = Path(objective_scales.source)
    if not scale_source.exists() or sha256_file(scale_source) != objective_scales.source_sha256:
        raise RuntimeError("objective scale source CSV hash mismatch")
    return objective_scales


@dataclass(frozen=True)
class CandidateAction:
    job: object
    operation: object
    machine: object


class PPOScheduleEnv:
    """Sparse-terminal-reward episodic wrapper around the shared decision loop."""

    def __init__(self, objective_scales: ObjectiveScales) -> None:
        self.objective_scales = objective_scales
        self._reset_fields()

    def _reset_fields(self) -> None:
        self.instance = None
        self.preference: tuple[float, ...] = ()
        self.level = ""
        self.seed = 0
        self.mode = ""
        self.preference_id = ""
        self.state: _RolloutState | None = None
        self.operations: list = []
        self.transports: list = []
        self.candidates: tuple[CandidateAction, ...] = ()
        self.done = False
        self.terminal_value: float | None = None
        self._time_scale = 1.0
        self._energy_scale = 1.0
        self._max_ops_per_job = 1

    def reset(
        self,
        *,
        instance: object,
        preference: tuple[float, ...],
        level: str,
        seed: int,
        mode: str,
        preference_id: str,
    ) -> dict[str, np.ndarray]:
        self._reset_fields()
        self.instance = instance
        self.preference = tuple(float(v) for v in preference)
        self.level, self.seed, self.mode, self.preference_id = level, seed, mode, preference_id
        self.state = _RolloutState(
            machine_available={machine.machine_id: 0 for machine in instance.machines},
            job_ready={job.job_id: job.release_time for job in instance.jobs},
            next_op_index={job.job_id: 0 for job in instance.jobs},
            agv_available={agv.agv_id: 0 for agv in instance.agvs},
            agv_node={agv.agv_id: agv.start_node_id for agv in instance.agvs},
            edge_reservations={edge.edge_id: [] for edge in instance.path_edges},
            total_operations=sum(len(job.operations) for job in instance.jobs),
        )
        self._time_scale = max(
            1.0,
            float(
                sum(
                    min(operation.processing_time.values())
                    for job in instance.jobs
                    for operation in job.operations
                )
            ),
        )
        self._energy_scale = max(
            1.0,
            float(
                sum(
                    min(operation.processing_energy.values())
                    for job in instance.jobs
                    for operation in job.operations
                )
            ),
        )
        self._max_ops_per_job = max(len(job.operations) for job in instance.jobs)
        self._build_candidates()
        return self._observation()

    def _build_candidates(self) -> None:
        available = _available_operations(self.instance.jobs, self.state.next_op_index)
        machine_by_id = {machine.machine_id: machine for machine in self.instance.machines}
        candidates = []
        for job, operation in available:
            for machine_id in operation.eligible_machines:
                candidates.append(CandidateAction(job, operation, machine_by_id[machine_id]))
        if not candidates:
            raise ValueError("no schedulable candidates remain before episode termination")
        self.candidates = tuple(candidates)

    # ------------------------------------------------------------------ features
    def _candidate_features(self, cand: CandidateAction) -> np.ndarray:
        state = self.state
        job, operation, machine = cand.job, cand.operation, cand.machine
        ts, es = self._time_scale, self._energy_scale
        proc = float(operation.processing_time[machine.machine_id])
        energy = float(operation.processing_energy[machine.machine_id])
        job_ready = float(state.job_ready[job.job_id])
        machine_free = float(state.machine_available[machine.machine_id])
        est_start = max(job_ready, machine_free)
        remaining_ops = len(job.operations) - state.next_op_index[job.job_id]
        remaining_work = float(
            sum(
                min(op.processing_time.values())
                for op in job.operations[state.next_op_index[job.job_id] :]
            )
        )
        slack = float(job.due_date) - (est_start + proc)
        min_proc = min(operation.processing_time.values())
        return np.array(
            [
                proc / ts,
                energy / es,
                machine_free / ts,
                job_ready / ts,
                est_start / ts,
                slack / ts,
                remaining_ops / self._max_ops_per_job,
                remaining_work / ts,
                len(operation.eligible_machines) / max(1, len(self.instance.machines)),
                operation.op_index / self._max_ops_per_job,
                (est_start - machine_free) / ts,  # machine idle gap if chosen
                1.0 if proc <= min_proc + 1e-9 else 0.0,
            ],
            dtype=np.float32,
        )

    def _context_features(self) -> np.ndarray:
        state = self.state
        ts = self._time_scale
        done_ops = len(self.operations)
        machine_avail = [float(v) for v in state.machine_available.values()]
        agv_avail = [float(v) for v in state.agv_available.values()] or [0.0]
        now = max(machine_avail) if machine_avail else 0.0
        available = _available_operations(self.instance.jobs, state.next_op_index)
        events = self.instance.events or ()
        group_counts = [0.0, 0.0, 0.0, 0.0]
        future_events = 0
        for event in events:
            if float(event.event_time) > now:
                future_events += 1
                group = _EVENT_GROUPS.get(event.event_type)
                if group is not None:
                    group_counts[group] += 1.0
        n_events = max(1, len(events))
        preference = list(self.preference)[:N_OBJECTIVES_MAX]
        preference += [0.0] * (N_OBJECTIVES_MAX - len(preference))
        context = [
            done_ops / max(1, state.total_operations),
            len(available) / max(1, len(self.instance.jobs)),
            float(np.mean(machine_avail)) / ts,
            float(np.max(machine_avail)) / ts,
            float(np.min(agv_avail)) / ts,
            float(np.mean(agv_avail)) / ts,
            float(np.max(agv_avail)) / ts,
            now / ts,
            future_events / n_events,
            group_counts[0] / n_events,
            group_counts[1] / n_events,
            group_counts[2] / n_events,
            group_counts[3] / n_events,
            *preference,
        ]
        return np.asarray(context, dtype=np.float32)

    def _observation(self) -> dict[str, np.ndarray]:
        cand_matrix = np.stack([self._candidate_features(c) for c in self.candidates])
        return {"candidates": cand_matrix, "context": self._context_features()}

    # ------------------------------------------------------------------ dynamics
    def step(self, action_index: int) -> tuple[dict[str, np.ndarray] | None, float, bool, dict]:
        if self.done:
            raise RuntimeError("step() called on a finished episode")
        if not 0 <= action_index < len(self.candidates):
            raise IndexError(f"action {action_index} outside candidate set of size {len(self.candidates)}")
        cand = self.candidates[action_index]
        _append_fixed_joint_preference_transport_action(
            instance=self.instance,
            preference=self.preference,
            job=cand.job,
            operation=cand.operation,
            machine=cand.machine,
            machine_available=self.state.machine_available,
            job_ready=self.state.job_ready,
            next_op_index=self.state.next_op_index,
            agv_available=self.state.agv_available,
            agv_node=self.state.agv_node,
            edge_reservations=self.state.edge_reservations,
            scheduled_operations=self.operations,
            transport_moves=self.transports,
        )
        if len(self.operations) >= self.state.total_operations:
            self.done = True
            value = self._evaluate(self.schedule_result())
            self.terminal_value = value
            return None, -value, True, {"weighted_value": value}
        self._build_candidates()
        return self._observation(), 0.0, False, {}

    def schedule_result(self) -> ScheduleResult:
        return ScheduleResult(
            algorithm="native-ppo",
            path_aware=bool(self.instance.path_edges),
            operations=tuple(self.operations),
            transports=tuple(self.transports),
            event_aware=bool(self.instance.events),
        )

    def _evaluate(self, schedule: ScheduleResult) -> float:
        row = _detail_row(
            instance=self.instance,
            schedule=schedule,
            teacher_operations=(),
            level=self.level,
            seed=self.seed,
            mode=self.mode,
            preference_id=self.preference_id,
            preference=self.preference,
            method="native-ppo",
            objective_scales=self.objective_scales,
        )
        return float(row["target_weighted_value"])
