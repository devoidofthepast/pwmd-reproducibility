from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np

from su_modfjsp_ha_pc.deterministic_dispatch_rules import select_operation_by_rule
from su_modfjsp_ha_pc.exact_advantage_learnability import build_inference_rows_from_candidates
from su_modfjsp_ha_pc.hierarchical_transport_attention_policy import (
    _append_fixed_joint_preference_transport_action,
    _best_fixed_machine_preference_transport,
)
from su_modfjsp_ha_pc.joint_operation_machine_value_policy import (
    _append_selected_preference_action,
    _resolve_joint_candidate,
)
from su_modfjsp_ha_pc.neural_operation_rollout import _RolloutState, _detail_row
from su_modfjsp_ha_pc.objective_contract import ObjectiveScales
from su_modfjsp_ha_pc.path_aware_selection_attention_policy import (
    _complete_schedule_after_forced_path_aware_joint_action,
    _path_aware_joint_group_rows,
)
from su_modfjsp_ha_pc.path_conflict_neural_dataset import (
    _best_preference_action,
)
from su_modfjsp_ha_pc.pgnd_v4_model import V4Artifact, V4Group, build_v4_group, predict_v4_model
from su_modfjsp_ha_pc.pgnd_v4_online import _assert_row_matches_selected_action, _assert_row_matches_transport
from su_modfjsp_ha_pc.rules import _available_operations
from su_modfjsp_ha_pc.schedule import ScheduleResult, ScheduledOperation, TransportMove


ProposalSelector = Callable[[tuple[dict[str, object], ...], V4Group, int], int | None]


def neural_ensemble_proposal_selector(
    artifacts: tuple[V4Artifact, ...],
) -> ProposalSelector:
    if not artifacts:
        raise ValueError("at least one model artifact is required")

    def select(candidate_rows, group, base_index):
        member_predictions = [predict_v4_model(artifact, (group,))[0] for artifact in artifacts]
        mean_prediction = np.mean(np.asarray(member_predictions, dtype=float), axis=0)
        if not np.all(np.isfinite(mean_prediction)):
            return None
        candidate_indexes = [index for index in range(len(candidate_rows)) if index != base_index]
        if not candidate_indexes:
            return None
        best_value = max(mean_prediction[index] for index in candidate_indexes)
        best = [
            index
            for index in candidate_indexes
            if abs(mean_prediction[index] - best_value) <= 1.0e-12
        ]
        return best[0] if len(best) == 1 else None

    return select


def decode_pgnd_v5_exact_certificate(
    *,
    instance: object,
    preference: tuple[float, ...],
    level: str,
    seed: int,
    mode: str,
    preference_id: str,
    proposal_selector: ProposalSelector,
    trace_rows: list[dict[str, object]] | None = None,
    objective_scales: ObjectiveScales | None = None,
    base_rule: str = "lookahead",
    continuation_rule: str = "lookahead",
) -> ScheduleResult:
    state = _RolloutState(
        machine_available={machine.machine_id: 0 for machine in instance.machines},
        job_ready={job.job_id: job.release_time for job in instance.jobs},
        next_op_index={job.job_id: 0 for job in instance.jobs},
        agv_available={agv.agv_id: 0 for agv in instance.agvs},
        agv_node={agv.agv_id: agv.start_node_id for agv in instance.agvs},
        edge_reservations={edge.edge_id: [] for edge in instance.path_edges},
        total_operations=sum(len(job.operations) for job in instance.jobs),
    )
    operations: list[ScheduledOperation] = []
    transports: list[TransportMove] = []
    step = 0
    while len(operations) < state.total_operations:
        available = _available_operations(instance.jobs, state.next_op_index)
        if not available:
            raise ValueError("no schedulable operations remain")
        base_job, base_operation = select_operation_by_rule(
            rule=base_rule,
            instance=instance,
            available=available,
            machine_available=state.machine_available,
            job_ready=state.job_ready,
            agv_available=state.agv_available,
            agv_node=state.agv_node,
            edge_reservations=state.edge_reservations,
        )
        base_action = _best_preference_action(
            instance=instance,
            available=((base_job, base_operation),),
            preference=preference,
            machine_available=state.machine_available,
            job_ready=state.job_ready,
            agv_available=state.agv_available,
            agv_node=state.agv_node,
            edge_reservations=state.edge_reservations,
        )
        candidate_count = 0
        proposal_index = None
        proposal_operation_id = ""
        proposal_machine_id = ""
        accepted = False
        base_value = None
        proposal_value = None
        rollout_evaluations = 0
        neural_scoring_ms = 0.0
        exact_rollout_ms = 0.0
        fallback_reason = "no_unique_nonbase_proposal"
        invariant_failure = 0
        try:
            group_id = f"pgnd-v5-{level}-{seed:03d}-{mode}-{preference_id}-{step:03d}"
            candidate_rows = _path_aware_joint_group_rows(
                instance=instance,
                available=available,
                preference=preference,
                preference_id=preference_id,
                level=level,
                seed=seed,
                mode=mode,
                group_id=group_id,
                candidate_uid_start=step * 10000,
                step_index=step,
                selected_operation_id="",
                selected_machine_id="",
                total_operations=state.total_operations,
                machine_available=state.machine_available,
                job_ready=state.job_ready,
                agv_available=state.agv_available,
                agv_node=state.agv_node,
                edge_reservations=state.edge_reservations,
            )
            inference_rows = build_inference_rows_from_candidates(
                candidate_rows,
                base_operation_id=base_operation.operation_id,
                base_machine_id=base_action["machine"].machine_id,
            )
            group = build_v4_group(inference_rows)
            ordered_candidates = tuple(
                sorted(
                    candidate_rows,
                    key=lambda row: (
                        str(row.get("candidate_operation_id", "")),
                        str(row.get("candidate_machine_id", "")),
                    ),
                )
            )
            candidate_count = len(ordered_candidates)
            base_index = group.base_index
            base_row = ordered_candidates[base_index]
            _assert_row_matches_selected_action(base_row, base_action)
            scoring_started = perf_counter()
            proposal_index = proposal_selector(ordered_candidates, group, base_index)
            neural_scoring_ms = (perf_counter() - scoring_started) * 1000.0
            if proposal_index is not None:
                if (
                    proposal_index < 0
                    or proposal_index >= len(ordered_candidates)
                    or proposal_index == base_index
                ):
                    raise ValueError("proposal selector returned an invalid index")
                proposal_row = ordered_candidates[proposal_index]
                proposal_operation_id = str(proposal_row.get("candidate_operation_id", ""))
                proposal_machine_id = str(proposal_row.get("candidate_machine_id", ""))
                proposal_job, proposal_operation, proposal_machine = _resolve_joint_candidate(
                    instance, available, proposal_row
                )
                proposal_transport = _best_fixed_machine_preference_transport(
                    instance=instance,
                    preference=preference,
                    job=proposal_job,
                    operation=proposal_operation,
                    machine=proposal_machine,
                    machine_available=state.machine_available,
                    job_ready=state.job_ready,
                    agv_available=state.agv_available,
                    agv_node=state.agv_node,
                    edge_reservations=state.edge_reservations,
                )
                _assert_row_matches_transport(proposal_row, proposal_transport.get("transport"))
                rollout_started = perf_counter()
                base_schedule = _complete_schedule_after_forced_path_aware_joint_action(
                    instance=instance,
                    preference=preference,
                    forced_job=base_job,
                    forced_operation=base_operation,
                    forced_machine=base_action["machine"],
                    machine_available=state.machine_available,
                    job_ready=state.job_ready,
                    next_op_index=state.next_op_index,
                    agv_available=state.agv_available,
                    agv_node=state.agv_node,
                    edge_reservations=state.edge_reservations,
                    scheduled_prefix=tuple(operations),
                    transport_prefix=tuple(transports),
                    continuation_rule=continuation_rule,
                    continuation_level=level,
                    continuation_seed=seed,
                    continuation_mode=mode,
                    continuation_preference_id=preference_id,
                )
                proposal_schedule = _complete_schedule_after_forced_path_aware_joint_action(
                    instance=instance,
                    preference=preference,
                    forced_job=proposal_job,
                    forced_operation=proposal_operation,
                    forced_machine=proposal_machine,
                    machine_available=state.machine_available,
                    job_ready=state.job_ready,
                    next_op_index=state.next_op_index,
                    agv_available=state.agv_available,
                    agv_node=state.agv_node,
                    edge_reservations=state.edge_reservations,
                    scheduled_prefix=tuple(operations),
                    transport_prefix=tuple(transports),
                    continuation_rule=continuation_rule,
                    continuation_level=level,
                    continuation_seed=seed,
                    continuation_mode=mode,
                    continuation_preference_id=preference_id,
                )
                rollout_evaluations = 2
                base_value = _weighted_value(
                    instance,
                    base_schedule,
                    preference,
                    level,
                    seed,
                    mode,
                    preference_id,
                    "base",
                    objective_scales,
                )
                proposal_value = _weighted_value(
                    instance,
                    proposal_schedule,
                    preference,
                    level,
                    seed,
                    mode,
                    preference_id,
                    "proposal",
                    objective_scales,
                )
                exact_rollout_ms = (perf_counter() - rollout_started) * 1000.0
                if not np.isfinite(base_value) or not np.isfinite(proposal_value):
                    raise ValueError("exact certificate produced a non-finite value")
                accepted = proposal_value < base_value - 1.0e-12
                fallback_reason = "accepted" if accepted else "certificate_rejected"
                if accepted:
                    next_machine_available = dict(state.machine_available)
                    next_job_ready = dict(state.job_ready)
                    next_next_op_index = dict(state.next_op_index)
                    next_agv_available = dict(state.agv_available)
                    next_agv_node = dict(state.agv_node)
                    next_edge_reservations = {
                        edge_id: list(items) for edge_id, items in state.edge_reservations.items()
                    }
                    next_operations = list(operations)
                    next_transports = list(transports)
                    _append_fixed_joint_preference_transport_action(
                        instance=instance,
                        preference=preference,
                        job=proposal_job,
                        operation=proposal_operation,
                        machine=proposal_machine,
                        machine_available=next_machine_available,
                        job_ready=next_job_ready,
                        next_op_index=next_next_op_index,
                        agv_available=next_agv_available,
                        agv_node=next_agv_node,
                        edge_reservations=next_edge_reservations,
                        scheduled_operations=next_operations,
                        transport_moves=next_transports,
                    )
                    state.machine_available = next_machine_available
                    state.job_ready = next_job_ready
                    state.next_op_index = next_next_op_index
                    state.agv_available = next_agv_available
                    state.agv_node = next_agv_node
                    state.edge_reservations = next_edge_reservations
                    operations = next_operations
                    transports = next_transports
        except Exception as error:
            accepted = False
            invariant_failure = 1
            fallback_reason = f"{type(error).__name__}: {error}"
        if not accepted:
            _append_selected_preference_action(
                instance=instance,
                selected=base_action,
                machine_available=state.machine_available,
                job_ready=state.job_ready,
                next_op_index=state.next_op_index,
                agv_available=state.agv_available,
                agv_node=state.agv_node,
                edge_reservations=state.edge_reservations,
                scheduled_operations=operations,
                transport_moves=transports,
            )
        if trace_rows is not None:
            committed_operation = operations[-1]
            committed_transport = next(
                (
                    move
                    for move in reversed(transports)
                    if move.operation_id == committed_operation.operation_id
                ),
                None,
            )
            trace_rows.append(
                {
                    "step_index": step,
                    "base_rule": base_rule,
                    "continuation_rule": continuation_rule,
                    "candidate_count": candidate_count,
                    "proposal_available": int(proposal_index is not None),
                    "proposal_operation_id": proposal_operation_id,
                    "proposal_machine_id": proposal_machine_id,
                    "chosen_operation_id": committed_operation.operation_id,
                    "chosen_machine_id": committed_operation.machine_id,
                    "chosen_agv_id": (
                        "" if committed_transport is None else committed_transport.agv_id
                    ),
                    "chosen_route": (
                        ""
                        if committed_transport is None
                        else ";".join(committed_transport.edge_ids)
                    ),
                    "accepted": int(accepted),
                    "base_value": base_value,
                    "proposal_value": proposal_value,
                    "certified_gain": (
                        None
                        if base_value is None or proposal_value is None
                        else float(base_value - proposal_value)
                    ),
                    "rollout_evaluations": rollout_evaluations,
                    "neural_scoring_ms": round(neural_scoring_ms, 6),
                    "exact_rollout_ms": round(exact_rollout_ms, 6),
                    "fallback_reason": fallback_reason,
                    "invariant_failure": invariant_failure,
                }
            )
        step += 1
    return ScheduleResult(
        algorithm="pgnd-v5-neural-proposal-exact-certificate",
        path_aware=bool(instance.path_edges),
        operations=tuple(operations),
        transports=tuple(transports),
        event_aware=bool(instance.events),
    )


def _weighted_value(
    instance,
    schedule,
    preference,
    level,
    seed,
    mode,
    preference_id,
    method,
    objective_scales,
) -> float:
    row = _detail_row(
        instance=instance,
        schedule=schedule,
        teacher_operations=(),
        level=level,
        seed=seed,
        mode=mode,
        preference_id=preference_id,
        preference=preference,
        method=method,
        objective_scales=objective_scales,
    )
    return float(row["target_weighted_value"])
