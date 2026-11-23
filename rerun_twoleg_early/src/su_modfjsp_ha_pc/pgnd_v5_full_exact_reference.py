from __future__ import annotations

from time import perf_counter

from su_modfjsp_ha_pc.hierarchical_transport_attention_policy import (
    _append_fixed_joint_preference_transport_action,
)
from su_modfjsp_ha_pc.joint_operation_machine_value_policy import (
    _append_selected_preference_action,
    _resolve_joint_candidate,
)
from su_modfjsp_ha_pc.neural_operation_rollout import _RolloutState, _candidate_completion
from su_modfjsp_ha_pc.objective_contract import OBJECTIVE_NAMES as CONTRACT_OBJECTIVE_NAMES, ObjectiveScales
from su_modfjsp_ha_pc.oracle_value_policy import OBJECTIVE_NAMES as LEGACY_OBJECTIVE_NAMES
from su_modfjsp_ha_pc.path_aware_selection_attention_policy import (
    _attach_future_path_aware_value_labels,
    _path_aware_joint_group_rows,
)
from su_modfjsp_ha_pc.path_conflict_neural_dataset import (
    _best_preference_action,
    _operation_stage_lookahead,
)
from su_modfjsp_ha_pc.rules import _available_operations
from su_modfjsp_ha_pc.schedule import ScheduleResult, ScheduledOperation, TransportMove


def decode_pgnd_v5_full_exact_reference(
    *,
    instance: object,
    preference: tuple[float, ...],
    level: str,
    seed: int,
    mode: str,
    preference_id: str,
    trace_rows: list[dict[str, object]] | None = None,
    objective_scales: ObjectiveScales | None = None,
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
        rows = _path_aware_joint_group_rows(
            instance=instance,
            available=available,
            preference=preference,
            preference_id=preference_id,
            level=level,
            seed=seed,
            mode=mode,
            group_id=f"pgnd-v5-full-exact-{level}-{seed}-{mode}-{preference_id}-{step}",
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
        base_job, base_operation = min(
            available,
            key=lambda item: (
                _operation_stage_lookahead(
                    instance=instance,
                    job=item[0],
                    operation=item[1],
                    machine_available=state.machine_available,
                    job_ready_time=state.job_ready[item[0].job_id],
                    agv_available=state.agv_available,
                    agv_node=state.agv_node,
                    edge_reservations=state.edge_reservations,
                )["score"],
                item[1].operation_id,
            ),
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
        base_key = (base_operation.operation_id, base_action["machine"].machine_id)
        rollout_started = perf_counter()
        labeled = _attach_future_path_aware_value_labels(
            instance=instance,
            rows=rows,
            available=available,
            preference=preference,
            machine_available=state.machine_available,
            job_ready=state.job_ready,
            next_op_index=state.next_op_index,
            agv_available=state.agv_available,
            agv_node=state.agv_node,
            edge_reservations=state.edge_reservations,
            scheduled_prefix=tuple(operations),
            transport_prefix=tuple(transports),
            continuation_rule="lookahead",
            continuation_level=level,
            continuation_seed=seed,
            continuation_mode=mode,
            continuation_preference_id=preference_id,
            objective_value_mode="contract" if objective_scales is not None else "rollout",
            objective_scales=objective_scales,
        )
        exact_rollout_ms = (perf_counter() - rollout_started) * 1000.0
        by_key = {
            (
                str(row.get("candidate_operation_id", "")),
                str(row.get("candidate_machine_id", "")),
            ): row
            for row in labeled
        }
        base_row = by_key[base_key]
        best_value = min(_weighted_value(row, preference) for row in labeled)
        best_rows = tuple(
            row
            for row in labeled
            if abs(_weighted_value(row, preference) - best_value) <= 1.0e-12
        )
        best = min(
            best_rows,
            key=lambda row: (
                _candidate_completion(row),
                str(row.get("candidate_operation_id", "")),
                str(row.get("candidate_machine_id", "")),
            ),
        )
        base_value = _weighted_value(base_row, preference)
        chosen = best if len(best_rows) == 1 and best_value < base_value - 1.0e-12 else base_row
        chosen_key = (
            str(chosen.get("candidate_operation_id", "")),
            str(chosen.get("candidate_machine_id", "")),
        )
        if chosen_key == base_key:
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
        else:
            job, operation, machine = _resolve_joint_candidate(instance, available, chosen)
            _append_fixed_joint_preference_transport_action(
                instance=instance,
                preference=preference,
                job=job,
                operation=operation,
                machine=machine,
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
                    "candidate_count": len(labeled),
                    "complete_rollout_evaluations": len(labeled),
                    "base_value": base_value,
                    "best_value": best_value,
                    "best_tie_count": len(best_rows),
                    "selected_by_lookahead": int(chosen_key == base_key),
                    "exact_override": int(chosen_key != base_key),
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
                    "exact_rollout_ms": round(exact_rollout_ms, 6),
                }
            )
        step += 1
    return ScheduleResult(
        algorithm="pgnd-v5-full-exact-operation-machine-reference",
        path_aware=bool(instance.path_edges),
        operations=tuple(operations),
        transports=tuple(transports),
        event_aware=bool(instance.events),
    )


def _weighted_value(row: dict[str, object], preference: tuple[float, ...]) -> float:
    objective_names = (
        CONTRACT_OBJECTIVE_NAMES
        if all(f"future_{objective}_after_action" in row for objective in CONTRACT_OBJECTIVE_NAMES)
        else LEGACY_OBJECTIVE_NAMES
    )
    return sum(
        float(preference[index])
        * float(row.get(f"future_{objective}_after_action", 0.0))
        for index, objective in enumerate(objective_names)
    )
