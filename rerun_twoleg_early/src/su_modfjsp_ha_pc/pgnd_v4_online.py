from __future__ import annotations

from collections.abc import Callable

from su_modfjsp_ha_pc.exact_advantage_learnability import build_inference_rows_from_candidates
from su_modfjsp_ha_pc.hierarchical_transport_attention_policy import (
    _append_fixed_joint_preference_transport_action,
    _best_fixed_machine_preference_transport,
)
from su_modfjsp_ha_pc.joint_operation_machine_value_policy import (
    _append_selected_preference_action,
    _resolve_joint_candidate,
)
from su_modfjsp_ha_pc.neural_operation_rollout import _RolloutState
from su_modfjsp_ha_pc.path_aware_selection_attention_policy import _path_aware_joint_group_rows
from su_modfjsp_ha_pc.path_conflict_neural_dataset import (
    _best_preference_action,
    _operation_stage_lookahead,
)
from su_modfjsp_ha_pc.pgnd_v4_model import V4Group, build_v4_group
from su_modfjsp_ha_pc.rules import _available_operations
from su_modfjsp_ha_pc.schedule import ScheduleResult, ScheduledOperation, TransportMove


OverrideSelector = Callable[[tuple[dict[str, object], ...], V4Group, int], int | None]


def decode_pgnd_v4_online(
    *,
    instance: object,
    preference: tuple[float, ...],
    level: str,
    seed: int,
    mode: str,
    preference_id: str,
    override_selector: OverrideSelector | None = None,
    trace_rows: list[dict[str, object]] | None = None,
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
        group_id = f"pgnd-v4-{level}-{seed:03d}-{mode}-{preference_id}-{step:03d}"
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
        base_machine_id = base_action["machine"].machine_id
        inference_rows = build_inference_rows_from_candidates(
            candidate_rows,
            base_operation_id=base_operation.operation_id,
            base_machine_id=base_machine_id,
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
        base_index = next(index for index, row in enumerate(inference_rows) if row.is_base)
        _assert_row_matches_selected_action(ordered_candidates[base_index], base_action)
        selected_index = None if override_selector is None else override_selector(ordered_candidates, group, base_index)
        if selected_index is None:
            chosen_index = base_index
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
            if selected_index < 0 or selected_index >= len(ordered_candidates) or selected_index == base_index:
                raise ValueError("override selector returned an invalid candidate index")
            chosen_index = selected_index
            chosen = ordered_candidates[selected_index]
            job, operation, machine = _resolve_joint_candidate(instance, available, chosen)
            selected = _best_fixed_machine_preference_transport(
                instance=instance,
                preference=preference,
                job=job,
                operation=operation,
                machine=machine,
                machine_available=state.machine_available,
                job_ready=state.job_ready,
                agv_available=state.agv_available,
                agv_node=state.agv_node,
                edge_reservations=state.edge_reservations,
            )
            _assert_row_matches_transport(chosen, selected.get("transport"))
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
            trace_rows.append(
                {
                    "step_index": step,
                    "candidate_count": len(ordered_candidates),
                    "base_action": _row_action_key(ordered_candidates[base_index]),
                    "executed_action": _row_action_key(ordered_candidates[chosen_index]),
                    "override": int(chosen_index != base_index),
                }
            )
        step += 1
    return ScheduleResult(
        algorithm="pgnd-v4-online",
        path_aware=bool(instance.path_edges),
        operations=tuple(operations),
        transports=tuple(transports),
        event_aware=bool(instance.events),
    )


def schedules_identical(left: ScheduleResult, right: ScheduleResult) -> bool:
    return left.operations == right.operations and left.transports == right.transports


def _assert_row_matches_selected_action(row: dict[str, object], selected: dict[str, object]) -> None:
    if str(row.get("candidate_operation_id", "")) != selected["operation"].operation_id:
        raise ValueError("base candidate operation differs from standalone lookahead")
    if str(row.get("candidate_machine_id", "")) != selected["machine"].machine_id:
        raise ValueError("base candidate machine differs from standalone lookahead")
    _assert_row_matches_transport(row, selected.get("transport"))


def _assert_row_matches_transport(row: dict[str, object], transport: object) -> None:
    expected_agv = "" if transport is None else str(transport["agv_id"])
    expected_route = "" if transport is None else ";".join(str(edge) for edge in transport["route"])
    if str(row.get("candidate_agv_id", "")) != expected_agv:
        raise ValueError(
            f"candidate AGV differs from recomputed transport: "
            f"row={row.get('candidate_agv_id', '')!r} expected={expected_agv!r}"
        )
    if str(row.get("candidate_route_edges", "")) != expected_route:
        raise ValueError(
            f"candidate route differs from recomputed transport: "
            f"row={row.get('candidate_route_edges', '')!r} expected={expected_route!r}"
        )


def _row_action_key(row: dict[str, object]) -> str:
    return "|".join(
        str(row.get(field, ""))
        for field in (
            "candidate_operation_id",
            "candidate_machine_id",
            "candidate_agv_id",
            "candidate_route_edges",
        )
    )
