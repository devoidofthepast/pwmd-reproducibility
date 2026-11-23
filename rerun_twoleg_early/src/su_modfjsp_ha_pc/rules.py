from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass

from su_modfjsp_ha_pc.conflicts import EdgeUsage, count_capacity_violations
from su_modfjsp_ha_pc.models import AGV, Instance, Job, Machine, Operation, PathEdge
from su_modfjsp_ha_pc.schedule import ScheduleResult, ScheduledOperation, TransportMove


@dataclass(frozen=True)
class SymbolicRuleSpec:
    name: str
    dispatch_weights: dict[str, float]
    machine_weights: dict[str, float]
    path_aware: bool = True
    event_aware: bool = True
    source_preference: tuple[float, ...] = ()


def schedule_rule_baseline(
    instance: Instance,
    rule: str = "fifo",
    path_aware: bool = False,
    event_aware: bool = False,
    repair_process_log: list[dict[str, object]] | None = None,
    repair_process_context: dict[str, object] | None = None,
) -> ScheduleResult:
    if rule not in {"fifo", "spt", "edd"}:
        raise ValueError("rule must be one of fifo, spt, edd")

    return _schedule_greedy(
        instance=instance,
        algorithm=f"rule-{rule}",
        path_aware=path_aware,
        event_aware=event_aware,
        select_operation=lambda available: min(
            available,
            key=lambda item: _dispatch_key(rule, item[0], item[1]),
        ),
        select_machine=lambda machine_by_id, job, operation: min(
            operation.eligible_machines,
            key=lambda candidate: (operation.processing_time[candidate], candidate),
        ),
        repair_process_log=repair_process_log,
        repair_process_context=repair_process_context,
    )


def schedule_preference_rule_baseline(
    instance: Instance,
    preference: tuple[float, ...],
    path_aware: bool | None = None,
    event_aware: bool | None = None,
    repair_process_log: list[dict[str, object]] | None = None,
    repair_process_context: dict[str, object] | None = None,
) -> ScheduleResult:
    """PSLGP-style lightweight proxy: one rule adapts its scores to a preference vector."""

    weights = _preference_weights(instance, preference)
    resolved_path_aware = (
        path_aware
        if path_aware is not None
        else bool(instance.path_edges and weights.get("path_conflict_delay", 0.0) > 0)
    )
    resolved_event_aware = (
        event_aware
        if event_aware is not None
        else bool(instance.events and weights.get("stability", 0.0) > 0)
    )

    return _schedule_greedy(
        instance=instance,
        algorithm=f"preference-rule-{_preference_label(preference)}",
        path_aware=resolved_path_aware,
        event_aware=resolved_event_aware,
        select_operation=lambda available: min(
            available,
            key=lambda item: (_preference_dispatch_score(weights, item[0], item[1]), item[1].operation_id),
        ),
        select_machine=lambda machine_by_id, job, operation: min(
            operation.eligible_machines,
            key=lambda candidate: (
                _preference_machine_score(weights, operation, machine_by_id[candidate]),
                candidate,
            ),
        ),
        repair_process_log=repair_process_log,
        repair_process_context=repair_process_context,
    )


def schedule_symbolic_rule_baseline(
    instance: Instance,
    rule: SymbolicRuleSpec,
    repair_process_log: list[dict[str, object]] | None = None,
    repair_process_context: dict[str, object] | None = None,
) -> ScheduleResult:
    resolved_path_aware = bool(instance.path_edges) and rule.path_aware
    resolved_event_aware = bool(instance.events) and rule.event_aware
    return _schedule_greedy(
        instance=instance,
        algorithm=rule.name,
        path_aware=resolved_path_aware,
        event_aware=resolved_event_aware,
        select_operation=lambda available: min(
            available,
            key=lambda item: (_symbolic_dispatch_score(rule.dispatch_weights, item[0], item[1]), item[1].operation_id),
        ),
        select_machine=lambda machine_by_id, job, operation: min(
            operation.eligible_machines,
            key=lambda candidate: (
                _symbolic_machine_score(rule.machine_weights, operation, machine_by_id[candidate]),
                candidate,
            ),
        ),
        repair_process_log=repair_process_log,
        repair_process_context=repair_process_context,
    )


def _schedule_greedy(
    instance: Instance,
    algorithm: str,
    path_aware: bool,
    event_aware: bool,
    select_operation: Callable[[tuple[tuple[Job, Operation], ...]], tuple[Job, Operation]],
    select_machine: Callable[[dict[str, Machine], Job, Operation], str],
    repair_process_log: list[dict[str, object]] | None = None,
    repair_process_context: dict[str, object] | None = None,
) -> ScheduleResult:
    machine_by_id = {machine.machine_id: machine for machine in instance.machines}

    machine_available = {machine.machine_id: 0 for machine in instance.machines}
    job_ready = {job.job_id: job.release_time for job in instance.jobs}
    next_op_index = {job.job_id: 0 for job in instance.jobs}
    agv_available = {agv.agv_id: 0 for agv in instance.agvs}
    agv_node = {agv.agv_id: agv.start_node_id for agv in instance.agvs}
    edge_reservations: dict[str, list[tuple[int, int]]] = {edge.edge_id: [] for edge in instance.path_edges}

    scheduled_operations: list[ScheduledOperation] = []
    transport_moves: list[TransportMove] = []
    total_operations = sum(len(job.operations) for job in instance.jobs)

    while len(scheduled_operations) < total_operations:
        available = _available_operations(instance.jobs, next_op_index)
        if not available:
            raise ValueError("no schedulable operations remain")
        job, operation = select_operation(available)
        machine_id = select_machine(machine_by_id, job, operation)
        machine = machine_by_id[machine_id]
        transport_end = job_ready[job.job_id]
        if instance.agvs and instance.path_edges:
            move = _schedule_transport(
                instance=instance,
                job=job,
                operation=operation,
                destination=machine,
                agv_available=agv_available,
                agv_node=agv_node,
                edge_reservations=edge_reservations,
                path_aware=path_aware,
                event_aware=event_aware,
                move_index=len(transport_moves),
                repair_process_log=repair_process_log,
                repair_process_context=repair_process_context,
            )
            transport_moves.append(move)
            transport_end = move.end

        start = max(job_ready[job.job_id], machine_available[machine_id], transport_end)
        if event_aware:
            start = _first_resource_event_feasible_start(
                start,
                operation.processing_time[machine_id],
                machine_id,
                "machine_breakdown",
                instance,
                repair_process_log=repair_process_log,
                repair_process_context=repair_process_context,
                subject={
                    "job_id": job.job_id,
                    "operation_id": operation.operation_id,
                    "machine_id": machine_id,
                },
            )
        end = start + operation.processing_time[machine_id]
        scheduled_operations.append(
            ScheduledOperation(
                operation_id=operation.operation_id,
                job_id=operation.job_id,
                op_index=operation.op_index,
                machine_id=machine_id,
                start=start,
                end=end,
                processing_energy=operation.processing_energy[machine_id],
            )
        )
        machine_available[machine_id] = end
        job_ready[job.job_id] = end
        next_op_index[job.job_id] += 1

    return ScheduleResult(
        algorithm=algorithm,
        path_aware=path_aware,
        operations=tuple(scheduled_operations),
        transports=tuple(transport_moves),
        event_aware=event_aware,
    )


def _available_operations(
    jobs: tuple[Job, ...],
    next_op_index: dict[str, int],
) -> tuple[tuple[Job, Operation], ...]:
    available = []
    for job in jobs:
        index = next_op_index[job.job_id]
        if index < len(job.operations):
            available.append((job, job.operations[index]))
    return tuple(available)


def _dispatch_key(rule: str, job: Job, operation: Operation) -> tuple[float, str]:
    if rule == "spt":
        return (min(operation.processing_time.values()), operation.operation_id)
    if rule == "edd":
        return (job.due_date, operation.operation_id)
    return (job.release_time, operation.operation_id)


def _preference_weights(instance: Instance, preference: tuple[float, ...]) -> dict[str, float]:
    if len(preference) != len(instance.objectives):
        raise ValueError("preference length must match instance objectives")
    if any(weight < 0 for weight in preference):
        raise ValueError("preference weights must be non-negative")
    total = sum(preference)
    if total <= 0:
        raise ValueError("preference weights must contain positive mass")
    return {
        objective: weight / total
        for objective, weight in zip(instance.objectives, preference)
    }


def _preference_label(preference: tuple[float, ...]) -> str:
    return "-".join(f"{weight:.2f}" for weight in preference)


def _preference_dispatch_score(weights: dict[str, float], job: Job, operation: Operation) -> float:
    best_processing = min(operation.processing_time.values())
    best_energy = min(operation.processing_energy.values())
    due_pressure = job.due_date
    release_pressure = job.release_time

    return (
        weights.get("makespan", 0.0) * best_processing
        + weights.get("tardiness", 0.0) * due_pressure
        + weights.get("energy", 0.0) * best_energy
        + weights.get("machine_idle_time", 0.0) * best_processing
        + weights.get("path_conflict_delay", 0.0) * release_pressure
        + weights.get("stability", 0.0) * release_pressure
    )


def _preference_machine_score(weights: dict[str, float], operation: Operation, machine: Machine) -> float:
    processing = operation.processing_time[machine.machine_id]
    energy = operation.processing_energy[machine.machine_id]
    return (
        weights.get("makespan", 0.0) * processing
        + weights.get("tardiness", 0.0) * processing
        + weights.get("machine_idle_time", 0.0) * processing
        + weights.get("energy", 0.0) * energy
        + weights.get("path_conflict_delay", 0.0) * processing
        + weights.get("stability", 0.0) * processing
    )


def _symbolic_dispatch_score(weights: dict[str, float], job: Job, operation: Operation) -> float:
    best_processing = min(operation.processing_time.values())
    best_energy = min(operation.processing_energy.values())
    remaining_ops = len(job.operations) - operation.op_index
    slack = job.due_date - job.release_time - best_processing
    features = {
        "processing": float(best_processing),
        "energy": float(best_energy),
        "due_date": float(job.due_date),
        "release_time": float(job.release_time),
        "remaining_ops": float(remaining_ops),
        "slack": float(slack),
        "op_index": float(operation.op_index),
    }
    return sum(weights.get(name, 0.0) * value for name, value in features.items())


def _symbolic_machine_score(weights: dict[str, float], operation: Operation, machine: Machine) -> float:
    processing = operation.processing_time[machine.machine_id]
    energy = operation.processing_energy[machine.machine_id]
    features = {
        "processing": float(processing),
        "energy": float(energy),
        "machine_energy_rate": float(machine.energy_rate),
    }
    return sum(weights.get(name, 0.0) * value for name, value in features.items())


def _schedule_transport(
    instance: Instance,
    job: Job,
    operation: Operation,
    destination: Machine,
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
    path_aware: bool,
    event_aware: bool,
    move_index: int,
    repair_process_log: list[dict[str, object]] | None = None,
    repair_process_context: dict[str, object] | None = None,
) -> TransportMove:
    raise NotImplementedError("early-departure patch: rules._schedule_transport is not patched")
    edge_by_id = {edge.edge_id: edge for edge in instance.path_edges}
    agv = _choose_agv(instance.agvs, agv_available)
    origin_node = agv_node[agv.agv_id]
    _pickup = agv_node.get("__job__:" + str(job.job_id), instance.agvs[0].start_node_id)
    route_edges = tuple(_shortest_path_edges(instance.path_edges, origin_node, _pickup)) + tuple(_shortest_path_edges(instance.path_edges, _pickup, destination.node_id))
    route_length = sum(edge_by_id[edge_id].length for edge_id in route_edges)
    travel_time = max(1, math.ceil(route_length / agv.speed))
    start = max(job.release_time, agv_available[agv.agv_id])
    if path_aware or event_aware:
        start = _first_transport_feasible_start(
            earliest_start=start,
            travel_time=travel_time,
            route_edges=route_edges,
            agv_id=agv.agv_id,
            instance=instance,
            edge_reservations=edge_reservations,
            check_capacity=path_aware,
            check_events=event_aware,
            repair_process_log=repair_process_log,
            repair_process_context=repair_process_context,
            subject={
                "job_id": job.job_id,
                "operation_id": operation.operation_id,
                "machine_id": destination.machine_id,
                "move_id": f"T{move_index}",
                "agv_id": agv.agv_id,
                "route_edges": ";".join(route_edges),
            },
        )
    end = start + travel_time

    for edge_id in dict.fromkeys(route_edges):
        edge_reservations.setdefault(edge_id, []).append((start, end))
    agv_available[agv.agv_id] = end
    agv_node[agv.agv_id] = destination.node_id
    agv_node["__job__:" + str(job.job_id)] = destination.node_id

    return TransportMove(
        move_id=f"T{move_index}",
        job_id=job.job_id,
        operation_id=operation.operation_id,
        agv_id=agv.agv_id,
        from_node=origin_node,
        to_node=destination.node_id,
        edge_ids=route_edges,
        start=start,
        end=end,
        transport_energy=round(route_length * agv.consumption_rate, 6),
    )


def _choose_agv(agvs: tuple[AGV, ...], agv_available: dict[str, int]) -> AGV:
    return min(agvs, key=lambda agv: (agv_available[agv.agv_id], agv.agv_id))


def _first_capacity_feasible_start(
    earliest_start: int,
    travel_time: int,
    route_edges: tuple[str, ...],
    path_edges: tuple[PathEdge, ...],
    edge_reservations: dict[str, list[tuple[int, int]]],
) -> int:
    capacities = {edge.edge_id: edge.capacity for edge in path_edges}
    start = earliest_start
    while True:
        proposed_usages = [
            EdgeUsage(edge_id=edge_id, start=start, end=start + travel_time)
            for edge_id in dict.fromkeys(route_edges)
        ]
        existing_usages = [
            EdgeUsage(edge_id=edge_id, start=left, end=right)
            for edge_id, reservations in edge_reservations.items()
            for left, right in reservations
        ]
        if count_capacity_violations(tuple(existing_usages + proposed_usages), capacities) == 0:
            return start
        start += 1


def _first_transport_feasible_start(
    earliest_start: int,
    travel_time: int,
    route_edges: tuple[str, ...],
    agv_id: str,
    instance: Instance,
    edge_reservations: dict[str, list[tuple[int, int]]],
    check_capacity: bool,
    check_events: bool,
    repair_process_log: list[dict[str, object]] | None = None,
    repair_process_context: dict[str, object] | None = None,
    subject: dict[str, object] | None = None,
) -> int:
    start = earliest_start
    causes: dict[tuple[str, str, str], dict[str, object]] = {}
    while True:
        if check_events:
            event_causes = _transport_event_conflicts(start, travel_time, route_edges, agv_id, instance)
            shifted_start = max((int(cause["shifted_start"]) for cause in event_causes), default=start)
            if shifted_start != start:
                for cause in event_causes:
                    _remember_repair_cause(causes, cause)
                start = shifted_start
                continue
        if check_capacity:
            capacity_causes = _capacity_conflicts(
                start,
                travel_time,
                route_edges,
                instance.path_edges,
                edge_reservations,
            )
            if capacity_causes:
                for cause in capacity_causes:
                    _remember_repair_cause(causes, cause)
                start = max(
                    int(cause["shifted_start"])
                    for cause in capacity_causes
                )
                continue
        if repair_process_log is not None and start > earliest_start:
            for cause in causes.values():
                _append_repair_process_row(
                    repair_process_log,
                    repair_process_context,
                    repair_type=str(cause["repair_type"]),
                    entity_id=str(cause.get("entity_id", "")),
                    earliest_start=earliest_start,
                    final_start=start,
                    event_id=str(cause.get("event_id", "")),
                    edge_id=str(cause.get("edge_id", "")),
                    agv_id=str(subject.get("agv_id", agv_id) if subject else agv_id),
                    machine_id=str(subject.get("machine_id", "") if subject else ""),
                    subject=subject,
                    reason=f"{cause.get('reason', '')}; shift_checks={cause.get('shift_checks', 1)}",
                )
        return start


def _next_capacity_feasible_start(
    start: int,
    travel_time: int,
    route_edges: tuple[str, ...],
    path_edges: tuple[PathEdge, ...],
    edge_reservations: dict[str, list[tuple[int, int]]],
) -> int:
    capacities = {edge.edge_id: edge.capacity for edge in path_edges}
    proposed_usages = [
        EdgeUsage(edge_id=edge_id, start=start, end=start + travel_time)
        for edge_id in dict.fromkeys(route_edges)
    ]
    existing_usages = [
        EdgeUsage(edge_id=edge_id, start=left, end=right)
        for edge_id, reservations in edge_reservations.items()
        for left, right in reservations
    ]
    if count_capacity_violations(tuple(existing_usages + proposed_usages), capacities) == 0:
        return start
    return start + 1


def _next_transport_event_feasible_start(
    start: int,
    travel_time: int,
    route_edges: tuple[str, ...],
    agv_id: str,
    instance: Instance,
) -> int:
    end = start + travel_time
    shifted_start = start
    for event in instance.events:
        if event.duration <= 0:
            continue
        event_start = event.event_time
        event_end = event.event_time + event.duration
        if event.event_type == "agv_failure" and event.affected_entity == agv_id:
            if _overlaps(start, end, event_start, event_end):
                shifted_start = max(shifted_start, event_end)
        if event.event_type in {"path_blockage", "congestion_spike"} and event.affected_entity in route_edges:
            if _overlaps(start, end, event_start, event_end):
                shifted_start = max(shifted_start, event_end)
    return shifted_start


def _transport_event_conflicts(
    start: int,
    travel_time: int,
    route_edges: tuple[str, ...],
    agv_id: str,
    instance: Instance,
) -> tuple[dict[str, object], ...]:
    end = start + travel_time
    rows = []
    for event in instance.events:
        if event.duration <= 0:
            continue
        event_start = event.event_time
        event_end = event.event_time + event.duration
        if event.event_type == "agv_failure" and event.affected_entity == agv_id:
            if _overlaps(start, end, event_start, event_end):
                rows.append(
                    {
                        "repair_type": "agv-event",
                        "entity_id": agv_id,
                        "event_id": event.event_id,
                        "shifted_start": event_end,
                        "reason": f"transport overlaps {event.event_type}",
                    }
                )
        if event.event_type in {"path_blockage", "congestion_spike"} and event.affected_entity in route_edges:
            if _overlaps(start, end, event_start, event_end):
                rows.append(
                    {
                        "repair_type": "path-event",
                        "entity_id": event.affected_entity,
                        "edge_id": event.affected_entity,
                        "event_id": event.event_id,
                        "shifted_start": event_end,
                        "reason": f"transport uses event-affected edge during {event.event_type}",
                    }
                )
    return tuple(rows)


def _capacity_conflicts(
    start: int,
    travel_time: int,
    route_edges: tuple[str, ...],
    path_edges: tuple[PathEdge, ...],
    edge_reservations: dict[str, list[tuple[int, int]]],
) -> tuple[dict[str, object], ...]:
    if isinstance(path_edges, tuple):
        _ct = _memo_table(path_edges, "cap")
        capacities = _ct.get("v")
        if capacities is None:
            capacities = _ct["v"] = {edge.edge_id: edge.capacity for edge in path_edges}
    else:
        capacities = {edge.edge_id: edge.capacity for edge in path_edges}
    end = start + travel_time
    rows = []
    for edge_id in dict.fromkeys(route_edges):
        capacity = capacities.get(edge_id, 1)
        active = tuple(
            (left, right)
            for left, right in edge_reservations.get(edge_id, [])
            if start < right and left < end
        )
        if len(active) + 1 > capacity:
            # With a fixed interval and capacity violation, every start before
            # the earliest active end remains infeasible for this edge.
            next_start = min((right for _, right in active), default=start + 1)
            rows.append(
                {
                    "repair_type": "path-capacity",
                    "entity_id": edge_id,
                    "edge_id": edge_id,
                    "shifted_start": max(start + 1, next_start),
                    "reason": f"edge {edge_id} would have {len(active) + 1} active moves over capacity {capacity}",
                }
            )
    return tuple(rows)


def _first_resource_event_feasible_start(
    earliest_start: int,
    duration: int,
    resource_id: str,
    event_type: str,
    instance: Instance,
    repair_process_log: list[dict[str, object]] | None = None,
    repair_process_context: dict[str, object] | None = None,
    subject: dict[str, object] | None = None,
) -> int:
    start = earliest_start
    causes: dict[tuple[str, str, str], dict[str, object]] = {}
    while True:
        shifted_start = start
        end = start + duration
        for event in instance.events:
            if event.event_type != event_type or event.affected_entity != resource_id or event.duration <= 0:
                continue
            event_start = event.event_time
            event_end = event.event_time + event.duration
            if _overlaps(start, end, event_start, event_end):
                shifted_start = max(shifted_start, event_end)
                _remember_repair_cause(
                    causes,
                    {
                        "repair_type": "machine-event"
                        if event_type == "machine_breakdown"
                        else f"{event_type}-event",
                        "entity_id": resource_id,
                        "event_id": event.event_id,
                        "shifted_start": event_end,
                        "reason": f"operation overlaps {event.event_type}",
                    },
                )
        if shifted_start == start:
            if repair_process_log is not None and start > earliest_start:
                for cause in causes.values():
                    _append_repair_process_row(
                        repair_process_log,
                        repair_process_context,
                        repair_type=str(cause["repair_type"]),
                        entity_id=str(cause.get("entity_id", "")),
                        earliest_start=earliest_start,
                        final_start=start,
                        event_id=str(cause.get("event_id", "")),
                        edge_id="",
                        agv_id="",
                        machine_id=str(subject.get("machine_id", resource_id) if subject else resource_id),
                        subject=subject,
                        reason=f"{cause.get('reason', '')}; shift_checks={cause.get('shift_checks', 1)}",
                    )
            return start
        start = shifted_start


def _overlaps(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


def _remember_repair_cause(
    causes: dict[tuple[str, str, str], dict[str, object]],
    cause: dict[str, object],
) -> None:
    key = (
        str(cause.get("repair_type", "")),
        str(cause.get("entity_id", "")),
        str(cause.get("event_id", "")),
    )
    if key not in causes:
        causes[key] = {**cause, "shift_checks": 1}
        return
    causes[key]["shift_checks"] = int(causes[key].get("shift_checks", 1)) + 1
    causes[key]["shifted_start"] = max(
        int(causes[key].get("shifted_start", 0)),
        int(cause.get("shifted_start", 0)),
    )


def _append_repair_process_row(
    rows: list[dict[str, object]],
    context: dict[str, object] | None,
    *,
    repair_type: str,
    entity_id: str,
    earliest_start: int,
    final_start: int,
    event_id: str,
    edge_id: str,
    agv_id: str,
    machine_id: str,
    subject: dict[str, object] | None,
    reason: str,
) -> None:
    context = context or {}
    subject = subject or {}
    rows.append(
        {
            "context_id": str(context.get("context_id", "")),
            "solution_id": str(context.get("solution_id", "")),
            "level": str(context.get("level", "")),
            "seed": str(context.get("seed", "")),
            "mode": str(context.get("mode", "")),
            "algorithm": str(context.get("algorithm", "")),
            "generation": context.get("generation", ""),
            "eval_uid": context.get("eval_uid", ""),
            "repair_type": repair_type,
            "entity_id": entity_id,
            "operation_id": str(subject.get("operation_id", "")),
            "job_id": str(subject.get("job_id", "")),
            "move_id": str(subject.get("move_id", "")),
            "agv_id": agv_id,
            "machine_id": machine_id,
            "edge_id": edge_id,
            "earliest_start": earliest_start,
            "final_start": final_start,
            "repair_delay": final_start - earliest_start,
            "event_id": event_id,
            "route_edges": str(subject.get("route_edges", "")),
            "reason": reason,
        }
    )


def _shortest_path_edges(path_edges: tuple[PathEdge, ...], source: str, target: str) -> tuple[str, ...]:
    if not isinstance(path_edges, tuple):
        return _shortest_path_edges_impl(path_edges, source, target)
    table = _memo_table(path_edges, "sp")
    key = (source, target)
    if key not in table:
        try:
            table[key] = (True, _shortest_path_edges_impl(path_edges, source, target))
        except ValueError as exc:
            table[key] = (False, str(exc))
    ok, value = table[key]
    if not ok:
        raise ValueError(value)
    return value


def _shortest_path_edges_impl(path_edges: tuple[PathEdge, ...], source: str, target: str) -> tuple[str, ...]:
    if source == target:
        return ()
    adjacency: dict[str, list[tuple[float, str, str]]] = {}
    for edge in path_edges:
        adjacency.setdefault(edge.from_node, []).append((edge.length, edge.to_node, edge.edge_id))
        if edge.bidirectional:
            adjacency.setdefault(edge.to_node, []).append((edge.length, edge.from_node, edge.edge_id))

    queue: list[tuple[float, str, tuple[str, ...]]] = [(0.0, source, ())]
    visited: set[str] = set()
    while queue:
        distance, node, route = heapq.heappop(queue)
        if node == target:
            return route
        if node in visited:
            continue
        visited.add(node)
        for length, neighbor, edge_id in adjacency.get(node, []):
            if neighbor not in visited:
                heapq.heappush(queue, (distance + length, neighbor, route + (edge_id,)))
    raise ValueError(f"no path from {source} to {target}")



# ---- early departure two-leg transport (2026-11-09, 提前出发试验) ----
def early_leg_split(instance, origin, pickup, route):
    """Number of leading edges in route that form the empty leg (walk from origin until pickup is first reached)."""
    if origin == pickup or not route:
        return 0
    edge_by_id = _edge_by_id(instance)
    node = origin
    for index, edge_id in enumerate(route):
        edge = edge_by_id[edge_id]
        if edge.from_node == node:
            node = edge.to_node
        elif edge.to_node == node:
            node = edge.from_node
        else:
            raise ValueError(f"route edge {edge_id} not incident to {node}")
        if node == pickup:
            return index + 1
    raise ValueError(f"route from {origin} never reaches pickup {pickup}")


def early_leg_times(instance, agv, origin, pickup, route):
    path_edges = instance.path_edges
    if isinstance(path_edges, tuple):
        table = _memo_table(path_edges, "legt")
        key = (origin, pickup, tuple(route), agv.speed)
        hit = table.get(key)
        if hit is None:
            hit = table[key] = _early_leg_times_impl(instance, agv, origin, pickup, route)
        return hit
    return _early_leg_times_impl(instance, agv, origin, pickup, route)


def _early_leg_times_impl(instance, agv, origin, pickup, route):
    n_empty = early_leg_split(instance, origin, pickup, route)
    edge_by_id = _edge_by_id(instance)
    empty_length = sum(edge_by_id[e].length for e in route[:n_empty])
    loaded_length = sum(edge_by_id[e].length for e in route[n_empty:])
    te = math.ceil(empty_length / agv.speed) if n_empty else 0
    tl = max(1, math.ceil(loaded_length / agv.speed))
    if _DIAG_CEIL_TOTAL:  # diagnosis only (default off): one ceil over the total length
        tl = max(1, max(1, math.ceil((empty_length + loaded_length) / agv.speed)) - te)
    return n_empty, te, tl


def two_leg_early_schedule(instance, agv, origin, pickup, route, agv_free, job_ready, edge_reservations):
    route = tuple(route)
    n_empty, te, tl = early_leg_times(instance, agv, origin, pickup, route)
    if te > 0:
        tse = _first_transport_feasible_start(
            earliest_start=agv_free, travel_time=te, route_edges=route[:n_empty], agv_id=agv.agv_id,
            instance=instance, edge_reservations=edge_reservations, check_capacity=True, check_events=True)
    else:
        tse = agv_free
    ete = tse + te
    loaded_earliest = max(ete, job_ready)
    tlb = _first_transport_feasible_start(
        earliest_start=loaded_earliest, travel_time=tl, route_edges=route[n_empty:], agv_id=agv.agv_id,
        instance=instance, edge_reservations=edge_reservations, check_capacity=True, check_events=True)
    return {"start": tse, "end": tlb + tl, "empty_end": ete, "loaded_start": tlb, "n_empty": n_empty,
            "delay": (tse - agv_free) + (tlb - loaded_earliest)}


def move_leg_usage(instance, move, pickup):
    """((edges, start, end), ...) for the empty and the loaded leg of a committed move."""
    agv = next(a for a in instance.agvs if a.agv_id == move.agv_id)
    route = tuple(move.edge_ids)
    n_empty, te, tl = early_leg_times(instance, agv, move.from_node, pickup, route)
    ete, tlb = move.start + te, move.end - tl
    if tlb < ete:
        raise ValueError(f"{move.move_id}: loaded leg starts at {tlb} before empty leg ends at {ete}")
    legs = []
    if n_empty:
        legs.append((route[:n_empty], move.start, ete))
    legs.append((route[n_empty:], tlb, move.end))
    return tuple(legs)


def job_pickup_of(instance, node_map, job_id):
    return node_map.get("__job__:" + str(job_id), instance.agvs[0].start_node_id)


# ---- exact memoization (2026-11-09, early_fast) ----
_MEMO = {}


def _memo_table(path_edges, name):
    key = (id(path_edges), name)
    entry = _MEMO.get(key)
    if entry is None or entry[0] is not path_edges:
        entry = (path_edges, {})
        _MEMO[key] = entry
    return entry[1]


def _edge_by_id(instance):
    path_edges = instance.path_edges
    if not isinstance(path_edges, tuple):
        return {edge.edge_id: edge for edge in path_edges}
    table = _memo_table(path_edges, "eb")
    if "v" not in table:
        table["v"] = {edge.edge_id: edge for edge in path_edges}
    return table["v"]


import os as _os  # noqa: E402

_DIAG_CEIL_TOTAL = _os.environ.get("EARLY_DIAG_CEIL_TOTAL") == "1"  # diagnosis only, default off


# ---- exact speed-ups (2026-11-12, 全部重跑规则_冻结v3 6b; V3_FAST_PROJ=0 restores the functions above) ----
# _capacity_conflicts: one pass per edge counting active reservations and their earliest end, no intermediate tuple.
# _transport_event_conflicts: scans only the events that can match (positive duration, AGV or path type), in event order.
_FAST = _os.environ.get("V3_FAST_PROJ", "1") != "0"
_capacity_conflicts_ref = _capacity_conflicts
_transport_event_conflicts_ref = _transport_event_conflicts


def _capacity_conflicts_fast(start, travel_time, route_edges, path_edges, edge_reservations):
    if not isinstance(path_edges, tuple):
        return _capacity_conflicts_ref(start, travel_time, route_edges, path_edges, edge_reservations)
    _ct = _memo_table(path_edges, "cap")
    capacities = _ct.get("v")
    if capacities is None:
        capacities = _ct["v"] = {edge.edge_id: edge.capacity for edge in path_edges}
    end = start + travel_time
    rows = []
    for edge_id in dict.fromkeys(route_edges):
        capacity = capacities.get(edge_id, 1)
        n = 0
        first_end = None
        for left, right in edge_reservations.get(edge_id, ()):
            if start < right and left < end:
                n += 1
                if first_end is None or right < first_end:
                    first_end = right
        if n + 1 > capacity:
            next_start = start + 1 if first_end is None else first_end
            rows.append(
                {
                    "repair_type": "path-capacity",
                    "entity_id": edge_id,
                    "edge_id": edge_id,
                    "shifted_start": max(start + 1, next_start),
                    "reason": f"edge {edge_id} would have {n + 1} active moves over capacity {capacity}",
                }
            )
    return tuple(rows)


def _transport_event_conflicts_fast(start, travel_time, route_edges, agv_id, instance):
    events = instance.events
    if not isinstance(events, tuple):
        return _transport_event_conflicts_ref(start, travel_time, route_edges, agv_id, instance)
    table = _memo_table(events, "tev")
    rel = table.get("v")
    if rel is None:
        rel = table["v"] = tuple(e for e in events if e.duration > 0 and e.event_type in {"agv_failure", "path_blockage", "congestion_spike"})
    if not rel:
        return ()
    return _transport_event_conflicts_ref(start, travel_time, route_edges, agv_id, _EventsOnly(rel))


class _EventsOnly:
    __slots__ = ("events",)

    def __init__(self, events):
        self.events = events


def _capacity_conflicts(start, travel_time, route_edges, path_edges, edge_reservations):
    return (_capacity_conflicts_fast if _FAST else _capacity_conflicts_ref)(start, travel_time, route_edges, path_edges, edge_reservations)


def _transport_event_conflicts(start, travel_time, route_edges, agv_id, instance):
    return (_transport_event_conflicts_fast if _FAST else _transport_event_conflicts_ref)(start, travel_time, route_edges, agv_id, instance)
