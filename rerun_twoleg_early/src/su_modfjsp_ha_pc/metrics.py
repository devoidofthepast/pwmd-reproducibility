from __future__ import annotations

import math
from dataclasses import dataclass

from su_modfjsp_ha_pc.conflicts import EdgeUsage, count_capacity_violations
from su_modfjsp_ha_pc.models import Instance
from su_modfjsp_ha_pc.schedule import ScheduleResult


@dataclass(frozen=True)
class FeasibilityReport:
    precedence_violations: int
    machine_overlap_violations: int
    agv_overlap_violations: int
    edge_capacity_violations: int
    machine_unavailability_violations: int
    agv_unavailability_violations: int
    path_event_violations: int

    @property
    def is_feasible(self) -> bool:
        return (
            self.precedence_violations == 0
            and self.machine_overlap_violations == 0
            and self.agv_overlap_violations == 0
            and self.edge_capacity_violations == 0
            and self.machine_unavailability_violations == 0
            and self.agv_unavailability_violations == 0
            and self.path_event_violations == 0
        )

    @property
    def dynamic_event_violations(self) -> int:
        return (
            self.machine_unavailability_violations
            + self.agv_unavailability_violations
            + self.path_event_violations
        )


def evaluate_schedule(instance: Instance, schedule: ScheduleResult) -> dict[str, float]:
    operations_by_job = _operations_by_job(schedule)
    job_completion = {
        job.job_id: max(operation.end for operation in operations_by_job.get(job.job_id, ()))
        for job in instance.jobs
        if operations_by_job.get(job.job_id)
    }
    makespan = max((operation.end for operation in schedule.operations), default=0)
    tardiness = sum(
        max(0, job_completion.get(job.job_id, 0) - job.due_date)
        for job in instance.jobs
    )
    processing_energy = sum(operation.processing_energy for operation in schedule.operations)
    transport_energy = sum(move.transport_energy for move in schedule.transports)
    machine_busy_time = {
        machine.machine_id: sum(
            operation.end - operation.start
            for operation in schedule.operations
            if operation.machine_id == machine.machine_id
        )
        for machine in instance.machines
    }
    machine_idle_time = sum(max(0, makespan - busy_time) for busy_time in machine_busy_time.values())
    feasibility = evaluate_feasibility(instance, schedule)

    return {
        "makespan": float(makespan),
        "tardiness": float(tardiness),
        "energy": round(processing_energy + transport_energy, 6),
        "machine_idle_time": float(machine_idle_time),
        "path_conflict_delay": float(feasibility.edge_capacity_violations + feasibility.path_event_violations),
        "stability": float(feasibility.dynamic_event_violations),
        "precedence_violations": float(feasibility.precedence_violations),
        "machine_overlap_violations": float(feasibility.machine_overlap_violations),
        "agv_overlap_violations": float(feasibility.agv_overlap_violations),
        "edge_capacity_violations": float(feasibility.edge_capacity_violations),
        "machine_unavailability_violations": float(feasibility.machine_unavailability_violations),
        "agv_unavailability_violations": float(feasibility.agv_unavailability_violations),
        "path_event_violations": float(feasibility.path_event_violations),
        "dynamic_event_violations": float(feasibility.dynamic_event_violations),
        "feasible": 1.0 if feasibility.is_feasible else 0.0,
    }


def evaluate_feasibility(instance: Instance, schedule: ScheduleResult) -> FeasibilityReport:
    return FeasibilityReport(
        precedence_violations=_count_precedence_violations(instance, schedule),
        machine_overlap_violations=_count_resource_overlaps(
            ((operation.machine_id, operation.start, operation.end) for operation in schedule.operations)
        ),
        agv_overlap_violations=_count_resource_overlaps(
            ((move.agv_id, move.start, move.end) for move in schedule.transports)
        ),
        edge_capacity_violations=_count_edge_capacity_violations(instance, schedule),
        machine_unavailability_violations=_count_machine_unavailability_violations(instance, schedule),
        agv_unavailability_violations=_count_agv_unavailability_violations(instance, schedule),
        path_event_violations=_count_path_event_violations(instance, schedule),
    )


def nondominated(points: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    front = []
    for candidate in points:
        if not any(_dominates(other, candidate) for other in points if other != candidate):
            front.append(candidate)
    return tuple(sorted(set(front)))


def hypervolume_2d(points: tuple[tuple[float, float], ...], reference: tuple[float, float]) -> float:
    front = sorted(nondominated(points))
    total = 0.0
    best_y = reference[1]
    for x_value, y_value in front:
        if x_value >= reference[0] or y_value >= reference[1]:
            continue
        if y_value < best_y:
            total += (reference[0] - x_value) * (best_y - y_value)
            best_y = y_value
    return round(total, 10)


def igd(points: tuple[tuple[float, ...], ...], reference_front: tuple[tuple[float, ...], ...]) -> float:
    if not points:
        raise ValueError("points must not be empty")
    if not reference_front:
        raise ValueError("reference_front must not be empty")
    total = 0.0
    for reference_point in reference_front:
        total += min(_euclidean(reference_point, point) for point in points)
    return total / len(reference_front)


def _operations_by_job(schedule: ScheduleResult) -> dict[str, tuple]:
    grouped: dict[str, list] = {}
    for operation in schedule.operations:
        grouped.setdefault(operation.job_id, []).append(operation)
    return {
        job_id: tuple(sorted(operations, key=lambda operation: operation.op_index))
        for job_id, operations in grouped.items()
    }


def _count_precedence_violations(instance: Instance, schedule: ScheduleResult) -> int:
    scheduled = {operation.operation_id: operation for operation in schedule.operations}
    violations = 0
    for job in instance.jobs:
        previous_end = job.release_time
        for operation in job.operations:
            scheduled_operation = scheduled.get(operation.operation_id)
            if scheduled_operation is None:
                violations += 1
                continue
            if scheduled_operation.start < previous_end:
                violations += 1
            previous_end = scheduled_operation.end
    return violations


def _count_resource_overlaps(intervals: object) -> int:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for resource_id, start, end in intervals:
        if start >= end:
            grouped.setdefault(resource_id, []).append((start, end))
            continue
        grouped.setdefault(resource_id, []).append((start, end))

    violations = 0
    for resource_intervals in grouped.values():
        ordered = sorted(resource_intervals)
        for (_, previous_end), (current_start, _) in zip(ordered, ordered[1:]):
            if current_start < previous_end:
                violations += 1
    return violations


def _count_edge_capacity_violations(instance: Instance, schedule: ScheduleResult) -> int:
    capacities = {edge.edge_id: edge.capacity for edge in instance.path_edges}
    if not capacities:
        return 0
    usages = tuple(
        EdgeUsage(edge_id=edge_id, start=leg_start, end=leg_end)
        for move, legs in _early_legs(instance, schedule)
        for leg_edges, leg_start, leg_end in legs
        for edge_id in dict.fromkeys(leg_edges)
    )
    return count_capacity_violations(usages, capacities)


def _count_machine_unavailability_violations(instance: Instance, schedule: ScheduleResult) -> int:
    downtime_events = tuple(event for event in instance.events if event.event_type == "machine_breakdown")
    return sum(
        1
        for operation in schedule.operations
        for event in downtime_events
        if operation.machine_id == event.affected_entity
        and _overlaps(operation.start, operation.end, event.event_time, event.event_time + event.duration)
    )


def _count_agv_unavailability_violations(instance: Instance, schedule: ScheduleResult) -> int:
    failure_events = tuple(event for event in instance.events if event.event_type == "agv_failure")
    return sum(
        1
        for move, legs in _early_legs(instance, schedule)
        for leg_edges, leg_start, leg_end in legs
        for event in failure_events
        if move.agv_id == event.affected_entity
        and _overlaps(leg_start, leg_end, event.event_time, event.event_time + event.duration)
    )


def _count_path_event_violations(instance: Instance, schedule: ScheduleResult) -> int:
    path_events = tuple(event for event in instance.events if event.event_type in {"path_blockage", "congestion_spike"})
    return sum(
        1
        for move, legs in _early_legs(instance, schedule)
        for leg_edges, leg_start, leg_end in legs
        for event in path_events
        if event.affected_entity in leg_edges
        and _overlaps(leg_start, leg_end, event.event_time, event.event_time + event.duration)
    )


def _overlaps(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    if right_start >= right_end:
        return False
    return left_start < right_end and right_start < left_end


def _dominates(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return all(left_value <= right_value for left_value, right_value in zip(left, right)) and any(
        left_value < right_value for left_value, right_value in zip(left, right)
    )


def _euclidean(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("points must have the same dimension")
    return math.sqrt(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right)))


def _early_legs(instance: Instance, schedule: ScheduleResult):
    """Per move, its (edges, start, end) legs; the pickup node is the job's previous machine node or the depot."""
    from su_modfjsp_ha_pc.rules import move_leg_usage

    if not instance.agvs:
        return ()
    machine_node = {machine.machine_id: machine.node_id for machine in instance.machines}
    op_machine = {operation.operation_id: operation.machine_id for operation in schedule.operations}
    previous_op = {}
    for job in instance.jobs:
        for index, operation in enumerate(job.operations):
            previous_op[operation.operation_id] = job.operations[index - 1].operation_id if index else None
    depot = instance.agvs[0].start_node_id
    rows = []
    for move in schedule.transports:
        prev = previous_op.get(move.operation_id)
        pickup = depot if prev is None else machine_node[op_machine[prev]]
        rows.append((move, move_leg_usage(instance, move, pickup)))
    return tuple(rows)
