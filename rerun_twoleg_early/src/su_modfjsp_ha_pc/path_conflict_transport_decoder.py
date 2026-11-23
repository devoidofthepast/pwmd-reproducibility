from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from su_modfjsp_ha_pc.rules import (
    _available_operations,
    _first_resource_event_feasible_start,
    _first_transport_feasible_start,
    _shortest_path_edges,
)
from su_modfjsp_ha_pc.schedule import ScheduleResult, ScheduledOperation, TransportMove
from su_modfjsp_ha_pc.transport_action_repair_diagnostic import TransportActionGenome


@dataclass(frozen=True)
class PathConflictTransportDecoderConfig:
    wait_weight: float = 1.0
    event_risk_weight: float = 4.0
    bottleneck_weight: float = 1.5
    energy_weight: float = 0.02
    anchor_bonus: float = 0.25
    include_bottleneck_avoidance: bool = True
    include_event_avoidance: bool = True
    enumerate_agvs: bool = True
    shortest_route_only: bool = False
    honor_transport_start_anchor: bool = True


def decode_path_conflict_aware_transport_genome(
    instance: object,
    operations: tuple[object, ...],
    genome: TransportActionGenome,
    *,
    algorithm: str,
    config: PathConflictTransportDecoderConfig | None = None,
    repair_process_log: list[dict[str, object]] | None = None,
    repair_process_context: dict[str, object] | None = None,
) -> ScheduleResult:
    resolved = config or PathConflictTransportDecoderConfig()
    operation_index = {operation.operation_id: index for index, operation in enumerate(operations)}
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
        job, operation = min(
            available,
            key=lambda item: (
                _operation_priority(genome, operation_index[item[1].operation_id]),
                item[1].operation_id,
            ),
        )
        machine_id = _select_machine(operation, operation_index[operation.operation_id], genome)
        machine = machine_by_id[machine_id]
        transport_end = job_ready[job.job_id]
        if instance.agvs and instance.path_edges:
            move = _schedule_conflict_aware_transport(
                instance=instance,
                job=job,
                operation=operation,
                destination=machine,
                genome=genome,
                operation_index=operation_index,
                agv_available=agv_available,
                agv_node=agv_node,
                edge_reservations=edge_reservations,
                move_index=len(transport_moves),
                config=resolved,
                repair_process_log=repair_process_log,
                repair_process_context=repair_process_context,
            )
            transport_moves.append(move)
            transport_end = move.end

        target_start = _operation_start_anchor(genome, operation_index[operation.operation_id])
        start = max(job_ready[job.job_id], machine_available[machine_id], transport_end, target_start)
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
        path_aware=True,
        operations=tuple(scheduled_operations),
        transports=tuple(transport_moves),
        event_aware=True,
    )


def _candidate_routes_for_transport_impl(
    path_edges: tuple[object, ...],
    *,
    source: str,
    target: str,
    avoid_edges: frozenset[str] = frozenset(),
    include_bottleneck_avoidance: bool = True,
    shortest_route_only: bool = False,
) -> tuple[tuple[str, ...], ...]:
    routes: list[tuple[str, ...]] = []
    base = _safe_shortest_path(path_edges, source, target)
    if base is not None:
        routes.append(base)
    if shortest_route_only:
        return tuple(routes)
    if avoid_edges:
        avoiding = _shortest_path_avoiding(path_edges, source=source, target=target, avoid_edges=avoid_edges)
        if avoiding is not None:
            routes.append(avoiding)
    if include_bottleneck_avoidance:
        bottlenecks = frozenset(edge.edge_id for edge in path_edges if bool(getattr(edge, "is_bottleneck", False)))
        if bottlenecks:
            avoiding = _shortest_path_avoiding(path_edges, source=source, target=target, avoid_edges=bottlenecks)
            if avoiding is not None:
                routes.append(avoiding)
    deduped = []
    seen = set()
    for route in routes:
        if route not in seen:
            seen.add(route)
            deduped.append(route)
    return tuple(deduped)


# ---- two-leg transport (empty + loaded), 2026-11-09 ----
JOB_KEY = "__job__:"


def job_pickup_node(instance, agv_node, job_id):
    node = agv_node.get(JOB_KEY + str(job_id))
    if node is None:
        node = instance.agvs[0].start_node_id  # load/unload station (common AGV start node N0 at L4/L5)
    return node


def empty_leg_route(path_edges, source, pickup, avoid_edges=frozenset()):
    if source == pickup:
        return ()
    route = None
    if avoid_edges:
        route = _shortest_path_avoiding(path_edges, source=source, target=pickup, avoid_edges=avoid_edges)
    if route is None:
        route = _safe_shortest_path(path_edges, source, pickup)
    return route


def two_leg_routes(path_edges, *, source, pickup, target, avoid_edges=frozenset(), **kwargs):
    empty = empty_leg_route(path_edges, source, pickup, avoid_edges)
    if empty is None:
        return ()
    loaded = candidate_routes_for_transport(path_edges, source=pickup, target=target, avoid_edges=avoid_edges, **kwargs)
    return tuple(tuple(empty) + tuple(route) for route in loaded)


def two_leg_safe(path_edges, source, pickup, target):
    empty = empty_leg_route(path_edges, source, pickup)
    loaded = _safe_shortest_path(path_edges, pickup, target)
    if empty is None or loaded is None:
        return None
    return tuple(empty) + tuple(loaded)


def _schedule_conflict_aware_transport(
    *,
    instance: object,
    job: object,
    operation: object,
    destination: object,
    genome: TransportActionGenome,
    operation_index: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
    move_index: int,
    config: PathConflictTransportDecoderConfig,
    repair_process_log: list[dict[str, object]] | None,
    repair_process_context: dict[str, object] | None,
) -> TransportMove:
    edge_by_id = {edge.edge_id: edge for edge in instance.path_edges}
    index = operation_index[operation.operation_id]
    anchored_agv = _transport_agv_choice(genome, index)
    anchored_route = _transport_route_edges(genome, index)
    start_anchor = _transport_start_anchor(genome, index) if config.honor_transport_start_anchor else 0
    candidates = []
    agv_candidates = tuple(instance.agvs)
    if not config.enumerate_agvs and anchored_agv:
        anchored = tuple(agv for agv in instance.agvs if agv.agv_id == anchored_agv)
        if anchored:
            agv_candidates = anchored
    for agv in agv_candidates:
        origin = agv_node[agv.agv_id]
        avoid_edges = _event_edges_active_after(instance, earliest=max(job.release_time, agv_available[agv.agv_id]))
        routes = candidate_routes_for_transport(
            instance.path_edges,
            source=origin,
            target=destination.node_id,
            avoid_edges=avoid_edges if config.include_event_avoidance else frozenset(),
            include_bottleneck_avoidance=config.include_bottleneck_avoidance,
            shortest_route_only=config.shortest_route_only,
        )
        if anchored_route and agv.agv_id == anchored_agv and _route_connects(instance.path_edges, anchored_route, origin, destination.node_id):
            routes = (anchored_route,) + tuple(route for route in routes if route != anchored_route)
        for route in routes:
            route_length = sum(edge_by_id[edge_id].length for edge_id in route if edge_id in edge_by_id)
            travel_time = max(1, math.ceil(route_length / agv.speed))
            earliest_start = max(job.release_time, agv_available[agv.agv_id], start_anchor)
            final_start = _first_transport_feasible_start(
                earliest_start=earliest_start,
                travel_time=travel_time,
                route_edges=route,
                agv_id=agv.agv_id,
                instance=instance,
                edge_reservations=edge_reservations,
                check_capacity=True,
                check_events=True,
            )
            score = _transport_score(
                instance=instance,
                route=route,
                agv_id=agv.agv_id,
                start=final_start,
                travel_time=travel_time,
                earliest_start=earliest_start,
                route_length=route_length,
                anchored_agv=anchored_agv,
                anchored_route=anchored_route,
                config=config,
            )
            candidates.append((score, final_start, agv.agv_id, route, route_length, travel_time))
    if not candidates:
        raise ValueError(f"no transport candidate for {operation.operation_id}")
    _score, start, agv_id, route_edges, route_length, travel_time = min(candidates, key=lambda item: item[:4])
    origin_node = agv_node[agv_id]
    start = _first_transport_feasible_start(
        earliest_start=start,
        travel_time=travel_time,
        route_edges=route_edges,
        agv_id=agv_id,
        instance=instance,
        edge_reservations=edge_reservations,
        check_capacity=True,
        check_events=True,
        repair_process_log=repair_process_log,
        repair_process_context=repair_process_context,
        subject={
            "job_id": job.job_id,
            "operation_id": operation.operation_id,
            "machine_id": destination.machine_id,
            "move_id": f"T{move_index}",
            "agv_id": agv_id,
            "route_edges": ";".join(route_edges),
        },
    )
    end = start + travel_time
    for edge_id in dict.fromkeys(route_edges):
        edge_reservations.setdefault(edge_id, []).append((start, end))
    agv_available[agv_id] = end
    agv_node[agv_id] = destination.node_id
    agv = next(item for item in instance.agvs if item.agv_id == agv_id)
    return TransportMove(
        move_id=f"T{move_index}",
        job_id=job.job_id,
        operation_id=operation.operation_id,
        agv_id=agv_id,
        from_node=origin_node,
        to_node=destination.node_id,
        edge_ids=route_edges,
        start=start,
        end=end,
        transport_energy=round(route_length * agv.consumption_rate, 6),
    )


def _transport_score(
    *,
    instance: object,
    route: tuple[str, ...],
    agv_id: str,
    start: int,
    travel_time: int,
    earliest_start: int,
    route_length: float,
    anchored_agv: str,
    anchored_route: tuple[str, ...],
    config: PathConflictTransportDecoderConfig,
) -> float:
    end = start + travel_time
    wait = start - earliest_start
    event_risk = _event_overlap_duration(instance, route, agv_id, start, end)
    bottlenecks = sum(1 for edge in instance.path_edges if edge.edge_id in route and bool(getattr(edge, "is_bottleneck", False)))
    anchor_bonus = 0.0
    if anchored_agv and agv_id == anchored_agv:
        anchor_bonus += config.anchor_bonus
    if anchored_route and route == anchored_route:
        anchor_bonus += config.anchor_bonus
    return (
        end
        + config.wait_weight * wait
        + config.event_risk_weight * event_risk
        + config.bottleneck_weight * bottlenecks
        + config.energy_weight * route_length
        - anchor_bonus
    )


def _select_machine(operation: object, operation_index: int, genome: TransportActionGenome) -> str:
    target = _machine_choice_index(genome, operation_index, len(operation.eligible_machines))
    return min(
        operation.eligible_machines,
        key=lambda machine_id: (
            abs(operation.eligible_machines.index(machine_id) - target),
            machine_id,
        ),
    )


def _event_edges_active_after(instance: object, *, earliest: int) -> frozenset[str]:
    edges = set()
    for event in instance.events:
        if event.event_type not in {"path_blockage", "congestion_spike"} or event.duration <= 0:
            continue
        if event.event_time + event.duration >= earliest:
            edges.add(event.affected_entity)
    return frozenset(edges)


def _event_overlap_duration(instance: object, route: tuple[str, ...], agv_id: str, start: int, end: int) -> int:
    overlap = 0
    for event in instance.events:
        if event.duration <= 0:
            continue
        event_start = int(event.event_time)
        event_end = int(event.event_time + event.duration)
        if not _overlaps(start, end, event_start, event_end):
            continue
        if event.event_type == "agv_failure" and event.affected_entity == agv_id:
            overlap += event_end - event_start
        if event.event_type in {"path_blockage", "congestion_spike"} and event.affected_entity in route:
            overlap += event_end - event_start
    return overlap


def _safe_shortest_path(path_edges: tuple[object, ...], source: str, target: str) -> tuple[str, ...] | None:
    try:
        return _shortest_path_edges(path_edges, source, target)
    except ValueError:
        return None


def _shortest_path_avoiding_impl(
    path_edges: tuple[object, ...],
    *,
    source: str,
    target: str,
    avoid_edges: frozenset[str],
) -> tuple[str, ...] | None:
    if source == target:
        return ()
    adjacency: dict[str, list[tuple[float, str, str]]] = {}
    for edge in path_edges:
        if edge.edge_id in avoid_edges:
            continue
        adjacency.setdefault(edge.from_node, []).append((float(edge.length), edge.to_node, edge.edge_id))
        if edge.bidirectional:
            adjacency.setdefault(edge.to_node, []).append((float(edge.length), edge.from_node, edge.edge_id))
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
    return None


def _route_connects(path_edges: tuple[object, ...], route_edges: tuple[str, ...], source: str, target: str) -> bool:
    if not route_edges:
        return source == target
    edges = {edge.edge_id: edge for edge in path_edges}
    node = source
    for edge_id in route_edges:
        edge = edges.get(edge_id)
        if edge is None:
            return False
        if edge.from_node == node:
            node = edge.to_node
            continue
        if edge.bidirectional and edge.to_node == node:
            node = edge.from_node
            continue
        return False
    return node == target


def _operation_priority(genome: TransportActionGenome, index: int) -> float:
    if index >= len(genome.operation_priority):
        return 1.0
    return genome.operation_priority[index]


def _machine_choice_index(genome: TransportActionGenome, index: int, machine_count: int) -> int:
    if machine_count <= 1:
        return 0
    value = genome.operation_machine_choice[index] if index < len(genome.operation_machine_choice) else 0.0
    return min(machine_count - 1, max(0, round(value * (machine_count - 1))))


def _operation_start_anchor(genome: TransportActionGenome, index: int) -> int:
    if index >= len(genome.operation_start_anchor):
        return 0
    return max(0, int(round(genome.operation_start_anchor[index])))


def _transport_agv_choice(genome: TransportActionGenome, index: int) -> str:
    if index >= len(genome.transport_agv_choice):
        return ""
    return genome.transport_agv_choice[index]


def _transport_start_anchor(genome: TransportActionGenome, index: int) -> int:
    if index >= len(genome.transport_start_anchor):
        return 0
    return max(0, int(round(genome.transport_start_anchor[index])))


def _transport_route_edges(genome: TransportActionGenome, index: int) -> tuple[str, ...]:
    if index >= len(genome.transport_route_edges):
        return ()
    return tuple(genome.transport_route_edges[index])


def _overlaps(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


# ---- exact memoization (2026-11-09, early_fast) ----
from su_modfjsp_ha_pc.rules import _memo_table  # noqa: E402


def candidate_routes_for_transport(path_edges, *, source, target, avoid_edges=frozenset(),
                                   include_bottleneck_avoidance=True, shortest_route_only=False):
    if not isinstance(path_edges, tuple):
        return _candidate_routes_for_transport_impl(path_edges, source=source, target=target, avoid_edges=avoid_edges,
                                                    include_bottleneck_avoidance=include_bottleneck_avoidance,
                                                    shortest_route_only=shortest_route_only)
    table = _memo_table(path_edges, "cand")
    key = (source, target, frozenset(avoid_edges), bool(include_bottleneck_avoidance), bool(shortest_route_only))
    if key not in table:
        table[key] = _candidate_routes_for_transport_impl(path_edges, source=source, target=target, avoid_edges=avoid_edges,
                                                          include_bottleneck_avoidance=include_bottleneck_avoidance,
                                                          shortest_route_only=shortest_route_only)
    return table[key]


def _shortest_path_avoiding(path_edges, *, source, target, avoid_edges):
    if not isinstance(path_edges, tuple):
        return _shortest_path_avoiding_impl(path_edges, source=source, target=target, avoid_edges=avoid_edges)
    table = _memo_table(path_edges, "avoid")
    key = (source, target, frozenset(avoid_edges))
    if key not in table:
        table[key] = _shortest_path_avoiding_impl(path_edges, source=source, target=target, avoid_edges=avoid_edges)
    return table[key]
