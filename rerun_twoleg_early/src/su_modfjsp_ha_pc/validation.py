from __future__ import annotations

from collections import deque

from su_modfjsp_ha_pc.models import Instance


def validate_instance(instance: Instance) -> None:
    machine_ids = {machine.machine_id for machine in instance.machines}
    node_ids = {node.node_id for node in instance.path_nodes}

    if not instance.machines:
        raise ValueError("instance must include at least one machine")
    if not instance.jobs:
        raise ValueError("instance must include at least one job")
    if not instance.objectives:
        raise ValueError("instance must include at least one objective")

    for machine in instance.machines:
        if machine.energy_rate <= 0:
            raise ValueError(f"machine {machine.machine_id} has non-positive energy rate")
        if instance.level in {"L2", "L3", "L4", "L5"} and machine.node_id not in node_ids:
            raise ValueError(f"machine {machine.machine_id} references unknown node {machine.node_id}")

    for job in instance.jobs:
        if job.release_time < 0:
            raise ValueError(f"job {job.job_id} has negative release time")
        if job.due_date < job.release_time:
            raise ValueError(f"job {job.job_id} due date is earlier than release time")
        if not job.operations:
            raise ValueError(f"job {job.job_id} has no operations")
        op_indices = [operation.op_index for operation in job.operations]
        if op_indices != list(range(len(job.operations))):
            raise ValueError(f"job {job.job_id} operation indices must be contiguous from 0")
        for operation in job.operations:
            if operation.job_id != job.job_id:
                raise ValueError(f"operation {operation.operation_id} has mismatched job id")
            if not operation.eligible_machines:
                raise ValueError(f"operation {operation.operation_id} has no eligible machines")
            for machine_id in operation.eligible_machines:
                if machine_id not in machine_ids:
                    raise ValueError(
                        f"operation {operation.operation_id} references unknown machine {machine_id}"
                    )
                if machine_id not in operation.processing_time:
                    raise ValueError(
                        f"operation {operation.operation_id} lacks processing time for machine {machine_id}"
                    )
                if machine_id not in operation.processing_energy:
                    raise ValueError(
                        f"operation {operation.operation_id} lacks processing energy for machine {machine_id}"
                    )
                if operation.processing_time[machine_id] <= 0:
                    raise ValueError(
                        f"operation {operation.operation_id} has non-positive processing time on {machine_id}"
                    )
                if operation.processing_energy[machine_id] <= 0:
                    raise ValueError(
                        f"operation {operation.operation_id} has non-positive processing energy on {machine_id}"
                    )

    for preference in instance.preferences:
        if len(preference) != len(instance.objectives):
            raise ValueError("preference vector length must match objective count")
        if any(weight < 0 for weight in preference):
            raise ValueError("preference weights must be non-negative")
        if abs(sum(preference) - 1.0) > 1e-9:
            raise ValueError("preference weights must sum to 1")

    _validate_events(instance)

    if instance.level in {"L2", "L3", "L4", "L5"}:
        _validate_transport_layer(instance, node_ids)

    if instance.level in {"L3", "L4", "L5"}:
        if "path_conflict_delay" not in instance.objectives:
            raise ValueError(f"{instance.level} instances must include path_conflict_delay objective")
        if not any(edge.is_bottleneck for edge in instance.path_edges):
            raise ValueError(f"{instance.level} instances must include at least one bottleneck path edge")

    if instance.level == "L1":
        required = {"new_job", "machine_breakdown", "machine_recovery"}
        present = {event.event_type for event in instance.events}
        if not required <= present:
            raise ValueError("L1 instances must include new_job, machine_breakdown, and machine_recovery events")

    if instance.level in {"L4", "L5"}:
        required = {"route_branch", "quality_rework", "path_blockage", "agv_failure", "congestion_spike"}
        present = {event.event_type for event in instance.events}
        if not required <= present:
            raise ValueError(f"{instance.level} instances must include route/path/AGV structural uncertainty events")

    if instance.level == "L5":
        scenario_ids = {event.scenario_id for event in instance.events}
        if not any(scenario_id.startswith("parametric-") for scenario_id in scenario_ids):
            raise ValueError("L5 instances must include parametric structural event scenarios")
        if len(instance.preferences) < 100:
            raise ValueError("L5 instances must include a dense preference grid")


def _validate_events(instance: Instance) -> None:
    allowed = {
        "new_job",
        "machine_breakdown",
        "machine_recovery",
        "agv_failure",
        "path_blockage",
        "congestion_spike",
        "quality_rework",
        "route_branch",
    }
    for event in instance.events:
        if event.event_type not in allowed:
            raise ValueError(f"event {event.event_id} has unknown type {event.event_type}")
        if event.event_time < 0:
            raise ValueError(f"event {event.event_id} has negative event time")
        if event.duration < 0:
            raise ValueError(f"event {event.event_id} has negative duration")
        if not 0 <= event.severity <= 1:
            raise ValueError(f"event {event.event_id} severity must be within [0, 1]")
        if not event.affected_entity:
            raise ValueError(f"event {event.event_id} has empty affected entity")


def _validate_transport_layer(instance: Instance, node_ids: set[str]) -> None:
    if not instance.agvs:
        raise ValueError(f"{instance.level} instances must include AGVs")
    if not instance.path_nodes:
        raise ValueError(f"{instance.level} instances must include path nodes")
    if not instance.path_edges:
        raise ValueError(f"{instance.level} instances must include path edges")

    for agv in instance.agvs:
        if agv.speed <= 0:
            raise ValueError(f"AGV {agv.agv_id} has non-positive speed")
        if agv.battery_capacity <= 0:
            raise ValueError(f"AGV {agv.agv_id} has non-positive battery capacity")
        if not 0 < agv.initial_battery <= agv.battery_capacity:
            raise ValueError(f"AGV {agv.agv_id} initial battery must be within capacity")
        if agv.consumption_rate <= 0:
            raise ValueError(f"AGV {agv.agv_id} has non-positive consumption rate")
        if agv.charging_rate <= 0:
            raise ValueError(f"AGV {agv.agv_id} has non-positive charging rate")
        if agv.load_capacity <= 0:
            raise ValueError(f"AGV {agv.agv_id} has non-positive load capacity")
        if agv.start_node_id not in node_ids:
            raise ValueError(f"AGV {agv.agv_id} references unknown start node {agv.start_node_id}")

    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in instance.path_edges:
        if edge.from_node not in node_ids:
            raise ValueError(f"path edge {edge.edge_id} references unknown node {edge.from_node}")
        if edge.to_node not in node_ids:
            raise ValueError(f"path edge {edge.edge_id} references unknown node {edge.to_node}")
        if edge.length <= 0:
            raise ValueError(f"path edge {edge.edge_id} has non-positive length")
        if edge.capacity <= 0:
            raise ValueError(f"path edge {edge.edge_id} has non-positive capacity")
        if not 0 <= edge.blockage_probability <= 1:
            raise ValueError(f"path edge {edge.edge_id} blockage probability must be within [0, 1]")
        adjacency[edge.from_node].add(edge.to_node)
        if edge.bidirectional:
            adjacency[edge.to_node].add(edge.from_node)

    start = next(iter(node_ids))
    visited = {start}
    queue: deque[str] = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    if visited != node_ids:
        raise ValueError("path graph must be connected")
