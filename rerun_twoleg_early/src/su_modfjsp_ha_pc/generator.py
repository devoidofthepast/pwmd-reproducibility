from __future__ import annotations

import random

from su_modfjsp_ha_pc.models import AGV, DynamicEvent, Instance, Job, Machine, Operation, PathEdge, PathNode
from su_modfjsp_ha_pc.preferences import simplex_grid
from su_modfjsp_ha_pc.validation import validate_instance


SIZE_PRESETS = {
    "small": {
        "n_jobs": 10,
        "ops_per_job": (3, 5),
        "n_machines": 5,
        "n_agvs": 2,
        "n_path_nodes": 10,
        "preference_resolution": 4,
    },
    "medium": {
        "n_jobs": 20,
        "ops_per_job": (5, 8),
        "n_machines": 10,
        "n_agvs": 5,
        "n_path_nodes": 25,
        "preference_resolution": 5,
    },
}


def generate_instance(level: str, size: str, seed: int) -> Instance:
    if level not in {"L0", "L1", "L2", "L3", "L4", "L5"}:
        raise ValueError("level must be one of L0, L1, L2, L3, L4, L5")
    if size not in SIZE_PRESETS:
        raise ValueError("size must be one of small, medium")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    rng = random.Random(seed)
    preset = SIZE_PRESETS[size]
    node_ids = _machine_node_ids(level, preset["n_machines"])
    machines = _generate_machines(preset["n_machines"], node_ids, rng)
    jobs = _generate_jobs(preset["n_jobs"], preset["ops_per_job"], machines, rng)

    objectives = ("makespan", "tardiness", "energy")
    agvs: tuple[AGV, ...] = ()
    path_nodes: tuple[PathNode, ...] = ()
    path_edges: tuple[PathEdge, ...] = ()
    events: tuple[DynamicEvent, ...] = ()

    if level in {"L2", "L3", "L4", "L5"}:
        path_nodes = _generate_path_nodes(preset["n_path_nodes"])
        machines = _attach_machines_to_path_nodes(machines, path_nodes)
        if level in {"L3", "L4", "L5"}:
            jobs = _inject_l3_bottleneck_pressure(jobs, machines)
        agvs = _generate_agvs(
            preset["n_agvs"],
            path_nodes,
            common_start_node="N0" if level in {"L3", "L4", "L5"} else None,
        )
        path_edges = _generate_path_edges(path_nodes, level)

    if level == "L1":
        objectives = ("makespan", "tardiness", "energy", "machine_idle_time")
        events = _generate_l1_events(jobs, machines)
    if level == "L3":
        objectives = ("makespan", "tardiness", "energy", "path_conflict_delay")
    if level == "L4":
        objectives = ("makespan", "tardiness", "energy", "path_conflict_delay", "stability")
        events = _generate_l4_events(jobs, machines, agvs, path_edges)
    if level == "L5":
        objectives = ("makespan", "tardiness", "energy", "path_conflict_delay", "stability")
        events = _generate_l5_events(jobs, machines, agvs, path_edges)

    preference_resolution = preset["preference_resolution"] + 2 if level == "L5" else preset["preference_resolution"]

    instance = Instance(
        instance_id=f"{level}_{size}_seed{seed:03d}",
        level=level,
        size=size,
        seed=seed,
        objectives=objectives,
        machines=machines,
        jobs=jobs,
        agvs=agvs,
        path_nodes=path_nodes,
        path_edges=path_edges,
        events=events,
        preferences=simplex_grid(len(objectives), preference_resolution),
    )
    validate_instance(instance)
    return instance


def _machine_node_ids(level: str, n_machines: int) -> tuple[str, ...]:
    if level == "L0":
        return tuple(f"LOCAL_M{i}" for i in range(n_machines))
    return tuple(f"N{i}" for i in range(n_machines))


def _generate_machines(n_machines: int, node_ids: tuple[str, ...], rng: random.Random) -> tuple[Machine, ...]:
    machines = []
    for index in range(n_machines):
        machines.append(
            Machine(
                machine_id=f"M{index}",
                node_id=node_ids[index],
                energy_rate=round(rng.uniform(0.8, 1.6), 3),
            )
        )
    return tuple(machines)


def _attach_machines_to_path_nodes(
    machines: tuple[Machine, ...], path_nodes: tuple[PathNode, ...]
) -> tuple[Machine, ...]:
    node_ids = [node.node_id for node in path_nodes]
    attached = []
    for index, machine in enumerate(machines):
        attached.append(
            Machine(
                machine_id=machine.machine_id,
                node_id=node_ids[index % len(node_ids)],
                energy_rate=machine.energy_rate,
            )
        )
    return tuple(attached)


def _generate_jobs(
    n_jobs: int,
    ops_per_job: tuple[int, int],
    machines: tuple[Machine, ...],
    rng: random.Random,
) -> tuple[Job, ...]:
    machine_ids = tuple(machine.machine_id for machine in machines)
    jobs = []
    for job_index in range(n_jobs):
        operation_count = rng.randint(ops_per_job[0], ops_per_job[1])
        operations = []
        total_min_processing = 0
        for op_index in range(operation_count):
            eligibility_count = rng.randint(1, max(1, len(machine_ids) - 1))
            eligible = tuple(sorted(rng.sample(machine_ids, eligibility_count)))
            processing_time = {machine_id: rng.randint(3, 30) for machine_id in eligible}
            processing_energy = {
                machine_id: round(processing_time[machine_id] * _machine_energy_rate(machines, machine_id), 3)
                for machine_id in eligible
            }
            total_min_processing += min(processing_time.values())
            operations.append(
                Operation(
                    operation_id=f"J{job_index}_O{op_index}",
                    job_id=f"J{job_index}",
                    op_index=op_index,
                    eligible_machines=eligible,
                    processing_time=processing_time,
                    processing_energy=processing_energy,
                )
            )
        due_date = total_min_processing + rng.randint(20, 80)
        jobs.append(
            Job(
                job_id=f"J{job_index}",
                release_time=0,
                due_date=due_date,
                operations=tuple(operations),
            )
        )
    return tuple(jobs)


def _inject_l3_bottleneck_pressure(
    jobs: tuple[Job, ...],
    machines: tuple[Machine, ...],
) -> tuple[Job, ...]:
    if len(jobs) < 2 or len(machines) < 2:
        return jobs
    bottleneck_machine = machines[1]
    updated_jobs = []
    for job_index, job in enumerate(jobs):
        if job_index >= 2:
            updated_jobs.append(job)
            continue
        first_operation = job.operations[0]
        processing_time = first_operation.processing_time.get(bottleneck_machine.machine_id, 10)
        processing_energy = round(processing_time * bottleneck_machine.energy_rate, 3)
        pressured_operation = Operation(
            operation_id=first_operation.operation_id,
            job_id=first_operation.job_id,
            op_index=first_operation.op_index,
            eligible_machines=(bottleneck_machine.machine_id,),
            processing_time={bottleneck_machine.machine_id: processing_time},
            processing_energy={bottleneck_machine.machine_id: processing_energy},
        )
        updated_jobs.append(
            Job(
                job_id=job.job_id,
                release_time=job.release_time,
                due_date=job.due_date,
                operations=(pressured_operation,) + job.operations[1:],
            )
        )
    return tuple(updated_jobs)


def _generate_l1_events(
    jobs: tuple[Job, ...],
    machines: tuple[Machine, ...],
) -> tuple[DynamicEvent, ...]:
    affected_machine = machines[0].machine_id
    return (
        DynamicEvent(
            event_id="EV0",
            event_time=10,
            event_type="new_job",
            affected_entity=f"future_job:J{len(jobs)}",
            duration=0,
            severity=0.3,
            scenario_id="dynamic-basic",
        ),
        DynamicEvent(
            event_id="EV1",
            event_time=20,
            event_type="machine_breakdown",
            affected_entity=affected_machine,
            duration=10,
            severity=0.7,
            scenario_id="dynamic-basic",
        ),
        DynamicEvent(
            event_id="EV2",
            event_time=30,
            event_type="machine_recovery",
            affected_entity=affected_machine,
            duration=0,
            severity=0.0,
            scenario_id="dynamic-basic",
        ),
    )


def _generate_l4_events(
    jobs: tuple[Job, ...],
    machines: tuple[Machine, ...],
    agvs: tuple[AGV, ...],
    path_edges: tuple[PathEdge, ...],
) -> tuple[DynamicEvent, ...]:
    first_operation = jobs[0].operations[0].operation_id
    second_operation = jobs[1].operations[0].operation_id
    bottleneck_edge = next((edge.edge_id for edge in path_edges if edge.is_bottleneck), path_edges[0].edge_id)
    events = list(_generate_l1_events(jobs, machines))
    events.extend(
        (
            DynamicEvent("EV3", 35, "route_branch", first_operation, 0, 0.5, "structural-basic"),
            DynamicEvent("EV4", 42, "quality_rework", second_operation, 8, 0.6, "structural-basic"),
            DynamicEvent("EV5", 50, "path_blockage", bottleneck_edge, 12, 0.8, "structural-basic"),
            DynamicEvent("EV6", 65, "agv_failure", agvs[0].agv_id, 15, 0.7, "structural-basic"),
            DynamicEvent("EV7", 80, "congestion_spike", bottleneck_edge, 20, 0.6, "structural-basic"),
        )
    )
    return tuple(events)


def _generate_l5_events(
    jobs: tuple[Job, ...],
    machines: tuple[Machine, ...],
    agvs: tuple[AGV, ...],
    path_edges: tuple[PathEdge, ...],
) -> tuple[DynamicEvent, ...]:
    events = list(_generate_l4_events(jobs, machines, agvs, path_edges))
    bottleneck_edge = next((edge.edge_id for edge in path_edges if edge.is_bottleneck), path_edges[0].edge_id)
    alternate_edge = path_edges[min(1, len(path_edges) - 1)].edge_id
    second_machine = machines[min(1, len(machines) - 1)].machine_id
    second_agv = agvs[min(1, len(agvs) - 1)].agv_id
    route_operation = jobs[min(2, len(jobs) - 1)].operations[0].operation_id
    rework_operation = jobs[min(3, len(jobs) - 1)].operations[0].operation_id
    next_event_index = len(events)
    events.extend(
        (
            DynamicEvent(
                f"EV{next_event_index}",
                18,
                "new_job",
                f"future_job:J{len(jobs) + 1}",
                0,
                0.5,
                "parametric-high-arrival",
            ),
            DynamicEvent(
                f"EV{next_event_index + 1}",
                28,
                "machine_breakdown",
                second_machine,
                18,
                0.9,
                "parametric-high-breakdown",
            ),
            DynamicEvent(
                f"EV{next_event_index + 2}",
                52,
                "machine_recovery",
                second_machine,
                0,
                0.0,
                "parametric-high-breakdown",
            ),
            DynamicEvent(
                f"EV{next_event_index + 3}",
                58,
                "path_blockage",
                alternate_edge,
                20,
                0.9,
                "parametric-high-congestion",
            ),
            DynamicEvent(
                f"EV{next_event_index + 4}",
                72,
                "agv_failure",
                second_agv,
                18,
                0.8,
                "parametric-high-agv",
            ),
            DynamicEvent(
                f"EV{next_event_index + 5}",
                88,
                "congestion_spike",
                bottleneck_edge,
                30,
                0.9,
                "parametric-high-congestion",
            ),
            DynamicEvent(
                f"EV{next_event_index + 6}",
                96,
                "route_branch",
                route_operation,
                0,
                0.7,
                "parametric-route-shift",
            ),
            DynamicEvent(
                f"EV{next_event_index + 7}",
                108,
                "quality_rework",
                rework_operation,
                12,
                0.8,
                "parametric-route-shift",
            ),
        )
    )
    return tuple(events)


def _machine_energy_rate(machines: tuple[Machine, ...], machine_id: str) -> float:
    for machine in machines:
        if machine.machine_id == machine_id:
            return machine.energy_rate
    raise ValueError(f"unknown machine {machine_id}")


def _generate_path_nodes(n_path_nodes: int) -> tuple[PathNode, ...]:
    width = 5
    nodes = []
    for index in range(n_path_nodes):
        row = index // width
        column = index % width
        nodes.append(
            PathNode(
                node_id=f"N{index}",
                node_type="machine" if index < width else "junction",
                x=float(column),
                y=float(row),
            )
        )
    return tuple(nodes)


def _generate_agvs(
    n_agvs: int,
    path_nodes: tuple[PathNode, ...],
    common_start_node: str | None = None,
) -> tuple[AGV, ...]:
    templates = (
        ("fast-light", 2.0, 80.0, 70.0, 1.20, 4.0, 1.0),
        ("slow-heavy", 0.8, 140.0, 120.0, 0.80, 3.0, 2.0),
        ("balanced", 1.3, 110.0, 100.0, 1.00, 3.5, 1.5),
    )
    agvs = []
    for index in range(n_agvs):
        agv_type, speed, capacity, initial, consumption, charging, load = templates[index % len(templates)]
        agvs.append(
            AGV(
                agv_id=f"A{index}",
                agv_type=agv_type,
                speed=speed,
                battery_capacity=capacity,
                initial_battery=initial,
                consumption_rate=consumption,
                charging_rate=charging,
                load_capacity=load,
                start_node_id=common_start_node or path_nodes[index % len(path_nodes)].node_id,
            )
        )
    return tuple(agvs)


def _generate_path_edges(path_nodes: tuple[PathNode, ...], level: str) -> tuple[PathEdge, ...]:
    edges = []
    for index in range(len(path_nodes) - 1):
        edges.append(
            PathEdge(
                edge_id=f"E{index}",
                from_node=path_nodes[index].node_id,
                to_node=path_nodes[index + 1].node_id,
                length=10.0 + index,
                capacity=1 if level in {"L3", "L4", "L5"} and index == 0 else 2,
                bidirectional=True,
                blockage_probability=0.05 if level in {"L3", "L4", "L5"} and index == 0 else 0.01,
                is_bottleneck=level in {"L3", "L4", "L5"} and index == 0,
            )
        )

    edge_index = len(edges)
    for index in range(0, len(path_nodes) - 5, 5):
        edges.append(
            PathEdge(
                edge_id=f"E{edge_index}",
                from_node=path_nodes[index].node_id,
                to_node=path_nodes[index + 5].node_id,
                length=12.0,
                capacity=2,
                bidirectional=True,
                blockage_probability=0.01,
                is_bottleneck=False,
            )
        )
        edge_index += 1
    return tuple(edges)
