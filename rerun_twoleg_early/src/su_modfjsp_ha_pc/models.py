from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


Level = Literal["L0", "L1", "L2", "L3", "L4", "L5"]
Size = Literal["small", "medium"]
Objective = Literal[
    "makespan",
    "tardiness",
    "energy",
    "machine_idle_time",
    "agv_waiting_time",
    "path_conflict_delay",
    "stability",
]


@dataclass(frozen=True)
class Machine:
    machine_id: str
    node_id: str
    energy_rate: float


@dataclass(frozen=True)
class Operation:
    operation_id: str
    job_id: str
    op_index: int
    eligible_machines: tuple[str, ...]
    processing_time: dict[str, int]
    processing_energy: dict[str, float]


@dataclass(frozen=True)
class Job:
    job_id: str
    release_time: int
    due_date: int
    operations: tuple[Operation, ...]

    def replace(self, **changes: object) -> "Job":
        return replace(self, **changes)


@dataclass(frozen=True)
class AGV:
    agv_id: str
    agv_type: str
    speed: float
    battery_capacity: float
    initial_battery: float
    consumption_rate: float
    charging_rate: float
    load_capacity: float
    start_node_id: str


@dataclass(frozen=True)
class PathNode:
    node_id: str
    node_type: str
    x: float
    y: float


@dataclass(frozen=True)
class PathEdge:
    edge_id: str
    from_node: str
    to_node: str
    length: float
    capacity: int
    bidirectional: bool
    blockage_probability: float
    is_bottleneck: bool = False


@dataclass(frozen=True)
class DynamicEvent:
    event_id: str
    event_time: int
    event_type: str
    affected_entity: str
    duration: int
    severity: float
    scenario_id: str


@dataclass(frozen=True)
class Instance:
    instance_id: str
    level: Level
    size: Size
    seed: int
    objectives: tuple[Objective, ...]
    machines: tuple[Machine, ...]
    jobs: tuple[Job, ...]
    agvs: tuple[AGV, ...] = ()
    path_nodes: tuple[PathNode, ...] = ()
    path_edges: tuple[PathEdge, ...] = ()
    events: tuple[DynamicEvent, ...] = ()
    preferences: tuple[tuple[float, ...], ...] = ()

    def replace(self, **changes: object) -> "Instance":
        return replace(self, **changes)
