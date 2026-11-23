from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduledOperation:
    operation_id: str
    job_id: str
    op_index: int
    machine_id: str
    start: int
    end: int
    processing_energy: float


@dataclass(frozen=True)
class TransportMove:
    move_id: str
    job_id: str
    operation_id: str
    agv_id: str
    from_node: str
    to_node: str
    edge_ids: tuple[str, ...]
    start: int
    end: int
    transport_energy: float


@dataclass(frozen=True)
class ScheduleResult:
    algorithm: str
    path_aware: bool
    operations: tuple[ScheduledOperation, ...]
    transports: tuple[TransportMove, ...]
    event_aware: bool = False
