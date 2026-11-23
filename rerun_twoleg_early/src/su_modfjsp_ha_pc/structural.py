from __future__ import annotations

import math
import random
from dataclasses import dataclass

from su_modfjsp_ha_pc.models import DynamicEvent, Instance, Job, Machine, Operation
from su_modfjsp_ha_pc.validation import validate_instance


@dataclass(frozen=True)
class StructuralRealizationReport:
    original_jobs: int
    realized_jobs: int
    original_operations: int
    realized_operations: int
    new_jobs: int
    rework_operations: int
    route_branches: int
    affected_events: tuple[str, ...]

    @property
    def added_jobs(self) -> int:
        return self.realized_jobs - self.original_jobs

    @property
    def added_operations(self) -> int:
        return self.realized_operations - self.original_operations

    @property
    def total_updates(self) -> int:
        return self.new_jobs + self.rework_operations + self.route_branches


@dataclass(frozen=True)
class StructuralRealization:
    instance: Instance
    report: StructuralRealizationReport


def empty_structural_report(instance: Instance) -> StructuralRealizationReport:
    operation_count = _operation_count(instance.jobs)
    return StructuralRealizationReport(
        original_jobs=len(instance.jobs),
        realized_jobs=len(instance.jobs),
        original_operations=operation_count,
        realized_operations=operation_count,
        new_jobs=0,
        rework_operations=0,
        route_branches=0,
        affected_events=(),
    )


def realize_structural_events(
    instance: Instance, *, validate_result: bool = True
) -> StructuralRealization:
    jobs = list(instance.jobs)
    original_jobs = len(jobs)
    original_operations = _operation_count(jobs)
    new_jobs = 0
    rework_operations = 0
    route_branches = 0
    affected_events: list[str] = []

    machine_by_id = {machine.machine_id: machine for machine in instance.machines}

    for event in sorted(instance.events, key=lambda item: (item.event_time, item.event_id)):
        if event.event_type == "route_branch":
            updated_jobs, changed = _apply_route_branch(jobs, event, instance.machines, instance.seed)
            jobs = updated_jobs
            if changed:
                route_branches += 1
                affected_events.append(event.event_id)
        elif event.event_type == "quality_rework":
            updated_jobs, changed = _apply_quality_rework(jobs, event, machine_by_id)
            jobs = updated_jobs
            if changed:
                rework_operations += 1
                affected_events.append(event.event_id)
        elif event.event_type == "new_job":
            job_id = _future_job_id(event)
            if not any(job.job_id == job_id for job in jobs):
                jobs.append(_new_job_from_event(job_id, event, instance.machines, instance.seed))
                new_jobs += 1
                affected_events.append(event.event_id)

    realized = instance.replace(
        instance_id=f"{instance.instance_id}_realized",
        jobs=tuple(sorted(jobs, key=lambda job: _job_sort_key(job.job_id))),
    )
    if validate_result:
        validate_instance(realized)
    report = StructuralRealizationReport(
        original_jobs=original_jobs,
        realized_jobs=len(realized.jobs),
        original_operations=original_operations,
        realized_operations=_operation_count(realized.jobs),
        new_jobs=new_jobs,
        rework_operations=rework_operations,
        route_branches=route_branches,
        affected_events=tuple(affected_events),
    )
    return StructuralRealization(instance=realized, report=report)


def _apply_route_branch(
    jobs: list[Job],
    event: DynamicEvent,
    machines: tuple[Machine, ...],
    seed: int,
) -> tuple[list[Job], bool]:
    changed = False
    updated_jobs = []
    all_machine_ids = tuple(machine.machine_id for machine in machines)
    machine_by_id = {machine.machine_id: machine for machine in machines}

    for job in jobs:
        operations = []
        for operation in job.operations:
            if operation.operation_id != event.affected_entity:
                operations.append(operation)
                continue
            alternatives = tuple(machine_id for machine_id in all_machine_ids if machine_id not in operation.eligible_machines)
            if alternatives:
                alternate = alternatives[_stable_index(event, seed, len(alternatives))]
                base_processing = min(operation.processing_time.values())
                processing_time = max(1, int(math.ceil(base_processing * (1.0 + event.severity))))
                operations.append(
                    Operation(
                        operation_id=operation.operation_id,
                        job_id=operation.job_id,
                        op_index=operation.op_index,
                        eligible_machines=(alternate,),
                        processing_time={alternate: processing_time},
                        processing_energy={alternate: round(processing_time * machine_by_id[alternate].energy_rate, 3)},
                    )
                )
            else:
                operations.append(
                    Operation(
                        operation_id=operation.operation_id,
                        job_id=operation.job_id,
                        op_index=operation.op_index,
                        eligible_machines=operation.eligible_machines,
                        processing_time={
                            machine_id: max(1, int(math.ceil(value * (1.0 + event.severity))))
                            for machine_id, value in operation.processing_time.items()
                        },
                        processing_energy={
                            machine_id: round(
                                max(1, int(math.ceil(operation.processing_time[machine_id] * (1.0 + event.severity))))
                                * machine_by_id[machine_id].energy_rate,
                                3,
                            )
                            for machine_id in operation.eligible_machines
                        },
                    )
                )
            changed = True
        updated_jobs.append(job if not changed or tuple(operations) == job.operations else job.replace(operations=tuple(operations)))
    return updated_jobs, changed


def _apply_quality_rework(
    jobs: list[Job],
    event: DynamicEvent,
    machine_by_id: dict[str, Machine],
) -> tuple[list[Job], bool]:
    updated_jobs = []
    changed = False
    for job in jobs:
        operations: list[Operation] = []
        inserted = False
        for operation in job.operations:
            operations.append(operation)
            if operation.operation_id != event.affected_entity:
                continue
            duration = event.duration or max(1, int(math.ceil(min(operation.processing_time.values()) * event.severity)))
            rework = Operation(
                operation_id=f"{job.job_id}_RW_{event.event_id}",
                job_id=job.job_id,
                op_index=operation.op_index + 1,
                eligible_machines=operation.eligible_machines,
                processing_time={machine_id: duration for machine_id in operation.eligible_machines},
                processing_energy={
                    machine_id: round(duration * machine_by_id[machine_id].energy_rate, 3)
                    for machine_id in operation.eligible_machines
                },
            )
            operations.append(rework)
            inserted = True
            changed = True
        if inserted:
            updated_jobs.append(_reindex_job(job, operations))
        else:
            updated_jobs.append(job)
    return updated_jobs, changed


def _new_job_from_event(
    job_id: str,
    event: DynamicEvent,
    machines: tuple[Machine, ...],
    seed: int,
) -> Job:
    rng = random.Random(seed + _stable_text_value(event.event_id) + int(event.event_time))
    operation_count = 1 + int(event.severity >= 0.5)
    operations = []
    total_min_processing = 0
    for op_index in range(operation_count):
        eligibility_count = min(max(1, 1 + int(event.severity >= 0.5)), len(machines))
        machine_ids = tuple(machine.machine_id for machine in rng.sample(list(machines), eligibility_count))
        processing_time = {}
        processing_energy = {}
        for machine_id in sorted(machine_ids):
            machine = next(machine for machine in machines if machine.machine_id == machine_id)
            duration = rng.randint(5, 14) + int(math.ceil(event.severity * 8))
            processing_time[machine_id] = duration
            processing_energy[machine_id] = round(duration * machine.energy_rate, 3)
        total_min_processing += min(processing_time.values())
        operations.append(
            Operation(
                operation_id=f"{job_id}_O{op_index}",
                job_id=job_id,
                op_index=op_index,
                eligible_machines=tuple(sorted(machine_ids)),
                processing_time=processing_time,
                processing_energy=processing_energy,
            )
        )
    due_date = event.event_time + total_min_processing + 30 + int(math.ceil((1.0 - event.severity) * 20))
    return Job(
        job_id=job_id,
        release_time=event.event_time,
        due_date=due_date,
        operations=tuple(operations),
    )


def _future_job_id(event: DynamicEvent) -> str:
    if ":" in event.affected_entity:
        return event.affected_entity.split(":", 1)[1]
    return event.affected_entity


def _reindex_job(job: Job, operations: list[Operation]) -> Job:
    reindexed = []
    for index, operation in enumerate(operations):
        reindexed.append(
            Operation(
                operation_id=operation.operation_id,
                job_id=operation.job_id,
                op_index=index,
                eligible_machines=operation.eligible_machines,
                processing_time=operation.processing_time,
                processing_energy=operation.processing_energy,
            )
        )
    return job.replace(operations=tuple(reindexed))


def _operation_count(jobs: object) -> int:
    return sum(len(job.operations) for job in jobs)


def _stable_index(event: DynamicEvent, seed: int, modulo: int) -> int:
    return (seed + _stable_text_value(event.event_id) + _stable_text_value(event.affected_entity)) % modulo


def _stable_text_value(text: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(text))


def _job_sort_key(job_id: str) -> tuple[int, str]:
    if len(job_id) > 1 and job_id[0] == "J" and job_id[1:].isdigit():
        return int(job_id[1:]), job_id
    return 10**9, job_id
