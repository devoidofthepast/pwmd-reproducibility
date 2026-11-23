from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from su_modfjsp_ha_pc.generator import (
    _generate_agvs,
    _generate_l1_events,
    _generate_l4_events,
    _generate_l5_events,
    _generate_path_edges,
    _generate_path_nodes,
    _inject_l3_bottleneck_pressure,
)
from su_modfjsp_ha_pc.models import Instance, Job, Machine, Operation
from su_modfjsp_ha_pc.preferences import simplex_grid
from su_modfjsp_ha_pc.validation import validate_instance


@dataclass(frozen=True)
class PublicFJSPCore:
    source_name: str
    n_jobs: int
    n_machines: int
    jobs: tuple[Job, ...]


def load_public_fjsp_core(path: str | Path, *, source_name: str | None = None) -> PublicFJSPCore:
    """Load a Brandimarte/Hurink/FJSPLib-style FJSP instance.

    Expected format:
    first line: n_jobs n_machines [optional metadata]
    each job: n_operations, followed by per-operation blocks
    operation block: n_eligible, then n_eligible pairs of machine_id processing_time.

    Machine ids may be zero- or one-based and are converted to M0, M1, ...
    """

    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path} does not contain an FJSP header")
    header = lines[0].replace("\t", " ").split()
    if len(header) < 2:
        raise ValueError(f"{path} does not contain an FJSP header")
    n_jobs = int(float(header[0]))
    n_machines = int(float(header[1]))
    integers = _body_integers(lines[1:])
    machine_index_base = _infer_machine_index_base(
        integers,
        n_jobs=n_jobs,
        n_machines=n_machines,
    )
    cursor = 0
    if n_jobs <= 0 or n_machines <= 0:
        raise ValueError(f"{path} has invalid job or machine count")

    machine_rates = _machine_energy_rates(n_machines)
    jobs: list[Job] = []
    for job_index in range(n_jobs):
        if cursor >= len(integers):
            raise ValueError(f"{path} ended before job {job_index}")
        operation_count = integers[cursor]
        cursor += 1
        operations: list[Operation] = []
        total_min_processing = 0
        for op_index in range(operation_count):
            if cursor >= len(integers):
                raise ValueError(f"{path} ended before job {job_index} operation {op_index}")
            eligible_count = integers[cursor]
            cursor += 1
            if eligible_count <= 0:
                raise ValueError(f"{path} has operation with no eligible machine")
            processing_time: dict[str, int] = {}
            processing_energy: dict[str, float] = {}
            for _ in range(eligible_count):
                if cursor + 1 >= len(integers):
                    raise ValueError(f"{path} has incomplete machine-time pair")
                machine_number = integers[cursor]
                duration = integers[cursor + 1]
                cursor += 2
                machine_index = machine_number - machine_index_base
                if not 0 <= machine_index < n_machines:
                    raise ValueError(
                        f"{path} uses machine id {machine_number} outside its "
                        f"detected base-{machine_index_base} range"
                    )
                if duration <= 0:
                    raise ValueError(f"{path} uses non-positive processing time {duration}")
                machine_id = f"M{machine_index}"
                processing_time[machine_id] = duration
                processing_energy[machine_id] = round(duration * machine_rates[machine_id], 3)
            total_min_processing += min(processing_time.values())
            operations.append(
                Operation(
                    operation_id=f"J{job_index}_O{op_index}",
                    job_id=f"J{job_index}",
                    op_index=op_index,
                    eligible_machines=tuple(sorted(processing_time)),
                    processing_time=processing_time,
                    processing_energy=processing_energy,
                )
            )
        jobs.append(
            Job(
                job_id=f"J{job_index}",
                release_time=0,
                due_date=max(1, int(round(total_min_processing * 1.5))),
                operations=tuple(operations),
            )
        )
    return PublicFJSPCore(
        source_name=source_name or path.stem,
        n_jobs=n_jobs,
        n_machines=n_machines,
        jobs=tuple(jobs),
    )


def extend_public_fjsp_core(
    core: PublicFJSPCore,
    *,
    level: str,
    seed: int,
    n_agvs: int | None = None,
    preference_resolution: int = 4,
    inject_bottleneck_pressure: bool = False,
) -> Instance:
    if level not in {"L0", "L1", "L2", "L3", "L4", "L5"}:
        raise ValueError("level must be one of L0, L1, L2, L3, L4, L5")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if preference_resolution <= 0:
        raise ValueError("preference_resolution must be positive")

    machine_rates = _machine_energy_rates(core.n_machines)
    node_count = max(core.n_machines, 10)
    node_ids = tuple(f"LOCAL_M{index}" for index in range(core.n_machines))
    machines = tuple(
        Machine(machine_id=f"M{index}", node_id=node_ids[index], energy_rate=machine_rates[f"M{index}"])
        for index in range(core.n_machines)
    )
    jobs = core.jobs
    objectives = ("makespan", "tardiness", "energy")
    agvs = ()
    path_nodes = ()
    path_edges = ()
    events = ()

    if level in {"L2", "L3", "L4", "L5"}:
        path_nodes = _generate_path_nodes(node_count)
        machines = tuple(
            Machine(machine_id=machine.machine_id, node_id=path_nodes[index].node_id, energy_rate=machine.energy_rate)
            for index, machine in enumerate(machines)
        )
        if inject_bottleneck_pressure and level in {"L3", "L4", "L5"}:
            jobs = _inject_l3_bottleneck_pressure(jobs, machines)
        agv_count = n_agvs if n_agvs is not None else max(2, min(5, core.n_machines // 3))
        agvs = _generate_agvs(agv_count, path_nodes, common_start_node="N0" if level in {"L3", "L4", "L5"} else None)
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

    resolution = preference_resolution + 2 if level == "L5" else preference_resolution
    instance = Instance(
        instance_id=f"{level}_public_{_safe_name(core.source_name)}_seed{seed:03d}",
        level=level,
        size="small" if core.n_jobs <= 10 and core.n_machines <= 10 else "medium",
        seed=seed,
        objectives=objectives,
        machines=machines,
        jobs=jobs,
        agvs=agvs,
        path_nodes=path_nodes,
        path_edges=path_edges,
        events=events,
        preferences=simplex_grid(len(objectives), resolution),
    )
    validate_instance(instance)
    return instance


def load_and_extend_public_fjsp(
    path: str | Path,
    *,
    level: str,
    seed: int,
    n_agvs: int | None = None,
    preference_resolution: int = 4,
    inject_bottleneck_pressure: bool = False,
) -> Instance:
    core = load_public_fjsp_core(path)
    return extend_public_fjsp_core(
        core,
        level=level,
        seed=seed,
        n_agvs=n_agvs,
        preference_resolution=preference_resolution,
        inject_bottleneck_pressure=inject_bottleneck_pressure,
    )


def _body_integers(lines: list[str]) -> tuple[int, ...]:
    values: list[int] = []
    for token in " ".join(lines).replace("\t", " ").split():
        try:
            values.append(int(token))
        except ValueError:
            continue
    return tuple(values)


def _infer_machine_index_base(
    integers: tuple[int, ...],
    *,
    n_jobs: int,
    n_machines: int,
) -> int:
    """Infer the two common public formats without confusing counts for ids."""

    cursor = 0
    machine_numbers = []
    for _ in range(n_jobs):
        if cursor >= len(integers):
            raise ValueError("public FJSP body ended before all jobs")
        operation_count = integers[cursor]
        cursor += 1
        for _ in range(operation_count):
            if cursor >= len(integers):
                raise ValueError("public FJSP body ended before all operations")
            eligible_count = integers[cursor]
            cursor += 1
            for _ in range(eligible_count):
                if cursor + 1 >= len(integers):
                    raise ValueError("public FJSP body has an incomplete pair")
                machine_numbers.append(integers[cursor])
                cursor += 2
    if not machine_numbers:
        raise ValueError("public FJSP body contains no machine assignments")
    if 0 in machine_numbers:
        if any(number < 0 or number >= n_machines for number in machine_numbers):
            raise ValueError("mixed or invalid zero-based machine ids")
        return 0
    if any(number < 1 or number > n_machines for number in machine_numbers):
        raise ValueError("invalid one-based machine ids")
    return 1


def _machine_energy_rates(n_machines: int) -> dict[str, float]:
    return {f"M{index}": round(0.8 + 0.05 * (index % 9), 3) for index in range(n_machines)}


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
