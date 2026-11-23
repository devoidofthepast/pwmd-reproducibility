from __future__ import annotations

from su_modfjsp_ha_pc.path_conflict_neural_dataset import _operation_stage_lookahead


DETERMINISTIC_RULES = ("lookahead", "spt", "edd", "fifo", "mwkr")


def select_operation_by_rule(
    *,
    rule: str,
    instance: object,
    available: tuple[tuple[object, object], ...],
    machine_available: dict[str, int],
    job_ready: dict[str, int],
    agv_available: dict[str, int],
    agv_node: dict[str, str],
    edge_reservations: dict[str, list[tuple[int, int]]],
) -> tuple[object, object]:
    if rule not in DETERMINISTIC_RULES:
        raise ValueError(f"unsupported deterministic dispatch rule: {rule}")
    if not available:
        raise ValueError("no operation is available")
    if rule == "lookahead":
        return min(
            available,
            key=lambda item: (
                _operation_stage_lookahead(
                    instance=instance,
                    job=item[0],
                    operation=item[1],
                    machine_available=machine_available,
                    job_ready_time=job_ready[item[0].job_id],
                    agv_available=agv_available,
                    agv_node=agv_node,
                    edge_reservations=edge_reservations,
                )["score"],
                item[1].operation_id,
            ),
        )
    if rule == "spt":
        return min(
            available,
            key=lambda item: (
                min(item[1].processing_time.values()),
                item[1].operation_id,
            ),
        )
    if rule == "edd":
        return min(
            available,
            key=lambda item: (
                float(getattr(item[0], "due_date", 0)),
                item[1].operation_id,
            ),
        )
    if rule == "fifo":
        return min(
            available,
            key=lambda item: (
                float(getattr(item[0], "release_time", 0)),
                item[0].job_id,
                item[1].operation_id,
            ),
        )

    def remaining_work(item: tuple[object, object]) -> float:
        job, operation = item
        index = job.operations.index(operation)
        return float(
            sum(
                min(candidate.processing_time.values())
                for candidate in job.operations[index:]
            )
        )

    return min(
        available,
        key=lambda item: (-remaining_work(item), item[1].operation_id),
    )
