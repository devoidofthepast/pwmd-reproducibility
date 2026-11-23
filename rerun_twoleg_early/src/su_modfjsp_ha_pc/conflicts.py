from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeUsage:
    edge_id: str
    start: int
    end: int


def count_capacity_violations(
    usages: tuple[EdgeUsage, ...],
    edge_capacities: dict[str, int],
) -> int:
    violations = 0
    for edge_id, capacity in edge_capacities.items():
        if capacity <= 0:
            raise ValueError(f"edge {edge_id} has non-positive capacity")
        edge_usages = [usage for usage in usages if usage.edge_id == edge_id]
        for usage in edge_usages:
            if usage.start >= usage.end:
                raise ValueError(f"edge usage on {usage.edge_id} must have start < end")
        time_points = sorted({point for usage in edge_usages for point in (usage.start, usage.end)})
        for left, right in zip(time_points, time_points[1:]):
            if left == right:
                continue
            active = sum(1 for usage in edge_usages if usage.start < right and usage.end > left)
            if active > capacity:
                violations += 1
    return violations
