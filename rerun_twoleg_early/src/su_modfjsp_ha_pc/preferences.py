from __future__ import annotations


def simplex_grid(n_objectives: int, resolution: int) -> tuple[tuple[float, ...], ...]:
    if n_objectives < 2:
        raise ValueError("n_objectives must be at least 2")
    if resolution < 1:
        raise ValueError("resolution must be at least 1")

    integer_vectors: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...], remaining_objectives: int, remaining_mass: int) -> None:
        if remaining_objectives == 1:
            integer_vectors.append(prefix + (remaining_mass,))
            return
        for mass in range(remaining_mass + 1):
            visit(prefix + (mass,), remaining_objectives - 1, remaining_mass - mass)

    visit((), n_objectives, resolution)
    vectors = {
        tuple(round(component / resolution, 10) for component in integer_vector)
        for integer_vector in integer_vectors
    }

    balanced_weight = round(1.0 / n_objectives, 10)
    vectors.add(tuple(balanced_weight for _ in range(n_objectives)))

    for objective_index in range(n_objectives):
        vector = [0.0] * n_objectives
        vector[objective_index] = 1.0
        vectors.add(tuple(vector))

    return tuple(sorted(vectors))
