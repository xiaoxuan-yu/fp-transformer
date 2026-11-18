"""Numerical primitives for the Tiwary--Berne transport analysis.

1. a reversible finite-volume (square-root approximation) reference for the
   linear Fokker--Planck flow, propagated by matrix-exponential action or by
   implicit Euler;
2. the continuum probability-flow velocity reconstructed from the
   finite-volume edge current, for the numerical continuum oracle; and
3. coarse-grid exact discrete quadratic optimal transport (W_2).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse.linalg import expm_multiply
from scipy.sparse.linalg import factorized
from scipy.spatial.distance import cdist

from examples.tb.common import TB_CENTERS, TB_DEPTHS



@dataclass(frozen=True)
class TBGrid:
    """Uniform cell-centered grid and Tiwary--Berne fields."""

    x: np.ndarray
    y: np.ndarray
    energy: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    target_mass: np.ndarray
    beta: float

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def dy(self) -> float:
        return float(self.y[1] - self.y[0])

    @property
    def shape(self) -> tuple[int, int]:
        return self.energy.shape


def tb_energy_and_gradient(
    x_coordinate: np.ndarray,
    y_coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the Tiwary--Berne energy and its analytic gradient."""

    points = np.stack((x_coordinate, y_coordinate), axis=-1)
    displacement = points[..., None, :] - TB_CENTERS
    squared = np.sum(np.square(displacement), axis=-1)
    exponential = np.exp(-2.0 * squared)
    energy = (
        20.0
        - np.sum(TB_DEPTHS * exponential, axis=-1)
        + 0.5 * (np.power(x_coordinate, 6) + np.power(y_coordinate, 6))
    )
    well_gradient = np.sum(
        4.0 * (TB_DEPTHS * exponential)[..., None] * displacement,
        axis=-2,
    )
    gradient_x = well_gradient[..., 0] + 3.0 * np.power(x_coordinate, 5)
    gradient_y = well_gradient[..., 1] + 3.0 * np.power(y_coordinate, 5)
    return energy, gradient_x, gradient_y


def make_tb_grid(
    *,
    beta: float,
    resolution: int,
    x_bounds: tuple[float, float] = (-1.45, 1.55),
    y_bounds: tuple[float, float] = (-1.15, 1.85),
) -> TBGrid:
    """Construct the common cell-centered grid and discrete target mass."""

    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if resolution < 16:
        raise ValueError("resolution must be at least 16")
    x = np.linspace(*x_bounds, resolution)
    y = np.linspace(*y_bounds, resolution)
    xx, yy = np.meshgrid(x, y)
    energy, gradient_x, gradient_y = tb_energy_and_gradient(xx, yy)
    target = np.exp(-beta * (energy - float(np.min(energy))))
    target /= np.sum(target)
    return TBGrid(
        x=x,
        y=y,
        energy=energy,
        gradient_x=gradient_x,
        gradient_y=gradient_y,
        target_mass=target,
        beta=float(beta),
    )


def gaussian_initial_mass(
    grid: TBGrid,
    *,
    center: tuple[float, float],
    standard_deviation: float,
) -> np.ndarray:
    """Return a smooth, strictly positive non-equilibrium initial mass."""

    if standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be positive")
    xx, yy = np.meshgrid(grid.x, grid.y)
    squared = np.square(xx - center[0]) + np.square(yy - center[1])
    mass = np.exp(-0.5 * squared / standard_deviation**2)
    mass /= np.sum(mass)
    return mass


def reversible_fp_generator(grid: TBGrid) -> sparse.csr_matrix:
    """Build a nearest-neighbor reversible Fokker--Planck generator.

    The jump rate from cell i to a neighboring cell j is

        r(i -> j) = delta^{-2} exp[-beta (U_j - U_i) / 2].

    The resulting column generator conserves probability and has the discrete
    Boltzmann mass as an exact stationary distribution.
    """

    ny, nx = grid.shape
    cell_count = nx * ny
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    diagonal = np.zeros(cell_count, dtype=float)
    flat_energy = grid.energy.ravel()

    def add_edges(source: np.ndarray, target: np.ndarray, spacing: float) -> None:
        difference = flat_energy[target] - flat_energy[source]
        forward = np.exp(-0.5 * grid.beta * difference) / spacing**2
        backward = np.exp(0.5 * grid.beta * difference) / spacing**2
        rows.extend((target, source))
        columns.extend((source, target))
        values.extend((forward, backward))
        np.add.at(diagonal, source, -forward)
        np.add.at(diagonal, target, -backward)

    index = np.arange(cell_count, dtype=int).reshape(ny, nx)
    add_edges(index[:, :-1].ravel(), index[:, 1:].ravel(), grid.dx)
    add_edges(index[:-1, :].ravel(), index[1:, :].ravel(), grid.dy)
    diagonal_index = np.arange(cell_count, dtype=int)
    rows.append(diagonal_index)
    columns.append(diagonal_index)
    values.append(diagonal)
    generator = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(cell_count, cell_count),
    ).tocsr()
    return generator


def exact_fp_snapshots(
    generator: sparse.csr_matrix,
    initial_mass: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Propagate the reversible grid reference at evenly spaced times."""

    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must contain at least two checkpoints")
    if not np.allclose(np.diff(times), np.diff(times)[0], atol=1e-13, rtol=0.0):
        raise ValueError("expm_multiply reference currently requires even checkpoints")
    if not math.isclose(float(times[0]), 0.0, abs_tol=1e-14):
        raise ValueError("times must start at zero")
    propagated = expm_multiply(
        generator,
        np.asarray(initial_mass, dtype=float).ravel(),
        start=float(times[0]),
        stop=float(times[-1]),
        num=int(times.size),
        endpoint=True,
    )
    snapshots = np.asarray(propagated, dtype=float).reshape(
        times.size,
        *initial_mass.shape,
    )
    snapshots = np.maximum(snapshots, 0.0)
    snapshots /= np.sum(snapshots, axis=(1, 2), keepdims=True)
    return snapshots


def implicit_fp_snapshots(
    generator: sparse.csr_matrix,
    initial_mass: np.ndarray,
    times: np.ndarray,
    *,
    output_memmap_path: str | None = None,
) -> np.ndarray:
    """Propagate a finite-volume mass by stable implicit Euler increments.

    The routine is intended for dense oracle time grids.  When increments
    repeat, sparse factorizations of ``I - dt Q`` are cached and reused.
    """

    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must contain at least two checkpoints")
    if not np.isclose(times[0], 0.0) or np.any(np.diff(times) <= 0.0):
        raise ValueError("times must start at zero and be strictly increasing")
    initial = np.asarray(initial_mass, dtype=float).ravel()
    if output_memmap_path is None:
        propagated = np.empty((times.size, initial.size), dtype=float)
    else:
        propagated = np.memmap(
            output_memmap_path,
            mode="w+",
            dtype=np.float64,
            shape=(times.size, initial.size),
        )
    propagated[0] = initial
    identity = sparse.eye(generator.shape[0], format="csc")
    solvers: dict[float, Callable[[np.ndarray], np.ndarray]] = {}
    current = initial
    for index, step in enumerate(np.diff(times), start=1):
        key = round(float(step), 14)
        solver = solvers.get(key)
        if solver is None:
            solver = factorized((identity - float(step) * generator).tocsc())
            solvers[key] = solver
        current = np.asarray(solver(current), dtype=float)
        current = np.maximum(current, 0.0)
        current /= np.sum(current)
        propagated[index] = current
    return propagated.reshape(times.size, *initial_mass.shape)


def finite_volume_probability_flow_velocity(
    mass: np.ndarray,
    *,
    grid: TBGrid,
    relative_floor: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a cell velocity reconstructed from finite-volume edge currents.

    For an edge from cell ``i`` to ``j``, the antisymmetric probability
    current is ``J_ij = r(i->j) m_i - r(j->i) m_j``.  Dividing by the
    arithmetic edge mass gives a velocity whose centered divergence reproduces
    the reference redistribution up to the interpolation used for particles.
    """

    mass = np.asarray(mass, dtype=float)
    if mass.shape != grid.shape:
        raise ValueError("mass shape must match the probability-flow grid")
    floor = max(relative_floor * float(np.max(mass)), np.finfo(float).tiny)
    velocity_x_edge = np.zeros((grid.shape[0], grid.shape[1] - 1), dtype=float)
    velocity_y_edge = np.zeros((grid.shape[0] - 1, grid.shape[1]), dtype=float)

    energy_difference_x = grid.energy[:, 1:] - grid.energy[:, :-1]
    forward_x = np.exp(-0.5 * grid.beta * energy_difference_x) / grid.dx**2
    backward_x = np.exp(0.5 * grid.beta * energy_difference_x) / grid.dx**2
    current_x = forward_x * mass[:, :-1] - backward_x * mass[:, 1:]
    edge_mass_x = 0.5 * (mass[:, :-1] + mass[:, 1:])
    velocity_x_edge = grid.dx * current_x / np.maximum(edge_mass_x, floor)

    energy_difference_y = grid.energy[1:, :] - grid.energy[:-1, :]
    forward_y = np.exp(-0.5 * grid.beta * energy_difference_y) / grid.dy**2
    backward_y = np.exp(0.5 * grid.beta * energy_difference_y) / grid.dy**2
    current_y = forward_y * mass[:-1, :] - backward_y * mass[1:, :]
    edge_mass_y = 0.5 * (mass[:-1, :] + mass[1:, :])
    velocity_y_edge = grid.dy * current_y / np.maximum(edge_mass_y, floor)

    velocity_x = np.zeros_like(mass)
    velocity_y = np.zeros_like(mass)
    velocity_x[:, 1:-1] = 0.5 * (
        velocity_x_edge[:, :-1] + velocity_x_edge[:, 1:]
    )
    velocity_x[:, 0] = velocity_x_edge[:, 0]
    velocity_x[:, -1] = velocity_x_edge[:, -1]
    velocity_y[1:-1, :] = 0.5 * (
        velocity_y_edge[:-1, :] + velocity_y_edge[1:, :]
    )
    velocity_y[0, :] = velocity_y_edge[0, :]
    velocity_y[-1, :] = velocity_y_edge[-1, :]
    velocity_x[:, 0] = 0.0
    velocity_x[:, -1] = 0.0
    velocity_y[0, :] = 0.0
    velocity_y[-1, :] = 0.0
    return velocity_x, velocity_y


def interpolate_vector_field(
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    *,
    grid: TBGrid,
    coordinate: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate a two-component field on a uniform grid."""

    coordinate = np.asarray(coordinate, dtype=float)
    if coordinate.ndim != 2 or coordinate.shape[1] != 2:
        raise ValueError("coordinate must have shape (N, 2)")
    if velocity_x.shape != grid.shape or velocity_y.shape != grid.shape:
        raise ValueError("velocity fields must match the grid")
    scaled_x = (coordinate[:, 0] - grid.x[0]) / grid.dx
    scaled_y = (coordinate[:, 1] - grid.y[0]) / grid.dy
    lower_x = np.floor(scaled_x).astype(int)
    lower_y = np.floor(scaled_y).astype(int)
    lower_x = np.clip(lower_x, 0, grid.x.size - 2)
    lower_y = np.clip(lower_y, 0, grid.y.size - 2)
    fraction_x = np.clip(scaled_x - lower_x, 0.0, 1.0)
    fraction_y = np.clip(scaled_y - lower_y, 0.0, 1.0)

    def interpolate(field: np.ndarray) -> np.ndarray:
        lower_left = field[lower_y, lower_x]
        lower_right = field[lower_y, lower_x + 1]
        upper_left = field[lower_y + 1, lower_x]
        upper_right = field[lower_y + 1, lower_x + 1]
        lower = lower_left + fraction_x * (lower_right - lower_left)
        upper = upper_left + fraction_x * (upper_right - upper_left)
        return lower + fraction_y * (upper - lower)

    return np.column_stack((interpolate(velocity_x), interpolate(velocity_y)))


def coarsen_mass(
    mass: np.ndarray,
    *,
    grid: TBGrid,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Block-coarsen a grid mass and return cell-center support."""

    ny, nx = grid.shape
    if ny % bins != 0 or nx % bins != 0:
        raise ValueError("OT bins must divide both grid dimensions")
    block_y = ny // bins
    block_x = nx // bins
    coarse = mass.reshape(bins, block_y, bins, block_x).sum(axis=(1, 3))
    coarse_x = grid.x.reshape(bins, block_x).mean(axis=1)
    coarse_y = grid.y.reshape(bins, block_y).mean(axis=1)
    xx, yy = np.meshgrid(coarse_x, coarse_y)
    support = np.column_stack((xx.ravel(), yy.ravel()))
    return coarse.ravel(), support


@lru_cache(maxsize=32)
def _dual_constraint_matrix(source_size: int, target_size: int) -> sparse.coo_array:
    upper = sparse.block_diag(
        (np.ones((1, target_size)),) * source_size,
        format="csr",
    )
    lower = sparse.hstack(
        (sparse.eye(target_size, format="csr"),) * source_size,
        format="csr",
    )
    return sparse.vstack((upper, lower), format="csr").T.tocoo()


def exact_discrete_w2(
    source_mass: np.ndarray,
    source_support: np.ndarray,
    target_mass: np.ndarray,
    target_support: np.ndarray,
    *,
    mass_threshold: float = 1e-12,
    time_limit: float = 120.0,
) -> tuple[float, dict[str, float | int]]:
    """Solve the finite-support quadratic OT dual and return its square root."""

    source_mass = np.asarray(source_mass, dtype=float)
    target_mass = np.asarray(target_mass, dtype=float)
    source_keep = source_mass > mass_threshold
    target_keep = target_mass > mass_threshold
    source_removed = float(np.sum(source_mass[~source_keep]))
    target_removed = float(np.sum(target_mass[~target_keep]))
    source_weight = source_mass[source_keep]
    target_weight = target_mass[target_keep]
    source_weight /= np.sum(source_weight)
    target_weight /= np.sum(target_weight)
    source_coordinate = np.asarray(source_support, dtype=float)[source_keep]
    target_coordinate = np.asarray(target_support, dtype=float)[target_keep]
    source_size = source_weight.size
    target_size = target_weight.size
    cost = np.square(cdist(source_coordinate, target_coordinate)).ravel()
    constraint = _dual_constraint_matrix(source_size, target_size)
    objective_weight = np.concatenate((source_weight, target_weight))
    lower_bound = np.full(objective_weight.size, -np.inf)
    upper_bound = np.full(objective_weight.size, np.inf)
    lower_bound[0] = 0.0
    upper_bound[0] = 0.0
    result = milp(
        c=-objective_weight,
        constraints=LinearConstraint(constraint, lb=-np.inf, ub=cost),
        bounds=Bounds(lower_bound, upper_bound),
        options={"time_limit": float(time_limit)},
    )
    if not result.success or result.fun is None:
        raise RuntimeError(f"Quadratic OT dual failed: {result.message}")
    squared_distance = max(0.0, float(-result.fun))
    return math.sqrt(squared_distance), {
        "source_support": int(source_size),
        "target_support": int(target_size),
        "source_removed_mass": source_removed,
        "target_removed_mass": target_removed,
        "solver_status": int(result.status),
    }


def grid_w2(
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    *,
    grid: TBGrid,
    bins: int,
    mass_threshold: float = 1e-12,
) -> tuple[float, dict[str, float | int]]:
    """Compute exact discrete W2 after common block coarsening."""

    source, support = coarsen_mass(source_mass, grid=grid, bins=bins)
    target, _ = coarsen_mass(target_mass, grid=grid, bins=bins)
    return exact_discrete_w2(
        source,
        support,
        target,
        support,
        mass_threshold=mass_threshold,
    )


def sample_grid_w2(
    samples: np.ndarray,
    target_mass: np.ndarray,
    *,
    grid: TBGrid,
    bins: int,
    mass_threshold: float = 1e-12,
) -> tuple[float, dict[str, float | int]]:
    """Compute discrete W2 from samples to a coarsened grid mass.

    Samples inside the grid are aggregated onto the common coarse support.
    Samples outside the grid remain individual support points, so tail escapes
    contribute their full transport cost rather than being discarded or
    conditionally renormalized.
    """

    samples = np.asarray(samples, dtype=float)
    ny, nx = grid.shape
    if ny % bins != 0 or nx % bins != 0:
        raise ValueError("OT bins must divide both grid dimensions")
    block_y = ny // bins
    block_x = nx // bins
    coarse_x = grid.x.reshape(bins, block_x).mean(axis=1)
    coarse_y = grid.y.reshape(bins, block_y).mean(axis=1)
    dx_coarse = float(np.mean(np.diff(coarse_x)))
    dy_coarse = float(np.mean(np.diff(coarse_y)))
    x_edges = np.concatenate(
        (
            [coarse_x[0] - 0.5 * dx_coarse],
            0.5 * (coarse_x[:-1] + coarse_x[1:]),
            [coarse_x[-1] + 0.5 * dx_coarse],
        )
    )
    y_edges = np.concatenate(
        (
            [coarse_y[0] - 0.5 * dy_coarse],
            0.5 * (coarse_y[:-1] + coarse_y[1:]),
            [coarse_y[-1] + 0.5 * dy_coarse],
        )
    )
    inside = (
        (samples[:, 0] >= x_edges[0])
        & (samples[:, 0] <= x_edges[-1])
        & (samples[:, 1] >= y_edges[0])
        & (samples[:, 1] <= y_edges[-1])
    )
    histogram, _, _ = np.histogram2d(
        samples[inside, 1],
        samples[inside, 0],
        bins=(y_edges, x_edges),
    )
    xx, yy = np.meshgrid(coarse_x, coarse_y)
    common_support = np.column_stack((xx.ravel(), yy.ravel()))
    inside_mass = histogram.ravel() / samples.shape[0]
    occupied = inside_mass > 0.0
    source = inside_mass[occupied]
    support = common_support[occupied]
    if np.any(~inside):
        source = np.concatenate(
            (source, np.full(np.count_nonzero(~inside), 1.0 / samples.shape[0]))
        )
        support = np.vstack((support, samples[~inside]))
    target, target_support = coarsen_mass(target_mass, grid=grid, bins=bins)
    distance, diagnostics = exact_discrete_w2(
        source,
        support,
        target,
        target_support,
        mass_threshold=mass_threshold,
    )
    diagnostics["outside_fraction"] = float(np.mean(~inside))
    diagnostics["outside_count"] = int(np.count_nonzero(~inside))
    return distance, diagnostics


def sample_sample_w2(
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    *,
    grid: TBGrid,
    bins: int,
    mass_threshold: float = 1e-12,
) -> tuple[float, dict[str, float | int]]:
    """Compute discrete W2 between two empirical measures after common binning.

    Inside-grid particles are aggregated on the same coarse support used by
    :func:`sample_grid_w2`; outside particles remain individual sites.
    """

    ny, nx = grid.shape
    if ny % bins != 0 or nx % bins != 0:
        raise ValueError("OT bins must divide both grid dimensions")
    block_y = ny // bins
    block_x = nx // bins
    coarse_x = grid.x.reshape(bins, block_x).mean(axis=1)
    coarse_y = grid.y.reshape(bins, block_y).mean(axis=1)
    dx_coarse = float(np.mean(np.diff(coarse_x)))
    dy_coarse = float(np.mean(np.diff(coarse_y)))
    x_edges = np.concatenate(
        (
            [coarse_x[0] - 0.5 * dx_coarse],
            0.5 * (coarse_x[:-1] + coarse_x[1:]),
            [coarse_x[-1] + 0.5 * dx_coarse],
        )
    )
    y_edges = np.concatenate(
        (
            [coarse_y[0] - 0.5 * dy_coarse],
            0.5 * (coarse_y[:-1] + coarse_y[1:]),
            [coarse_y[-1] + 0.5 * dy_coarse],
        )
    )
    xx, yy = np.meshgrid(coarse_x, coarse_y)
    common_support = np.column_stack((xx.ravel(), yy.ravel()))

    def empirical_measure(samples: np.ndarray):
        samples = np.asarray(samples, dtype=float)
        inside = (
            (samples[:, 0] >= x_edges[0])
            & (samples[:, 0] <= x_edges[-1])
            & (samples[:, 1] >= y_edges[0])
            & (samples[:, 1] <= y_edges[-1])
        )
        histogram, _, _ = np.histogram2d(
            samples[inside, 1],
            samples[inside, 0],
            bins=(y_edges, x_edges),
        )
        inside_mass = histogram.ravel() / samples.shape[0]
        occupied = inside_mass > 0.0
        mass = inside_mass[occupied]
        support = common_support[occupied]
        if np.any(~inside):
            mass = np.concatenate(
                (mass, np.full(np.count_nonzero(~inside), 1.0 / samples.shape[0]))
            )
            support = np.vstack((support, samples[~inside]))
        return mass, support, inside

    source_mass, source_support, source_inside = empirical_measure(source_samples)
    target_mass, target_support, target_inside = empirical_measure(target_samples)
    distance, diagnostics = exact_discrete_w2(
        source_mass,
        source_support,
        target_mass,
        target_support,
        mass_threshold=mass_threshold,
    )
    diagnostics.update(
        {
            "source_outside_fraction": float(np.mean(~source_inside)),
            "target_outside_fraction": float(np.mean(~target_inside)),
        }
    )
    return distance, diagnostics
