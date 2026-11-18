#!/usr/bin/env python3
"""Exact row-mass FPT transport on the Tiwary--Berne potential (main text Fig. 2).

Advances the particle system of the main text by explicit Euler,

    x_i^{n+1} = x_i^n - dt [s_{-i,h_i}(x_i^n) + beta grad U(x_i^n)],

with the pointwise Gaussian leave-one-out score and particlewise bandwidths
refitted at every step so that sum_{j != i} exp(-|x_i - x_j|^2 / 2 h_i^2) = K_N
(core ``solve_row_mass_bandwidth_hybrid``: six bisections followed by ten
safeguarded Newton steps on the bracket [h_0 f_min, h_0 f_max]).  Records
target metrics and score diagnostics at checkpoints, the integrated weak-flux
defects and discrete action over every interval, and all checkpoint samples.

Paper protocol (one run of the batch; see run_batch.py):

    python -m examples.tb.run_transport --output-dir OUT/n2048/seed20260820 \
        --replicas 2048 --seed 20260820 --row-mass 16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from fp_transformer.kernel.row_mass import solve_row_mass_bandwidth_hybrid  # noqa: E402

from examples.tb.common import (  # noqa: E402
    TB_CENTERS,
    TB_GRAD,
    grid_edges,
    metrics,
    tb_exact_grid,
    weighted_cdf_reference,
)

BUMP_WIDTH = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--initial-center", type=float, nargs=2, default=(-0.5, 0.5))
    parser.add_argument("--initial-standard-deviation", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=1.25)
    parser.add_argument("--time-step", type=float, default=2e-4)
    parser.add_argument("--final-time", type=float, default=20.0)
    parser.add_argument("--checkpoint-time", type=float, default=0.5)
    parser.add_argument("--row-mass", type=float, default=16.0, help="target off-diagonal row mass K_N")
    parser.add_argument("--base-bandwidth", type=float, default=0.03,
                        help="h_0; the root bracket is [h_0 f_min, h_0 f_max]")
    parser.add_argument("--bracket-factor-min", type=float, default=0.1)
    parser.add_argument("--bracket-factor-max", type=float, default=40.0)
    parser.add_argument("--fit-iterations", type=int, default=16)
    parser.add_argument("--row-mass-tolerance", type=float, default=1e-3,
                        help="reject the run when the relative root error exceeds this")
    parser.add_argument("--particle-compute-dtype", choices=("float64", "float32"), default="float32",
                        help="precision of pairwise reductions; coordinates and Euler accumulation stay float64")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


@jax.jit
def pairwise_squared_distance(coordinate: jax.Array) -> jax.Array:
    displacement = coordinate[:, None, :] - coordinate[None, :, :]
    return jnp.sum(displacement * displacement, axis=2)


def gaussian_loo_score_from_squared_distance(coordinate, bandwidth, squared_distance):
    """Leave-one-out score s_i = h_i^-2 (sum_j w_ij x_j - x_i) and weights w_ij."""
    bandwidth_squared = jnp.square(jnp.asarray(bandwidth))
    row_bandwidth_squared = bandwidth_squared if bandwidth_squared.ndim == 0 else bandwidth_squared[:, None]
    logits = -0.5 * squared_distance / row_bandwidth_squared
    logits = jnp.where(jnp.eye(coordinate.shape[0], dtype=bool), -jnp.inf, logits)
    weights = jax.nn.softmax(logits, axis=1)
    neighbor_mean = jnp.matmul(weights, coordinate, precision=jax.lax.Precision.HIGHEST)
    return (neighbor_mean - coordinate) / row_bandwidth_squared, weights


def bump_terms(coordinate):
    """Gradient and Laplacian of the three localized tests phi_r."""
    width = jnp.asarray(BUMP_WIDTH, dtype=coordinate.dtype)
    displacement = coordinate[:, None, :] - jnp.asarray(TB_CENTERS, dtype=coordinate.dtype)[None, :, :]
    value = jnp.exp(-0.5 * jnp.sum(jnp.square(displacement), axis=2) / jnp.square(width))
    gradient = -displacement * value[:, :, None] / jnp.square(width)
    laplacian = value * (jnp.sum(jnp.square(displacement), axis=2) / jnp.power(width, 4)
                         - coordinate.shape[1] / jnp.square(width))
    return gradient, laplacian


@partial(jax.jit, static_argnames=("fit_iterations", "particle_compute_dtype", "steps"))
def euler_chunk(coordinate, *, beta, base_bandwidth, factor_min, factor_max, row_mass, fit_iterations,
                particle_compute_dtype, time_step, steps):
    """Advance ``steps`` explicit Euler updates; returns the state and interval diagnostics."""

    def step(carry, _):
        (current, maximum_speed, maximum_update, minimum_bandwidth, maximum_bandwidth, action,
         linear_defect, quadratic_defect, bump_defect, saturation, root_error) = carry
        particle = current.astype(jnp.float32 if particle_compute_dtype == "float32" else current.dtype)
        squared_distance = pairwise_squared_distance(particle)
        bandwidth, fitted_mass, lower_saturated, upper_saturated = solve_row_mass_bandwidth_hybrid(
            squared_distance, base_bandwidth=base_bandwidth, selector_factor_min=factor_min,
            selector_factor_max=factor_max, target_kernel_mass=row_mass, fit_iterations=fit_iterations)
        step_saturation = jnp.any(lower_saturated | upper_saturated)
        step_root_error = jnp.max(jnp.abs(fitted_mass - row_mass)) / row_mass
        score, _ = gaussian_loo_score_from_squared_distance(particle, bandwidth, squared_distance)
        score = score.astype(current.dtype)
        bandwidth = bandwidth.astype(current.dtype)
        entropy_velocity = -score
        velocity = entropy_velocity - beta * TB_GRAD(current)
        step_linear = jnp.linalg.norm(jnp.mean(entropy_velocity, axis=0))
        step_quadratic = jnp.abs(jnp.mean(jnp.sum(current * entropy_velocity, axis=1)) - current.shape[1])
        bump_gradient, bump_laplacian = bump_terms(current)
        step_bump = jnp.max(jnp.abs(jnp.mean(
            jnp.sum(bump_gradient * entropy_velocity[:, None, :], axis=2) - bump_laplacian, axis=0)))
        update = time_step * velocity
        return (
            current + update,
            jnp.maximum(maximum_speed, jnp.max(jnp.linalg.norm(velocity, axis=1))),
            jnp.maximum(maximum_update, jnp.max(jnp.linalg.norm(update, axis=1))),
            jnp.minimum(minimum_bandwidth, jnp.min(bandwidth)),
            jnp.maximum(maximum_bandwidth, jnp.max(bandwidth)),
            action + time_step * jnp.mean(jnp.sum(jnp.square(velocity), axis=1)),
            linear_defect + time_step * step_linear,
            quadratic_defect + time_step * step_quadratic,
            bump_defect + time_step * step_bump,
            saturation | step_saturation,
            jnp.maximum(root_error, step_root_error),
        ), None

    zero = jnp.asarray(0.0, dtype=coordinate.dtype)
    initial = (coordinate, zero, zero, jnp.asarray(jnp.inf, dtype=coordinate.dtype), zero, zero, zero, zero, zero,
               jnp.asarray(False), zero)
    final, _ = jax.lax.scan(step, initial, xs=None, length=steps)
    return final


@jax.jit
def force_diagnostics(coordinate, *, beta, bandwidth):
    """Score stability and weak-generator diagnostics at a checkpoint."""
    squared_distance = pairwise_squared_distance(coordinate)
    score, weights = gaussian_loo_score_from_squared_distance(coordinate, bandwidth, squared_distance)
    potential_term = beta * TB_GRAD(coordinate)
    residual = score + potential_term
    entropy_velocity = -score
    velocity = entropy_velocity - potential_term
    residual_norm = jnp.linalg.norm(residual, axis=1)
    score_norm = jnp.linalg.norm(score, axis=1)
    potential_norm = jnp.linalg.norm(potential_term, axis=1)
    effective_neighbors = jnp.reciprocal(jnp.sum(weights * weights, axis=1))
    self_mask = jnp.eye(coordinate.shape[0], dtype=bool)
    nearest_distance = jnp.sqrt(jnp.min(jnp.where(self_mask, jnp.inf, squared_distance), axis=1))
    bandwidth_array = jnp.asarray(bandwidth)
    bandwidth_squared = jnp.square(bandwidth_array)
    row_bandwidth_squared = bandwidth_squared if bandwidth_squared.ndim == 0 else bandwidth_squared[:, None]
    logits = jnp.where(self_mask, -jnp.inf, -0.5 * squared_distance / row_bandwidth_squared)
    z_squared = squared_distance / row_bandwidth_squared
    displacement = coordinate[:, None, :] - coordinate[None, :, :]
    standardized = (displacement / bandwidth_array if bandwidth_array.ndim == 0
                    else displacement / bandwidth_array[:, None, None])
    radial_second = jnp.sum(weights * z_squared, axis=1)
    odd_vector = jnp.sum(weights[:, :, None] * (z_squared - 2.0)[:, :, None] * standardized, axis=1)
    column_load = jnp.sum(weights, axis=0)
    column_second = jnp.sum(weights * z_squared, axis=0)
    dimensions = coordinate.shape[1]
    bump_gradient, bump_laplacian = bump_terms(coordinate)
    bump_particle_derivative = jnp.mean(jnp.sum(bump_gradient * velocity[:, None, :], axis=2), axis=0)
    bump_target_generator = jnp.mean(
        bump_laplacian - jnp.sum(bump_gradient * potential_term[:, None, :], axis=2), axis=0)
    log_density = (jax.scipy.special.logsumexp(logits, axis=1) - jnp.log(coordinate.shape[0] - 1)
                   - dimensions * jnp.log(bandwidth_array) - 0.5 * dimensions * jnp.log(2.0 * jnp.pi))
    return {
        "residual_rms": jnp.sqrt(jnp.mean(residual_norm**2)),
        "residual_p95": jnp.quantile(residual_norm, 0.95),
        "residual_max": jnp.max(residual_norm),
        "score_rms": jnp.sqrt(jnp.mean(score_norm**2)),
        "potential_rms": jnp.sqrt(jnp.mean(potential_norm**2)),
        "effective_neighbors_min": jnp.min(effective_neighbors),
        "effective_neighbors_median": jnp.median(effective_neighbors),
        "effective_neighbors_p10": jnp.quantile(effective_neighbors, 0.10),
        "radial_second_min": jnp.min(radial_second),
        "radial_second_median": jnp.median(radial_second),
        "column_load_max": jnp.max(column_load),
        "column_second_max": jnp.max(column_second),
        "odd_vector_rms": jnp.sqrt(jnp.mean(jnp.sum(jnp.square(odd_vector), axis=1))),
        "weak_linear_defect": jnp.linalg.norm(jnp.mean(velocity, axis=0)),
        "weak_quadratic_defect": jnp.abs(jnp.mean(jnp.sum(coordinate * velocity, axis=1))),
        "weak_bump_defect_max": jnp.max(jnp.abs(bump_particle_derivative)),
        "weak_target_generator_linear": jnp.linalg.norm(jnp.mean(-potential_term, axis=0)),
        "weak_target_generator_quadratic": jnp.abs(
            jnp.mean(dimensions - jnp.sum(coordinate * potential_term, axis=1))),
        "weak_target_generator_bump_max": jnp.max(jnp.abs(bump_target_generator)),
        "weak_flux_linear_defect": jnp.linalg.norm(jnp.mean(entropy_velocity, axis=0)),
        "weak_flux_quadratic_defect": jnp.abs(
            jnp.mean(jnp.sum(coordinate * entropy_velocity, axis=1)) - dimensions),
        "weak_flux_bump_defect_max": jnp.max(jnp.abs(bump_particle_derivative - bump_target_generator)),
        "nearest_over_bandwidth_max": jnp.max(nearest_distance / bandwidth_array),
        "log_kde_min": jnp.min(log_density),
        "coordinate_radius_max": jnp.max(jnp.linalg.norm(coordinate, axis=1)),
    }


def checkpoint_steps(final_time: float, checkpoint_time: float, time_step: float) -> list[int]:
    total = int(round(final_time / time_step))
    interval = int(round(checkpoint_time / time_step))
    if not math.isclose(total * time_step, final_time, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("--final-time must be an integer multiple of --time-step")
    if not math.isclose(interval * time_step, checkpoint_time, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("--checkpoint-time must be an integer multiple of --time-step")
    steps = list(range(0, total + 1, interval))
    if steps[-1] != total:
        raise ValueError("--final-time must be an integer multiple of --checkpoint-time")
    return steps


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.output_dir} is not empty; choose a new directory or use --force")
    checkpoints = checkpoint_steps(args.final_time, args.checkpoint_time, args.time_step)
    row_mass = min(float(args.row_mass), float(args.replicas - 1))
    fit = dict(base_bandwidth=float(args.base_bandwidth), selector_factor_min=float(args.bracket_factor_min),
               selector_factor_max=float(args.bracket_factor_max), target_kernel_mass=row_mass,
               fit_iterations=int(args.fit_iterations))

    exact = tb_exact_grid(float(args.beta))
    exact_density = np.asarray(exact["density"], dtype=float)
    exact_energy = np.asarray(exact["energy"], dtype=float)
    exact_basin_mass = np.asarray(exact["basin_mass"], dtype=float)
    exact_q, exact_cdf = weighted_cdf_reference(exact_energy, exact_density, float(args.beta))
    reference = dict(beta=float(args.beta), energy_min=float(np.min(exact_energy)), exact_q=exact_q,
                     exact_cdf=exact_cdf, exact_density=exact_density, exact_basin_mass=exact_basin_mass,
                     x_edges=grid_edges(np.asarray(exact["x"], dtype=float)),
                     y_edges=grid_edges(np.asarray(exact["y"], dtype=float)))

    rng = np.random.default_rng(args.seed)
    initial = rng.normal(loc=tuple(float(v) for v in args.initial_center),
                         scale=float(args.initial_standard_deviation), size=(args.replicas, 2))
    coordinate = jnp.asarray(initial, dtype=jnp.float64)
    snapshots = [np.asarray(coordinate)]
    records, interval_records = [], []
    wall_start = time.perf_counter()
    compute_dtype = jnp.float32 if args.particle_compute_dtype == "float32" else jnp.float64

    def record(step: int, sample: np.ndarray) -> None:
        diagnostic = jnp.asarray(sample).astype(compute_dtype)
        bandwidth, fitted_mass, lower_saturated, upper_saturated = solve_row_mass_bandwidth_hybrid(
            pairwise_squared_distance(diagnostic), **fit)
        if bool(np.asarray(jnp.any(lower_saturated | upper_saturated))):
            raise RuntimeError("row-mass root outside the numerical bracket; widen the bracket factors")
        root_error = float(np.asarray(jnp.max(jnp.abs(fitted_mass - row_mass)) / row_mass))
        if root_error > args.row_mass_tolerance:
            raise RuntimeError(f"row-mass inversion exceeded relative tolerance {args.row_mass_tolerance:g}")
        centered = diagnostic - jnp.mean(diagnostic, axis=0, keepdims=True)
        force = {name: float(np.asarray(value))
                 for name, value in force_diagnostics(diagnostic, beta=float(args.beta), bandwidth=bandwidth).items()}
        bumps = {f"bump_observable_{r}": float(np.mean(np.exp(
            -0.5 * np.sum(np.square(sample - np.asarray(TB_CENTERS[r])), axis=1) / BUMP_WIDTH**2)))
            for r in range(len(TB_CENTERS))}
        records.append({
            "step": int(step),
            "time": float(step * args.time_step),
            "bandwidth": float(np.asarray(jnp.median(bandwidth))),
            "bandwidth_min": float(np.asarray(jnp.min(bandwidth))),
            "bandwidth_p05": float(np.asarray(jnp.quantile(bandwidth, 0.05))),
            "bandwidth_median": float(np.asarray(jnp.median(bandwidth))),
            "bandwidth_p95": float(np.asarray(jnp.quantile(bandwidth, 0.95))),
            "bandwidth_max": float(np.asarray(jnp.max(bandwidth))),
            "bandwidth_factor": float(np.asarray(jnp.median(bandwidth / args.base_bandwidth))),
            "ensemble_scale": float(np.asarray(jnp.sqrt(jnp.mean(jnp.square(centered)) + 1e-8**2))),
            "coordinate_mean_x": float(np.mean(sample[:, 0])),
            "coordinate_mean_y": float(np.mean(sample[:, 1])),
            "quadratic_observable": float(0.5 * np.mean(np.sum(np.square(sample), axis=1))),
            **bumps,
            "displacement_rms": float(np.sqrt(np.mean(np.sum(np.square(sample - initial), axis=1)))),
            "displacement_max": float(np.max(np.linalg.norm(sample - initial, axis=1))),
            "selector_target_kernel_mass": row_mass,
            "selector_kernel_mass_min": float(np.asarray(jnp.min(fitted_mass))),
            "selector_kernel_mass_median": float(np.asarray(jnp.median(fitted_mass))),
            "selector_kernel_mass_max": float(np.asarray(jnp.max(fitted_mass))),
            "selector_lower_saturation_fraction": float(np.asarray(jnp.mean(lower_saturated))),
            "selector_upper_saturation_fraction": float(np.asarray(jnp.mean(upper_saturated))),
            "selector_kernel_mass_relative_error_max": root_error,
            **metrics(sample, **reference),
            **force,
        })

    record(0, snapshots[0])
    previous = 0
    for checkpoint in checkpoints[1:]:
        (coordinate, maximum_speed, maximum_update, minimum_bandwidth, maximum_bandwidth, action, linear_defect,
         quadratic_defect, bump_defect, saturation, root_error) = euler_chunk(
            coordinate, beta=float(args.beta), base_bandwidth=fit["base_bandwidth"],
            factor_min=fit["selector_factor_min"], factor_max=fit["selector_factor_max"], row_mass=row_mass,
            fit_iterations=fit["fit_iterations"], particle_compute_dtype=args.particle_compute_dtype,
            time_step=float(args.time_step), steps=int(checkpoint - previous))
        sample = np.asarray(coordinate)
        if not np.all(np.isfinite(sample)):
            raise FloatingPointError(f"Non-finite coordinate at step {checkpoint}")
        if bool(np.asarray(saturation)):
            raise RuntimeError("row-mass root left the numerical bracket during an Euler interval")
        if float(np.asarray(root_error)) > args.row_mass_tolerance:
            raise RuntimeError(f"row-mass inversion exceeded tolerance in the interval ending at step {checkpoint}")
        snapshots.append(sample.copy())
        interval_records.append({
            "start_step": int(previous),
            "end_step": int(checkpoint),
            "maximum_speed": float(np.asarray(maximum_speed)),
            "maximum_update": float(np.asarray(maximum_update)),
            "minimum_bandwidth": float(np.asarray(minimum_bandwidth)),
            "maximum_bandwidth": float(np.asarray(maximum_bandwidth)),
            "discrete_action": float(np.asarray(action)),
            "integrated_linear_flux_defect": float(np.asarray(linear_defect)),
            "integrated_quadratic_flux_defect": float(np.asarray(quadratic_defect)),
            "integrated_bump_flux_defect": float(np.asarray(bump_defect)),
            "selector_saturation_occurred": int(bool(np.asarray(saturation))),
            "maximum_row_mass_relative_error": float(np.asarray(root_error)),
        })
        record(checkpoint, sample)
        previous = checkpoint

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("metrics.csv", records), ("interval_diagnostics.csv", interval_records)):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "protocol": {
            "algorithm": "explicit Euler discretization of the particle ODE",
            "score_estimator": "pointwise Gaussian leave-one-out KDE score",
            "bandwidth_mode": "fresh_particle_mass",
            "selector": "fresh particlewise exact Gaussian row-mass inverse",
            "selector_fit_method": "hybrid",
            "replicas": int(args.replicas),
            "seed": int(args.seed),
            "initialization": "gaussian",
            "initial_center": [float(v) for v in args.initial_center],
            "initial_standard_deviation": float(args.initial_standard_deviation),
            "beta": float(args.beta),
            "time_step": float(args.time_step),
            "final_time": float(args.final_time),
            "checkpoint_time": float(args.checkpoint_time),
            "checkpoints": checkpoints,
            "base_bandwidth": fit["base_bandwidth"],
            "selector_factor_min": fit["selector_factor_min"],
            "selector_factor_max": fit["selector_factor_max"],
            "selector_target_kernel_mass": row_mass,
            "selector_fit_iterations": fit["fit_iterations"],
            "row_mass_fit_tolerance": float(args.row_mass_tolerance),
            "particle_compute_dtype": args.particle_compute_dtype,
            "state_dtype": str(coordinate.dtype),
            "jax_devices": [str(device) for device in jax.devices()],
            "wall_time_s": float(time.perf_counter() - wall_start),
        },
        "exact": {"basin_mass": exact_basin_mass.tolist()},
        "initial": records[0],
        "final": records[-1],
        "interval_diagnostics": interval_records,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "samples.npz",
        checkpoints=np.asarray(checkpoints, dtype=int),
        times=np.asarray(checkpoints, dtype=float) * float(args.time_step),
        samples=np.asarray(snapshots),
        grid_x=np.asarray(exact["x"]),
        grid_y=np.asarray(exact["y"]),
        exact_energy=exact_energy,
        exact_density=exact_density,
        exact_basin_mass=exact_basin_mass,
    )
    print(f"Saved {args.output_dir}; final JS={records[-1]['density_js']:.6f}, "
          f"KS={records[-1]['energy_ks']:.6f}, TV={records[-1]['basin_tv']:.6f}")


if __name__ == "__main__":
    main()
