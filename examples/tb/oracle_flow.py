#!/usr/bin/env python3
"""Segmented same-initial oracle-flow decomposition with bounded memory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np



from examples.tb.numerics import (
    finite_volume_probability_flow_velocity,
    gaussian_initial_mass,
    implicit_fp_snapshots,
    interpolate_vector_field,
    make_tb_grid,
    reversible_fp_generator,
    sample_grid_w2,
    sample_sample_w2,
)


DEFAULT_ROOT = Path("data/tb_transport")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-resolution", type=int, default=160)
    parser.add_argument("--reference-time-step", type=float, default=0.00125)
    parser.add_argument("--oracle-time-step", type=float, default=0.0025)
    parser.add_argument("--ot-bins", type=int, default=16)
    parser.add_argument("--relative-floor", type=float, default=1.0e-14)
    parser.add_argument(
        "--reference-step-after",
        nargs=2,
        type=float,
        metavar=("TIME", "STEP"),
        default=(10.0, 0.0025),
        help="Use a second reference step for intervals beginning at TIME",
    )
    parser.add_argument(
        "--w2-times",
        type=float,
        nargs="+",
        default=(0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0),
    )
    parser.add_argument("--limit-runs", type=int)
    parser.add_argument("--only-n", type=int, nargs="+")
    parser.add_argument("--seed", type=int, nargs="+")
    return parser.parse_args()


def mean_se(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "standard_error": float(
            np.std(array, ddof=1) / np.sqrt(array.size)
            if array.size > 1
            else 0.0
        ),
        "count": int(array.size),
    }


def integrate_interval(
    coordinates: list[np.ndarray],
    *,
    mass_snapshots: np.ndarray,
    reference_step: float,
    oracle_step: float,
    grid,
    relative_floor: float,
) -> tuple[list[np.ndarray], float, int]:
    velocity_x = np.empty_like(mass_snapshots, dtype=np.float32)
    velocity_y = np.empty_like(mass_snapshots, dtype=np.float32)
    for index, mass in enumerate(mass_snapshots):
        vx, vy = finite_volume_probability_flow_velocity(
            mass,
            grid=grid,
            relative_floor=relative_floor,
        )
        velocity_x[index] = vx
        velocity_y[index] = vy

    interval = reference_step * (mass_snapshots.shape[0] - 1)
    step_count = int(round(interval / oracle_step))
    if not np.isclose(step_count * oracle_step, interval):
        raise ValueError("oracle step must divide every checkpoint interval")
    maximum_speed = 0.0
    clipped_count = 0
    coordinate_list = [np.asarray(value, dtype=float).copy() for value in coordinates]

    def velocity(time_value: float, coordinate: np.ndarray) -> np.ndarray:
        scaled = np.clip(time_value / reference_step, 0.0, mass_snapshots.shape[0] - 1)
        left = min(int(np.floor(scaled)), mass_snapshots.shape[0] - 2)
        fraction = float(scaled - left)
        first = interpolate_vector_field(
            velocity_x[left], velocity_y[left], grid=grid, coordinate=coordinate
        )
        if fraction <= 1.0e-14:
            return first
        second = interpolate_vector_field(
            velocity_x[left + 1],
            velocity_y[left + 1],
            grid=grid,
            coordinate=coordinate,
        )
        return first + fraction * (second - first)

    for step_index in range(step_count):
        time_value = step_index * oracle_step
        for list_index, coordinate in enumerate(coordinate_list):
            first = velocity(time_value, coordinate)
            midpoint = coordinate + 0.5 * oracle_step * first
            midpoint[:, 0] = np.clip(midpoint[:, 0], grid.x[0], grid.x[-1])
            midpoint[:, 1] = np.clip(midpoint[:, 1], grid.y[0], grid.y[-1])
            midpoint_velocity = velocity(time_value + 0.5 * oracle_step, midpoint)
            proposed = coordinate + oracle_step * midpoint_velocity
            outside = (
                (proposed[:, 0] < grid.x[0])
                | (proposed[:, 0] > grid.x[-1])
                | (proposed[:, 1] < grid.y[0])
                | (proposed[:, 1] > grid.y[-1])
            )
            clipped_count += int(np.count_nonzero(outside))
            proposed[:, 0] = np.clip(proposed[:, 0], grid.x[0], grid.x[-1])
            proposed[:, 1] = np.clip(proposed[:, 1], grid.y[0], grid.y[-1])
            maximum_speed = max(
                maximum_speed,
                float(np.max(np.linalg.norm(midpoint_velocity, axis=1))),
            )
            coordinate_list[list_index] = proposed
    return coordinate_list, maximum_speed, clipped_count


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    output = (args.output_dir or (root / "oracle_flow_analysis")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(path.parent for path in root.glob("n*/seed*/summary.json"))
    if args.only_n is not None:
        allowed_sizes = {int(value) for value in args.only_n}
        run_dirs = [
            path
            for path in run_dirs
            if int(path.parent.name.removeprefix("n")) in allowed_sizes
        ]
    if args.seed is not None:
        allowed_seeds = {int(value) for value in args.seed}
        run_dirs = [
            path
            for path in run_dirs
            if int(path.name.removeprefix("seed")) in allowed_seeds
        ]
    if args.limit_runs is not None:
        run_dirs = run_dirs[: int(args.limit_runs)]
    if not run_dirs:
        raise FileNotFoundError(f"No runs under {root}")
    summaries = [
        json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    protocols = [summary["protocol"] for summary in summaries]
    beta = float(protocols[0]["beta"])
    center = tuple(float(value) for value in protocols[0]["initial_center"])
    standard_deviation = float(protocols[0]["initial_standard_deviation"])
    output_times = np.asarray(args.w2_times, dtype=float)
    if not np.isclose(output_times[0], 0.0) or np.any(np.diff(output_times) <= 0.0):
        raise ValueError("w2-times must start at zero and increase")

    grid = make_tb_grid(
        beta=beta,
        resolution=int(args.reference_resolution),
        x_bounds=(-1.85, 1.65),
        y_bounds=(-1.25, 1.95),
    )
    generator = reversible_fp_generator(grid)
    current_mass = gaussian_initial_mass(
        grid,
        center=center,
        standard_deviation=standard_deviation,
    )
    particle_by_run: dict[tuple[int, int], np.ndarray] = {}
    coordinate_by_run: dict[tuple[int, int], np.ndarray] = {}
    for run_dir, protocol in zip(run_dirs, protocols, strict=True):
        key = (int(protocol["replicas"]), int(protocol["seed"]))
        with np.load(run_dir / "samples.npz") as archive:
            stored_times = np.asarray(archive["times"], dtype=float)
            stored_samples = np.asarray(archive["samples"], dtype=float)
        indices = [
            int(np.flatnonzero(np.isclose(stored_times, value))[0])
            for value in output_times
        ]
        particle_by_run[key] = stored_samples[np.asarray(indices)]
        coordinate_by_run[key] = particle_by_run[key][0].copy()

    oracle_by_time: dict[tuple[int, int], list[np.ndarray]] = {
        key: [coordinate.copy()] for key, coordinate in coordinate_by_run.items()
    }
    reference_by_time = [current_mass.copy()]
    maximum_speed = 0.0
    clipped_updates = 0
    for left_time, right_time in zip(output_times[:-1], output_times[1:], strict=True):
        interval = float(right_time - left_time)
        local_reference_step = float(args.reference_time_step)
        if (
            args.reference_step_after is not None
            and left_time >= float(args.reference_step_after[0])
        ):
            local_reference_step = float(args.reference_step_after[1])
        reference_count = int(round(interval / local_reference_step)) + 1
        local_times = np.linspace(0.0, interval, reference_count)
        mass_snapshots = implicit_fp_snapshots(generator, current_mass, local_times)
        keys = list(coordinate_by_run)
        next_coordinates, interval_speed, interval_clipped = integrate_interval(
            [coordinate_by_run[key] for key in keys],
            mass_snapshots=mass_snapshots,
            reference_step=local_reference_step,
            oracle_step=float(args.oracle_time_step),
            grid=grid,
            relative_floor=float(args.relative_floor),
        )
        for key, coordinate in zip(keys, next_coordinates, strict=True):
            coordinate_by_run[key] = coordinate
            oracle_by_time[key].append(coordinate.copy())
        current_mass = np.asarray(mass_snapshots[-1], dtype=float).copy()
        reference_by_time.append(current_mass)
        maximum_speed = max(maximum_speed, interval_speed)
        clipped_updates += interval_clipped
        print(f"completed oracle interval [{left_time:g}, {right_time:g}]")

    reference_by_time = np.asarray(reference_by_time)
    records: list[dict[str, float | int]] = []
    for key, particle_samples in particle_by_run.items():
        n_particles, seed = key
        oracle_samples = np.asarray(oracle_by_time[key])
        for time_value, particle, oracle, reference_mass in zip(
            output_times,
            particle_samples,
            oracle_samples,
            reference_by_time,
            strict=True,
        ):
            total_w2 = sample_grid_w2(
                particle, reference_mass, grid=grid, bins=int(args.ot_bins)
            )[0]
            oracle_sampling_w2 = sample_grid_w2(
                oracle, reference_mass, grid=grid, bins=int(args.ot_bins)
            )[0]
            dynamic_w2 = sample_sample_w2(
                particle, oracle, grid=grid, bins=int(args.ot_bins)
            )[0]
            paired_rms = float(
                np.sqrt(np.mean(np.sum(np.square(particle - oracle), axis=1)))
            )
            records.append(
                {
                    "n_particles": n_particles,
                    "seed": seed,
                    "time": float(time_value),
                    "total_w2": float(total_w2),
                    "dynamic_w2": float(dynamic_w2),
                    "oracle_sampling_w2": float(oracle_sampling_w2),
                    "triangle_w2": float(dynamic_w2 + oracle_sampling_w2),
                    "paired_rms": paired_rms,
                    "paired_triangle_w2": float(paired_rms + oracle_sampling_w2),
                }
            )

    with (output / "oracle_flow_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    fields = (
        "total_w2",
        "dynamic_w2",
        "oracle_sampling_w2",
        "triangle_w2",
        "paired_rms",
        "paired_triangle_w2",
    )
    sizes = sorted({int(row["n_particles"]) for row in records})
    by_size = {}
    for size in sizes:
        size_rows = [row for row in records if int(row["n_particles"]) == size]
        by_run: dict[int, list[dict[str, float | int]]] = {}
        for row in size_rows:
            by_run.setdefault(int(row["seed"]), []).append(row)
        by_size[str(size)] = {
            "by_time": {
                f"{time_value:g}": {
                    field: mean_se(
                        [
                            float(row[field])
                            for row in size_rows
                            if np.isclose(float(row["time"]), time_value)
                        ]
                    )
                    for field in fields
                }
                for time_value in output_times
            },
            "checkpoint_maximum": {
                field: mean_se(
                    [max(float(row[field]) for row in run) for run in by_run.values()]
                )
                for field in fields
            },
        }
    payload = {
        "protocol": {
            "definition": "same-initial numerical probability flow",
            "velocity": "finite-volume edge current divided by edge mass",
            "reference_resolution": int(args.reference_resolution),
            "reference_time_step": float(args.reference_time_step),
            "reference_step_after": (
                list(args.reference_step_after)
                if args.reference_step_after is not None
                else None
            ),
            "oracle_time_step": float(args.oracle_time_step),
            "reference_propagator": "implicit Euler, segmented by checkpoint",
            "oracle_integrator": "explicit midpoint Runge-Kutta",
            "spatial_interpolation": "bilinear",
            "relative_mass_floor": float(args.relative_floor),
            "ot_bins": int(args.ot_bins),
            "times": output_times.tolist(),
        },
        "by_size": by_size,
        "maximum_speed": maximum_speed,
        "clipped_updates": clipped_updates,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    maximum_key = max(oracle_by_time)
    np.savez_compressed(
        output / "representative_oracle_snapshots.npz",
        times=output_times,
        oracle_samples=np.asarray(oracle_by_time[maximum_key]),
        n_particles=np.asarray(maximum_key[0]),
        seed=np.asarray(maximum_key[1]),
    )
    print(f"Saved segmented oracle-flow decomposition to {output}")


if __name__ == "__main__":
    main()
