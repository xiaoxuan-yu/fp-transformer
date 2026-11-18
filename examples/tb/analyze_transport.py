#!/usr/bin/env python3
"""Analyze the common-horizon exact-row-mass transport experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np



from examples.tb.numerics import (
    TB_CENTERS,
    exact_fp_snapshots,
    gaussian_initial_mass,
    grid_w2,
    make_tb_grid,
    reversible_fp_generator,
    sample_grid_w2,
)


DEFAULT_ROOT = Path("data/tb_transport")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference-resolution", type=int, default=160)
    parser.add_argument("--ot-bins", type=int, default=16)
    parser.add_argument("--floor-replicates", type=int, default=4)
    parser.add_argument(
        "--checkpoint-floor-seed-offset",
        type=int,
        default=5_005,
    )
    parser.add_argument(
        "--w2-times",
        type=float,
        nargs="+",
        default=(0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def basin_labels(points: np.ndarray) -> np.ndarray:
    displacement = points[..., None, :] - TB_CENTERS
    return np.argmin(np.sum(np.square(displacement), axis=-1), axis=-1)


def basin_mass_from_grid(mass: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.asarray([np.sum(mass[labels == index]) for index in range(3)])


def basin_mass_from_samples(samples: np.ndarray) -> np.ndarray:
    labels = basin_labels(samples)
    return np.asarray([np.mean(labels == index) for index in range(3)])


def observable_values_samples(samples: np.ndarray) -> np.ndarray:
    displacement = samples[:, None, :] - TB_CENTERS[None, :, :]
    bumps = np.exp(-0.5 * np.sum(displacement**2, axis=2) / 0.35**2)
    return np.concatenate(
        (
            np.mean(samples, axis=0),
            [0.5 * np.mean(np.sum(samples**2, axis=1))],
            np.mean(bumps, axis=0),
        )
    )


def observable_values_grid(mass: np.ndarray, grid) -> np.ndarray:
    xx, yy = np.meshgrid(grid.x, grid.y)
    points = np.stack((xx, yy), axis=-1)
    displacement = points[..., None, :] - TB_CENTERS
    bumps = np.exp(-0.5 * np.sum(displacement**2, axis=-1) / 0.35**2)
    return np.asarray(
        (
            np.sum(mass * xx),
            np.sum(mass * yy),
            0.5 * np.sum(mass * (xx**2 + yy**2)),
            *(np.sum(mass[..., None] * bumps, axis=(0, 1))),
        ),
        dtype=float,
    )


def sample_grid_mass(rng: np.random.Generator, mass: np.ndarray, grid, size: int):
    flat = np.asarray(mass, dtype=float).ravel()
    flat /= np.sum(flat)
    indices = rng.choice(flat.size, size=size, replace=True, p=flat)
    yy, xx = np.unravel_index(indices, mass.shape)
    samples = np.column_stack((grid.x[xx], grid.y[yy]))
    samples += rng.uniform(
        low=(-0.5 * grid.dx, -0.5 * grid.dy),
        high=(0.5 * grid.dx, 0.5 * grid.dy),
        size=samples.shape,
    )
    return samples


def mean_se(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "standard_error": float(
            np.std(array, ddof=1) / np.sqrt(array.size) if array.size > 1 else 0.0
        ),
        "count": int(array.size),
    }


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    output = (args.output_dir or (root / "analysis")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(path.parent for path in root.glob("n*/seed*/summary.json"))
    if not run_dirs:
        raise FileNotFoundError(f"No completed runs under {root}")

    summaries = [
        json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    protocols = [summary["protocol"] for summary in summaries]
    initialization = str(protocols[0]["initialization"])
    if any(str(protocol["initialization"]) != initialization for protocol in protocols):
        raise ValueError("Complete-transport runs must share one initial law")
    beta = float(protocols[0]["beta"])
    with np.load(run_dirs[0] / "samples.npz") as archive:
        times = np.asarray(archive["times"], dtype=float)
    for run_dir in run_dirs[1:]:
        with np.load(run_dir / "samples.npz") as archive:
            if not np.allclose(archive["times"], times, atol=1.0e-12, rtol=0.0):
                raise ValueError(f"Checkpoint mismatch in {run_dir}")

    grid = make_tb_grid(
        beta=beta,
        resolution=int(args.reference_resolution),
        x_bounds=(-1.85, 1.65),
        y_bounds=(-1.25, 1.95),
    )
    if initialization == "gaussian":
        center = tuple(float(value) for value in protocols[0]["initial_center"])
        standard_deviation = float(protocols[0]["initial_standard_deviation"])
        initial_mass = gaussian_initial_mass(
            grid,
            center=center,
            standard_deviation=standard_deviation,
        )
        initial_law = (
            f"isotropic Gaussian centered at ({center[0]:g}, {center[1]:g}) "
            f"with standard deviation {standard_deviation:g}"
        )
    else:
        raise ValueError(f"Unsupported complete-transport initialization {initialization!r}")
    reference = exact_fp_snapshots(
        reversible_fp_generator(grid),
        initial_mass,
        times,
    )
    xx, yy = np.meshgrid(grid.x, grid.y)
    grid_labels = basin_labels(np.stack((xx, yy), axis=-1))
    reference_basin = np.asarray(
        [basin_mass_from_grid(mass, grid_labels) for mass in reference]
    )
    target_basin = basin_mass_from_grid(grid.target_mass, grid_labels)
    keep = (reference > 0.0) & (grid.target_mass[None, ...] > 0.0)
    reference_kl = np.sum(
        np.where(
            keep,
            reference * np.log(reference / grid.target_mass[None, ...]),
            0.0,
        ),
        axis=(1, 2),
    )
    w2_indices = []
    for value in args.w2_times:
        matches = np.flatnonzero(np.isclose(times, value, atol=1.0e-12, rtol=0.0))
        if matches.size != 1:
            raise ValueError(f"Requested W2 time {value:g} is unavailable")
        w2_indices.append(int(matches[0]))
    reference_target_w2 = {
        int(index): grid_w2(
            reference[index], grid.target_mass, grid=grid, bins=int(args.ot_bins)
        )[0]
        for index in w2_indices
    }

    records: list[dict[str, float | int]] = []
    endpoint_rows: list[dict[str, float | int]] = []
    representative_samples = None
    maximum_n = max(int(protocol["replicas"]) for protocol in protocols)
    for run_dir, summary in zip(run_dirs, summaries, strict=True):
        protocol = summary["protocol"]
        n_particles = int(protocol["replicas"])
        seed = int(protocol["seed"])
        with np.load(run_dir / "samples.npz") as archive:
            samples = np.asarray(archive["samples"], dtype=float)
        metric_rows = read_csv(run_dir / "metrics.csv")
        interval_rows = read_csv(run_dir / "interval_diagnostics.csv")
        cumulative_action = np.concatenate(
            ([0.0], np.cumsum([float(row["discrete_action"]) for row in interval_rows]))
        )
        cumulative_linear = np.concatenate(
            ([0.0], np.cumsum([float(row["integrated_linear_flux_defect"]) for row in interval_rows]))
        )
        cumulative_quadratic = np.concatenate(
            ([0.0], np.cumsum([float(row["integrated_quadratic_flux_defect"]) for row in interval_rows]))
        )
        cumulative_bump = np.concatenate(
            ([0.0], np.cumsum([float(row["integrated_bump_flux_defect"]) for row in interval_rows]))
        )
        if len(metric_rows) != len(times) or len(cumulative_action) != len(times):
            raise ValueError(f"Malformed checkpoint diagnostics in {run_dir}")
        for index, (time_value, coordinate, metric_row) in enumerate(
            zip(times, samples, metric_rows, strict=True)
        ):
            particle_basin = basin_mass_from_samples(coordinate)
            observable_error = np.abs(
                observable_values_samples(coordinate)
                - observable_values_grid(reference[index], grid)
            )
            row: dict[str, float | int] = {
                "n_particles": n_particles,
                "seed": seed,
                "time": float(time_value),
                "reference_kl_to_target": float(reference_kl[index]),
                "reference_basin_0": float(reference_basin[index, 0]),
                "reference_basin_1": float(reference_basin[index, 1]),
                "reference_basin_2": float(reference_basin[index, 2]),
                "particle_basin_0": float(particle_basin[0]),
                "particle_basin_1": float(particle_basin[1]),
                "particle_basin_2": float(particle_basin[2]),
                "particle_target_density_js": float(metric_row["density_js"]),
                "particle_target_energy_ks": float(metric_row["energy_ks"]),
                "particle_target_basin_tv": float(metric_row["basin_tv"]),
                "time_matched_observable_error_max": float(np.max(observable_error)),
                "time_matched_bump_error_max": float(np.max(observable_error[3:])),
                "cumulative_weak_linear_residual": float(cumulative_linear[index]),
                "cumulative_weak_quadratic_residual": float(cumulative_quadratic[index]),
                "cumulative_weak_bump_residual": float(cumulative_bump[index]),
                "cumulative_action": float(cumulative_action[index]),
                "bandwidth_median": float(metric_row["bandwidth_median"]),
            }
            if index in w2_indices:
                row["particle_reference_w2"] = float(
                    sample_grid_w2(
                        coordinate,
                        reference[index],
                        grid=grid,
                        bins=int(args.ot_bins),
                    )[0]
                )
                row["particle_target_w2"] = float(
                    sample_grid_w2(
                        coordinate,
                        grid.target_mass,
                        grid=grid,
                        bins=int(args.ot_bins),
                    )[0]
                )
                row["reference_target_w2"] = float(reference_target_w2[index])
            else:
                row["particle_reference_w2"] = float("nan")
                row["particle_target_w2"] = float("nan")
                row["reference_target_w2"] = float("nan")
            records.append(row)

        run_records = [
            row
            for row in records
            if int(row["n_particles"]) == n_particles and int(row["seed"]) == seed
        ]
        terminal = run_records[-1]
        finite_w2_rows = [
            row
            for row in run_records
            if np.isfinite(float(row["particle_reference_w2"]))
        ]
        maximum_w2_row = max(
            finite_w2_rows,
            key=lambda row: float(row["particle_reference_w2"]),
        )
        maximum_w2_index = int(
            np.flatnonzero(
                np.isclose(
                    times,
                    float(maximum_w2_row["time"]),
                    atol=1.0e-12,
                    rtol=0.0,
                )
            )[0]
        )
        checkpoint_floor_rng = np.random.default_rng(
            seed + 4_100_003 + int(args.checkpoint_floor_seed_offset)
        )
        checkpoint_floor_values = []
        for _ in range(int(args.floor_replicates)):
            reference_sample = sample_grid_mass(
                checkpoint_floor_rng,
                reference[maximum_w2_index],
                grid,
                n_particles,
            )
            checkpoint_floor_values.append(
                sample_grid_w2(
                    reference_sample,
                    reference[maximum_w2_index],
                    grid=grid,
                    bins=int(args.ot_bins),
                )[0]
            )
        floor_rng = np.random.default_rng(seed + 4_100_003)
        floor_values = []
        for _ in range(int(args.floor_replicates)):
            reference_sample = sample_grid_mass(
                floor_rng,
                reference[-1],
                grid,
                n_particles,
            )
            floor_values.append(
                sample_grid_w2(
                    reference_sample,
                    reference[-1],
                    grid=grid,
                    bins=int(args.ot_bins),
                )[0]
            )
        maximum_row_mass_error = max(
            float(row["maximum_row_mass_relative_error"]) for row in interval_rows
        )
        endpoint_rows.append(
            {
                **terminal,
                "independent_reference_w2_mean": float(np.mean(floor_values)),
                "independent_reference_w2_standard_error": float(
                    np.std(floor_values, ddof=1) / np.sqrt(len(floor_values))
                    if len(floor_values) > 1
                    else 0.0
                ),
                "maximum_row_mass_relative_error": maximum_row_mass_error,
                "selector_saturation_occurred": int(
                    any(int(row["selector_saturation_occurred"]) for row in interval_rows)
                ),
                "maximum_update_over_bandwidth": float(
                    max(
                        float(row["maximum_update"]) / float(row["minimum_bandwidth"])
                        for row in interval_rows
                    )
                ),
                "checkpoint_maximum_particle_reference_w2": float(
                    maximum_w2_row["particle_reference_w2"]
                ),
                "checkpoint_maximum_particle_reference_w2_time": float(
                    maximum_w2_row["time"]
                ),
                "checkpoint_maximum_independent_reference_w2_mean": float(
                    np.mean(checkpoint_floor_values)
                ),
                "checkpoint_maximum_independent_reference_w2_standard_error": float(
                    np.std(checkpoint_floor_values, ddof=1)
                    / np.sqrt(len(checkpoint_floor_values))
                    if len(checkpoint_floor_values) > 1
                    else 0.0
                ),
            }
        )
        if n_particles == maximum_n and representative_samples is None:
            representative_samples = samples[np.asarray(w2_indices, dtype=int)]

    fieldnames = list(records[0])
    with (output / "time_series.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    endpoint_fields = list(endpoint_rows[0])
    with (output / "endpoint_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=endpoint_fields)
        writer.writeheader()
        writer.writerows(endpoint_rows)

    sizes = sorted({int(row["n_particles"]) for row in endpoint_rows})
    endpoint_summary = {}
    for size in sizes:
        group = [row for row in endpoint_rows if int(row["n_particles"]) == size]
        endpoint_summary[str(size)] = {
            field: mean_se([float(row[field]) for row in group])
            for field in (
                "particle_reference_w2",
                "particle_target_w2",
                "independent_reference_w2_mean",
                "particle_target_density_js",
                "particle_target_basin_tv",
                "particle_target_energy_ks",
                "time_matched_observable_error_max",
                "cumulative_weak_linear_residual",
                "cumulative_weak_quadratic_residual",
                "cumulative_weak_bump_residual",
                "cumulative_action",
                "maximum_row_mass_relative_error",
                "maximum_update_over_bandwidth",
                "checkpoint_maximum_particle_reference_w2",
                "checkpoint_maximum_particle_reference_w2_time",
                "checkpoint_maximum_independent_reference_w2_mean",
            )
        }
    payload = {
        "protocol": {
            "initialization": initialization,
            "initial_law": initial_law,
            "initial_center": (
                list(center) if initialization == "gaussian" else None
            ),
            "initial_standard_deviation": (
                standard_deviation if initialization == "gaussian" else None
            ),
            "beta": beta,
            "horizon": float(times[-1]),
            "reference_resolution": int(args.reference_resolution),
            "ot_bins": int(args.ot_bins),
            "w2_times": [float(times[index]) for index in w2_indices],
            "sizes": sizes,
            "seeds": sorted({int(row["seed"]) for row in endpoint_rows}),
        },
        "reference": {
            "initial_basin_mass": reference_basin[0].tolist(),
            "terminal_basin_mass": reference_basin[-1].tolist(),
            "target_basin_mass": target_basin.tolist(),
            "terminal_basin_tv_to_target": float(
                0.5 * np.sum(np.abs(reference_basin[-1] - target_basin))
            ),
            "initial_kl_to_target": float(reference_kl[0]),
            "terminal_kl_to_target": float(reference_kl[-1]),
            "initial_w2_to_target": float(reference_target_w2[w2_indices[0]]),
            "terminal_w2_to_target": float(reference_target_w2[w2_indices[-1]]),
        },
        "endpoint_by_size": endpoint_summary,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    if representative_samples is not None:
        np.savez_compressed(
            output / "overview_snapshots.npz",
            x=grid.x,
            y=grid.y,
            energy=grid.energy,
            target_mass=grid.target_mass,
            initial_mass=initial_mass,
            times=times[np.asarray(w2_indices, dtype=int)],
            reference_mass=reference[np.asarray(w2_indices, dtype=int)],
            particle_samples=representative_samples,
            n_particles=np.asarray(maximum_n),
        )
    print(f"Saved complete-transport analysis to {output}")


if __name__ == "__main__":
    main()
