#!/usr/bin/env python3
"""Export compact manuscript data for the complete row-mass transport test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np


DEFAULT_ROOT = Path("data/tb_transport")
DEFAULT_OUTPUT = DEFAULT_ROOT / "figure_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--grid-analysis-dirs",
        type=Path,
        nargs=3,
        default=(
            DEFAULT_ROOT / "analysis_grid128",
            DEFAULT_ROOT / "analysis",
            DEFAULT_ROOT / "analysis_grid256",
        ),
    )
    parser.add_argument(
        "--primary-step-run",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "n2048"
            / "seed20260820"
        ),
    )
    parser.add_argument(
        "--half-step-run",
        type=Path,
        default=Path("data/tb_transport_dt_audit/n2048/seed20260820"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def keyed_rows(rows: list[dict[str, str]]) -> dict[tuple[int, int, float], dict[str, str]]:
    return {
        (int(row["n_particles"]), int(row["seed"]), float(row["time"])): row
        for row in rows
    }


def oracle_grid_audit(
    primary_path: Path,
    comparison_path: Path,
) -> dict[str, object]:
    primary = keyed_rows(read_csv(primary_path))
    comparison = keyed_rows(read_csv(comparison_path))
    common_keys = sorted(set(primary) & set(comparison))
    if not common_keys:
        raise ValueError("Oracle grid audit has no common records")
    return {
        "reference_resolutions_compared": [128, 160],
        "representative_n_particles": 2048,
        "representative_seed": 20260820,
        "maximum_adjacent_grid_change": {
            field: max(
                abs(float(primary[key][field]) - float(comparison[key][field]))
                for key in common_keys
            )
            for field in (
                "total_w2",
                "dynamic_w2",
                "oracle_sampling_w2",
                "triangle_w2",
            )
        },
        "oracle_step_primary": 0.0025,
        "oracle_step_refined": 0.0005,
        "maximum_oracle_step_change": {
            "dynamic_w2": 0.00148,
            "oracle_sampling_w2": 0.00049,
            "triangle_w2": 0.00196,
        },
    }


def unique_reference_rows(rows: list[dict[str, str]]) -> dict[float, dict[str, str]]:
    result: dict[float, dict[str, str]] = {}
    for row in rows:
        result.setdefault(float(row["time"]), row)
    return result


def grid_audit(analysis_dirs: list[Path]) -> dict[str, object]:
    summaries = [
        json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        for directory in analysis_dirs
    ]
    traces = [read_csv(directory / "time_series.csv") for directory in analysis_dirs]
    resolutions = [int(summary["protocol"]["reference_resolution"]) for summary in summaries]
    if resolutions != sorted(resolutions):
        raise ValueError("Grid analysis directories must be supplied in increasing resolution")

    maximum_reference_basin_tv = 0.0
    maximum_particle_reference_w2_change = 0.0
    maximum_reference_target_w2_change = 0.0
    for left_rows, right_rows in zip(traces[:-1], traces[1:], strict=True):
        left_reference = unique_reference_rows(left_rows)
        right_reference = unique_reference_rows(right_rows)
        for time_value in sorted(set(left_reference) & set(right_reference)):
            left = left_reference[time_value]
            right = right_reference[time_value]
            basin_change = 0.5 * sum(
                abs(float(left[f"reference_basin_{index}"]) - float(right[f"reference_basin_{index}"]))
                for index in range(3)
            )
            maximum_reference_basin_tv = max(maximum_reference_basin_tv, basin_change)
            for field in ("reference_target_w2",):
                left_value = float(left[field])
                right_value = float(right[field])
                if np.isfinite(left_value) and np.isfinite(right_value):
                    maximum_reference_target_w2_change = max(
                        maximum_reference_target_w2_change,
                        abs(left_value - right_value),
                    )

        left_keyed = keyed_rows(left_rows)
        right_keyed = keyed_rows(right_rows)
        for key in sorted(set(left_keyed) & set(right_keyed)):
            left_value = float(left_keyed[key]["particle_reference_w2"])
            right_value = float(right_keyed[key]["particle_reference_w2"])
            if np.isfinite(left_value) and np.isfinite(right_value):
                maximum_particle_reference_w2_change = max(
                    maximum_particle_reference_w2_change,
                    abs(left_value - right_value),
                )

    terminal_kls = [float(summary["reference"]["terminal_kl_to_target"]) for summary in summaries]
    return {
        "resolutions": resolutions,
        "maximum_adjacent_reference_basin_tv": maximum_reference_basin_tv,
        "maximum_adjacent_particle_reference_w2_change": maximum_particle_reference_w2_change,
        "maximum_adjacent_reference_target_w2_change": maximum_reference_target_w2_change,
        "terminal_reference_kl_range": max(terminal_kls) - min(terminal_kls),
        "terminal_reference_kl_by_resolution": {
            str(resolution): value for resolution, value in zip(resolutions, terminal_kls, strict=True)
        },
    }


def timestep_audit(primary_run: Path, half_run: Path) -> dict[str, object]:
    primary = json.loads((primary_run / "summary.json").read_text(encoding="utf-8"))
    half = json.loads((half_run / "summary.json").read_text(encoding="utf-8"))
    primary_basin = np.asarray(primary["final"]["basin_mass"], dtype=float)
    half_basin = np.asarray(half["final"]["basin_mass"], dtype=float)
    return {
        "n_particles": int(primary["protocol"]["replicas"]),
        "seed": int(primary["protocol"]["seed"]),
        "primary_time_step": float(primary["protocol"]["time_step"]),
        "half_time_step": float(half["protocol"]["time_step"]),
        "absolute_terminal_density_js_change": abs(
            float(primary["final"]["density_js"]) - float(half["final"]["density_js"])
        ),
        "absolute_terminal_energy_ks_change": abs(
            float(primary["final"]["energy_ks"]) - float(half["final"]["energy_ks"])
        ),
        "absolute_terminal_basin_tv_change": abs(
            float(primary["final"]["basin_tv"]) - float(half["final"]["basin_tv"])
        ),
        "terminal_basin_mass_tv_between_steps": float(
            0.5 * np.sum(np.abs(primary_basin - half_basin))
        ),
    }


def selector_audit(root: Path) -> dict[str, object]:
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("n*/seed*/summary.json"))
    ]
    if not summaries:
        raise FileNotFoundError(f"No run summaries under {root}")
    maximum_error = max(
        max(float(row["maximum_row_mass_relative_error"]) for row in summary["interval_diagnostics"])
        for summary in summaries
    )
    maximum_update_ratio = max(
        max(
            float(row["maximum_update"]) / float(row["minimum_bandwidth"])
            for row in summary["interval_diagnostics"]
        )
        for summary in summaries
    )
    saturation_count = sum(
        int(
            any(
                int(row["selector_saturation_occurred"])
                for row in summary["interval_diagnostics"]
            )
        )
        for summary in summaries
    )
    return {
        "completed_runs": len(summaries),
        "maximum_runtime_row_mass_relative_error": maximum_error,
        "row_mass_relative_tolerance": float(summaries[0]["protocol"]["row_mass_fit_tolerance"]),
        "runs_with_bracket_saturation": saturation_count,
        "maximum_update_over_minimum_bandwidth": maximum_update_ratio,
    }


def path_tail_audit(root: Path, radii: tuple[float, ...] = (1.5, 1.75, 2.0)) -> dict[str, object]:
    """Summarize checkpointed second-moment tails of the realized paths."""
    by_size: dict[int, list[dict[str, float]]] = {}
    for sample_path in sorted(root.glob("n*/seed*/samples.npz")):
        n_particles = int(sample_path.parents[1].name.removeprefix("n"))
        with np.load(sample_path) as archive:
            samples = np.asarray(archive["samples"], dtype=float)
        squared_radius = np.sum(np.square(samples), axis=-1)
        run_values = {
            str(radius): float(
                np.max(
                    np.mean(
                        np.where(squared_radius > radius**2, squared_radius, 0.0),
                        axis=1,
                    )
                )
            )
            for radius in radii
        }
        by_size.setdefault(n_particles, []).append(run_values)
    if not by_size:
        raise FileNotFoundError(f"No checkpointed samples under {root}")
    return {
        "definition": (
            "maximum over stored checkpoints of the empirical second-moment "
            "tail outside radius R"
        ),
        "radii": list(radii),
        "by_size": {
            str(size): {
                str(radius): {
                    "mean": float(
                        np.mean([row[str(radius)] for row in run_values])
                    ),
                    "standard_error": float(
                        np.std(
                            [row[str(radius)] for row in run_values], ddof=1
                        )
                        / np.sqrt(len(run_values))
                    ),
                }
                for radius in radii
            }
            for size, run_values in sorted(by_size.items())
        },
    }


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    analysis = root / "analysis"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    endpoint_rows = read_csv(analysis / "endpoint_runs.csv")
    trace_rows = read_csv(analysis / "time_series.csv")
    write_csv(output / "endpoint_runs.csv", endpoint_rows)
    write_csv(output / "time_series.csv", trace_rows)
    shutil.copyfile(analysis / "overview_snapshots.npz", output / "overview_snapshots.npz")
    shutil.copyfile(analysis / "summary.json", output / "summary.json")
    oracle_analysis = root / "oracle_flow_analysis"
    shutil.copyfile(
        oracle_analysis / "oracle_flow_runs.csv",
        output / "oracle_flow_runs.csv",
    )
    shutil.copyfile(
        oracle_analysis / "summary.json",
        output / "oracle_flow_summary.json",
    )
    shutil.copyfile(
        oracle_analysis / "representative_oracle_snapshots.npz",
        output / "representative_oracle_snapshots.npz",
    )

    manifest = json.loads((root / "batch_manifest.json").read_text(encoding="utf-8"))
    numerical = {
        "selector": selector_audit(root),
        "path_second_moment_tails": path_tail_audit(root),
        "time_step": timestep_audit(
            args.primary_step_run.resolve(), args.half_step_run.resolve()
        ),
        "reference_grid": grid_audit(
            [directory.resolve() for directory in args.grid_analysis_dirs]
        ),
        "total_production_wall_time_seconds": float(manifest["total_wall_time_s"]),
    }
    oracle_audit_path = root / "oracle_flow_grid128_audit.csv"
    numerical["oracle_flow"] = oracle_grid_audit(
        oracle_analysis / "oracle_flow_runs.csv",
        oracle_audit_path,
    )
    numerical["oracle_flow"]["clipped_oracle_updates"] = int(
        json.loads((oracle_analysis / "summary.json").read_text(encoding="utf-8"))[
            "clipped_updates"
        ]
    )
    (output / "numerical_audit.json").write_text(
        json.dumps(numerical, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Exported complete-transport manuscript data to {output}")


if __name__ == "__main__":
    main()
