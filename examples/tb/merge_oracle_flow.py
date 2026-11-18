#!/usr/bin/env python3
"""Merge seed-wise oracle-flow analyses into one manuscript dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("input_dirs", type=Path, nargs="+")
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


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    summaries = []
    for directory in args.input_dirs:
        directory = directory.resolve()
        with (directory / "oracle_flow_runs.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows.extend(csv.DictReader(handle))
        summaries.append(
            json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        )
    rows.sort(
        key=lambda row: (
            int(row["n_particles"]),
            int(row["seed"]),
            float(row["time"]),
        )
    )
    with (output / "oracle_flow_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fields = (
        "total_w2",
        "dynamic_w2",
        "oracle_sampling_w2",
        "triangle_w2",
        "paired_rms",
        "paired_triangle_w2",
    )
    times = sorted({float(row["time"]) for row in rows})
    sizes = sorted({int(row["n_particles"]) for row in rows})
    by_size = {}
    for size in sizes:
        size_rows = [row for row in rows if int(row["n_particles"]) == size]
        by_run: dict[int, list[dict[str, str]]] = {}
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
                for time_value in times
            },
            "checkpoint_maximum": {
                field: mean_se(
                    [max(float(row[field]) for row in run) for run in by_run.values()]
                )
                for field in fields
            },
        }
    payload = {
        "protocol": summaries[0]["protocol"],
        "by_size": by_size,
        "maximum_speed": max(float(summary["maximum_speed"]) for summary in summaries),
        "clipped_updates": sum(int(summary["clipped_updates"]) for summary in summaries),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(
        args.input_dirs[-1] / "representative_oracle_snapshots.npz",
        output / "representative_oracle_snapshots.npz",
    )
    print(f"Merged {len(rows)} oracle-flow records into {output}")


if __name__ == "__main__":
    main()
