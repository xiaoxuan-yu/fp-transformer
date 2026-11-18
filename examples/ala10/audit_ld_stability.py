#!/usr/bin/env python3
"""Langevin step-size stability audit for Ala10 (SI Ala10).

For every Langevin run the saved checkpoints must contain finite coordinates,
energies and dimensionless gradients, the largest saved energy must stay below
1000 kcal/mol, and the largest deterministic componentwise drift
dt * max|beta grad U| must stay below pi.

    python -m examples.ala10.audit_ld_stability --output audit.csv \
        --tuning DT5e-5_RUN DT2.5e-5_RUN --production data/ala10/ld_controls/seed*/ld_m128
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ENERGY_GATE_KCAL_MOL = 1_000.0
DRIFT_GATE_RAD = math.pi


def scan_run(scope: str, run_dir: Path) -> dict[str, object]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    seed = int(config["seed"])
    time_step = float(config["time_step"])
    updates = int(config["steps"])
    checkpoints = sorted(run_dir.glob("checkpoint_*.npz"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found under {run_dir}")

    all_finite = True
    maximum_energy = -math.inf
    maximum_drift = 0.0
    for checkpoint in checkpoints:
        with np.load(checkpoint, allow_pickle=False) as data:
            coordinates = np.asarray(data["coordinates"])
            energies = np.asarray(data["energy_kcal_mol"])
            gradient = np.asarray(data["dimensionless_gradient"])
        all_finite = all_finite and bool(
            np.isfinite(coordinates).all()
            and np.isfinite(energies).all()
            and np.isfinite(gradient).all()
        )
        maximum_energy = max(maximum_energy, float(np.nanmax(energies)))
        maximum_drift = max(
            maximum_drift,
            time_step * float(np.nanmax(np.abs(gradient))),
        )

    passed = bool(
        all_finite
        and maximum_energy < ENERGY_GATE_KCAL_MOL
        and maximum_drift < DRIFT_GATE_RAD
    )
    return {
        "scope": scope,
        "seed": seed,
        "time_step": time_step,
        "updates": updates,
        "langevin_time": time_step * updates,
        "checkpoint_count": len(checkpoints),
        "all_checkpoint_arrays_finite": all_finite,
        "max_saved_energy_kcal_mol": maximum_energy,
        "max_saved_deterministic_drift_rad": maximum_drift,
        "energy_gate_kcal_mol": ENERGY_GATE_KCAL_MOL,
        "drift_gate_rad": DRIFT_GATE_RAD,
        "passed": passed,
        "source_directory": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tuning", type=Path, nargs="*", default=[], help="step-selection runs")
    parser.add_argument("--production", type=Path, nargs="*", default=[], help="production runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [scan_run("tuning", run) for run in args.tuning] + [scan_run("production", run) for run in args.production]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row["scope"], row["seed"], row["time_step"], "passed" if row["passed"] else "FAILED",
              f"max E {row['max_saved_energy_kcal_mol']:.1f}", f"max drift {row['max_saved_deterministic_drift_rad']:.3g}")


if __name__ == "__main__":
    main()
