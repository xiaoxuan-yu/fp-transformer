#!/usr/bin/env python3
"""Langevin step-size audit on the 3D eight-well target (SI Table: LD step).

For every step size, Euler--Maruyama Langevin dynamics is started from iid
target ensembles (four seeds) and run for 13500 updates at the target beta.
A step is admissible when the run stays finite and the four-seed mean energy
KS and basin TV do not exceed the 99th percentiles of four-ensemble means of
iid target ensembles; the largest admissible step is selected.

Paper protocol:

    python run_ld_step_audit.py --output-dir DATA/n3_ula_step_audit_large
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.periodic3d.common import (
    AUDIT_SEEDS,
    BETA_0,
    NUM_UPDATES,
    langevin_chunk,
    make_reference,
    reference_mean_bands,
    sample_metrics,
    sample_reference,
)


DEFAULT_STEPS = (0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.012, 0.016, 0.024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step-sizes", type=float, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=AUDIT_SEEDS)
    parser.add_argument("--replicas", type=int, default=1024)
    parser.add_argument("--num-updates", type=int, default=NUM_UPDATES)
    parser.add_argument("--reference-resolution", type=int, default=128)
    parser.add_argument("--reference-groups", type=int, default=400)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--beta-scale", type=float, default=3.0,
                        help="target beta in units of beta_0; barrier = beta_scale k_BT")
    return parser.parse_args()


def run_equilibrium_audit(initial, *, noise_seed, beta, step_size, total_steps, reference) -> dict:
    coordinate, _ = langevin_chunk(jnp.asarray(initial), jax.random.PRNGKey(noise_seed), beta=beta,
                                   step_size=step_size, steps=total_steps)
    snapshot = np.asarray(coordinate)
    return {
        "stable": bool(np.all(np.isfinite(snapshot))),
        "initial": sample_metrics(initial, reference),
        "endpoint": sample_metrics(snapshot, reference),
    }


def metric_summary(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(array)), "minimum": float(np.min(array)),
            "maximum": float(np.max(array)), "values": array.tolist()}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.output_dir} is not empty; choose another directory or use --force")
    if any(step <= 0.0 for step in args.step_sizes):
        raise ValueError("step sizes must be positive")
    beta = args.beta_scale * BETA_0
    reference = make_reference(beta, args.reference_resolution)
    iid_bands = reference_mean_bands(reference=reference, replicas=args.replicas,
                                     seed_count=len(args.seeds), group_count=args.reference_groups)

    run_rows: list[dict[str, object]] = []
    grouped: dict[str, object] = {}
    admissible_steps: list[float] = []
    for step_size in sorted(set(float(value) for value in args.step_sizes)):
        rows = []
        for seed in args.seeds:
            initial = sample_reference(np.random.default_rng(int(seed) + 100_000), reference, args.replicas)
            result = run_equilibrium_audit(initial, noise_seed=int(seed) + 2 + 100_000, beta=beta,
                                           step_size=step_size, total_steps=args.num_updates,
                                           reference=reference)
            rows.append({"step_size": step_size, "seed": int(seed), "stable": result["stable"],
                         "energy_ks": result["endpoint"]["energy_ks"],
                         "basin_tv": result["endpoint"]["basin_tv"]})
        run_rows.extend(rows)
        energy = [float(r["energy_ks"]) for r in rows]
        basin = [float(r["basin_tv"]) for r in rows]
        admissible = (all(r["stable"] for r in rows)
                      and float(np.mean(energy)) <= float(iid_bands["energy_ks_mean_99_upper"])
                      and float(np.mean(basin)) <= float(iid_bands["basin_tv_mean_99_upper"]))
        if admissible:
            admissible_steps.append(step_size)
        grouped[f"step={step_size:g}"] = {"step_size": step_size, "admissible": admissible,
                                          "equilibrium_energy_ks": metric_summary(energy),
                                          "equilibrium_basin_tv": metric_summary(basin)}
        print(f"step={step_size:g} KS={np.mean(energy):.4f} TV={np.mean(basin):.4f} "
              f"admissible={admissible}", flush=True)

    output = {
        "protocol": {
            "replicas": args.replicas,
            "seeds": list(args.seeds),
            "initial_rng_seed_offset": 100_000,
            "noise_seed_offset": 100_002,
            "step_sizes": sorted(set(float(value) for value in args.step_sizes)),
            "beta": beta,
            "total_steps": args.num_updates,
            "reference_resolution": args.reference_resolution,
            "selection_rule": (
                "largest numerically stable step whose four-seed mean "
                "equilibrium-start energy KS and basin TV do not exceed the "
                "99th percentiles of four-ensemble iid-reference means"
            ),
        },
        "iid_reference": iid_bands,
        "selected_step_size": max(admissible_steps) if admissible_steps else None,
        "groups": grouped,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved Langevin step-size audit to {args.output_dir}; selected step={output['selected_step_size']}")


if __name__ == "__main__":
    main()
