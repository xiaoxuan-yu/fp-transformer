#!/usr/bin/env python3
"""Single-update displacement screen for FPT parameter selection (SI FPT selection table).

From the restricted-Boltzmann start of run_discovery.py, FPT runs ``--updates`` updates for every row-mass
band and learning rate; a setting passes when the 99th percentile of the
single-update toroidal displacement stays below ``--limit`` rad.  Records
the largest 99th percentile (max_p99), the largest single displacement
(max_dx), and the largest beta<U> over the screened updates (E_max_50).

Paper protocol:

    python screen_fpt.py --output DATA/n3_rms_screen.json \
        --lrs 0.015 0.03 0.06 0.09 0.12 0.15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.periodic3d.common import (
    BETA_0,
    basin_boltzmann_initialization,
    build_fpt_sampler,
    energy_numpy,
    sampler_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260737)
    parser.add_argument("--replicas", type=int, default=1024)
    parser.add_argument("--beta-scale", type=float, default=3.0)
    parser.add_argument("--bands", type=str, nargs="+", default=["2,4", "5,10", "20,40"])
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.015, 0.03, 0.06])
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--limit", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    beta = args.beta_scale * BETA_0
    initial = basin_boltzmann_initialization(args.seed, replicas=args.replicas, beta=beta)
    results = {}
    for band in args.bands:
        low, high = (float(v) for v in band.split(","))
        for lr in args.lrs:
            sampler = build_fpt_sampler(initial, beta=beta, seed=args.seed + 1, learning_rate=lr,
                                        kernel_min=low, kernel_max=high, update_pace=1)
            previous = initial
            p99, dx, energy = [], [], []
            for update in range(args.updates):
                sampler.run(1, verbose=False, continue_from_last_state=update > 0)
                current = sampler_snapshot(sampler)
                step = np.abs(np.angle(np.exp(1j * (current - previous))))
                p99.append(float(np.quantile(step, 0.99)))
                dx.append(float(step.max()))
                energy.append(float(beta * np.mean(energy_numpy(current))))
                previous = current
            key = f"[{low:g},{high:g}] lr={lr:g}"
            results[key] = {"max_p99": max(p99), "max_dx": max(dx), "E_max_50": max(energy),
                            "pass": max(p99) < args.limit}
            print(key, results[key], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
