#!/usr/bin/env python3
"""Exact row-mass estimator on the 3D eight-well target (integrator-matched diagnosis).

Every update solves the per-particle row mass sum_{j!=i} exp(-d_ij^2/2h_i^2) = K
and advances theta <- wrap(theta - dt (s_i + beta grad U)) by explicit Euler
from the restricted-Boltzmann start of run_discovery.py.  Records beta<U>, well
counts, distinct visitors per well, the median bandwidth and the largest
99th-percentile and maximum single-update displacements every --interval
updates.

Paper protocol (K = 3, the midpoint of the shared row-mass band; seeds 20260737..67):

    python run_exact_row_mass.py --seed S --beta-scale 3 --dt 0.008 --row-mass 3 \
        --steps 13500 --interval 5 --bisection-iters 40 \
        --output DATA/oracle_n3_exact_k3_bb_visits/seedS.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fp_transformer.kernel.row_mass import torus_row_mass_score

from examples.periodic3d.common import (
    BETA_0,
    ENERGY_GRAD,
    basin_boltzmann_initialization,
    energy_numpy,
    well_assignments,
    wrap_to_pi,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20260737)
    parser.add_argument("--replicas", type=int, default=1024)
    parser.add_argument("--beta-scale", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.008)
    parser.add_argument("--row-mass", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=13500)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--bisection-iters", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-endpoint", action="store_true",
                        help="also write the final configuration next to --output (.npz)")
    return parser.parse_args()


def make_advance(dt, row_mass, iterations, beta):
    def step(theta, _):
        score, h, _ = torus_row_mass_score(
            theta, row_mass, log_lower=jnp.log(1e-4), iterations=iterations
        )
        new = wrap_to_pi(theta - dt * (score + beta * ENERGY_GRAD(theta)))
        move = jnp.max(jnp.abs(wrap_to_pi(new - theta)), axis=1)
        return new, (jnp.median(h), jnp.percentile(move, 99.0), jnp.max(move))

    return jax.jit(lambda th, m: jax.lax.scan(step, th, xs=None, length=m), static_argnums=1)


def main() -> None:
    args = parse_args()
    beta = BETA_0 * args.beta_scale
    theta = jnp.asarray(basin_boltzmann_initialization(args.seed, replicas=args.replicas, beta=beta))
    advance = make_advance(args.dt, args.row_mass, args.bisection_iters, beta)
    visited = np.zeros(8, bool)
    visit_mask = np.zeros((args.replicas, 8), bool)
    records = []

    def record(update, th, h_med, p99=None, dmax=None):
        nonlocal visited
        x = np.asarray(th)
        assignment = well_assignments(x)
        counts = np.bincount(assignment, minlength=8)
        visited |= counts > 0
        visit_mask[np.arange(len(x)), assignment] = True
        records.append({"update": update, "time": update * args.dt,
                        "beta_mean_energy": float(beta * np.mean(energy_numpy(x))),
                        "cumulative_covered": int(visited.sum()), "counts": counts.tolist(),
                        "distinct_visitors": visit_mask.sum(0).tolist(),
                        "median_h": h_med, "p99_step_displacement": p99,
                        "max_step_displacement": dmax})

    record(0, theta, None)
    for update in range(args.interval, args.steps + 1, args.interval):
        theta, (h, p99, dmax) = advance(theta, args.interval)
        record(update, theta, float(h[-1]), float(jnp.max(p99)), float(jnp.max(dmax)))
    t8 = next((r["time"] for r in records if r["cumulative_covered"] == 8), None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    protocol = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    args.output.write_text(json.dumps({"protocol": protocol, "t8_time": t8, "records": records}) + "\n")
    if args.save_endpoint:
        np.savez_compressed(args.output.with_suffix(".npz"), endpoint=np.asarray(theta))
    print(f"N={args.replicas} K={args.row_mass:g} dt={args.dt:g}: t8 time={t8}, "
          f"final beta<U>={records[-1]['beta_mean_energy']:.3f}", flush=True)


if __name__ == "__main__":
    main()
