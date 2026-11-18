#!/usr/bin/env python3
"""Grid Fokker--Planck reference for the 3D eight-well target (SI exact FP reference).

Solves d rho/dt = Lap rho + div(rho beta grad U) on the 3-torus at the fixed
target inverse temperature, from the Boltzmann law restricted to the initial
well (all sin theta_k < 0), the start of the particle runs.  Time t corresponds to Langevin updates of
step epsilon via t = n * epsilon.

Discretization: square-root approximation (SQRA) finite volumes, i.e. a
continuous-time Markov chain on cell centres with nearest-neighbour rates
h^-2 exp(-(u_j - u_i)/2), u = beta U, which preserves exp(-u) exactly.  u is
capped at --u-cap (the excluded region carries mass below exp(-u_cap)) to bound
the stiffness; explicit Euler steps with dt * (max outflow rate) <= --cfl.

Records beta<U> (uncapped U) and the eight basin masses (basin = sign pattern
of sin theta, as in ``common.well_assignments``) at the requested times.

Paper protocol:

    python oracle_fp.py --output-dir DATA/oracle_n3_bb
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.periodic3d.common import BETA_0

jax.config.update("jax_enable_x64", False)

# Recording times: update 0 and 140 log-spaced updates up to 13500.
DEFAULT_UPDATES = [0] + sorted({int(v) for v in np.round(np.geomspace(1, 13500, 140))})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--u-cap", type=float, default=25.0)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--beta-scale", type=float, default=3.0,
                        help="beta in units of beta_0 (barrier = beta_scale k_BT)")
    parser.add_argument("--epsilon", type=float, default=0.008,
                        help="Langevin step used to convert recording updates to time")
    parser.add_argument("--updates", type=int, nargs="+", default=DEFAULT_UPDATES)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def potential(theta):
    return np.exp(3.0 * (3.0 - np.sum(np.sin(theta) ** 4, axis=0))) - 1.0


def main() -> None:
    args = parse_args()
    n = args.grid
    h = 2.0 * math.pi / n
    axis = -math.pi + h * (np.arange(n) + 0.5)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"))
    beta = BETA_0 * args.beta_scale
    u_true = beta * potential(grid)
    u = np.minimum(u_true, args.u_cap)

    # Boltzmann law restricted to the initial well, normalized as cell masses.
    log_p = np.where(np.all(np.sin(grid) < 0, axis=0), -u_true, -np.inf)
    p0 = np.exp(log_p - log_p.max())
    p0 /= p0.sum()

    basin = ((np.sin(grid[0]) > 0).astype(int) * 4 + (np.sin(grid[1]) > 0).astype(int) * 2
             + (np.sin(grid[2]) > 0).astype(int))

    rate0 = 1.0 / h**2
    outflow = np.zeros_like(u)
    for ax in range(3):
        for shift in (1, -1):
            outflow += rate0 * np.exp(-(np.roll(u, shift, axis=ax) - u) / 2.0)
    dt = args.cfl / float(outflow.max())

    u_j = jnp.asarray(u, dtype=jnp.float32)
    half = jnp.exp(-u_j / 2.0)          # rate i->j = rate0 * half_j / half_i
    inv_half = jnp.exp(u_j / 2.0)
    out_j = jnp.asarray(outflow, dtype=jnp.float32)

    @jax.jit
    def advance(p, steps):
        def body(_, p):
            q = p * inv_half
            inflow = jnp.zeros_like(p)
            for ax in range(3):
                for shift in (1, -1):
                    inflow = inflow + jnp.roll(q, shift, axis=ax)
            inflow = rate0 * half * inflow
            return p + dt * (inflow - out_j * p)
        return jax.lax.fori_loop(0, steps, body, p)

    basin_j = jnp.asarray(basin)
    u_true_j = jnp.asarray(u_true, dtype=jnp.float32)
    p = jnp.asarray(p0, dtype=jnp.float32)
    records, done_time = [], 0.0
    for update in sorted(args.updates):
        target_time = update * args.epsilon
        steps = int(round((target_time - done_time) / dt))
        if steps > 0:
            p = advance(p, steps)
            done_time += steps * dt
        masses = np.asarray(jax.ops.segment_sum(p.ravel(), basin_j.ravel(), num_segments=8))
        records.append({
            "update": update, "time": done_time,
            "beta_mean_energy": float(jnp.sum(p * u_true_j)),
            "total_mass": float(jnp.sum(p)),
            "basin_mass": masses.tolist(),
            "min_basin_mass": float(masses.min()),
        })
        print(f"update {update:5d} t={done_time:.4f} beta<U>={records[-1]['beta_mean_energy']:.3f} "
              f"min basin mass={masses.min():.2e} total={records[-1]['total_mass']:.6f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oracle.json").write_text(json.dumps({
        "protocol": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        | {"beta_target": beta, "dt": dt, "max_outflow": float(outflow.max()),
           "equilibrium_beta_mean_energy": float(np.sum(np.exp(-u_true) * u_true) / np.sum(np.exp(-u_true)))},
        "records": records}, indent=1) + "\n")


if __name__ == "__main__":
    main()
