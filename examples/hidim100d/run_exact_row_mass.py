#!/usr/bin/env python3
"""Exact row-mass FPT on the D-dimensional benchmark (SI thermal state and smoothing).

Every update solves sum_{j != i} exp(-d_ij^2 / 2 h_i^2) = K for every particle
and advances theta <- wrap(theta - dt (s_i + beta grad U)) by explicit Euler.
Discovery metrics and checkpoints are those of run_discovery.py.

Paper protocols (DATA = data/numerical_hidim_periodic_discovery):

    # beta = 5 equilibrium start, D = 100
    python -m examples.hidim100d.run_exact_row_mass --beta 5 --seeds 0 --num-steps 2000 \
        --dts 0.03 0.1 --row-masses 27 100 400 \
        --init-file DATA/equilibrium_beta5/ld_beta5_seed0.npy \
        --output-dir DATA/equilibrium_beta5/perparticle_euler
    # dimension scan, e.g. D = 10 (beta 4.3); D = 5 (3.53), D = 20 (5.35)
    python -m examples.hidim100d.run_exact_row_mass --dim 10 --beta 4.3 --seeds 0 \
        --num-steps 2000 --dts 0.03 0.1 --row-masses 27 100 400 \
        --init-file DATA/equilibrium_dimscan/D10_beta4.3/ld_beta4.3_seed0.npy \
        --output-dir DATA/equilibrium_dimscan/D10_beta4.3/perparticle_euler
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fp_transformer.kernel.row_mass import torus_row_mass_score

from examples.hidim100d.common import Evaluator, build_seeded_coords, checkpoints, make_grad, wrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--replicas", type=int, default=5120)
    parser.add_argument("--init-scale", type=float, default=0.4)
    parser.add_argument("--init-file", type=Path, default=None,
                        help="(N, D) .npy start ensemble replacing the seeded start")
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--num-steps", type=int, default=20000)
    parser.add_argument("--dts", type=float, nargs="+", default=[0.01])
    parser.add_argument("--row-masses", type=float, nargs="+", default=[27.0])
    parser.add_argument("--bisection-iters", type=int, default=40)
    parser.add_argument("--radius", type=float, default=1.25)
    parser.add_argument("--quench-step", type=float, default=0.1)
    parser.add_argument("--quench-iters", type=int, default=1500)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def make_advance(dim, beta, dt, row_mass, iterations):
    grad_fn = make_grad(dim)

    def step(theta, _):
        score, h, mass = torus_row_mass_score(theta, row_mass, log_lower=jnp.log(1e-3), iterations=iterations)
        theta = wrap(theta - dt * (score + beta * grad_fn(theta)))
        return theta, (jnp.median(h), jnp.max(jnp.abs(mass / row_mass - 1.0)))

    return jax.jit(lambda th, n: jax.lax.scan(step, th, xs=None, length=n), static_argnums=1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(args.dim, args.radius, args.quench_step, args.quench_iters)
    points = checkpoints(args.num_steps)
    protocol = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    output = {"protocol": protocol | {"checkpoints": points}, "runs": []}
    out_path = args.output_dir / f"summary{args.tag}.json"
    for dt in args.dts:
        for row_mass in args.row_masses:
            for seed in args.seeds:
                advance = make_advance(args.dim, args.beta, dt, row_mass, args.bisection_iters)
                theta = (jnp.asarray(np.load(args.init_file)) if args.init_file is not None
                         else build_seeded_coords(dim=args.dim, replicas=args.replicas,
                                                  init_scale=args.init_scale, seed=seed))
                evaluator.reset()
                evaluator.record(0, theta)
                h_trace, root_err = [], 0.0
                start = time.perf_counter()
                current = 0
                for target in points[1:]:
                    theta, (h_med, err) = advance(theta, target - current)
                    current = target
                    evaluator.record(current, theta)
                    h_trace.append(float(h_med[-1]))
                    root_err = max(root_err, float(jnp.max(err)))
                result = evaluator.summary() | {
                    "method": "fpt_rowmass_euler", "dt": dt, "row_mass": row_mass, "seed": seed,
                    "median_bandwidth_trace": h_trace, "max_relative_row_mass_error": root_err,
                    "runtime_s": time.perf_counter() - start,
                }
                output["runs"].append(result)
                print(f"[fpt dt={dt:g} K={row_mass:g} seed {seed}] quench_full={result['quench_first_full']} "
                      f"E/dim end={result['final_energy_per_dim']:.4f} h_med end={h_trace[-1]:.3f} "
                      f"root_err={root_err:.1e} {result['runtime_s']:.0f}s", flush=True)
                out_path.write_text(json.dumps(output) + "\n")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
