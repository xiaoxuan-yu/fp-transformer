#!/usr/bin/env python3
"""FPT, Langevin step scan, and no-score control on the 100D benchmark (main-text stable-basin discovery).

Every run starts 5120 particles from the seeded single-zero ensemble (seeds
0--3, scale 0.4) unless ``--init-file`` supplies a start ensemble.

* fpt: shared-bandwidth FPT with the fused leave-one-out row pass
  (``fp_transformer.kernel.fused.torus_row_score``).  Before every
  ``--fpt-update-pace``-th update the common bandwidth is multiplied or divided
  by ``--fpt-rescale-factor`` until the mean off-diagonal row mass lies in
  [kernel_min, kernel_max]; particles move by RMSprop (or explicit Euler) on
  beta grad U - s.
* ld: Euler--Maruyama, theta <- wrap(theta - eps beta grad U + sqrt(2 eps) xi).
* noscore: optax.rmsprop(lr, decay=0.9, eps=1e-8) on beta grad U alone.

Every recorded state stores the quench counts and the distinct visitors of each
of the 101 stable basins (``Evaluator``), from which split-half identification
is evaluated.

Paper protocols at beta = 5 (DATA = data/numerical_hidim_periodic_discovery):

    python -m examples.hidim100d.run_discovery --methods fpt \
        --output-dir DATA/tuned/beta5_fpt_fused_row_k20_40_p1_visits
    python -m examples.hidim100d.run_discovery --methods ld noscore \
        --ld-steps 0.003 0.01 0.02 0.03 --output-dir DATA/tuned/beta5_ld_noscore_visits
    # equilibrium reference of split-half identification
    python -m examples.hidim100d.run_discovery --methods ld --ld-steps 0.003 \
        --init-file DATA/equilibrium_beta5/ld_beta5_seed0.npy \
        --output-dir DATA/equilibrium_beta5/ld_split_half
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax

from fp_transformer.kernel.fused import torus_row_score
from fp_transformer.potential import HiDimPeriodic

from examples.hidim100d.common import (
    SIGMA,
    Evaluator,
    build_seeded_coords,
    checkpoints,
    energy_single,
    make_grad,
    wrap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--replicas", type=int, default=5120)
    parser.add_argument("--init-scale", type=float, default=0.4)
    parser.add_argument("--init-file", type=Path, default=None,
                        help="(N, D) .npy start ensemble replacing the seeded start")
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--num-steps", type=int, default=20000)
    parser.add_argument("--methods", nargs="+", choices=("fpt", "ld", "noscore"), default=["fpt"])
    parser.add_argument("--ld-steps", type=float, nargs="+", default=[0.003, 0.01, 0.02, 0.03])
    parser.add_argument("--lr", type=float, default=0.14)
    parser.add_argument("--fpt-bandwidth", type=float, default=1.0)
    parser.add_argument("--fpt-kernel-min", type=float, default=20.0)
    parser.add_argument("--fpt-kernel-max", type=float, default=40.0)
    parser.add_argument("--fpt-update-pace", type=int, default=1)
    parser.add_argument("--fpt-rescale-factor", type=float, default=2.0**0.1)
    parser.add_argument("--fpt-optimizer", choices=("rmsprop", "euler"), default="rmsprop")
    parser.add_argument("--fpt-no-tf32", action="store_true", help="IEEE fp32 dots in the fused row pass")
    parser.add_argument("--radius", type=float, default=1.25)
    parser.add_argument("--quench-step", type=float, default=0.1)
    parser.add_argument("--quench-iters", type=int, default=1500)
    return parser.parse_args()


def run_simple(kind, coord0, *, key, beta, param, grad_fn, evaluator, points):
    if kind == "ld":
        def step(carry, _):
            x, k = carry
            k, sub = jax.random.split(k)
            x = wrap(x - param * beta * grad_fn(x) + jnp.sqrt(2.0 * param) * jax.random.normal(sub, x.shape))
            return (x, k), None

        carry = (coord0, key)
    else:
        opt = optax.rmsprop(param, decay=0.9, eps=1e-8)

        def step(carry, _):
            x, s = carry
            upd, s = opt.update(beta * grad_fn(x), s)
            return (wrap(optax.apply_updates(x, upd)), s), None

        carry = (coord0, opt.init(coord0))
    advance = jax.jit(lambda c, n: jax.lax.scan(step, c, xs=None, length=n)[0], static_argnums=1)
    evaluator.record(0, coord0)
    current = 0
    for target in points[1:]:
        carry = advance(carry, target - current)
        current = target
        evaluator.record(current, carry[0])


def run_fpt(coord0, *, beta, lr, grad_fn, evaluator, points, bandwidth, kernel_min, kernel_max,
            update_pace, rescale_factor, optimizer, allow_tf32):
    """Shared-bandwidth FPT; returns the bandwidth at every checkpoint."""

    def row_stats(x, h):
        score, log_mass = torus_row_score(x[:, None, :], h, allow_tf32=allow_tf32)
        return score[:, 0, :], jnp.mean(jnp.exp(log_mass))

    def refresh(x, h):
        # One-directional rescale: the initial mean row mass fixes the direction.
        score, mass = row_stats(x, h)
        shrink = mass > kernel_max

        def cond(c):
            _, _, mass, it = c
            return jnp.where(shrink, mass > kernel_max, mass < kernel_min) & (it < 128)

        def body(c):
            h, _, _, it = c
            h = jnp.where(shrink, h / rescale_factor, h * rescale_factor)
            score, mass = row_stats(x, h)
            return h, score, mass, it + 1

        h, score, _, _ = lax.while_loop(cond, body, (h, score, mass, 0))
        return h, score

    opt = optax.rmsprop(lr, decay=0.9, eps=1e-8) if optimizer == "rmsprop" else optax.sgd(lr)

    def step(carry, _):
        x, s, h, i = carry
        h, score = lax.cond(i % update_pace == 0, lambda: refresh(x, h), lambda: (h, row_stats(x, h)[0]))
        upd, s = opt.update(beta * grad_fn(x) + score, s)
        return (wrap(optax.apply_updates(x, upd)), s, h, i + 1), h

    advance = jax.jit(lambda c, m: jax.lax.scan(step, c, xs=None, length=m), static_argnums=1)
    carry = (coord0, opt.init(coord0), jnp.float32(bandwidth), jnp.int32(0))
    evaluator.record(0, coord0)
    current, bandwidths = 0, []
    for target in points[1:]:
        carry, hs = advance(carry, target - current)
        current = target
        evaluator.record(current, carry[0])
        bandwidths.append(float(hs[-1]))
    return bandwidths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dim = args.dim
    probe = np.asarray(build_seeded_coords(dim=dim, replicas=256, init_scale=1.5, seed=123))
    package_energy = np.asarray(HiDimPeriodic(dim=dim, sigma=SIGMA)(jnp.asarray(probe[:, None, :])))[:, 0]
    local_energy = np.asarray(jax.vmap(lambda t: energy_single(t, dim))(jnp.asarray(probe)))
    assert np.allclose(package_energy, local_energy, rtol=1e-4, atol=1e-3)

    evaluator = Evaluator(dim, args.radius, args.quench_step, args.quench_iters)
    grad_fn = make_grad(dim)
    points = checkpoints(args.num_steps)
    jobs = [("fpt", args.lr)] if "fpt" in args.methods else []
    jobs += [("ld", s) for s in args.ld_steps] if "ld" in args.methods else []
    jobs += [("noscore", args.lr)] if "noscore" in args.methods else []

    protocol = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    output = {"protocol": protocol | {"sigma": SIGMA, "checkpoints": points}, "runs": []}
    out_path = args.output_dir / "summary.json"
    for kind, param in jobs:
        for seed in args.seeds:
            coord0 = (jnp.asarray(np.load(args.init_file)) if args.init_file is not None
                      else build_seeded_coords(dim=dim, replicas=args.replicas, init_scale=args.init_scale, seed=seed))
            evaluator.reset()
            start = time.perf_counter()
            extra = {}
            if kind == "fpt":
                extra["bandwidth_trace"] = run_fpt(
                    coord0, beta=args.beta, lr=param, grad_fn=grad_fn, evaluator=evaluator, points=points,
                    bandwidth=args.fpt_bandwidth, kernel_min=args.fpt_kernel_min,
                    kernel_max=args.fpt_kernel_max, update_pace=args.fpt_update_pace,
                    rescale_factor=args.fpt_rescale_factor, optimizer=args.fpt_optimizer,
                    allow_tf32=not args.fpt_no_tf32)
                extra |= {"optimizer": args.fpt_optimizer, "allow_tf32": not args.fpt_no_tf32,
                          "rescale_factor": args.fpt_rescale_factor}
            else:
                run_simple(kind, coord0, key=jax.random.PRNGKey(10_000 + seed), beta=args.beta, param=param,
                           grad_fn=grad_fn, evaluator=evaluator, points=points)
            result = evaluator.summary() | extra | {"method": kind, "param": param, "seed": seed,
                                                    "runtime_s": time.perf_counter() - start}
            output["runs"].append(result)
            print(f"[{kind} {param:g} seed {seed}] quench_full={result['quench_first_full']} "
                  f"quench_cum={result['quench_final_cum']} E/dim={result['final_energy_per_dim']:.4f} "
                  f"{result['runtime_s']:.0f}s", flush=True)
            out_path.write_text(json.dumps(output) + "\n")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
