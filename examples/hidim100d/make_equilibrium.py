#!/usr/bin/env python3
"""Pre-equilibrated ensemble of U_D from long overdamped Langevin dynamics (SI).

LD (eps=0.003, ~1% energy bias at beta=5) starts from the seeded satellite-0
ensemble and runs 60000 updates; the stationarity of energy per coordinate and
of the quenched central-basin fraction is checked between the second and last
quarters before the final coordinates are saved.

Paper protocols (DATA = data/numerical_hidim_periodic_discovery):

    python -m examples.hidim100d.make_equilibrium --beta 5 --out-dir DATA/equilibrium_beta5
    python -m examples.hidim100d.make_equilibrium --dim 10 --beta 4.3 \
        --out-dir DATA/equilibrium_dimscan/D10_beta4.3     # also --replicas 10240, 20480
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.hidim100d.common import build_seeded_coords, energy_single, make_grad, make_labels, wrap

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--beta", type=float, default=5.0)
parser.add_argument("--eps", type=float, default=0.003)
parser.add_argument("--steps", type=int, default=60000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dim", type=int, default=100)
parser.add_argument("--replicas", type=int, default=5120)
parser.add_argument("--out-dir", type=Path, required=True)
args = parser.parse_args()

D, N = args.dim, args.replicas
OUT = args.out_dir
grad_fn = make_grad(D)
labels = make_labels(D, 1.25)
energy = jax.vmap(lambda t: energy_single(t, D))


@jax.jit
def quench_labels(x):
    q = jax.lax.fori_loop(0, 1500, lambda i, y: wrap(y - 0.1 * grad_fn(y)), x)
    return labels(q)


def block(x, key, n):
    def step(c, _):
        x, k = c
        k, s = jax.random.split(k)
        return (wrap(x - args.eps * args.beta * grad_fn(x)
                     + jnp.sqrt(2 * args.eps) * jax.random.normal(s, x.shape)), k), None
    return jax.lax.scan(step, (x, key), None, length=n)[0]


block = jax.jit(block, static_argnums=2)
x = build_seeded_coords(dim=D, replicas=N, init_scale=0.4, seed=args.seed)
key = jax.random.PRNGKey(20261012 + args.seed)
rows = []
for t in range(2000, args.steps + 1, 2000):
    x, key = block(x, key, 2000)
    lab = np.asarray(quench_labels(x))
    rows.append({"step": t, "energy_per_dim": float(jnp.mean(energy(x)) / D),
                 "central_fraction": float(np.mean(lab == D)),
                 "assigned_fraction": float(np.mean(lab >= 0)),
                 "occupied_basins": int(np.unique(lab[lab >= 0]).size)})
    print(rows[-1], flush=True)
q = len(rows) // 4
summary = {k: (float(np.mean([r[k] for r in rows[q:2 * q]])), float(np.mean([r[k] for r in rows[3 * q:]])))
           for k in ("energy_per_dim", "central_fraction")}
OUT.mkdir(parents=True, exist_ok=True)
np.save(OUT / f"ld_beta{args.beta:g}_seed{args.seed}{'' if N == 5120 else f'_N{N}'}.npy", np.asarray(x, dtype=np.float32))
(OUT / f"ld_beta{args.beta:g}_seed{args.seed}{'' if N == 5120 else f'_N{N}'}.json").write_text(json.dumps(
    {"protocol": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "trace": rows, "quarter2_vs_quarter4": summary}, indent=1) + "\n")
print("quarter 2 vs 4 means:", summary)
