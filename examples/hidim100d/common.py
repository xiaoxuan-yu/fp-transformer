"""Shared pieces of the one-hundred-dimensional periodic benchmark.

U_D(theta) = (D + sum_j cos theta_j) / sigma^2 - log sum_i exp(2 cos theta_i / sigma^2),
sigma = pi sqrt(2) / 6.  For D = 100 the stable set is the all-pi corner and the
100 corners with one zero coordinate (101 basins).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

SIGMA = math.pi * math.sqrt(2.0) / 6.0


def checkpoints(num_steps: int) -> list[int]:
    """Every update to 200, every 10 updates to 2000, every 100 updates thereafter."""
    points = list(range(0, min(200, num_steps) + 1))
    points += list(range(210, min(2000, num_steps) + 1, 10))
    points += list(range(2100, num_steps + 1, 100))
    return points


def wrap(x):
    return jnp.arctan2(jnp.sin(x), jnp.cos(x))


def energy_single(theta, dim):
    inv_s2 = 1.0 / SIGMA**2
    c = jnp.cos(theta)
    return (dim + jnp.sum(c)) * inv_s2 - logsumexp(2.0 * c * inv_s2)


def make_grad(dim):
    return jax.vmap(jax.grad(lambda t: energy_single(t, dim)))


def build_seeded_coords(*, dim: int, replicas: int, init_scale: float, seed: int, mode_index: int = 0):
    """Wrapped Gaussian of scale ``init_scale`` about the single-zero minimum ``mode_index``."""
    center = np.full((dim,), np.pi, dtype=np.float32)
    center[mode_index] = 0.0
    noise = jax.random.normal(jax.random.PRNGKey(seed), (replicas, 1, dim), dtype=jnp.float32)
    coord = jnp.asarray(center, dtype=jnp.float32).reshape(1, 1, dim) + noise * float(init_scale)
    return wrap(coord)[:, 0, :]


def make_labels(dim, radius):
    """Analytic stable-basin label per particle: 0..D-1 single-zero corner, D all-pi, -1 none."""
    theta_star = math.acos(-1.0 + 0.5 * SIGMA**2 * math.log(dim - 1))

    def labels(theta):
        d0 = jnp.abs(theta)
        dpi = jnp.pi - d0
        zero_count = jnp.sum(d0 < dpi, axis=1)
        binary = jnp.sqrt(jnp.sum(jnp.minimum(d0, dpi) ** 2, axis=1))
        all_pi_sq = jnp.sum(dpi**2, axis=1, keepdims=True)
        ds = jnp.minimum(jnp.abs(wrap(theta - theta_star)), jnp.abs(wrap(theta + theta_star)))
        barrier = jnp.sqrt(jnp.min(all_pi_sq - dpi**2 + ds**2, axis=1))
        barrier_mask = (barrier <= radius) & (barrier < binary)
        assigned = (binary <= radius) & ~barrier_mask
        sat = assigned & (zero_count == 1)
        cen = assigned & (zero_count == 0)
        return jnp.where(sat, jnp.argmin(d0, axis=1), jnp.where(cen, dim, -1))

    return labels


class Evaluator:
    """Discovery metrics at checkpoints.

    ``raw``: the configuration lies within toroidal distance ``radius`` of an
    analytic stable corner (saddle-near configurations excluded); ``quench``:
    the same rule after gradient descent on U to the inherent structure.
    ``quench_counts`` are the particles currently quenched into each basin and
    ``distinct_visitors`` the replicas that have been quenched into it at any
    recorded state, the inputs of split-half identification.
    """

    def __init__(self, dim, radius, quench_step, quench_iters):
        self.dim = dim
        grad_fn = make_grad(dim)
        labels = make_labels(dim, radius)
        energy = jax.vmap(lambda t: energy_single(t, dim))

        @jax.jit
        def evaluate(theta):
            theta = wrap(theta)
            quenched = jax.lax.fori_loop(0, quench_iters, lambda i, y: wrap(y - quench_step * grad_fn(y)), theta)
            raw = labels(theta)
            q = labels(quenched)

            def hit(lab):
                return jnp.zeros(dim + 1, bool).at[jnp.where(lab >= 0, lab, dim + 1)].set(True, mode="drop")

            return (hit(raw), hit(q), jnp.mean(energy(theta)) / dim, jnp.mean(q >= 0),
                    jnp.sum(q == dim), q)

        self._evaluate = evaluate
        self.reset()

    def reset(self):
        self.raw_visited = np.zeros(self.dim + 1, bool)
        self.q_visited = np.zeros(self.dim + 1, bool)
        # Per-replica record of the quenched basins visited at recorded states; the number
        # of distinct replicas per basin defines split-half identification.
        self.visit_mask = None
        self.trace = {k: [] for k in ("step", "raw_cum", "quench_cum", "raw_inst", "quench_inst",
                                      "energy_per_dim", "quench_assigned", "quench_central",
                                      "distinct_visitors", "quench_counts")}

    def record(self, step, theta):
        raw, q, e, assigned, central, lab = (np.asarray(v)
                                             for v in self._evaluate(jnp.asarray(theta)))
        if self.visit_mask is None:
            self.visit_mask = np.zeros((len(lab), self.dim + 1), bool)
        ok = lab >= 0
        self.visit_mask[np.nonzero(ok)[0], lab[ok]] = True
        self.raw_visited |= raw
        self.q_visited |= q
        t = self.trace
        t["step"].append(int(step))
        t["raw_cum"].append(int(self.raw_visited.sum()))
        t["quench_cum"].append(int(self.q_visited.sum()))
        t["raw_inst"].append(int(raw.sum()))
        t["quench_inst"].append(int(q.sum()))
        t["energy_per_dim"].append(float(e))
        t["quench_assigned"].append(float(assigned))
        t["quench_central"].append(int(central))
        t["distinct_visitors"].append(self.visit_mask.sum(0).tolist())
        t["quench_counts"].append(np.bincount(lab[ok], minlength=self.dim + 1).tolist())

    def summary(self):
        t = self.trace
        full = self.dim + 1

        def first(key):
            return next((s for s, c in zip(t["step"], t[key]) if c == full), None)

        return {
            "raw_first_full": first("raw_cum"),
            "quench_first_full": first("quench_cum"),
            "raw_final_cum": t["raw_cum"][-1],
            "quench_final_cum": t["quench_cum"][-1],
            "quench_final_inst": t["quench_inst"][-1],
            "final_energy_per_dim": t["energy_per_dim"][-1],
            "trace": t,
        }
