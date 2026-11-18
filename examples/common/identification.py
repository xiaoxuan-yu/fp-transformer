"""Split-half basin identification (SI Eq. for P_id).

A basin is identified when both halves of a random equal division of the N
particles contain a particle that supplies a configuration in it.  For FPT the
supplying particles are those of the current ensemble (well or quench counts);
for Langevin dynamics they are the independent trajectories that have visited
the basin at any recorded state (``distinct_visitors``).  The discovery time
t_id is the first recorded update at which the expected identified fraction
reaches IDENTIFICATION_SHARE of its value for an equilibrium ensemble of N
particles.
"""

from __future__ import annotations

import numpy as np

IDENTIFICATION_SHARE = 0.99


def split_half_probability(visitors, replicas: int) -> np.ndarray:
    """P_id(m) = 1 - 2 C(N-m, N/2) / C(N, N/2) for m supplying particles."""
    from scipy.special import gammaln

    m = np.asarray(visitors, dtype=float)
    half = replicas // 2

    def log_choose(a, b):
        return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)

    small = m <= half
    out = np.ones_like(m)
    out[small] = 1.0 - 2.0 * np.exp(log_choose(replicas - m[small], half)
                                    - log_choose(float(replicas), half))
    out[m == 0] = 0.0
    return out


def equilibrium_identification(weights, replicas: int) -> float:
    """Expected identified fraction of an iid target sample with basin weights ``weights``."""
    from scipy.stats import binom

    m = np.arange(replicas + 1)
    p_id = split_half_probability(m, replicas)
    return float(np.mean([np.sum(binom.pmf(m, replicas, w) * p_id) for w in weights]))


def identification(steps, visitors, replicas: int, reference: float):
    """Expected identified fraction per recorded state, and t_id (None if never reached)."""
    fraction = split_half_probability(visitors, replicas).mean(axis=-1)
    complete = fraction >= IDENTIFICATION_SHARE * reference
    first = float(np.asarray(steps)[np.argmax(complete)]) if complete.any() else None
    return fraction, first
