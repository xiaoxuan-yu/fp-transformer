"""Weighted leave-one-out product-von-Mises density on the dihedral torus (SI Ala10).

With angular dimension d, pair scales h_ij^2 = (h_i^2 + h_j^2) / 2 and
kappa_ij = 1 / (d h_ij^2),

    q_{-i}(x) = sum_{j != i} w_j prod_l exp[kappa_ij cos(x_l - x_jl)] / (2 pi I_0(kappa_ij))
                / sum_{j != i} w_j.

The same estimator supplies the coordinate score (kernel centers frozen, so
the derivative is the pointwise score at each particle), the log density, and
the gradient of the discrete free energy sum_i w_i [log q_{-i}(x_i) + beta U_i]
with respect to the weights used by the Fisher--Rao step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import i0e

from fp_transformer.utils import torus_pairwise_distance


def query_center_torus_distance(value):
    """``torus_pairwise_distance`` with the kernel centers held fixed.

    Same value, 2 - 2 mean(cos(x_i - x_j)); only the query rows carry a
    derivative.
    """

    flat = value.reshape(value.shape[0], -1)
    centers = jax.lax.stop_gradient(flat)
    dots = jnp.cos(flat) @ jnp.cos(centers).T + jnp.sin(flat) @ jnp.sin(centers).T
    return 2.0 - 2.0 * dots / flat.shape[1]


def make_von_mises_estimator(*, freeze_kernel_centers: bool = True):
    """Return (coordinate_score, log_density, free_energy_weight_value_and_gradient)."""

    def log_density_from_distance(value, bandwidth, particle_weights, distance):
        bandwidth_squared = jnp.square(jnp.asarray(bandwidth).reshape(-1))
        pair_bandwidth_squared = 0.5 * (bandwidth_squared[:, None] + bandwidth_squared[None, :])
        n_dimensions = value.shape[1] * value.shape[2]
        concentration = 1.0 / (n_dimensions * pair_bandwidth_squared)
        # torus_pairwise_distance = 2 - 2 mean(cos(delta)); i0e keeps
        # log(I0(kappa)) - kappa stable for narrow bandwidths.
        logits = n_dimensions * (
            -0.5 * concentration * distance - jnp.log(2.0 * jnp.pi) - jnp.log(i0e(concentration))
        )
        logits = logits + jnp.log(jnp.maximum(particle_weights, jnp.finfo(value.dtype).tiny))[None, :]
        logits = jnp.where(jnp.eye(logits.shape[0], dtype=bool), -jnp.inf, logits)
        normalizer = jnp.sum(particle_weights) - particle_weights
        return jax.nn.logsumexp(logits, axis=1) - jnp.log(
            jnp.maximum(normalizer, jnp.finfo(value.dtype).tiny)
        )

    checkpointed = jax.checkpoint(log_density_from_distance)

    def log_density(value, bandwidth, particle_weights):
        distance = (query_center_torus_distance(value) if freeze_kernel_centers
                    else torus_pairwise_distance(value))
        return checkpointed(value, bandwidth, particle_weights, distance)

    def weighted_objective(value, bandwidth, particle_weights):
        return jnp.sum(particle_weights * log_density(value, bandwidth, particle_weights))

    raw_gradient = jax.grad(weighted_objective, argnums=0)

    @jax.jit
    def coordinate_score(value, bandwidth, particle_weights):
        normalized = particle_weights / jnp.sum(particle_weights)
        gradient = raw_gradient(value, bandwidth, normalized)
        inverse_mass = jnp.reciprocal(jnp.maximum(normalized, jnp.finfo(value.dtype).tiny))
        return gradient * inverse_mass.reshape((len(inverse_mass),) + (1,) * (value.ndim - 1))

    def discrete_free_energy(value, bandwidth, particle_weights, dimensionless_energy):
        normalized = particle_weights / jnp.sum(particle_weights)
        return jnp.sum(normalized * (log_density(value, bandwidth, normalized) + dimensionless_energy))

    return (
        coordinate_score,
        jax.jit(log_density),
        jax.jit(jax.value_and_grad(discrete_free_energy, argnums=2)),
    )
