import jax
from jax import numpy as jnp
import typing as _t


from ..type import Array, JaxArray
from .its_utils import (
    compute_next_anneal_factor,
    its_grad_coeff,
    its_grad_coeff_naive,
    its_grad_coeff_per_sample,
)
from .mathfunc import (
    logsumexp,
    pairwise_distance,
    pairwise_distance_between,
    torus_pairwise_distance,
    torus_pairwise_distance_between,
    dihedral,
    dist_mat_to_gaussian_kernel_exp,
)
from .periodic_multidim import (
    generate_subspace_aware_params,
    PotentialParameters,
    init_params_by_nearest_ci,
)
from .minimization import minimize_with_amber_system, minimize_with_energy_fn

# get sharding


def get_jax_array(
    x: _t.Union[float, int, Array, JaxArray, _t.List[_t.Union[int, float]]],
    requires_grad: bool = True,
    sharding=None,
) -> JaxArray:
    """
    Convert input into JaxArray.
    Args:
        x: input.
        requires_grad: whether require gradient.
    Returns:
        JaxArray: JaxArray.
    """
    try:
        x = jnp.asarray(x, device=sharding)
    except Exception:
        x = jnp.asarray(x)
    if not requires_grad:
        x = jax.lax.stop_gradient(x)
    # for float and int, reshape
    if x.ndim == 0:
        x = x.reshape(1)
    return x


def any_nan_in_tree(x):
    return jnp.any(jnp.isnan(x))


__all__ = [
    "pairwise_distance",
    "torus_pairwise_distance",
    "logsumexp",
    "dihedral",
    "get_jax_array",
    "its_grad_coeff",
    "its_grad_coeff_per_sample",
    "its_grad_coeff_naive",
    "compute_next_anneal_factor",
    "generate_subspace_aware_params",
    "init_params_by_nearest_ci",
    "any_nan_in_tree",
    "PotentialParameters",
    "dist_mat_to_gaussian_kernel_exp",
    "minimize_with_amber_system",
    "minimize_with_energy_fn",
]
