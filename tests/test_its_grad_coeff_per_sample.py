import os

# Force CPU backend. This must happen before importing jax.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest

import jax
from jax import numpy as jnp
from jax import random

from fp_transformer.utils import its_grad_coeff_per_sample


def _bruteforce_coeff(beta_target, beta_array, potential_array, f_array, mask):
    beta_array = np.asarray(beta_array)
    f_array = np.asarray(f_array)
    mask = np.asarray(mask)
    u = np.asarray(potential_array)[..., 0]  # (T, N)

    w = np.exp(f_array[:, None] - beta_array[:, None] * u)
    w = w * (mask[:, None] > 0)
    denom = w.sum(axis=0)
    denom = np.where(denom == 0, 1e-300, denom)
    numer = (beta_array[:, None] * w).sum(axis=0)
    return (numer / denom) / float(beta_target)


def test_cpu_backend():
    assert jax.devices()[0].platform == "cpu"


@pytest.mark.parametrize("T,N", [(5, 7), (3, 1), (8, 16)])
def test_its_grad_coeff_per_sample_matches_bruteforce(T, N):
    key = random.PRNGKey(0)
    key, k1, k2, k3, k4 = random.split(key, 5)

    beta_target = 1.5
    beta_array = random.uniform(k1, (T,), minval=0.1, maxval=3.0)

    # Ensure at least one active temperature.
    mask = random.bernoulli(k2, p=0.7, shape=(T,)).astype(jnp.int32)
    mask = mask.at[0].set(1)

    # Use a wider range to exercise numerical stability.
    potential_array = random.normal(k3, (T, N, 1)) * 8.0
    f_array = random.normal(k4, (T,)) * 2.0

    coeff = its_grad_coeff_per_sample(
        beta_target, beta_array, potential_array, f_array, mask
    )
    assert coeff.shape == (N,)

    expected = _bruteforce_coeff(beta_target, beta_array, potential_array, f_array, mask)
    np.testing.assert_allclose(np.asarray(coeff), expected, rtol=1e-5, atol=1e-6)
