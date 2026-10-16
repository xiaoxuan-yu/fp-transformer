import os

# Force CPU backend. This must happen before importing jax.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from fp_transformer import kernel
from fp_transformer.utils import torus_pairwise_distance


def test_gaussian_state_non_rff_matches_legacy():
    n = 16
    coord = random.normal(random.PRNGKey(0), (n, 1, 2))
    k = kernel.Gaussian(n, bandwidth=0.2, update_bw_method="trick", rff=False)

    # legacy wrapper path
    g_legacy = k.kernel_grad(coord)

    # explicit-state path should match
    state = k.get_state()
    g_state, state2 = k.kernel_grad_with_state(coord, state)
    np.testing.assert_allclose(
        np.asarray(g_state), np.asarray(g_legacy), rtol=1e-6, atol=1e-6
    )

    # non-rff path should not touch key/state shape
    assert state2.rff_key.shape == state.rff_key.shape


def test_gaussian_exact_kernel_excludes_self_term():
    coord = jnp.array([[[0.0]], [[0.0]], [[2.0]]], dtype=jnp.float32)
    g = kernel.Gaussian(3, bandwidth=1.0, update_bw_method="rescale", rff=False)

    log_density = g.kernel(coord, jnp.asarray(1.0)).reshape(-1)
    expected_row_sum = jnp.array(
        [1.0 + jnp.exp(-2.0), 1.0 + jnp.exp(-2.0), 2.0 * jnp.exp(-2.0)]
    )

    np.testing.assert_allclose(
        np.asarray(jnp.exp(log_density)),
        np.asarray(expected_row_sum),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(g.compute_kernel_row_sum(coord, jnp.asarray(1.0))),
        np.asarray(expected_row_sum),
        rtol=1e-6,
        atol=1e-6,
    )


def test_gaussian_rescale_row_sum_excludes_diagonal():
    n = 4
    g = kernel.Gaussian(
        n,
        update_bw_method="rescale",
        rff=False,
        kernel_min=0.1,
        kernel_max=0.2,
        max_rescale_iters=1,
    )
    kernel_mat = jnp.eye(n, dtype=jnp.float32)

    _, row_sum = g.rescale_kernel(kernel_mat, jnp.ones((n,), dtype=jnp.float32))

    assert float(row_sum) == 0.0


def test_gaussian_state_rff_key_progresses():
    n = 16
    coord = random.normal(random.PRNGKey(1), (n, 1, 2))
    k = kernel.Gaussian(
        n,
        bandwidth=0.2,
        update_bw_method="rescale",
        rff=True,
        rff_dim=64,
        rff_seed=123,
    )
    state0 = k.get_state()

    _, state1 = k.kernel_grad_with_state(coord, state0)
    _, state2 = k.kernel_grad_with_state(coord, state1)

    assert not jnp.array_equal(state0.rff_key, state1.rff_key)
    assert not jnp.array_equal(state1.rff_key, state2.rff_key)


def test_gaussian_rff_kernel_excludes_represented_self_term():
    n = 8
    coord = random.normal(random.PRNGKey(19), (n, 1, 2))
    g = kernel.Gaussian(
        n,
        bandwidth=0.7,
        update_bw_method="rescale",
        rff=True,
        rff_dim=256,
        rff_seed=31,
    )
    state = g.get_state()
    _, subkey = random.split(state.rff_key)
    features = g._compute_rff_features(coord, jnp.asarray(0.7), subkey)
    full_row_sum = features @ jnp.sum(features, axis=0)
    self_term = jnp.sum(jnp.square(features), axis=-1)
    expected = jnp.log(jnp.maximum(full_row_sum - self_term, 1e-10))

    actual, _ = g.kernel_with_state(coord, state)

    np.testing.assert_allclose(
        np.asarray(actual.reshape(-1)),
        np.asarray(expected),
        rtol=1e-6,
        atol=1e-6,
    )


def test_gaussian_state_rff_grad_jittable():
    n = 12
    coord = random.normal(random.PRNGKey(2), (n, 1, 2))
    k = kernel.Gaussian(
        n,
        bandwidth=0.15,
        update_bw_method="rescale",
        rff=True,
        rff_dim=32,
        rff_seed=42,
    )
    state = k.get_state()

    @jax.jit
    def run_once(coord_, state_):
        return k.kernel_grad_with_state(coord_, state_)

    grad_val, next_state = run_once(coord, state)
    assert grad_val.shape == coord.shape
    assert next_state.rff_key.shape == state.rff_key.shape


def test_bw_reset_does_not_trigger_kernel_compute():
    n = 8
    k = kernel.Gaussian(n, bandwidth=0.3, update_bw_method="trick", rff=False)
    original = k.bandwidth
    k.bandwidth = jnp.ones_like(k.bandwidth) * 1.7

    # If reset path accidentally computes kernel, this will fail.
    k.kernel_jit = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("kernel_jit should not be called")
    )

    k.bw_reset()
    np.testing.assert_allclose(
        np.asarray(k.bandwidth), np.asarray(original), rtol=1e-6, atol=1e-6
    )

    # State API should also be pure reset.
    st = k.get_state()
    st2 = k.bw_reset_with_state(st)
    np.testing.assert_allclose(
        np.asarray(st2.bandwidth), np.asarray(original), rtol=1e-6, atol=1e-6
    )


def test_rescale_per_replica_flag_controls_independent_scaling():
    n = 4
    # Synthetic kernel matrix with mixed row-sum regimes:
    # row0 low, row1 high, row2 in-range, row3 low.
    kernel_mat = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.2, 0.2, 0.2, 0.2],
        ],
        dtype=jnp.float32,
    )
    init_scale = jnp.ones((n,), dtype=jnp.float32)

    g_global = kernel.Gaussian(
        n,
        update_bw_method="rescale",
        rff=False,
        kernel_min=1.5,
        kernel_max=2.5,
        rescale_per_replica=False,
    )
    scale_global, _ = g_global.rescale_kernel(kernel_mat, init_scale)
    np.testing.assert_allclose(
        np.asarray(scale_global),
        np.ones((n,), dtype=np.float32) * np.asarray(scale_global)[0],
        rtol=1e-6,
        atol=1e-6,
    )

    g_local = kernel.Gaussian(
        n,
        update_bw_method="rescale",
        rff=False,
        kernel_min=1.5,
        kernel_max=2.5,
        rescale_per_replica=True,
    )
    scale_local, _ = g_local.rescale_kernel(kernel_mat, init_scale)
    assert np.std(np.asarray(scale_local)) > 0.0


def test_rescale_per_replica_accepts_scalar_bandwidth_runtime():
    n = 8
    coord = random.normal(random.PRNGKey(7), (n, 1, 2))
    g = kernel.Gaussian(
        n,
        bandwidth=0.2,
        update_bw_method="rescale",
        rff=False,
        rescale_per_replica=True,
    )

    # Simulate runtime scalar assignment from user code.
    g.bandwidth = jnp.asarray(0.1, dtype=jnp.float32)
    state = g.get_state()
    state2 = g.update_with_state(coord, state)

    assert state2.bandwidth.shape == (n,)


def test_per_replica_rescale_config_interface_compatibility():
    n = 6
    g = kernel.Gaussian(
        n,
        bandwidth=0.2,
        update_bw_method="rescale",
        rff=False,
        rescale_per_replica=True,
    )
    assert g.rescale_per_replica is True
    assert g.rescale_per_replica_config.enabled is True


def test_per_replica_rescale_config_fixed_bounds_override_defaults():
    n = 6
    cfg = kernel.PerReplicaRescaleConfig(
        enabled=True,
        target_mode="fixed",
        fixed_row_sum_min=1.1,
        fixed_row_sum_max=1.9,
    )
    g = kernel.Gaussian(
        n,
        bandwidth=0.2,
        update_bw_method="rescale",
        rff=False,
        rescale_per_replica_config=cfg,
    )
    assert g.rescale_per_replica is True
    assert np.isclose(float(g.ksum_min), 1.1)
    assert np.isclose(float(g.ksum_max), 1.9)


def test_per_replica_rescale_config_validation():
    n = 4
    bad_cfg = kernel.PerReplicaRescaleConfig(
        enabled=True,
        bw_min=1.0,
        bw_max=1e-3,
    )
    with pytest.raises(ValueError):
        kernel.Gaussian(
            n,
            update_bw_method="rescale",
            rff=False,
            rescale_per_replica_config=bad_cfg,
        )


def test_per_replica_rescale_config_enforces_bw_bounds():
    n = 4
    kernel_mat = jnp.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.2, 0.2, 0.2, 0.2],
        ],
        dtype=jnp.float32,
    )
    init_scale = jnp.ones((n,), dtype=jnp.float32)
    cfg = kernel.PerReplicaRescaleConfig(
        enabled=True,
        target_mode="fixed",
        fixed_row_sum_min=1.5,
        fixed_row_sum_max=2.5,
        bw_min=0.2,
        bw_max=0.3,
    )
    g = kernel.Gaussian(
        n,
        update_bw_method="rescale",
        rff=False,
        rescale_per_replica_config=cfg,
    )
    scale, _ = g.rescale_kernel(kernel_mat, init_scale)
    assert float(jnp.min(scale)) >= 0.2 - 1e-6
    assert float(jnp.max(scale)) <= 0.3 + 1e-6


def test_per_replica_rescale_adaptive_quantile_runs_and_is_bounded():
    n = 8
    coord = random.normal(random.PRNGKey(11), (n, 1, 2))
    cfg = kernel.PerReplicaRescaleConfig(
        enabled=True,
        target_mode="adaptive_quantile",
        adaptive_ref_quantile=0.5,
        adaptive_band_low=0.8,
        adaptive_band_high=1.2,
        bw_min=1e-2,
        bw_max=2e-1,
    )
    g = kernel.Gaussian(
        n,
        bandwidth=0.1,
        update_bw_method="rescale",
        rff=False,
        distance_func=torus_pairwise_distance,
        rescale_per_replica_config=cfg,
    )
    st = g.get_state()
    st2 = g.update_with_state(coord, st)
    bw = jnp.asarray(st2.bandwidth)
    assert float(jnp.min(bw)) >= 1e-2 - 1e-8
    assert float(jnp.max(bw)) <= 2e-1 + 1e-8
