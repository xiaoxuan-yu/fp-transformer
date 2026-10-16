import sys
from dataclasses import replace

import optax
from jax import numpy as jnp
from jax import random

sys.path.append(".")

from src.fp_transformer import CoordTopology, EnergySystem, Sampler, kernel, potential
from src.fp_transformer.core.state import KernelState
from src.fp_transformer.utils import torus_pairwise_distance


def _build_sampler():
    num_replicas = 16
    key = random.PRNGKey(0)
    coord = random.normal(key, (num_replicas, 1, 2)) * 0.1

    init_dist2 = torus_pairwise_distance(coord)
    bw_init = jnp.sort(jnp.triu(init_dist2, k=1).ravel())[::-1][
        (num_replicas - 1) * (num_replicas - 2) // 2 // 2
    ]
    k = kernel.Gaussian(
        num_replicas=num_replicas,
        bandwidth=bw_init,
        update_bw_method="rescale",
        distance_func=torus_pairwise_distance,
    )
    top = CoordTopology(coord)
    energy_ = potential.PeriodicMultidim(
        centers=jnp.array([[0.0, 0.0]]),
        depths=jnp.array([5.0]),
        concentrations=jnp.array([[3.0, 3.0]]),
    )
    sys_ = EnergySystem(topology=top, potential_=energy_)
    opt = optax.adagrad(0.05, 0.1)
    sampler = Sampler(system=sys_, kernel=k, optimizer=opt, temperature=1.0, seed=42)
    sampler.set_beta(2.0)
    return sampler


def test_sampler_init_state_contains_runtime_fields():
    sampler = _build_sampler()
    st = sampler.init_state(sampler.system.coordinate)
    assert st.coordinate.shape == sampler.system.coordinate.shape
    assert st.rng_key.shape == sampler.key.shape
    assert st.nb_list == sampler.system.nbList
    assert st.kernel_state is not None


def test_sampler_step_core_jit_matches_nonjit_with_bandwidth_updates():
    sampler = _build_sampler()
    st0 = sampler.init_state(sampler.system.coordinate)
    noise_scale = jnp.asarray(0.0, dtype=st0.coordinate.dtype)

    assert isinstance(st0.kernel_state, KernelState) or hasattr(
        st0.kernel_state, "bandwidth"
    )

    bw1 = st0.kernel_state.bandwidth * 0.8
    st1 = replace(st0, kernel_state=replace(st0.kernel_state, bandwidth=bw1))
    jit_st1, jit_m1 = sampler.step_core(st1, sampler.beta, noise_scale, sampler.gamma)
    py_st1, py_m1 = sampler._step_core_(st1, sampler.beta, noise_scale, sampler.gamma)

    assert jnp.allclose(jit_st1.coordinate, py_st1.coordinate, rtol=1e-6, atol=1e-7)
    assert jnp.allclose(jit_m1["grad_norm"], py_m1["grad_norm"], rtol=1e-6, atol=1e-7)

    bw2 = st0.kernel_state.bandwidth * 1.2
    st2 = replace(st0, kernel_state=replace(st0.kernel_state, bandwidth=bw2))
    jit_st2, jit_m2 = sampler.step_core(st2, sampler.beta, noise_scale, sampler.gamma)
    py_st2, py_m2 = sampler._step_core_(st2, sampler.beta, noise_scale, sampler.gamma)

    assert jnp.allclose(jit_st2.coordinate, py_st2.coordinate, rtol=1e-6, atol=1e-7)
    assert jnp.allclose(jit_m2["grad_norm"], py_m2["grad_norm"], rtol=1e-6, atol=1e-7)

    # Bandwidth update should actually affect the result.
    assert not jnp.allclose(jit_st1.coordinate, jit_st2.coordinate)


def test_sampler_rff_step_core_jit_matches_nonjit_and_advances_key():
    num_replicas = 16
    key = random.PRNGKey(0)
    coord = random.normal(key, (num_replicas, 1, 2)) * 0.1

    init_dist2 = torus_pairwise_distance(coord)
    bw_init = jnp.sort(jnp.triu(init_dist2, k=1).ravel())[::-1][
        (num_replicas - 1) * (num_replicas - 2) // 2 // 2
    ]
    k = kernel.Gaussian(
        num_replicas=num_replicas,
        bandwidth=bw_init,
        update_bw_method="rescale",
        distance_func=torus_pairwise_distance,
        rff=True,
        rff_dim=64,
        rff_seed=7,
    )
    top = CoordTopology(coord)
    energy_ = potential.PeriodicMultidim(
        centers=jnp.array([[0.0, 0.0]]),
        depths=jnp.array([5.0]),
        concentrations=jnp.array([[3.0, 3.0]]),
    )
    sys_ = EnergySystem(topology=top, potential_=energy_)
    opt = optax.adagrad(0.05, 0.1)
    sampler = Sampler(system=sys_, kernel=k, optimizer=opt, temperature=1.0, seed=99)
    sampler.set_beta(2.0)

    st = sampler.init_state(sampler.system.coordinate)
    noise_scale = jnp.asarray(0.0, dtype=st.coordinate.dtype)

    jit_st, jit_m = sampler.step_core(st, sampler.beta, noise_scale, sampler.gamma)
    py_st, py_m = sampler._step_core_(st, sampler.beta, noise_scale, sampler.gamma)

    assert jnp.allclose(jit_st.coordinate, py_st.coordinate, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(jit_m["grad_norm"], py_m["grad_norm"], rtol=1e-5, atol=1e-6)
    assert jit_st.kernel_state is not None
    assert py_st.kernel_state is not None
    assert not jnp.array_equal(jit_st.kernel_state.rff_key, st.kernel_state.rff_key)
    assert jnp.array_equal(jit_st.kernel_state.rff_key, py_st.kernel_state.rff_key)


def test_sampler_rff_grad_freee_api_uses_state_threaded_key_progression():
    num_replicas = 12
    key = random.PRNGKey(21)
    coord = random.normal(key, (num_replicas, 1, 2)) * 0.1

    init_dist2 = torus_pairwise_distance(coord)
    bw_init = jnp.sort(jnp.triu(init_dist2, k=1).ravel())[::-1][
        (num_replicas - 1) * (num_replicas - 2) // 2 // 2
    ]
    k = kernel.Gaussian(
        num_replicas=num_replicas,
        bandwidth=bw_init,
        update_bw_method="rescale",
        distance_func=torus_pairwise_distance,
        rff=True,
        rff_dim=64,
        rff_seed=17,
    )
    top = CoordTopology(coord)
    energy_ = potential.PeriodicMultidim(
        centers=jnp.array([[0.0, 0.0]]),
        depths=jnp.array([5.0]),
        concentrations=jnp.array([[3.0, 3.0]]),
    )
    sys_ = EnergySystem(topology=top, potential_=energy_)
    opt = optax.adagrad(0.05, 0.1)
    sampler = Sampler(system=sys_, kernel=k, optimizer=opt, temperature=1.0, seed=123)
    sampler.set_beta(2.0)

    coord0 = sampler.system.coordinate
    key0 = sampler.key
    ks0 = sampler._kernel_state_from_kernel()
    next_key, noise_key = random.split(key0)
    noise_scale = jnp.asarray(0.0, dtype=coord0.dtype)
    g_ref, ks_ref = sampler.grad_freeE_with_state_core(
        coord0,
        sampler.beta,
        sampler.system.nbList,
        ks0,
        ks0.bandwidth,
        noise_key,
        noise_scale,
        sampler.gamma,
    )

    sampler.key = key0
    sampler._sync_kernel_from_state(ks0)
    g_api = sampler.grad_freeE(coord0, noise=False)
    ks_api = sampler._kernel_state_from_kernel()

    assert jnp.allclose(g_api, g_ref, rtol=1e-5, atol=1e-6)
    assert jnp.array_equal(ks_api.rff_key, ks_ref.rff_key)
    assert jnp.array_equal(sampler.key, next_key)
