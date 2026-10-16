import sys
from dataclasses import replace

import optax
import pytest
import jax
from jax import numpy as jnp
from jax import random

sys.path.append(".")

from src.fp_transformer import (
    CoordTopology,
    DynamicITSSampler,
    EnergySystem,
    ITSSampler,
    kernel,
    potential,
)
from src.fp_transformer.utils import torus_pairwise_distance


def _build_system(num_replicas=24):
    key = random.PRNGKey(0)
    coord = (
        random.normal(key, (num_replicas, 1, 2)) * 0.1
        + jnp.array([[-1.0, 1.0]])[None, :, :]
    )
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
        centers=jnp.array([[1.5, 1.0], [-1.0, 1.0]]),
        depths=jnp.array([12.0, 10.0]),
        concentrations=jnp.array([[6.0, 2.0], [2.0, 6.0]]),
    )
    sys_ = EnergySystem(topology=top, potential_=energy_)
    opt = optax.adagrad(0.05, 0.1)
    return sys_, k, opt


def _build_system_rff(num_replicas=24):
    key = random.PRNGKey(123)
    coord = (
        random.normal(key, (num_replicas, 1, 2)) * 0.1
        + jnp.array([[-1.0, 1.0]])[None, :, :]
    )
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
        rff_seed=11,
    )
    top = CoordTopology(coord)
    energy_ = potential.PeriodicMultidim(
        centers=jnp.array([[1.5, 1.0], [-1.0, 1.0]]),
        depths=jnp.array([12.0, 10.0]),
        concentrations=jnp.array([[6.0, 2.0], [2.0, 6.0]]),
    )
    sys_ = EnergySystem(topology=top, potential_=energy_)
    opt = optax.adagrad(0.05, 0.1)
    return sys_, k, opt


def test_its_state_roundtrip_and_grad_core_equivalence():
    sys_, k, opt = _build_system()
    beta_array = jnp.array([0.5, 1.0, 1.5, 2.0], dtype=jnp.float32)
    sampler = ITSSampler(
        system=sys_,
        kernel=k,
        optimizer=opt,
        temperature=1.0,
        beta_array=beta_array,
        its_iterations=1,
    )
    sampler.min_energy = jnp.min(
        sampler.compute_potential(coord=sampler.system.coordinate)
    )

    st = sampler.get_its_state()
    new_st = replace(st, f_array=st.f_array + 0.3, mask=jnp.ones_like(st.mask))
    sampler.set_its_state(new_st)
    assert jnp.allclose(sampler.f_array, new_st.f_array)
    assert jnp.array_equal(sampler.mask, new_st.mask)

    g_state = sampler.grad_freeE_from_state(
        sampler.system.coordinate, new_st, sampler.min_energy
    )
    g_legacy = sampler.grad_freeE(
        sampler.system.coordinate, new_st.beta_array, new_st.f_array, sampler.min_energy
    )
    assert jnp.allclose(g_state, g_legacy, rtol=1e-6, atol=1e-7)


def test_dynamic_its_state_grad_and_ref_energy_equivalence():
    sys_, k, opt = _build_system()
    sampler = DynamicITSSampler(
        system=sys_,
        kernel=k,
        optimizer=opt,
        beta=2.0,
        max_num_temperatures=6,
        max_steps=20,
        grad_norm_threshold_mixing=1e-3,
        grad_norm_threshold_annealing=1e-3,
    )
    st = sampler.get_its_state()
    beta_array = st.beta_array.at[0].set(1.5).at[1].set(2.0)
    mask = st.mask.at[0].set(1).at[1].set(1)
    f_array = st.f_array.at[0].set(0.0).at[1].set(0.1)
    st = replace(st, beta_array=beta_array, mask=mask, f_array=f_array)
    sampler.set_its_state(st)

    g_state = sampler.grad_freeE_from_state(sampler.system.coordinate, st, sampler.beta)
    g_legacy = sampler.grad_freeE(
        sampler.system.coordinate, st.beta_array, sampler.beta, st.f_array, st.mask
    )
    assert jnp.allclose(g_state, g_legacy, rtol=1e-6, atol=1e-7)

    e_state = sampler.ref_energy_from_state(sampler.system.coordinate, sampler.beta, st)
    e_legacy = sampler.ref_energy(sampler.system.coordinate, sampler.beta)
    assert jnp.allclose(e_state, e_legacy, rtol=1e-6, atol=1e-7)


def test_its_rff_kernel_state_threading_jit():
    sys_, k, opt = _build_system_rff()
    beta_array = jnp.array([0.5, 1.0, 1.5, 2.0], dtype=jnp.float32)
    sampler = ITSSampler(
        system=sys_,
        kernel=k,
        optimizer=opt,
        temperature=1.0,
        beta_array=beta_array,
        its_iterations=1,
    )
    sampler.min_energy = jnp.min(
        sampler.compute_potential(coord=sampler.system.coordinate)
    )
    st = sampler.get_its_state()
    ks = sampler._kernel_state_from_kernel()

    g_jit, ks_jit = sampler.grad_freeE_from_state_with_kernel_state(
        sampler.system.coordinate, st, sampler.min_energy, ks
    )
    g_py, ks_py = sampler._grad_freeE_from_state_with_kernel_state_(
        sampler.system.coordinate, st, sampler.min_energy, ks
    )
    assert jnp.allclose(g_jit, g_py, rtol=1e-5, atol=1e-6)
    assert not jnp.array_equal(ks_jit.rff_key, ks.rff_key)
    assert jnp.array_equal(ks_jit.rff_key, ks_py.rff_key)


def test_dynamic_its_rff_kernel_state_threading_jit():
    sys_, k, opt = _build_system_rff()
    sampler = DynamicITSSampler(
        system=sys_,
        kernel=k,
        optimizer=opt,
        beta=2.0,
        max_num_temperatures=6,
        max_steps=20,
        grad_norm_threshold_mixing=1e-3,
        grad_norm_threshold_annealing=1e-3,
    )
    st = sampler.get_its_state()
    st = replace(
        st,
        beta_array=st.beta_array.at[0].set(1.5).at[1].set(2.0),
        f_array=st.f_array.at[0].set(0.0).at[1].set(0.1),
        mask=st.mask.at[0].set(1).at[1].set(1),
    )
    ks = sampler._kernel_state_from_kernel()

    g_jit, ks_jit = sampler.grad_freeE_from_state_with_kernel_state(
        sampler.system.coordinate, st, sampler.beta, ks
    )
    g_py, ks_py = sampler._grad_freeE_from_state_with_kernel_state_(
        sampler.system.coordinate, st, sampler.beta, ks
    )
    assert jnp.allclose(g_jit, g_py, rtol=1e-5, atol=1e-6)
    assert not jnp.array_equal(ks_jit.rff_key, ks.rff_key)
    assert jnp.array_equal(ks_jit.rff_key, ks_py.rff_key)


def test_its_rff_grad_freee_from_state_api_progression_matches_core():
    sys_, k, opt = _build_system_rff()
    beta_array = jnp.array([0.5, 1.0, 1.5, 2.0], dtype=jnp.float32)
    sampler = ITSSampler(
        system=sys_,
        kernel=k,
        optimizer=opt,
        temperature=1.0,
        beta_array=beta_array,
        its_iterations=1,
    )
    sampler.min_energy = jnp.min(
        sampler.compute_potential(coord=sampler.system.coordinate)
    )
    its_state = sampler.get_its_state()
    ks0 = sampler._kernel_state_from_kernel()
    g_ref, ks_ref = sampler.grad_freeE_from_state_with_kernel_state(
        sampler.system.coordinate, its_state, sampler.min_energy, ks0
    )

    sampler._sync_kernel_from_state(ks0)
    g_api = sampler.grad_freeE_from_state(
        sampler.system.coordinate, its_state, sampler.min_energy
    )
    ks_api = sampler._kernel_state_from_kernel()

    assert jnp.allclose(g_api, g_ref, rtol=1e-5, atol=1e-6)
    assert jnp.array_equal(ks_api.rff_key, ks_ref.rff_key)


def test_dynamic_its_rff_grad_freee_from_state_api_progression_matches_core():
    sys_, k, opt = _build_system_rff()
    sampler = DynamicITSSampler(
        system=sys_,
        kernel=k,
        optimizer=opt,
        beta=2.0,
        max_num_temperatures=6,
        max_steps=20,
        grad_norm_threshold_mixing=1e-3,
        grad_norm_threshold_annealing=1e-3,
    )
    its_state = sampler.get_its_state()
    its_state = replace(
        its_state,
        beta_array=its_state.beta_array.at[0].set(1.5).at[1].set(2.0),
        f_array=its_state.f_array.at[0].set(0.0).at[1].set(0.1),
        mask=its_state.mask.at[0].set(1).at[1].set(1),
    )
    ks0 = sampler._kernel_state_from_kernel()
    g_ref, ks_ref = sampler.grad_freeE_from_state_with_kernel_state(
        sampler.system.coordinate, its_state, sampler.beta, ks0
    )

    sampler._sync_kernel_from_state(ks0)
    g_api = sampler.grad_freeE_from_state(
        sampler.system.coordinate, its_state, sampler.beta
    )
    ks_api = sampler._kernel_state_from_kernel()

    assert jnp.allclose(g_api, g_ref, rtol=1e-5, atol=1e-6)
    assert jnp.array_equal(ks_api.rff_key, ks_ref.rff_key)


def test_its_verbose_flag_does_not_change_numerics():
    beta_array = jnp.array([0.5, 1.0, 1.5, 2.0], dtype=jnp.float32)

    sys1, k1, opt1 = _build_system()
    s1 = ITSSampler(
        system=sys1,
        kernel=k1,
        optimizer=opt1,
        temperature=1.0,
        beta_array=beta_array,
        its_iterations=1,
    )
    s1.min_energy = jnp.min(s1.compute_potential(coord=s1.system.coordinate))
    st1 = s1.get_its_state()
    st1 = s1.run_fix_ensemble(num_steps=4, verbose=False, verbose_pace=1, its_state=st1)

    sys2, k2, opt2 = _build_system()
    s2 = ITSSampler(
        system=sys2,
        kernel=k2,
        optimizer=opt2,
        temperature=1.0,
        beta_array=beta_array,
        its_iterations=1,
    )
    s2.min_energy = jnp.min(s2.compute_potential(coord=s2.system.coordinate))
    st2 = s2.get_its_state()
    st2 = s2.run_fix_ensemble(num_steps=4, verbose=True, verbose_pace=1, its_state=st2)

    assert jnp.allclose(
        s1.system.coordinate, s2.system.coordinate, rtol=1e-6, atol=1e-7
    )
    assert jnp.allclose(st1.f_array, st2.f_array, rtol=1e-6, atol=1e-7)
    assert jnp.allclose(st1.beta_array, st2.beta_array, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(
    jax.process_count() < 2,
    reason="Requires multi-process JAX runtime.",
)
def test_distributed_verbose_split_contract_placeholder():
    # Contract test placeholder for distributed runs:
    # rank0 may set verbose=True while others verbose=False, but compute paths
    # must stay numerically equivalent.
    assert True
