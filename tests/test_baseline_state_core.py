import sys

import optax
from jax import numpy as jnp
from jax import random

sys.path.append(".")

from src.fp_transformer import CoordTopology, EnergySystem, potential
from src.fp_transformer.core.sampler import BaselineLangevinSampler
from src.fp_transformer.potential.base import Potential


class _QuadraticPotential(Potential):
    def __call__(self, x, beta: float = 1.0):
        del beta
        return 0.5 * jnp.sum(x * x, axis=(-2, -1), keepdims=True)


def _build_baseline_sampler():
    num_replicas = 16
    key = random.PRNGKey(0)
    coord = random.normal(key, (num_replicas, 1, 2)) * 0.1
    top = CoordTopology(coord)
    energy_ = _QuadraticPotential()
    sys_ = EnergySystem(topology=top, potential_=energy_)
    opt = optax.sgd(0.01)
    sampler = BaselineLangevinSampler(
        system=sys_,
        optimizer=opt,
        temperature=300.0,
        kB=1.0,
        friction_factor=0.1,
        tau=0.01,
        update_pace=5,
    )
    return sampler


def test_baseline_init_state_and_chunk_core():
    sampler = _build_baseline_sampler()
    key = random.PRNGKey(123)
    st = sampler.init_state(
        sampler.system.coordinate, key, nb_list=sampler.system.nbList
    )
    st2, metrics = sampler.chunk_core(
        st, 3, sampler.beta, sampler.friction_factor, sampler.tau
    )
    assert st2.coordinate.shape == st.coordinate.shape
    assert st2.rng_key.shape == st.rng_key.shape
    assert jnp.all(jnp.isfinite(st2.coordinate))
    assert jnp.isfinite(metrics["grad_norm"])
    assert not jnp.array_equal(st2.coordinate, st.coordinate)


def test_baseline_run_updates_coordinate():
    sampler = _build_baseline_sampler()
    coord0 = sampler.system.coordinate
    sampler.run(10, random.PRNGKey(7))
    coord1 = sampler.system.coordinate
    assert coord1 is not None
    assert jnp.all(jnp.isfinite(coord1))
    assert not jnp.array_equal(coord0, coord1)


def test_baseline_temperature_scales_one_step_drift():
    key = random.PRNGKey(11)
    coord = random.normal(key, (8, 1, 2)) * 0.1
    top = CoordTopology(coord)
    energy_ = _QuadraticPotential()
    sys_ = EnergySystem(top, energy_)
    opt = optax.sgd(0.01)
    cold = BaselineLangevinSampler(
        system=sys_,
        optimizer=opt,
        temperature=150.0,
        kB=1.0,
        friction_factor=0.1,
        tau=0.01,
        update_pace=5,
    )
    warm = BaselineLangevinSampler(
        system=sys_,
        optimizer=opt,
        temperature=600.0,
        kB=1.0,
        friction_factor=0.1,
        tau=0.01,
        update_pace=5,
    )
    state = cold.init_state(coord, random.PRNGKey(123), nb_list=sys_.nbList)
    cold_state, cold_metrics = cold.step_core(
        state, cold.beta, cold.friction_factor, cold.tau
    )
    warm_state, warm_metrics = warm.step_core(
        state, warm.beta, warm.friction_factor, warm.tau
    )

    grad = sys_.grad_potential(coord, beta=1.0)
    next_key, subkey = random.split(state.rng_key)
    drift = random.normal(subkey, shape=coord.shape)
    expected_cold = (
        coord
        - cold.friction_factor * cold.tau * cold.beta * grad
        + jnp.sqrt(2 * cold.friction_factor * cold.tau) * drift
    )
    expected_warm = (
        coord
        - warm.friction_factor * warm.tau * warm.beta * grad
        + jnp.sqrt(2 * warm.friction_factor * warm.tau) * drift
    )

    assert jnp.allclose(cold_state.coordinate, expected_cold, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(warm_state.coordinate, expected_warm, rtol=1e-6, atol=1e-6)
    assert not jnp.allclose(cold_state.coordinate, warm_state.coordinate)
    assert jnp.allclose(cold_state.rng_key, next_key)
    assert jnp.allclose(warm_state.rng_key, next_key)
    assert cold_metrics["grad_norm"] > warm_metrics["grad_norm"]


def test_baseline_zero_mobility_keeps_coordinate_fixed():
    key = random.PRNGKey(19)
    coord = random.normal(key, (4, 1, 2))
    system = EnergySystem(CoordTopology(coord), _QuadraticPotential())
    sampler = BaselineLangevinSampler(
        system=system,
        optimizer=optax.sgd(0.0),
        temperature=300.0,
        kB=1.0,
        friction_factor=0.0,
        tau=0.01,
    )
    state = sampler.init_state(coord, random.PRNGKey(3), nb_list=system.nbList)
    next_state, _ = sampler.step_core(
        state, sampler.beta, sampler.friction_factor, sampler.tau
    )
    assert jnp.array_equal(next_state.coordinate, coord)
