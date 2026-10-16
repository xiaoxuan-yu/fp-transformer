import pytest
from fp_transformer.potential.BereSzabo import BerezhkovskiiSzabo

import jax.numpy as jnp


def test_berezhkovskii_szabo_init():
    potential = BerezhkovskiiSzabo(beta=1.0, x_0=2.2, omega=2.0, scale=1.01, d=1.0)
    assert jnp.allclose(potential.beta, 1.0)
    assert jnp.allclose(potential.x_0, 2.2)
    assert jnp.allclose(potential.omega2, 4.0)
    assert jnp.allclose(potential.Omega2, 4.04)
    assert jnp.allclose(potential.d, 1.0)


def test_potential_energy_calculation():
    potential = BerezhkovskiiSzabo()
    # 修改输入形状为(3, 1, 2)，表示3个副本，每个副本1个粒子，每个粒子2D坐标
    x = jnp.array([[[0.0, 0.0]], [[1.5, 1.5]], [[-1.5, -1.5]]])
    beta = 1.0
    energies = potential(x, beta)
    assert energies.shape == (3, 1)


def test_2d_contour():
    potential = BerezhkovskiiSzabo()
    x_range = (-3.0, 3.0)
    y_range = (-3.0, 3.0)
    num_grids = 50
    x, y, energy = potential.get_2dcontour(x_range, y_range, num_grids)
    assert x.shape == (num_grids, num_grids)
    assert y.shape == (num_grids, num_grids)
    assert energy.shape == (num_grids, num_grids)


def test_input_dimension_assertion():
    potential = BerezhkovskiiSzabo()
    with pytest.raises(AssertionError):
        # 修改输入形状为(1, 1, 3)，表示1个副本，1个粒子，但坐标维度为3，不符合要求的2D
        x = jnp.array([[[0.0, 0.0, 0.0]]])
        potential(x, 1.0)
