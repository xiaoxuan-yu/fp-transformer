from fp_transformer import potential, EnergySystem, CoordTopology
from jax import random
from jax import numpy as jnp


def test_coordtopology():
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    coord = random.normal(subkey, (10, 3))
    top = CoordTopology(coord)
    flag1 = jnp.all((top.coordinate == coord))
    key, subkey = random.split(key)
    new_coord = random.normal(subkey, (10, 3))
    top.coordinate = new_coord
    flag2 = jnp.all((top.coordinate == new_coord))
    assert flag1 and flag2


def test_energysystem():
    ene = potential.TiwaryBerne()
    shapes = [(2, 1, 2), (64, 1, 2), (384, 1, 2)]
    for shape in shapes:
        key = random.PRNGKey(0)
        coord = random.normal(key, shape)
        top = CoordTopology(coord)
        sys = EnergySystem(top, ene)
        flag1 = jnp.all((sys.coordinate == coord))
        key, subkey = random.split(key)
        new_coord = random.normal(subkey, shape)
        sys.coordinate = new_coord
        flag2 = jnp.all((sys.coordinate == new_coord))
        flag3 = jnp.all((sys.potential() == ene(new_coord)))
        flag4 = sys.grad_potential(sys.coordinate).shape == new_coord.shape
        assert flag1 and flag2 and flag3 and flag4


def test_energysystem_grad_minibatch_matches_full():
    ene = potential.TiwaryBerne()
    key = random.PRNGKey(42)
    coord = random.normal(key, (16, 1, 2))
    top = CoordTopology(coord)
    sys = EnergySystem(top, ene, minibatch_size=4, assert_divisible_minibatch=True)

    sys.minibatch_size = None
    g_full = sys.grad_potential(coord, beta=1.0)
    sys.minibatch_size = 4
    g_chunk = sys.grad_potential(coord, beta=1.0)

    assert g_chunk.shape == coord.shape
    assert jnp.allclose(g_full, g_chunk, rtol=1e-6, atol=1e-8)


def test_energysystem_potential_minibatch_matches_full():
    ene = potential.TiwaryBerne()
    key = random.PRNGKey(7)
    coord = random.normal(key, (16, 1, 2))
    top = CoordTopology(coord)
    sys = EnergySystem(top, ene, minibatch_size=4, assert_divisible_minibatch=True)

    sys.minibatch_size = None
    e_full = sys.potential(beta=1.0, coord=coord)
    sys.minibatch_size = 4
    e_chunk = sys.potential(beta=1.0, coord=coord)
    assert e_full.shape == e_chunk.shape
    assert jnp.allclose(e_full, e_chunk, rtol=1e-6, atol=1e-8)
