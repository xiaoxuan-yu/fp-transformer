import pytest
from fp_transformer.potential.Periodic import Periodic

import jax.numpy as jnp


def test_periodic_init():
    # Test default initialization
    potential = Periodic()
    assert jnp.array_equal(potential.p, jnp.array([1.0]))

    # Test custom period
    potential = Periodic(p=2.0)
    assert jnp.array_equal(potential.p, jnp.array([2.0]))


def test_periodic_call():
    potential = Periodic(p=1.0)

    # Test single point
    x = jnp.array([[[0.0, 0.0]]])
    E = potential(x, beta=1.0)
    assert E.shape == (1, 1)

    # Test multiple points
    x = jnp.array([[[0.0, 0.0]], [[1.0, 1.0]]])
    E = potential(x, beta=1.0)
    assert E.shape == (2, 1)

    # Test input validation
    with pytest.raises(AssertionError):
        x = jnp.array([[[0.0, 0.0, 0.0]]])
        potential(x, beta=1.0)


def test_periodic_2dcontour():
    potential = Periodic()
    x_range = (-2.0, 2.0)
    y_range = (-2.0, 2.0)

    X, Y, E = potential.get_2dcontour(x_range, y_range, num_points=10)

    assert X.shape == (10, 10)
    assert Y.shape == (10, 10)
    assert E.shape == (10, 10)
    assert jnp.min(E) == 0.0  # Due to E -= jnp.min(E)
