from fp_transformer.utils.mathfunc import (
    dist,
    torus_dist,
    pairwise_distance,
    torus_pairwise_distance,
)

import jax.numpy as jnp


def test_dist():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    expected = jnp.sum((x - y) ** 2)
    assert jnp.allclose(dist(x, y), expected)


def test_torus_dist():
    x = jnp.array([0.0, jnp.pi / 2])
    y = jnp.array([jnp.pi, 3 * jnp.pi / 2])
    result = torus_dist(x, y)
    assert result.shape == ()
    assert jnp.isfinite(result)


def test_pairwise_distance():
    batch = 3
    atoms = 2
    dims = 4
    xs = jnp.ones((batch, atoms, dims))
    result = pairwise_distance(xs)
    assert result.shape == (batch, batch)
    assert jnp.all(result >= -1e-6)
    assert jnp.allclose(result.diagonal(), jnp.zeros(batch), atol=1e-6)


def test_torus_pairwise_distance():
    batch = 3
    atoms = 2
    dims = 4
    xs = jnp.ones((batch, atoms, dims))
    result = torus_pairwise_distance(xs)
    assert result.shape == (batch, batch)
    assert jnp.all(result >= -1e-6)
    assert jnp.allclose(result.diagonal(), jnp.zeros(batch), atol=1e-6)


def test_torus_periodic_distance():
    x = jnp.array([0.0, 0.0])
    y = jnp.array([2 * jnp.pi, 2 * jnp.pi])
    result = torus_dist(x, y)
    assert result.shape == ()
    assert jnp.isfinite(result)
    assert jnp.allclose(result, 0.0)  # Expecting zero distance on the torus


def test_torus_pairwise_distance_shape():
    batch = 4
    atoms = 3
    dims = 2
    xs = jnp.ones((batch, atoms, dims))
    result = torus_pairwise_distance(xs)
    assert result.shape == (batch, batch)
    assert jnp.all(result >= -1e-6)
    assert jnp.allclose(result.diagonal(), jnp.zeros(batch), atol=1e-6)


def test_torus_pairwise_distance_symmetry():
    xs = jnp.array(
        [
            [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]],
            [[0.4, 0.5, 0.6], [1.4, 1.5, 1.6]],
            [[0.7, 0.8, 0.9], [1.7, 1.8, 1.9]],
        ]
    )
    result = torus_pairwise_distance(xs)
    # Check symmetry property: d(a, b) = d(b, a)
    assert jnp.allclose(result, result.T)


def test_torus_pairwise_distance_periodicity():
    # Test that points differing by 2π have distance 0
    batch = 2
    xs = jnp.array(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[2 * jnp.pi, 2 * jnp.pi], [1.0 + 2 * jnp.pi, 1.0 + 2 * jnp.pi]],
        ]
    )
    result = torus_pairwise_distance(xs)
    assert jnp.allclose(result, jnp.zeros((batch, batch)), atol=1e-6)
