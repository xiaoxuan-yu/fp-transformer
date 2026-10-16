import jax
import jax.numpy as jnp
import numpy as np
import optax

from fp_transformer.optim import replica_global_rms


def test_replica_global_rms_preserves_direction_per_replica():
    gradient = jax.random.normal(jax.random.PRNGKey(0), (5, 2, 3))
    gradient = gradient * jnp.arange(1.0, 6.0)[:, None, None]
    params = jnp.zeros_like(gradient)
    optimizer = replica_global_rms(0.1)
    update, _ = optimizer.update(gradient, optimizer.init(params))
    # First step: m/sqrt(v) with bias correction is g / rms(g) per replica.
    rms = jnp.sqrt(jnp.mean(jnp.square(gradient), axis=(1, 2), keepdims=True))
    np.testing.assert_allclose(update, -0.1 * gradient / rms, rtol=1e-5)


def test_replica_global_rms_matches_adam_for_one_coordinate():
    gradients = jax.random.normal(jax.random.PRNGKey(1), (4, 6, 1))
    params = jnp.zeros((6, 1))
    rms, adam = replica_global_rms(0.05), optax.adam(0.05)
    rms_state, adam_state = rms.init(params), adam.init(params)
    for gradient in gradients:
        rms_update, rms_state = rms.update(gradient, rms_state)
        adam_update, adam_state = adam.update(gradient, adam_state)
        np.testing.assert_allclose(rms_update, adam_update, rtol=1e-5, atol=1e-7)
