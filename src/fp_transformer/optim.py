"""Optax transformations for replica ensembles."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


class ScaleByReplicaGlobalRMSState(NamedTuple):
    """State for a direction-preserving per-replica RMS preconditioner."""

    count: jax.Array
    momentum: Any
    second_moment: Any


def scale_by_replica_global_rms(
    *, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8
) -> optax.GradientTransformation:
    """Precondition each replica by one scalar instead of per coordinate.

    A vector momentum is retained, but the bias-corrected second moment is
    averaged over all non-replica axes (axis 0 indexes replicas), so the
    update direction of every replica is that of its momentum.
    """

    def init_fn(params):
        def scalar_second_moment(parameter):
            if parameter.ndim < 1:
                raise ValueError("replica-global RMS parameters require a replica axis")
            return jnp.zeros(
                (parameter.shape[0],) + (1,) * (parameter.ndim - 1),
                dtype=parameter.dtype,
            )

        return ScaleByReplicaGlobalRMSState(
            count=jnp.zeros([], dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
            second_moment=jax.tree.map(scalar_second_moment, params),
        )

    def update_fn(updates, state, params=None):
        del params
        count = optax.safe_increment(state.count)
        momentum = jax.tree.map(
            lambda old, gradient: b1 * old + (1.0 - b1) * gradient,
            state.momentum,
            updates,
        )

        def update_second_moment(old, gradient):
            axes = tuple(range(1, gradient.ndim))
            squared_rms = jnp.mean(jnp.square(gradient), axis=axes, keepdims=True)
            return b2 * old + (1.0 - b2) * squared_rms

        second_moment = jax.tree.map(update_second_moment, state.second_moment, updates)
        b1_correction = 1.0 - jnp.asarray(b1) ** count
        b2_correction = 1.0 - jnp.asarray(b2) ** count
        normalized = jax.tree.map(
            lambda first, second: (first / b1_correction)
            / (jnp.sqrt(second / b2_correction) + eps),
            momentum,
            second_moment,
        )
        return normalized, ScaleByReplicaGlobalRMSState(
            count=count, momentum=momentum, second_moment=second_moment
        )

    return optax.GradientTransformation(init_fn, update_fn)


def replica_global_rms(
    learning_rate, *, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8
) -> optax.GradientTransformation:
    """Adam-like descent with one RMS normalization per replica."""

    return optax.chain(
        scale_by_replica_global_rms(b1=b1, b2=b2, eps=eps),
        optax.scale_by_learning_rate(learning_rate),
    )


__all__ = [
    "ScaleByReplicaGlobalRMSState",
    "replica_global_rms",
    "scale_by_replica_global_rms",
]
