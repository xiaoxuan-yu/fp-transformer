"""Fused leave-one-out torus KDE row score (Pallas Triton backend).

With the dimension-normalized chordal distance of ``torus_pairwise_distance``,

    d_ij = 2 - 2 G_ij / M,   G_ij = cos(x_i).cos(x_j) + sin(x_i).sin(x_j),

and a row-specific bandwidth h_i, the Gaussian logits are

    -d_ij / (2 h_i^2) = q_i . k_j - 1 / h_i^2,
    q_i = [cos x_i, sin x_i] / (M h_i^2),   k_j = [cos x_j, sin x_j],

i.e. a self-attention score.  The row constant drops out of the softmax and is
restored only for the row mass.  With w_ij the normalized weights, the score at
the evaluation point x_i (kernel centers held fixed) is

    s_i = 1 / (M h_i^2) sum_j w_ij sin(x_j - x_i)
        = 1 / (M h_i^2) [cos x_i (W S)_i - sin x_i (W C)_i],

so one online-softmax pass over V = [sin x, cos x] returns the score and the
log row mass log sum_j exp(-d_ij / (2 h_i^2)) without materializing an N x N
array.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax

from ..type import JaxArray

DEFAULT_BLOCK_SIZE = 16


def _features(coordinate: JaxArray, bandwidth: JaxArray):
    flat = coordinate.reshape(coordinate.shape[0], -1)
    dimension = flat.shape[1]
    bandwidth = jnp.broadcast_to(
        jnp.asarray(bandwidth, dtype=flat.dtype).reshape(-1), (flat.shape[0],)
    )
    inverse_squared = jnp.reciprocal(jnp.square(bandwidth))
    cosine, sine = jnp.cos(flat), jnp.sin(flat)
    keys = jnp.concatenate([cosine, sine], axis=1)
    values = jnp.concatenate([sine, cosine], axis=1)
    queries = keys * (inverse_squared / dimension)[:, None]
    return flat, cosine, sine, dimension, inverse_squared, queries, keys, values


def torus_row_score_reference(
    coordinate: JaxArray,
    bandwidth: JaxArray,
    *,
    include_self: bool = False,
) -> tuple[JaxArray, JaxArray]:
    """Dense JAX reference of ``torus_row_score``; returns (score, log row mass)."""

    flat, cosine, sine, dimension, inverse_squared, queries, keys, values = _features(
        coordinate, bandwidth
    )
    logits = queries @ keys.T
    if not include_self:
        logits = jnp.where(jnp.eye(flat.shape[0], dtype=bool), -jnp.inf, logits)
    log_normalizer = jax.nn.logsumexp(logits, axis=1)
    weighted = jax.nn.softmax(logits, axis=1) @ values
    score = (
        cosine * weighted[:, :dimension] - sine * weighted[:, dimension:]
    ) * (inverse_squared / dimension)[:, None]
    return score.reshape(coordinate.shape), log_normalizer - inverse_squared


def _row_pass_kernel(
    q_ref, k_ref, v_ref, o_ref, lse_ref, *, num_valid, block_q, block_k, include_self, allow_tf32
):
    from jax.experimental import pallas as pl

    rows = pl.program_id(0) * block_q + jnp.arange(block_q)
    q = q_ref[...]
    feature = v_ref.shape[1]

    def body(j, carry):
        o, m, l = carry
        cols = pl.ds(j * block_k, block_k)
        logits = pl.dot(q, k_ref[cols, :], trans_b=True, allow_tf32=allow_tf32)
        col_ids = j * block_k + jnp.arange(block_k)
        valid = (col_ids < num_valid)[None, :]
        if not include_self:
            valid = valid & (rows[:, None] != col_ids[None, :])
        logits = jnp.where(valid, logits, -jnp.inf)
        m_new = jnp.maximum(m, jnp.max(logits, axis=1))
        m_safe = jnp.where(jnp.isfinite(m_new), m_new, 0.0)
        correction = jnp.exp(m - m_safe)
        p = jnp.exp(logits - m_safe[:, None])
        l_new = correction * l + jnp.sum(p, axis=1)
        o_new = correction[:, None] * o + pl.dot(
            p, v_ref[cols, :], allow_tf32=allow_tf32
        )
        return o_new, m_new, l_new

    o0 = jnp.zeros((block_q, feature), jnp.float32)
    m0 = jnp.full((block_q,), -jnp.inf, jnp.float32)
    l0 = jnp.zeros((block_q,), jnp.float32)
    o, m, l = lax.fori_loop(0, k_ref.shape[0] // block_k, body, (o0, m0, l0))
    o_ref[...] = o / l[:, None]
    lse_ref[...] = m + jnp.log(l)


def _pad_to(array: JaxArray, rows: int, cols: int) -> JaxArray:
    return jnp.pad(array, ((0, rows - array.shape[0]), (0, cols - array.shape[1])))


@functools.partial(
    jax.jit,
    static_argnames=("include_self", "block_size", "num_warps", "num_stages", "allow_tf32"),
)
def _torus_row_score_pallas(
    coordinate, bandwidth, *, include_self, block_size, num_warps, num_stages, allow_tf32
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton

    flat, cosine, sine, dimension, inverse_squared, queries, keys, values = _features(
        coordinate, bandwidth
    )
    num_valid = flat.shape[0]
    if num_valid < 2 and not include_self:
        raise ValueError("leave-one-out KDE requires at least two particles")
    padded_n = math.ceil(num_valid / block_size) * block_size
    feature = max(16, 1 << math.ceil(math.log2(2 * dimension)))
    queries = _pad_to(queries.astype(jnp.float32), padded_n, feature)
    keys = _pad_to(keys.astype(jnp.float32), padded_n, feature)
    values = _pad_to(values.astype(jnp.float32), padded_n, feature)

    row_tile = pl.BlockSpec((block_size, feature), lambda i: (i, 0))
    full = pl.BlockSpec((padded_n, feature), lambda i: (0, 0))
    weighted, log_normalizer = pl.pallas_call(
        functools.partial(
            _row_pass_kernel,
            num_valid=num_valid,
            block_q=block_size,
            block_k=block_size,
            include_self=include_self,
            allow_tf32=allow_tf32,
        ),
        grid=(padded_n // block_size,),
        in_specs=[row_tile, full, full],
        out_specs=[row_tile, pl.BlockSpec((block_size,), lambda i: (i,))],
        out_shape=[
            jax.ShapeDtypeStruct((padded_n, feature), jnp.float32),
            jax.ShapeDtypeStruct((padded_n,), jnp.float32),
        ],
        compiler_params=pltriton.CompilerParams(
            num_warps=num_warps, num_stages=num_stages
        ),
        name="torus_row_score",
    )(queries, keys, values)
    weighted = weighted[:num_valid].astype(flat.dtype)
    score = (
        cosine * weighted[:, :dimension] - sine * weighted[:, dimension : 2 * dimension]
    ) * (inverse_squared / dimension)[:, None]
    log_mass = log_normalizer[:num_valid].astype(flat.dtype) - inverse_squared
    return score.reshape(coordinate.shape), log_mass


def torus_row_score(
    coordinate: JaxArray,
    bandwidth: JaxArray,
    *,
    include_self: bool = False,
    block_size: int = DEFAULT_BLOCK_SIZE,
    num_warps: int = 2,
    num_stages: int = 1,
    allow_tf32: bool = False,
) -> tuple[JaxArray, JaxArray]:
    """Pointwise torus KDE score and log row mass in one fused pass.

    Args:
        coordinate: (N, A, D) angles.
        bandwidth: scalar, (N,), or (N, 1) row bandwidths h_i.
        include_self: keep the j = i term (default: leave-one-out).
        allow_tf32: TF32 tensor-core dots (faster, about 1e-3 relative error).

    Returns:
        (score with the shape of ``coordinate``, log row mass (N,)).
        Off GPU the dense reference is used.
    """

    if jax.default_backend() != "gpu":
        return torus_row_score_reference(
            coordinate, bandwidth, include_self=include_self
        )
    return _torus_row_score_pallas(
        coordinate,
        bandwidth,
        include_self=include_self,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
        allow_tf32=allow_tf32,
    )


__all__ = ["torus_row_score", "torus_row_score_reference"]
