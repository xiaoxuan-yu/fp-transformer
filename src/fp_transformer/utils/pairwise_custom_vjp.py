# file: pairwise_custom_vjp.py
# import jax
import jax.numpy as jnp
from jax import jit, custom_vjp


# --------------------------
# Pairwise Euclidean (custom vjp)
# --------------------------
def make_pairwise_distance():
    @custom_vjp
    def f(xs):
        B, A, D = xs.shape
        M = A * D
        flat = xs.reshape(B, M)
        norm2 = jnp.sum(flat * flat, axis=1)
        dist2 = norm2[:, None] + norm2[None, :] - 2.0 * (flat @ flat.T)
        dist2 = jnp.maximum(dist2, 0.0)
        return dist2 / M

    def f_fwd(xs):
        B, A, D = xs.shape
        M = A * D
        flat = xs.reshape(B, M)
        norm2 = jnp.sum(flat * flat, axis=1)
        dist2 = norm2[:, None] + norm2[None, :] - 2.0 * (flat @ flat.T)
        dist2 = jnp.maximum(dist2, 0.0)
        # save flat and shape info for backward
        return dist2 / M, (flat, M, A, D)

    def f_bwd(res, g):
        flat, M, A, D = res
        B = flat.shape[0]
        # g: [B,B]
        S = g + g.T  # [B,B]
        row_sum = jnp.sum(S, axis=1)  # [B]
        # compute (2/M) * ( row_sum[:,None] * flat - S @ flat )
        grad_flat = (2.0 / M) * (row_sum[:, None] * flat - (S @ flat))  # [B, M]
        grad_xs = grad_flat.reshape((B, A, D))
        return (grad_xs,)

    f.defvjp(f_fwd, f_bwd)
    return jit(f)


# pairwise_distance = make_pairwise_distance()


# --------------------------
# Torus / circular distance (custom vjp)
# --------------------------
def make_torus_pairwise_distance():
    @custom_vjp
    def f(xs):
        B, A, D = xs.shape
        M = A * D
        flat = xs.reshape(B, M)
        cx = jnp.cos(flat)
        sx = jnp.sin(flat)
        dots = cx @ cx.T + sx @ sx.T  # equivalent to sum_k cos(ai_k - aj_k) across k
        dist = 2.0 - 2.0 * (dots / M)
        return dist

    def f_fwd(xs):
        B, A, D = xs.shape
        M = A * D
        flat = xs.reshape(B, M)
        cx = jnp.cos(flat)
        sx = jnp.sin(flat)
        dots = cx @ cx.T + sx @ sx.T
        dist = 2.0 - 2.0 * (dots / M)
        return dist, (cx, sx, M, A, D)

    def f_bwd(res, g):
        cx, sx, M, A, D = res
        B = cx.shape[0]
        S = g + g.T  # [B,B]
        Sc = S @ cx  # [B, M]
        Ss = S @ sx  # [B, M]
        # grad_flat = (2/M) * ( sx * (S @ cx)  -  cx * (S @ sx) )
        grad_flat = (2.0 / M) * (sx * Sc - cx * Ss)
        grad_xs = grad_flat.reshape((B, A, D))
        return (grad_xs,)

    f.defvjp(f_fwd, f_bwd)
    return jit(f)


# torus_pairwise_distance = make_torus_pairwise_distance()
