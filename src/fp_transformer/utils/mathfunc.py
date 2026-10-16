import jax
from jax import jit, vmap
from jax import numpy as jnp
# from .pairwise_custom_vjp import make_pairwise_distance, make_torus_pairwise_distance

from ..type import JaxArray


# Euclidean distance
# @jit
def dist(x: JaxArray, y: JaxArray) -> JaxArray:
    return jnp.sum((x - y) ** 2)


# torus distance
# @jit
def torus_dist(x: JaxArray, y: JaxArray) -> JaxArray:
    """
    Compute torus distance of x and y (D,).

    Args:
        x: input tensor (D,).
        y: input tensor (D,).
    Returns:
        distance: torus distance (1,).
    """
    # torus distance
    return jnp.mean(
        jnp.square(jnp.sin(x) - jnp.sin(y)) + jnp.square(jnp.cos(x) - jnp.cos(y))
    )


@jit
def pairwise_distance_(xs: JaxArray) -> JaxArray:
    """
    Compute pairwise distance of x (B * A * D).

    Args:
        x: input tensor (B * A * D).
    Returns:
        distance: pairwise distance (B * B).
    """
    # B * A * D -> B * (A * D)
    xs = xs.reshape(xs.shape[0], -1)

    # pairwise distance (B * (A * D) -> B * B)
    # distance = vmap(lambda x: vmap(lambda y: dist(x, y))(xs))(xs)
    pairwise_distances = vmap(vmap(dist, (0, None)), (None, 0))
    distance = pairwise_distances(xs, xs)

    dim = xs.shape[-1]

    return distance / dim


@jit
def pairwise_distance(xs):
    """
    xs: [B, A, D]
    return: [B, B]
    """
    B, A, D = xs.shape

    # flatten to [B, M]
    flat = xs.reshape(B, A * D)

    # squared norm of each row: [B]
    norm2 = jnp.sum(flat * flat, axis=1)

    # pairwise squared distance: [B, B]
    dist2 = norm2[:, None] + norm2[None, :] - 2.0 * (flat @ flat.T)

    # optional: numerical cleanup
    dist2 = jnp.maximum(dist2, 0.0)

    return dist2 / (A * D)


@jit
def torus_pairwise_distance(xs):
    """
    xs: [B, A, D]
    return: [B, B]
    """
    B, A, D = xs.shape

    # Flatten → [B, M]  (M=A*D)
    flat = xs.reshape(B, A * D)

    # compute differences: broadcasting [B,1,M] - [1,B,M] → [B,B,M]
    cx = jnp.cos(flat)
    sx = jnp.sin(flat)

    dots = cx @ cx.T + sx @ sx.T  # [B,B]
    dist = 2.0 - 2.0 * (dots / (A * D))
    # diff = flat[:, None, :] - flat[None, :, :]

    # torus identity: (sinΔ)^2 + (cosΔ)^2 = 2 - 2 cos(Δ)
    # dist = jnp.mean(2.0 - 2.0 * jnp.cos(diff), axis=-1)

    return dist


@jit
def pairwise_distance_between(xs, ys):
    """Normalized squared Euclidean distance between rows of xs [B,A,D] and ys [C,A,D]."""
    B, A, D = xs.shape
    flat_x = xs.reshape(B, A * D)
    flat_y = ys.reshape(ys.shape[0], A * D)
    dist2 = (
        jnp.sum(flat_x * flat_x, axis=1)[:, None]
        + jnp.sum(flat_y * flat_y, axis=1)[None, :]
        - 2.0 * (flat_x @ flat_y.T)
    )
    return jnp.maximum(dist2, 0.0) / (A * D)


@jit
def torus_pairwise_distance_between(xs, ys):
    """Normalized chordal torus distance between rows of xs [B,A,D] and ys [C,A,D]."""
    B, A, D = xs.shape
    flat_x = xs.reshape(B, A * D)
    flat_y = ys.reshape(ys.shape[0], A * D)
    dots = jnp.cos(flat_x) @ jnp.cos(flat_y).T + jnp.sin(flat_x) @ jnp.sin(flat_y).T
    return 2.0 - 2.0 * (dots / (A * D))


# torus pairwise distance
@jit
def torus_pairwise_distance_(xs: JaxArray) -> JaxArray:
    """
    Compute pairwise distance of x (B * A * D) in torus.

    Args:
        x: input tensor (B * A * D).
    Returns:
        distance: pairwise distance (B * B).
    """
    # B * A * D -> B * (A * D)
    xs = xs.reshape(xs.shape[0], -1)

    # pairwise distance (B * (A * D) -> B * B)
    pairwise_distances = vmap(vmap(torus_dist, (0, None)), (None, 0))
    distance = pairwise_distances(xs, xs)
    # distance = vmap(lambda x: vmap(lambda y: torus_dist(x=x, y=y))(xs))(xs)

    # """
    # x1 = jnp.expand_dims(xs, axis=1)  # (B, 1, A, D)
    # x2 = jnp.expand_dims(xs, axis=0)  # (1, B, A, D)
    # x2 = jax.lax.stop_gradient(x2)
    # distance = (jnp.sin(x1) - jnp.sin(x2)) ** 2 + (
    #     jnp.cos(x1) - jnp.cos(x2)
    # ) ** 2  # (B, B, A, D)

    # dim = xs.shape[-1]

    return distance  # / dim


logsumexp = jax.nn.logsumexp


@jit
def dist_mat_to_gaussian_kernel_exp(dist_mat: JaxArray, bandwidth: JaxArray):
    """
    Convert distance matrix to Gaussian kernel exponent.

    Args:
        dist_mat: Pairwise distance matrix (B, B).
        bandwidth: Bandwidths (B, 1) or (1,).
    Returns:
        kernel_exp: Gaussian kernel exponent matrix (B, B).
    """
    gaussian_exp = -0.5 * dist_mat
    kernel_exp = gaussian_exp * jnp.reciprocal(jnp.square(bandwidth))
    return kernel_exp


@jit
def dihedral(x1: JaxArray, x2: JaxArray, x3: JaxArray, x4: JaxArray) -> JaxArray:
    """
    计算四个三维点形成的二面角（torsion/dihedral）。

    参数:
        x1, x2, x3, x4: 各为一个三维点，形状可为 (3,) 或 (3,1)。

    返回:
        标量角度（弧度）。

    说明:
        使用稳定的 atan2 公式：
            - b1 = x2 - x1, b2 = x3 - x2, b3 = x4 - x3
            - n1 = cross(b1, b2), n2 = cross(b2, b3)
            - angle = atan2( dot(cross(n1̂, n2̂), b2̂), dot(n1̂, n2̂) )
        其中 v̂ 表示 v 的单位向量；为避免 0 长度导致的数值问题，分母加了微小 eps。
    """
    eps = 1e-12

    # 统一为 (3,) 向量，兼容 (3,) 与 (3,1)
    x1 = x1.reshape(3)
    x2 = x2.reshape(3)
    x3 = x3.reshape(3)
    x4 = x4.reshape(3)

    # 边向量
    b1 = x2 - x1
    b2 = x3 - x2
    b3 = x4 - x3

    # 法向量
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)

    # 归一化
    b2_u = b2 / (jnp.linalg.norm(b2) + eps)
    n1_u = n1 / (jnp.linalg.norm(n1) + eps)
    n2_u = n2 / (jnp.linalg.norm(n2) + eps)

    x = jnp.dot(n1_u, n2_u)
    # 标准定义：y = dot(b2̂, cross(n1̂, n2̂))
    y = jnp.dot(b2_u, jnp.cross(n1_u, n2_u))

    return jnp.arctan2(y, x)
