from functools import partial

from jax import jit, lax, random
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from ..type import JaxArray, KeyArray, Tuple, Union

tfd = tfp.distributions
tfm = tfp.math


def random_normal(key, shape):
    """
    生成符合标准正态分布的随机数

    Args:
        key: JAX随机数生成器的key
        shape: 输出张量的形状

    Returns:
        JaxArray: 符合正态分布的随机数组
    """
    return random.normal(key, shape, dtype=jnp.float32)


def random_uniform(key, shape):
    """
    生成符合均匀分布的随机数

    Args:
        key: JAX随机数生成器的key
        shape: 输出张量的形状

    Returns:
        JaxArray: 符合均匀分布的随机数组
    """
    return random.uniform(key, shape, dtype=jnp.float32)


def sample_rff_params(
    key: KeyArray,
    input_dim: int,
    n_features: int,
    bandwidth: float,
) -> Tuple[JaxArray, JaxArray]:
    """
    采样随机傅里叶特征(RFF)参数

    该函数生成用于Random Fourier Features的随机权重和偏置参数。
    为了与pairwise_distance的维度缩放保持一致，使用特定的sigma计算方式。

    Args:
        key (KeyArray): JAX随机数生成器的key
        input_dim (int): 输入维度 (A*D)
        n_features (int): RFF特征的维度
        bandwidth (float): 带宽参数

    Returns:
        Tuple[JaxArray, JaxArray]: 包含以下元素的元组：
            - weights: 随机权重矩阵 (n_features, input_dim)
            - bias: 随机偏置向量 (n_features,)
    """
    # 为了与pairwise_distance的维度缩放保持一致：
    # pairwise_distance返回 ||x-y||²/dim
    # Gaussian核计算 exp(-0.5 * distance/bandwidth²) = exp(-0.5 * ||x-y||²/(dim*bandwidth²))
    # 因此RFF的sigma应该是 sqrt(1/(dim*bandwidth))
    sigma = jnp.sqrt(1.0 / (input_dim * bandwidth))
    sigma = jnp.mean(sigma)  # 确保sigma是标量

    # 采样随机权重
    key_w, key_b = random.split(key)

    weights = random_normal(key_w, (n_features, input_dim)) * sigma

    # 采样随机偏置
    bias = random_uniform(key_b, (n_features,)) * 2 * jnp.pi

    weights = lax.stop_gradient(weights)
    bias = lax.stop_gradient(bias)

    return weights, bias


def create_torus_bessel_pmf_dist(
    gamma: Union[float, JaxArray], m_cutoff: int = 30, n_dim: int = 1
) -> Tuple[tfd.Categorical, JaxArray]:
    """
    创建用于圆环距离的高斯核RFF特征采样器

    基于圆环距离构建概率质量函数，用于采样RFF特征的频率参数。
    使用修正贝塞尔函数计算概率分布。参考 temp_torus_rff.py 中的实现。

    Args:
        gamma (float): gamma参数，等于 1/(2*bandwidth^2)
        m_cutoff (int, optional): 截断范围. Defaults to 30.
        n_dim (int, optional): 维度缩放因子. Defaults to 1.

    Returns:
        Tuple[tfd.Categorical, JaxArray]: 包含以下元素的元组：
            - dist: 采样器的概率质量函数 (PMF)
            - k_range: 频率范围数组
    """
    k_range = jnp.arange(-m_cutoff, m_cutoff + 1)

    gamma /= n_dim

    # p(k) 的未归一化值直接由 bessel_ive 给出
    # 我们需要将 k 转换为浮点数以匹配 bessel_ive 的 API
    probs = tfm.bessel_ive(k_range.astype(jnp.float32), 2 * gamma)

    # 创建 TFP Categorical 分布，它会自动处理归一化
    dist = tfd.Categorical(logits=jnp.log(probs), name="TorusBesselPMF")
    return dist, k_range


def create_torus_bessel_logits(
    gamma: Union[float, JaxArray], m_cutoff: int = 30, n_dim: int = 1
) -> Tuple[JaxArray, JaxArray]:
    """Create torus PMF logits and support (k_range) without constructing TFP objects."""
    k_range = jnp.arange(-m_cutoff, m_cutoff + 1)
    gamma = gamma / n_dim
    probs = tfm.bessel_ive(k_range.astype(jnp.float32), 2 * gamma)
    eps = jnp.asarray(1e-30, dtype=probs.dtype)
    logits = jnp.log(jnp.maximum(probs, eps))
    return logits, k_range


def sample_rff_params_torus(
    key: KeyArray,
    dist: tfd.Categorical,
    k_range: JaxArray,
    input_dim: int,
    n_features: int,
) -> JaxArray:
    """
    从圆环分布中采样RFF参数

    根据给定的概率分布采样频率参数，用于圆环距离的RFF计算。

    Args:
        key (KeyArray): JAX随机数生成器的key
        dist (tfd.Categorical): 圆环分布的概率质量函数
        k_range (JaxArray): 频率范围数组
        input_dim (int): 输入维度 (A*D)
        n_features (int): RFF特征的维度

    Returns:
        JaxArray: 采样得到的频率参数 (n_features, input_dim)
    """
    sampled_indices = dist.sample(sample_shape=(n_features, input_dim), seed=key)
    k_samples = k_range[sampled_indices]
    return k_samples


def sample_rff_params_torus_from_logits(
    key: KeyArray,
    logits: JaxArray,
    k_range: JaxArray,
    input_dim: int,
    n_features: int,
) -> JaxArray:
    """Sample torus RFF frequencies directly from cached logits via JAX categorical."""
    sampled_indices = random.categorical(
        key, logits=logits, shape=(n_features, input_dim)
    )
    return k_range[sampled_indices]


def rff_feature_map_torus(x: JaxArray, k_samples: JaxArray) -> JaxArray:
    """
    计算环面距离的RFF特征映射，参考 temp_torus_rff.py 中的实现

    基于圆环距离的频率参数计算RFF特征，同时生成余弦和正弦特征。

    Args:
        x (JaxArray): 输入数据 (B, A*D)
        k_samples (JaxArray): 采样的频率参数 (n_features, input_dim)

    Returns:
        JaxArray: 圆环RFF特征 (B, 2*n_features)，包含余弦和正弦特征
    """
    D = k_samples.shape[0]
    x = x.reshape(x.shape[0], -1)
    kx = x @ k_samples.T  # (B, A*D) @ (A*D, n_features).T = (B, n_features)
    cos_features = jnp.cos(kx)
    sin_features = jnp.sin(kx)
    return (1.0 / jnp.sqrt(D)) * jnp.concatenate([cos_features, sin_features], axis=-1)


@jit
def compute_rff_features(
    X: JaxArray,
    weights: JaxArray,
    bias: JaxArray,
) -> JaxArray:
    """
    计算随机傅里叶特征(RFF)

    使用随机权重和偏置计算输入数据的RFF特征，用于核方法的近似。

    Args:
        X (JaxArray): 输入数据 (B, A, D)
        weights (JaxArray): 随机权重矩阵 (n_features, A*D)
        bias (JaxArray): 随机偏置向量 (n_features,)

    Returns:
        JaxArray: RFF特征 (B, n_features)
    """
    # B, A, D = X.shape
    X_reshaped = X.reshape(X.shape[0], -1)  # (B, A*D)

    # 计算随机特征
    projection = jnp.dot(X_reshaped, weights.T)  # (B, n_features)

    features = jnp.cos(projection + bias) * jnp.sqrt(2.0 / weights.shape[0])

    return features


def compute_rff_feature_torus(
    X: JaxArray,
    k_samples: JaxArray,
) -> JaxArray:
    """
    计算圆环距离的RFF特征

    基于圆环距离的频率参数计算RFF特征，同时生成余弦和正弦特征。

    Args:
        X (JaxArray): 输入数据 (B, A, D)
        k_samples (JaxArray): 采样的频率参数 (A*D, n_features)

    Returns:
        JaxArray: 圆环RFF特征 (B, 2*n_features)，包含余弦和正弦特征
    """
    X = X.reshape(X.shape[0], -1)  # 确保X是二维的 (B, A*D)
    kx = jnp.dot(X, k_samples.T)  # (B, A*D) @ (A*D, n_features) -> (B, n_features)
    cos_features = jnp.cos(kx)
    sin_features = jnp.sin(kx)

    n_feat = k_samples.shape[0]
    return jnp.reciprocal(jnp.sqrt(n_feat)) * jnp.concatenate(
        [cos_features, sin_features], axis=-1
    )


@jit
def rff_kernel(
    X1: JaxArray,
    X2: JaxArray,
    weights: JaxArray,
    bias: JaxArray,
) -> JaxArray:
    """
    使用RFF特征计算核矩阵

    通过计算两组数据的RFF特征并计算其内积来近似核矩阵。

    Args:
        X1 (JaxArray): 第一组输入数据 (N1, A, D)
        X2 (JaxArray): 第二组输入数据 (N2, A, D)
        weights (JaxArray): 随机权重矩阵 (n_features, A*D)
        bias (JaxArray): 随机偏置向量 (n_features,)

    Returns:
        JaxArray: 核矩阵 (N1, N2)
    """
    # 计算两组数据的RFF特征
    features1 = compute_rff_features(X1, weights, bias)  # (N1, n_features)
    features2 = compute_rff_features(X2, weights, bias)  # (N2, n_features)

    # 核矩阵是特征向量的内积
    kernel_matrix = jnp.dot(features1, features2.T)  # (N1, N2)

    return kernel_matrix


@partial(jit, static_argnames=("exclude_self",))
def rff_row_sum(features: JaxArray, *, exclude_self: bool = False) -> JaxArray:
    """
    使用RFF特征近似核矩阵的行求和操作

    通过RFF特征的数学性质，避免显式计算核矩阵而直接计算行求和。
    利用 sum_j K[i,j] = sum_j (features[i] @ features[j]) = features[i] @ sum_j features[j]

    Args:
        features (JaxArray): RFF特征矩阵 (B, D)
        exclude_self: Whether to subtract each represented diagonal term.

    Returns:
        JaxArray: 相当于对核矩阵(B,B)每行求和的结果 (B,)
    """
    # 行求和：sum_j K[i,j] = sum_j (features[i] @ features[j])
    #                      = features[i] @ sum_j features[j]
    #                      = features[i] @ features_sum
    features_sum = jnp.sum(features, axis=0)  # (D,)
    # row_sum = jnp.dot(features, features_sum)  # (B,)
    row_sum = features @ features_sum  # 等价于上面 dot 的结果，保持 JIT 友好
    if exclude_self:
        # RFF does not guarantee K(x_i, x_i) == 1 exactly. Subtract the
        # represented diagonal rather than an idealized constant.
        row_sum = row_sum - jnp.sum(jnp.square(features), axis=-1)
    # stabilize the result to avoid numerical issues
    row_sum = jnp.maximum(row_sum, 1e-10)  # 避免极小值导致的数值问题
    return row_sum


@partial(jit, static_argnames=("exclude_self",))
def rff_power_row_sum(
    features, power, epsilon: float = 1e-8, *, exclude_self: bool = False
):
    B, D = features.shape

    C = features.T @ features
    eigenvalues, V = jnp.linalg.eigh(C)
    safe_eigenvalues = jnp.maximum(eigenvalues, 0.0)  # (D,)

    ones_vec = jnp.ones((B,))
    w = features.T @ ones_vec  # (D,)
    y = V.T @ w  # (D,)

    power = jnp.asarray(power)

    # 标量 power：走你原来的逻辑（保持输出 (B,)）
    if power.ndim == 0:
        lam_pow = jnp.where(
            power - 1 < 0,
            1.0 / (safe_eigenvalues ** (1 - power) + epsilon),
            safe_eigenvalues ** (power - 1),
        )  # (D,)
        p = lam_pow * y  # (D,)
        q = V @ p  # (D,)
        result = features @ q  # (B,)
        if exclude_self:
            projected = features @ V
            diagonal = jnp.sum(jnp.square(projected) * lam_pow[None, :], axis=1)
            result = result - diagonal
        return result

    # 向量 power：(B,) -> 每行一个幂次
    s = power[:, None]  # (B, 1)
    lam = safe_eigenvalues[None, :]  # (1, D)

    lam_pow = jnp.where(
        (s - 1) < 0,
        1.0 / (lam ** (1 - s) + epsilon),
        lam ** (s - 1),
    )  # (B, D)

    p = lam_pow * y[None, :]  # (B, D)
    q = p @ V.T  # (B, D)  （等价于对每行做 V @ p_i）
    result = jnp.sum(features * q, axis=1)  # (B,)
    if exclude_self:
        projected = features @ V
        diagonal = jnp.sum(jnp.square(projected) * lam_pow, axis=1)
        result = result - diagonal

    return result


@partial(jit, static_argnames=("exclude_self",))
def rff_logsum(features: JaxArray, *, exclude_self: bool = False) -> JaxArray:
    """
    使用RFF特征近似核矩阵的对数求和操作

    计算核矩阵每行的对数求和，支持分片计算。通过先计算行求和再取对数实现。

    Args:
        features (JaxArray): RFF特征矩阵 (B, D)
        exclude_self: Whether to use a leave-one-out row sum.

    Returns:
        JaxArray: 相当于对核矩阵(B,B)每行log(sum)的结果 (B, 1)

    Note:
        计算公式：log(sum(k)) = log(row_sum)
    """
    row_sum = rff_row_sum(features, exclude_self=exclude_self)
    return jnp.log(row_sum).reshape(-1, 1)
