from jax import jit
from jax import numpy as jnp


@jit
def its_grad_coeff(beta, beta_array, potential_array, f_array, mask):
    """
    计算 ITS 梯度系数。

    Args:
        beta: scalar, target inverse temperature (e.g., β_target)
        beta_array: shape (T,), list of inverse temperatures
        potential_array: shape (T, N, 1), potential energies
        mask: shape (T,), mask for valid temperatures (1 for used, 0 for unused)
        f_array: shape (T,), fk = -ln Zk
    Returns:
        coeff: scalar, coefficient for gradient of ITS effective potential energy
    """
    active = (mask > 0).astype(jnp.float32)  # (T,)

    # x[k,n,1] = -β_k * U_kn + f_k
    x = -beta_array[:, None, None] * potential_array  # (T, N, 1)
    x = x + f_array[:, None, None]  # (T, N, 1)
    # 局部 log-sum-exp（避免内部溢出）
    m_k = jnp.max(x, axis=1).squeeze(axis=1)  # (T,)
    s_k = jnp.sum(jnp.exp(x - m_k[:, None, None]), axis=1).squeeze(axis=1)
    s_k = jnp.where(s_k == 0, 1e-300, s_k)

    # 全局缩放：mask out padded temps so they don't dominate the max
    m_k_masked = jnp.where(active > 0, m_k, -jnp.inf)
    M = jnp.max(m_k_masked)
    scaled = jnp.exp(m_k - M) * s_k * active  # (T,) — zero out padded

    # 计算加权和
    numerator = jnp.sum(beta_array * scaled)
    denominator = jnp.sum(scaled)

    # 输出结果
    coeff = (1.0 / beta) * (numerator / denominator)
    return coeff


@jit
def its_grad_coeff_per_sample(beta, beta_array, potential_array, f_array, mask):
    """
    逐样本 ITS 梯度系数（每个 replica/样本一个系数）。

    Args:
        beta: scalar, target inverse temperature (e.g., β_target)
        beta_array: shape (T,), list of inverse temperatures
        potential_array: shape (T, N, 1), potential energies
        f_array: shape (T,), fk = -ln Zk
        mask: shape (T,), mask for valid temperatures (1 for used, 0 for unused)
    Returns:
        coeff: shape (N,), per-sample coefficient for gradient scaling
    """
    # x[k,n] = f_k - β_k * U_kn
    u = potential_array[..., 0]  # (T, N)
    x = -beta_array[:, None] * u + f_array[:, None]  # (T, N)
    x = jnp.where(mask[:, None] > 0, x, -jnp.inf)

    # Stable per-sample normalization across temperatures (axis=0)
    m_n = jnp.max(x, axis=0)  # (N,)
    m_n = jnp.where(jnp.isfinite(m_n), m_n, 0.0)
    w = jnp.exp(x - m_n[None, :])
    w = jnp.where(jnp.isfinite(w), w, 0.0)

    denom = jnp.sum(w, axis=0)  # (N,)
    denom = jnp.where(denom == 0, 1e-300, denom)
    numer = jnp.sum(beta_array[:, None] * w, axis=0)  # (N,)

    coeff = (1.0 / beta) * (numer / denom)
    return coeff


@jit
def its_grad_coeff_naive(beta_array, potential_array, f_array):
    """
    Calculate the gradient coefficient for the ITS effective potential energy.
    beta_array: shape (T,), list of inverse temperatures
    potential_array: shape (N, 1), potential energies
    f_array: shape (T,), fk = -ln Zk
    """
    potential_array = potential_array.reshape(1, -1)
    beta_array = beta_array.reshape(-1, 1)
    beta_U_array = beta_array * potential_array  # (T, N)
    beta_U_array = beta_U_array - f_array.reshape(-1, 1)  # (T, N)
    min_beta_U = jnp.min(beta_U_array, axis=0).reshape(1, -1)  # (1, N)
    exp_beta_U_array = jnp.exp(-(beta_U_array - min_beta_U))  # (T, N)
    numerator = jnp.sum(beta_array * exp_beta_U_array, axis=0)  # (N,)
    denominator = jnp.sum(exp_beta_U_array, axis=0)  # (N,)
    coeff = numerator / denominator
    return coeff  # (N,)


@jit
def kld_anneal(af_now, af_next, energy):
    """
    计算KL散度用于退火控制

    Args:
        af_now: the current anneal factor
        af_next: next anneal factor
        energy: potential energy

    Return:
        kld: KL divergence
    """
    exponent = (af_now - af_next) * energy
    exponent = exponent - jnp.max(exponent)
    wt = jnp.clip(jnp.exp(exponent), 1e-8, 1.0)
    wt = wt / jnp.sum(wt)  # 归一化权重

    n = energy.shape[0]
    kld = jnp.sum(wt * jnp.log(n * wt))  # KL散度
    return kld


@jit
def compute_next_anneal_factor(af_now, af_increment, energy):
    """
    计算下一个退火因子

    Args:
        af_now: the current anneal factor
        af_increment: increase of anneal factor
        energy: the energy of the system

    Return:
        af_next: the next anneal factor
        kld: KL divergence
    """
    af_next = af_now + af_increment
    kld = kld_anneal(af_now, af_next, energy)
    return af_next, kld
