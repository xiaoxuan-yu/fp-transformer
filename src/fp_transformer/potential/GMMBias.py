"""
GMM 偏置势能：使二面角子空间服从预设的高斯混合模型分布。

U^bias(φ) = -k_B T ln P(φ) + const

其中 P(φ) 为 GMM 目标分布，常数项不影响动力学。
"""

import jax.numpy as jnp
from jax import vmap
from typing import Optional, Sequence, Union

from .base import Potential
from ..type import JaxArray
from ..utils import logsumexp


def _wrap_delta(d: JaxArray, period: float = 2.0 * jnp.pi) -> JaxArray:
    """将周期变量的差值 wrap 到 [-period/2, period/2)。"""
    return (d + period / 2.0) % period - period / 2.0


class GMMBiasPotential(Potential):
    """
    基于 GMM 的偏置势能，用于控制二面角子空间的采样分布。

    P(φ) = sum_k C_k exp(- beta * sum_i (φ_i - φ*_{i,k})^2 / (2 σ_k^2))
    U^bias(φ) = -k_B T ln P(φ)

    Parameters
    ----------
    torsion_cv : callable
        二面角 CV，输入坐标 x，返回二面角 (R, M, 1) 或 (M, 1)
    dihedral_indices : sequence of int
        要使用的二面角在 torsion_cv 输出中的索引，如 [1, 2, 4, 6, 7, 8]
    centers : JaxArray, shape (K, n_dihedrals)
        各 GMM 分量中心
    weights : JaxArray, shape (K,)
        各分量权重 C_k
    sigmas : JaxArray, shape (K,)
        各分量宽度 σ_k
    eps : float
        ln P 的数值保护，避免 log(0)
    """

    def __init__(
        self,
        torsion_cv,
        dihedral_indices: Sequence[int],
        centers: JaxArray,
        weights: JaxArray,
        sigmas: JaxArray,
        eps: float = 1e-12,
        beta: float = 1.0,
    ):
        super().__init__()
        self.torsion_cv = torsion_cv
        self.dihedral_indices = list(dihedral_indices)
        self.centers = jnp.asarray(centers)
        self.weights = jnp.asarray(weights)
        self.sigmas = jnp.asarray(sigmas)
        self.eps = eps
        self.beta = beta

        K = self.centers.shape[0]
        n_dihe = self.centers.shape[1]
        if len(self.dihedral_indices) != n_dihe:
            raise ValueError(
                f"dihedral_indices 长度 {len(self.dihedral_indices)} 与 centers 列数 {n_dihe} 不一致"
            )
        if self.weights.shape != (K,):
            raise ValueError(f"weights 形状应为 ({K},)，得到 {self.weights.shape}")
        if self.sigmas.shape != (K,):
            raise ValueError(f"sigmas 形状应为 ({K},)，得到 {self.sigmas.shape}")

    def _log_gmm(self, phi: JaxArray) -> JaxArray:
        """
        计算 ln P(φ)，使用 log-sum-exp 保证数值稳定。

        phi: (R, n_dihedrals) 或 (n_dihedrals,)
        """
        # phi: (..., n_dihedrals), centers: (K, n_dihedrals)
        # delta: (..., K, n_dihedrals)
        delta = phi[..., jnp.newaxis, :] - self.centers[jnp.newaxis, ...]
        delta = _wrap_delta(delta)

        # dist_sq: (..., K)
        dist_sq = jnp.sum(delta**2, axis=-1)

        # log_term_k = log(C_k) - dist_sq / (2 σ_k^2)
        log_terms = jnp.log(self.weights + 1e-300) - dist_sq / (
            2.0 * (self.sigmas**2 + 1e-10)
        ) * self.beta

        # log P = logsumexp(log_terms)
        return logsumexp(log_terms, axis=-1)

    def __call__(self, x: JaxArray, beta: float = 1.0) -> JaxArray:
        """
        计算偏置势能。

        x: (num_replicas, n_atoms, 3)
        beta: 1/(k_B T)

        Returns
        -------
        (num_replicas, 1)，返回 U_bias * beta（与 GAFF 等单位）。
        """
        if len(x.shape) != 3:
            raise ValueError(
                f"输入坐标应为 (num_replicas, n_atoms, 3)，得到 {x.shape}"
            )

        # 计算二面角
        phi_all = self.torsion_cv(x)  # (R, M, 1)
        phi_all = phi_all.squeeze(-1)  # (R, M)

        # 选取指定二面角
        phi = phi_all[:, self.dihedral_indices]  # (R, n_dihedrals)

        # log P
        log_P = self._log_gmm(phi)


        # U^bias = -k_B T ln P = -ln P / beta
        # 返回 U_bias * beta = -ln P（与 GAFF 统一接口）
        U_bias_scaled = -log_P

        return U_bias_scaled[:, jnp.newaxis]
