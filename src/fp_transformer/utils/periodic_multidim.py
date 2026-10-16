import jax
import jax.numpy as jnp
from jax.random import PRNGKey, split, uniform, randint, choice

try:
    # ndtri 是标准正态分布 CDF 的反函数，比通过 erfinv 更稳定；某些 ROCm 环境下 erfinv 可能返回 NaN。
    from jax.scipy.special import ndtri as _ndtri
except Exception:  # pragma: no cover - 极端情况下导入失败
    _ndtri = None
from typing import Tuple, NamedTuple, Optional
from functools import partial
from tqdm import trange


# NamedTuple定义保持不变
class PotentialParameters(NamedTuple):
    depths: jnp.ndarray
    concentrations: jnp.ndarray
    centers: jnp.ndarray


# ------------------- 向量化的核心辅助函数 -------------------
@partial(
    jax.jit, static_argnames=["major_separation_factor", "minor_separation_factor"]
)
def _check_candidate(
    mu_candidate: jnp.ndarray,
    accepted_centers_batch: jnp.ndarray,
    candidate_major_mask: jnp.ndarray,
    existing_major_masks: jnp.ndarray,
    candidate_major_sigma: float,
    existing_major_sigmas: jnp.ndarray,
    candidate_minor_sigma: float,
    existing_minor_sigmas: jnp.ndarray,
    major_separation_factor: float,
    minor_separation_factor: float,
) -> bool:
    """
    JIT编译的辅助函数，用于向量化检查一个候选点与一批已接受点的距离。
    """
    # --- 高优先级检查 (向量化) ---
    # 计算所有交集子空间 (critical_masks)
    critical_masks = (
        candidate_major_mask & existing_major_masks
    )  # Shape: (num_accepted, N)
    num_crit_dims = jnp.sum(critical_masks, axis=1)

    # 计算所有周期性距离向量
    delta = jnp.abs(mu_candidate - accepted_centers_batch)
    periodic_delta = jnp.minimum(delta, 2 * jnp.pi - delta)

    # 仅在交集子空间内计算距离
    p_delta_crit = jnp.where(critical_masks, periodic_delta, 0.0)
    dist_crit = jnp.linalg.norm(p_delta_crit, axis=1)

    # 计算安全距离
    safe_dist_major = major_separation_factor * (
        candidate_major_sigma + existing_major_sigmas
    )

    # 检查是否有任何冲突
    is_major_conflict = (num_crit_dims > 0) & (dist_crit < safe_dist_major)
    any_major_conflict = jnp.any(is_major_conflict)

    # --- 低优先级检查 (向量化) ---
    dist_full = jnp.linalg.norm(periodic_delta, axis=1)
    safe_dist_minor = minor_separation_factor * (
        candidate_minor_sigma + existing_minor_sigmas
    )
    any_minor_conflict = jnp.any(dist_full < safe_dist_minor)

    # 候选点有效当且仅当两种冲突都不发生
    return ~any_major_conflict & ~any_minor_conflict


def generate_subspace_aware_params(
    key: PRNGKey,
    N: int,
    M: int,
    depth_range: Tuple[float, float],
    num_loose_dims_range: Tuple[int, int],
    tight_kappa_range: Tuple[float, float],
    loose_kappa_range: Tuple[float, float],
    tight_separation_factor: float = 5.0,
    loose_separation_factor: float = 1.5,
    max_retries: int = 1000,
    verbose=False,
) -> PotentialParameters:
    """
    使用向量化的、分层、子空间感知的拒绝采样算法生成势能函数参数。
    """

    if not (0 < num_loose_dims_range[0] <= num_loose_dims_range[1] < N):
        raise ValueError("Invalid loose dimensions range.")

    num_major_dims_range = (N - num_loose_dims_range[1], N - num_loose_dims_range[0])
    major_kappa_range = tight_kappa_range
    minor_kappa_range = loose_kappa_range
    major_separation_factor = tight_separation_factor
    minor_separation_factor = loose_separation_factor

    # --- 输入验证 (同前) ---
    if not loose_kappa_range[1] < tight_kappa_range[0]:
        raise ValueError(
            "Loose kappa range must be strictly less than tight kappa range."
        )

    key, depth_key, conc_key, center_key = split(key, 4)

    # --- 步骤 1: 生成深度 (同前) ---
    depths = uniform(
        depth_key, shape=(M,), minval=depth_range[0], maxval=depth_range[1]
    )

    # --- 步骤 2: 生成Concentrations并创建布尔掩码 ---
    concentrations = jnp.zeros((M, N))
    major_masks = jnp.zeros((M, N), dtype=bool)  # 使用布尔掩码矩阵
    conc_subkeys = split(conc_key, M)

    for i in range(M):
        loop_key, d_key, choice_key, major_k_key, minor_k_key = split(
            conc_subkeys[i], 5
        )
        num_major = randint(
            d_key,
            shape=(),
            minval=num_major_dims_range[0],
            maxval=num_major_dims_range[1] + 1,
        )
        major_indices = choice(choice_key, N, shape=(num_major,), replace=False)

        major_kappas = uniform(
            major_k_key,
            shape=(num_major,),
            minval=major_kappa_range[0],
            maxval=major_kappa_range[1],
        )
        minor_kappas = uniform(
            minor_k_key,
            shape=(N - num_major,),
            minval=minor_kappa_range[0],
            maxval=minor_kappa_range[1],
        )

        is_major_mask = jnp.zeros(N, dtype=bool).at[major_indices].set(True)
        major_masks = major_masks.at[i].set(is_major_mask)  # 填充掩码矩阵

        row = (
            jnp.zeros(N)
            .at[is_major_mask]
            .set(major_kappas)
            .at[~is_major_mask]
            .set(minor_kappas)
        )
        concentrations = concentrations.at[i].set(row)

    # --- 步骤 3: 计算两套有效宽度 (同前) ---
    # 需要一个小的epsilon防止当major_kappas为空时出错 (虽然我们的验证避免了这种情况)
    major_masked_kappas = jnp.where(major_masks, concentrations, jnp.inf)
    major_min_kappas = jnp.min(major_masked_kappas, axis=1)
    major_effective_sigmas = 1.0 / jnp.sqrt(major_min_kappas)

    minor_min_kappas = jnp.min(concentrations, axis=1)
    minor_effective_sigmas = 1.0 / jnp.sqrt(minor_min_kappas)

    # --- 步骤 4: 向量化拒绝采样 ---
    accepted_centers = []

    range_func = trange if verbose else range

    for i in range_func(M):
        center_key, loop_key = split(center_key)
        found_position = False
        for _ in range(max_retries):
            loop_key, candidate_key = split(loop_key)
            # choices = jnp.array([0, jnp.pi])
            # mu_candidate = jax.random.choice(candidate_key, choices, shape=(N,))
            mu_candidate = uniform(
                candidate_key, shape=(N,), minval=0, maxval=2 * jnp.pi
            )

            # 如果是第一个点，直接接受
            if i == 0:
                is_valid = True
            else:
                # 准备送入JIT函数的批量数据
                accepted_centers_batch = jnp.stack(accepted_centers)
                num_accepted = accepted_centers_batch.shape[0]

                # 调用向量化的检查函数
                is_valid = _check_candidate(
                    mu_candidate,
                    accepted_centers_batch,
                    major_masks[i],
                    major_masks[:num_accepted],
                    major_effective_sigmas[i],
                    major_effective_sigmas[:num_accepted],
                    minor_effective_sigmas[i],
                    minor_effective_sigmas[:num_accepted],
                    major_separation_factor,
                    minor_separation_factor,
                )

            if is_valid:
                accepted_centers.append(mu_candidate)
                found_position = True
                break

        if not found_position:
            raise RuntimeError(
                f"Failed to place center for well {i + 1}/{M} after {max_retries} retries."
            )

    centers = jnp.stack(accepted_centers)
    return PotentialParameters(
        depths=depths, concentrations=concentrations, centers=centers
    )


# ---------------------------------------------------------------------------
# 新的初始化方法：基于最近邻距离设定 kappa，使得最近的中心落在 0.95 置信区间之外
# ---------------------------------------------------------------------------
def _circular_distance(a: float, b: float) -> float:
    """最短圆距 [0, 2π)"""
    diff = jnp.abs(a - b)
    return jnp.minimum(diff, 2 * jnp.pi - diff)


def _nearest_neighbor_distance(centers_dim: jnp.ndarray) -> jnp.ndarray:
    """
    计算给定维度上每个 mode 的最近邻圆距。
    centers_dim: (M,)
    返回: (M,) 每个 mode 最近邻的角距。
    """
    M = centers_dim.shape[0]
    dists = []
    for m in range(M):
        others = jnp.delete(centers_dim, m)
        d = jnp.min(_circular_distance(centers_dim[m], others))
        dists.append(d)
    return jnp.array(dists)


def init_params_by_nearest_ci(
    key: PRNGKey,
    N: int,  # dimension
    M: int,  # number of modes
    depth_range: Tuple[float, float],
    alpha: float = 0.95,
    margin: float = 0.95,
    centers: Optional[jnp.ndarray] = None,  # (M, N)
    min_distance_clip: float = 1e-6,
    verbose: bool = False,
) -> PotentialParameters:
    """
    初始化多维 von Mises 势能参数，为每个 mode 单独计算局部 κ。
    """

    # 1️⃣ 初始化中心
    if centers is None:
        base = jnp.linspace(0, 2 * jnp.pi, M, endpoint=False)
        centers = jnp.stack([jnp.roll(base, i) for i in range(N)], axis=1)
    else:
        assert centers.shape == (M, N), "centers must have shape (M, N)"
    centers = jnp.mod(centers, 2 * jnp.pi)

    d_local = jnp.zeros((M, N))
    for i in range(N):
        d_i = _nearest_neighbor_distance(centers[:, i])
        d_local = d_local.at[:, i].set(jnp.clip(d_i, min_distance_clip, None))

    z_alpha = (
        1.96
        if jnp.isclose(alpha, 0.95)
        else jnp.sqrt(2) * jax.scipy.special.erfinv(alpha)
    )
    kappas = (2 * z_alpha / (d_local * margin)) ** 2  # shape (M, N)

    key_depth, _ = split(key)
    depths = uniform(key_depth, (M,), minval=depth_range[0], maxval=depth_range[1])

    if verbose:
        print("[init_params_by_nearest_ci]")
        print(f"  dims={N}, modes={M}")
        print(f"  d_local min/max = {d_local.min():.3f}/{d_local.max():.3f} rad")
        print(f"  kappas range   = {kappas.min():.2f}–{kappas.max():.2f}")
        print(f"  depth range={depth_range}")

    return PotentialParameters(depths=depths, concentrations=kappas, centers=centers)
