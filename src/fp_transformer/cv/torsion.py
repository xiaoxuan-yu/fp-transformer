from .base import BaseCV
from ..utils import dihedral
from jax import numpy as jnp
from jax import vmap


class Torsion(BaseCV):
    def __init__(self, torsion_indices):
        """
        二面角（torsion/dihedral）集体变量。

        Args:
            torsion_indices: 形状为 [M, 4] 的整型索引数组，每一行表示一个二面角的四个原子索引。
        """
        # 保持为 JAX 数组，便于后续索引与 JIT/vmap。
        self.torsion_indices = jnp.asarray(torsion_indices, dtype=jnp.int32)  # [M, 4]

    def __call__(self, x):
        """
        计算输入构型中所有指定二面角。

        Args:
            x: 原子坐标，支持形状 [..., N, 3]，其中 ... 是任意批次维。
               常见情形：
               - [R, N, 3]: R 组样本/副本；
               - [N, 3]: 单个样本。

        Returns:
            angles: 二面角（弧度）。
                - 若输入为 [..., N, 3]，返回形状 [..., M, 1]
        """
        x = jnp.asarray(x)
        if x.ndim < 2:
            raise ValueError(f"Torsion expects x with shape [..., N, 3], got {x.shape}")
        if x.shape[-1] != 3:
            raise ValueError(f"Torsion expects last dimension=3, got {x.shape}")

        idx = self.torsion_indices  # [M, 4]
        leading_shape = x.shape[:-2]
        x_flat = x.reshape((-1, x.shape[-2], 3))  # [B, N, 3]

        # 根据四元组索引收集点坐标 -> [B, M, 3]
        x1 = x_flat[:, idx[:, 0], :]
        x2 = x_flat[:, idx[:, 1], :]
        x3 = x_flat[:, idx[:, 2], :]
        x4 = x_flat[:, idx[:, 3], :]

        # 展平到 (B*M, 3)，用 vmap 批量计算，再恢复形状
        B, M = x1.shape[0], x1.shape[1]
        f = vmap(dihedral, in_axes=(0, 0, 0, 0), out_axes=0)
        angles = f(
            x1.reshape(-1, 3),
            x2.reshape(-1, 3),
            x3.reshape(-1, 3),
            x4.reshape(-1, 3),
        ).reshape(B, M)

        angles = angles.reshape((*leading_shape, M))
        return angles[..., jnp.newaxis]  # [..., M, 1]
