from .base import BaseCV
from jax import numpy as jnp


class SelectiveParticle(BaseCV):
    def __init__(self, select_index):
        """
        选择粒子（原子）坐标的集体变量。

        Args:
            select_index: 要选择的粒子索引，可为：
                - 标量（int），表示选择单个粒子；
                - 一维数组，形状为 [M]，表示选择多个粒子。
        """
        # 将索引保持为 JAX 数组，便于与 JIT/vmap/索引配合使用
        self.select_index = jnp.asarray(select_index, dtype=jnp.int32)

    def __call__(self, x):
        """
        从输入坐标中选择指定粒子的坐标。

        Args:
            x: 原子坐标，支持两种形状：
               - [R, N, 3]: R 组样本/副本；
               - [N, 3]: 单个样本。

        Returns:
            coords: 被选择的粒子坐标。
                - 若输入为 [R, N, 3]：
                    - 选择单个索引 -> 形状 [R, 1, 3]
                    - 选择多个索引 -> 形状 [R, M, 3]
                - 若输入为 [N, 3]：
                    - 选择单个索引 -> 形状 [1, 3]
                    - 选择多个索引 -> 形状 [M, 3]
        """
        idx = self.select_index

        # 统一批次维处理：若为 [N, 3]，先升维到 [1, N, 3]
        if x.ndim == 2:
            x = x[jnp.newaxis, ...]
            squeeze_out = True
        else:
            squeeze_out = False

        # 利用高级索引选择：
        # - 为了在标量索引时也保留粒子维（与向量索引一致），将 idx 至少提升到 1D。
        #   这样：
        #   - 当 idx 为标量时，x[:, idx[None], :] -> [R, 1, 3]
        #   - 当 idx 为向量时，x[:, idx, :]      -> [R, M, 3]
        idx_1d = jnp.atleast_1d(idx)
        selected = x[:, idx_1d, :]

        if squeeze_out:
            # 返回到无批次的形状
            return selected[0]
        return selected
