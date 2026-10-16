"""
势能组合：将多个势能相加得到总势能。
"""

import jax.numpy as jnp
from typing import Tuple

from .base import Potential
from ..type import JaxArray


class SumPotential(Potential):
    """
    多个势能之和：U_total = U_1 + U_2 + ...
    """

    def __init__(self, *potentials: Potential):
        super().__init__()
        self.potentials = potentials

    def __call__(self, x: JaxArray, beta: float = 1.0) -> JaxArray:
        total = jnp.zeros((x.shape[0], 1), dtype=x.dtype)
        for p in self.potentials:
            total = total + p(x, beta)
        return total
