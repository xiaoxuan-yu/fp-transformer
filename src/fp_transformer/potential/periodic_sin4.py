"""PeriodicSin4 potential for high-dimensional numerical benchmarks.

Formula: V(θ) = exp[c·(d - Σᵢ sin⁴(p·θᵢ))] - 1
where d is inferred from x.shape[-1].
"""

from jax import numpy as jnp

from .base import Potential


class PeriodicSin4(Potential):
    """High-dimensional periodic potential with sin⁴ wells.

    Args:
        c: Steepness parameter controlling barrier height.
        p: Frequency parameter (period = 2π/p).
    """

    def __init__(self, c: float = 3.0, p: float = 1.0):
        self.c = c
        self.p = p

    def __call__(self, x, beta=1.0):
        """Compute potential energy.

        Args:
            x: Coordinates, shape (M, P, d).
            beta: Inverse temperature (unused, kept for API compat).

        Returns:
            Energy, shape (M, 1).
        """
        d = x.shape[-1]
        sin4_sum = jnp.sum(jnp.sin(self.p * x) ** 4, axis=-1)  # (M, P)
        exponent = self.c * (d - sin4_sum)                        # (M, P)
        energy = jnp.exp(exponent) - 1.0                          # (M, P)
        return jnp.sum(energy, axis=-1, keepdims=True)             # (M, 1)

    def to_config(self, beta: float, dim: int) -> dict:
        """Return a JSON-serializable config dict."""
        return {
            "type": "PeriodicSin4",
            "c": self.c,
            "p": self.p,
            "dim": dim,
            "beta": beta,
        }

    @classmethod
    def from_config(cls, config: dict):
        """Reconstruct potential from config dict.

        Returns:
            (potential, beta, dim)
        """
        pot = cls(c=config["c"], p=config["p"])
        return pot, config["beta"], config["dim"]

    def get_2dcontour(self, x_range, y_range, num_points=100):
        """Generate 2D contour data for visualization.

        Args:
            x_range: (min, max) for x-axis.
            y_range: (min, max) for y-axis.
            num_points: Grid resolution per axis.

        Returns:
            (X, Y, E) meshgrid arrays.
        """
        x = jnp.linspace(x_range[0], x_range[1], num_points)
        y = jnp.linspace(y_range[0], y_range[1], num_points)
        X, Y = jnp.meshgrid(x, y)
        coord = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
        coord = jnp.expand_dims(coord, axis=-2)  # (N, 1, 2)
        E = self(coord, 1.0).reshape(X.shape)
        E -= jnp.min(E)
        return X, Y, E
