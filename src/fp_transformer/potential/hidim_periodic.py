"""HiDimPeriodic potential — hyper-cubic Gaussian mixture on torus.

Minima on a hyper-cubic lattice: minimum i has coordinate 0 on axis i
and π on all other axes.  All pairs are equidistant (NN dist = π√2).

Simplified formula (algebraically identical to generic logsumexp-of-Gaussians):
    V(θ) = (D + S)/σ² − logsumexp_i(2cos(θᵢ)/σ²)
where S = Σⱼ cos(θⱼ), D = dimensionality, σ = well width.

O(D) compute and memory — no centers matrix needed.
"""

import jax.numpy as jnp
from jax.nn import logsumexp

from .base import Potential

_DEFAULT_SIGMA_FACTOR = 6.0  # σ = π√2 / factor


class HiDimPeriodic(Potential):
    """D-dimensional periodic potential with D hyper-cubic minima.

    Args:
        dim: Dimensionality (and number of wells).
        sigma: Well width.  Defaults to π√2/6 (NN distance / 6).
    """

    def __init__(self, dim: int, sigma: float | None = None):
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim = int(dim)
        if sigma is None:
            sigma = float(jnp.pi * jnp.sqrt(2.0) / _DEFAULT_SIGMA_FACTOR)
        self._sigma = float(sigma)

    def __call__(self, x, beta=1.0):
        """Compute potential energy.

        Args:
            x: Coordinates, shape (M, P, d).
            beta: Inverse temperature (unused, kept for API compat).

        Returns:
            Energy, shape (M, 1).
        """
        cos_x = jnp.cos(x)                                    # (M, P, d)
        S = jnp.sum(cos_x, axis=-1)                           # (M, P)
        inv_s2 = 1.0 / (self._sigma ** 2)
        # V = (D + S)/σ² − logsumexp(2cos(θᵢ)/σ²)
        energy = (self._dim + S) * inv_s2 - logsumexp(
            2.0 * cos_x * inv_s2, axis=-1
        )                                                      # (M, P)
        return jnp.sum(energy, axis=-1, keepdims=True)         # (M, 1)

    def to_config(self, beta: float) -> dict:
        """Return a JSON-serializable config dict."""
        return {
            "type": "HiDimPeriodic",
            "dim": self._dim,
            "sigma": self._sigma,
            "beta": beta,
        }

    @classmethod
    def from_config(cls, config: dict):
        """Reconstruct potential from config dict.

        Returns:
            (potential, beta)
        """
        pot = cls(dim=config["dim"], sigma=config["sigma"])
        return pot, config["beta"]

    @property
    def minima_positions(self):
        """Well center positions, shape (D, D).  Computed on-the-fly."""
        if self._dim > 10000:
            raise ValueError(
                f"minima_positions not supported for dim={self._dim} > 10000"
            )
        pos = jnp.full((self._dim, self._dim), jnp.pi)
        return pos.at[jnp.diag_indices(self._dim)].set(0.0)

    @property
    def num_wells(self):
        return self._dim

    @property
    def dim(self):
        return self._dim

    def get_2dcontour(self, x_range, y_range, num_points=100):
        """Generate 2D contour data for visualization.

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
