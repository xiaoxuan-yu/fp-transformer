from .base import Potential
from ..type import JaxArray, Tuple
from ..utils import get_jax_array
from jax import numpy as jnp


class Periodic(Potential):
    """
    Periodic potential energy function.

    Args:
        amplitude (float): Amplitude of the potential.
            Default: 1.0

        period (float): Period of the potential.
            Default: 1.0

        phase (float): Phase shift of the potential.
            Default: 0.0
    """

    def __init__(self, p: float = 1.0):
        self.p = get_jax_array(p)

    def __call__(self, x: JaxArray, beta: float) -> JaxArray:
        """
        Args:
            x: JaxArray, shape=(num_replicas, num_particles, 2)
                The input coordinates for which to calculate the potential energy.
                The shape is (num_replicas, num_particles, 2), representing batch size,
                number of particles, and the 2D coordinates for each particle.
            beta: float
                The reverse temperature scaling factor.

        Returns:
            energy: JaxArray, shape=(num_replicas, 1)
                The potential energy calculated for the input coordinates, summed over particles.
        """

        assert x.shape[-1] == 2, "Only 2D potential is supported"

        x, y = jnp.split(x, 2, axis=-1)  # (num_replicas, num_particles, 1)
        u = 3 * (
            3 - jnp.sin(self.p * x) ** 4 - jnp.sin(self.p * y) ** 4
        )  # (num_replicas, num_particles, 1)
        E = jnp.exp(u) - 1  # (num_replicas, num_particles, 1)
        E = jnp.squeeze(E, axis=-1)  # (num_replicas, num_particles)

        # Sum over particles dimension and keep dimensions
        return jnp.sum(E, axis=-1, keepdims=True)  # shape (num_replicas, 1)

    def get_2dcontour(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        num_points: int = 100,
    ) -> Tuple[JaxArray, JaxArray, JaxArray]:
        """
        Generate a 2D contour plot of the potential energy.

        Args:
            x_range (Tuple[float, float]): Range of x values.
            y_range (Tuple[float, float]): Range of y values.
            num_points (int): Number of points in each dimension.
                Default: 100

        Returns:
            Tuple[JaxArray, JaxArray, JaxArray]: x-coordinates, y-coordinates, and potential energy values.
        """
        x = jnp.linspace(x_range[0], x_range[1], num_points)
        y = jnp.linspace(y_range[0], y_range[1], num_points)
        X, Y = jnp.meshgrid(x, y)
        coord = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
        coord = jnp.expand_dims(coord, axis=-2)
        E = self(coord, 1.0).reshape(X.shape)
        E -= jnp.min(E)

        return X, Y, E
