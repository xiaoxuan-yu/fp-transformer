from .base import Potential
from ..type import JaxArray, Tuple
from ..utils import get_jax_array
from jax import numpy as jnp


class BerezhkovskiiSzabo(Potential):
    """
    Potential energy of a toy model developed by Berezhkovskii and Szabo.

    Reference:

        Berezhkovskii, A.; Szabo, A.
        One-dimensional reaction coordinates for diffusive activated rate processes in many dimensions.
        The Journal of Chemical Physics, 2005, 122 (1): 014503.

    Args:
        beta (float): Inverse temperature.
            Default: 1.0

        x_0 (float): Location of the potential well.
            Default: 2.2

        omega (float): Frequency of the harmonic potential.
            Default: 2

        scale (float): Scale factor for the potential.
            Default: 1.01
    """

    def __init__(
        self,
        beta: float = 1.0,
        x_0: float = 2.2,
        omega: float = 2,
        scale: float = 1.01,
        d: float = 1.0,
    ):
        self.beta = get_jax_array(beta)
        self.x_0 = get_jax_array(x_0)
        self.omega2 = jnp.square(get_jax_array(omega))
        self.delta = self.omega2 * jnp.square(self.x_0) / 4
        self.Omega2 = scale * self.omega2
        self.d = get_jax_array(d)

    def __call__(self, x: JaxArray, beta: float) -> JaxArray:
        """
        Calculate the potential energy for the given input.

        Args:
            x : JaxArray, shape=(num_replicas, num_particles, 2)
                The input coordinates for which to calculate the potential energy.
                The shape is (num_replicas, num_particles, 2), representing batch size,
                number of particles, and the 2D coordinates for each particle.
            beta: float
                The reverse temperature scaling factor.

        Returns:
            energy : JaxArray, shape=(num_replicas, 1)
                The potential energy calculated for the input coordinates, summed over particles.
        """
        assert x.shape[-1] == 2, "Only 2D potential is supported"
        energy = (
            self.Omega2 * (x[..., 1] - x[..., 0] * self.d) ** 2 / (2 * beta)
        )  # shape (num_replicas, num_particles)
        # x < -x_0/2
        energy += jnp.where(
            x[..., 0] < -self.x_0 / 2,
            -self.delta + self.omega2 * (x[..., 0] + self.x_0) ** 2 / 2,
            0,
        )
        # x > x_0/2
        energy += jnp.where(
            x[..., 0] > self.x_0 / 2,
            -self.delta + self.omega2 * (x[..., 0] - self.x_0) ** 2 / 2,
            0,
        )
        # -x_0/2 < x < x_0/2
        energy += jnp.where(
            (x[..., 0] > -self.x_0 / 2) & (x[..., 0] < self.x_0 / 2),
            -self.omega2 * x[..., 0] ** 2 / 2,
            0,
        )
        # Sum over particles dimension and keep dimensions
        return jnp.sum(energy, axis=-1, keepdims=True)  # shape (num_replicas, 1)

    def get_2dcontour(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        num_grids: int = 100,
    ) -> JaxArray:
        """
        Calculate the potential energy of the Berezhkovskii-Szabo model on a 2D grid.

        Args:
            x_range: Tuple[float, float]
                The range of x-axis.
            y_range: Tuple[float, float]
                The range of y-axis.
            num_grids: int
                The number of grid points along each axis.

        Returns:
            energy: JaxArray, shape=(num_grids, num_grids)
                The potential energy values at each grid point.
        """
        x_grids = jnp.linspace(x_range[0], x_range[1], num_grids)
        y_grids = jnp.linspace(y_range[0], y_range[1], num_grids)
        x, y = jnp.meshgrid(x_grids, y_grids)

        # construct the coordinate
        coord = jnp.stack((x.ravel(), y.ravel()), axis=-1)
        coord = jnp.expand_dims(coord, axis=-2)

        # calculate the potential energy
        energy = self(coord, 1).reshape(num_grids, num_grids)
        energy -= jnp.min(energy)

        return x, y, energy
