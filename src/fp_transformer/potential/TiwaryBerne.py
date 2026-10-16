from .base import Potential
from ..type import Union, JaxArray, Array, List, Tuple
from ..utils import get_jax_array
from jax import numpy as jnp


class TiwaryBerne(Potential):
    """
    Potential energy of a toy model developed by Tiwary and Berne.

    Reference:

        Tiwary, P.; Berne, B. J.
        Predicting Reaction Coordinates in Energy Landscapes with Diffusion Anisotropy [J].
        The Journal of Chemical Physics, 2017, 147(15): 152701.

    Args:

        location: Union[JaxArray, Array, List[float], Tuple[float]]:
            Locations of metastable states on the potential energy surface (PES).
            The shape of the array is (num_metastable_states, dimension), and the data type is float64.
            Default: ((-0.5, 0.5), (0.8, 1.2), (0.5, -0.3))

        depth: Union[JaxArray, Array, List[float], Tuple[float]]:
            Depths of metastable states on the PES.
            The shape of the array is (num_metastable_states), and the data type is float64.
            Default: (16, 18, 16)

        base_energy (float): Base energy of the toy model.
            Default: 20
    """

    def __init__(
        self,
        location: Union[JaxArray, Array, List[float], Tuple[float]] = (
            (-0.5, 0.5),
            (0.8, 1.2),
            (0.5, -0.3),
        ),
        depth: Union[JaxArray, Array, List[float], Tuple[float]] = (16, 18, 16),
        base_energy: float = 20,
        in_sharding = None,
        out_sharding = None
    ):
        super().__init__()

        if in_sharding is not None:
            if out_sharding is None:
                out_sharding = in_sharding
            self.in_sharding = in_sharding
            self.out_sharding = out_sharding
        else:
            self.in_sharding = None
            self.out_sharding = None

        self.location: JaxArray = get_jax_array(location, sharding = self.in_sharding)
        self.dimension = self.location.shape[-1]

        if self.location.ndim == 1:
            self.location = jnp.expand_dims(self.location, 0)

        num_states = self.location.shape[-2]

        self.depth: JaxArray = get_jax_array(depth, sharding=self.in_sharding)
        if self.depth.shape[-1] != num_states and self.depth.shape[-1] != 1:
            raise ValueError(
                f"The number of depth {self.depth.shape[-1]} does not match the number of states {num_states}"
            )

        self.base_energy = get_jax_array(base_energy, sharding=self.in_sharding)

    def __call__(self, coordinate: JaxArray, beta: float = 1.0) -> JaxArray:
        """
        Calculate the potential energy of the Tiwary-Berne model.

        Args:
            coordinate: JaxArray, shape=(num_replicas, num_particles, dimension)
                The coordinates of particles.

        Returns:
            energy: JaxArray, shape=(num_replicas, 1)
                The potential energy of the Tiwary-Berne model.
        """
        diff = jnp.expand_dims(coordinate, axis=-2) - self.location
        diff2 = jnp.sum(jnp.square(diff), axis=-1)

        energy = -1 * self.depth * jnp.exp(-2 * diff2)
        energy = jnp.sum(energy, axis=-1)

        restraint = 0.5 * jnp.sum(jnp.pow(coordinate, 6), axis=-1)

        return jnp.sum(energy + self.base_energy + restraint, keepdims=True, axis=-1)

    def get_2dcontour(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        num_grids: int = 100,
    ) -> JaxArray:
        """
        Calculate the potential energy of the Tiwary-Berne model on a 2D grid.

        Args:
            x_range: Tuple[float, float]
                The range of x-axis.
            y_range: Tuple[float, float]
                The range of y-axis.
            num_grids: int
                The number of grids on each axis.

        Returns:
            energy: JaxArray, shape=(num_points, num_points)
                The potential energy of the Tiwary-Berne model on the 2D grid.
        """

        # construct the meshgrid
        x_grids = jnp.linspace(x_range[0], x_range[1], num_grids)
        y_grids = jnp.linspace(y_range[0], y_range[1], num_grids)
        x, y = jnp.meshgrid(x_grids, y_grids)

        # construct the coordinate
        coordinate = jnp.stack((x.ravel(), y.ravel()), axis=1)
        coordinate = jnp.expand_dims(coordinate, axis=-2)

        # calculate the potential energy
        energy = self(coordinate).reshape(num_grids, num_grids)
        energy -= jnp.min(energy)

        return x, y, energy
