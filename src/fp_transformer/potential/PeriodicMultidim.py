from .base import Potential
from ..type import JaxArray, Tuple
from ..utils import get_jax_array, logsumexp
from jax import numpy as jnp


class PeriodicMultidim(Potential):
    """
    An N-dimensional periodic potential energy function based on the multivariate
    von Mises distribution. The potential is constructed as a sum of M distinct
    potential wells (negative von Mises distributions).

    The functional form is:
    V(x) = sum_{i=1 to M} -A_i * exp(sum_{j=1 to N} kappa_{ij} * cos(x_j - mu_{ij}))

    Attributes:
        centers (JaxArray): Shape (M, N). The mean direction vectors mu_i for each of the M wells.
        depths (JaxArray): Shape (M,). The depth parameters A_i for each well.
        concentrations (JaxArray): Shape (M, N). The concentration parameters kappa_{ij} for each well and dimension.
        num_wells (int): M, the number of potential wells.
        dim (int): N, the dimensionality of the space.
    """

    def __init__(self, centers: JaxArray, depths: JaxArray, concentrations: JaxArray):
        """
        Initializes the vonMisesMultivar potential.

        Args:
            centers (JaxArray): A JAX array of shape (M, N) representing the center
                coordinates (mu) for each of the M potential wells in N-dimensional space.
                Values should be in the range [0, 2*pi].
            depths (JaxArray): A JAX array of shape (M,) representing the depth (A_i)
                of each potential well. Must be positive values.
            concentrations (JaxArray): A JAX array of shape (M, N) representing the
                concentration/sharpness (kappa_{ij}) of each well for each dimension.
                Must be positive values. This allows for anisotropic distributions.
        """
        super().__init__()
        self.centers = get_jax_array(centers)
        self.depths = get_jax_array(depths)
        self.concentrations = get_jax_array(concentrations)

        # Validate shapes
        if not (
            self.centers.ndim == 2
            and self.depths.ndim == 1
            and self.concentrations.ndim == 2
        ):
            raise ValueError(
                "Incompatible shapes: centers and concentrations must be 2D, depths must be 1D."
            )
        if not (
            self.centers.shape[0]
            == self.depths.shape[0]
            == self.concentrations.shape[0]
        ):
            raise ValueError(
                "Inconsistent number of wells (M) across centers, depths, and concentrations."
            )
        if not (self.centers.shape[1] == self.concentrations.shape[1]):
            raise ValueError(
                "Inconsistent dimensionality (N) between centers and concentrations."
            )

        self.num_wells, self.dim = self.centers.shape
        self._weighted_center_cos = self.concentrations * jnp.cos(self.centers)
        self._weighted_center_sin = self.concentrations * jnp.sin(self.centers)
        self._log_depths = jnp.log(self.depths)

    def __call__(self, x: JaxArray, beta: float = 1.0) -> JaxArray:
        return self.potential_optimized(x, beta)

    def potential_optimized(self, x: JaxArray, beta: float = 1.0) -> JaxArray:
        """
        Calculates the potential energy for a batch of configurations using
        a GEMM-friendly formulation of cos(x - c).

        Args:
            x (JaxArray): A JAX array of shape (num_replicas, num_particle, N) representing the
                coordinates in the N-dimensional periodic space for each particle.
            beta (float): Inverse temperature. This parameter is ignored in this
                potential function calculation but included for API consistency.

        Returns:
            energy (JaxArray): A JAX array of shape (num_replicas, 1) containing
                the total potential energy for each configuration.
        """
        del beta
        num_replicas, num_particle, dim = x.shape
        x_flat = x.reshape(-1, dim)

        x_cos = jnp.cos(x_flat)
        x_sin = jnp.sin(x_flat)

        # cos(x - c) = cos(x)cos(c) + sin(x)sin(c), which turns the reduction
        # over coordinates into two matrix multiplications.
        cos_sum = jnp.matmul(x_cos, self._weighted_center_cos.T)
        cos_sum += jnp.matmul(x_sin, self._weighted_center_sin.T)

        log_wells = self._log_depths + cos_sum
        log_particle_energies = logsumexp(log_wells, axis=1).reshape(
            num_replicas, num_particle
        )
        log_total_energy = logsumexp(log_particle_energies, axis=1, keepdims=True)
        return -log_total_energy

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
        # Add num_particle dimension (assuming single particle for 2D contour)
        coord = jnp.expand_dims(coord, axis=-2)  # shape: (num_points^2, 1, 2)
        E = self(coord, 1.0).reshape(X.shape)
        E -= jnp.min(E)

        return X, Y, E
