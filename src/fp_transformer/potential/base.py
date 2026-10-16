from abc import ABC, abstractmethod
from ..type import JaxArray


class Potential(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x: JaxArray, beta: float = 1.0) -> JaxArray:
        """Calculate potential energy for given coordinates and temperature."""
        pass

    def force(self, topology) -> JaxArray:
        """Calculate forces. Default implementation uses gradient of potential."""
        from jax import grad
        import jax.numpy as jnp

        # Get coordinates from topology
        coord = topology.coordinate
        if coord is None:
            raise ValueError("Topology coordinate is None")

        # Calculate negative gradient (force = -grad(potential))
        grad_fn = grad(lambda x: jnp.sum(self(x, 1.0)))
        return -grad_fn(coord)
