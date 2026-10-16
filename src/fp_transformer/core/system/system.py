from abc import ABC, abstractmethod
from typing import Optional
from ...potential import Potential
from ..topology import Topology
from ...type import JaxArray
from jax import grad, jit, lax, value_and_grad, vmap
from jax import numpy as jnp


class BaseSystem(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def coordinate(self) -> Optional[JaxArray]:
        """Get the coordinate array."""
        pass

    @coordinate.setter
    @abstractmethod
    def coordinate(self, coordinate: JaxArray) -> None:
        """Set the coordinate array."""
        pass

    @abstractmethod
    def potential(self, *args, **kwargs) -> JaxArray:
        """Calculate potential energy."""
        pass

    @abstractmethod
    def grad_potential(self, *args, **kwargs) -> JaxArray:
        """Calculate gradient of potential energy."""
        pass

    def project_coordinate(self, coordinate: JaxArray) -> JaxArray:
        """Project updated coordinates back onto the system manifold.

        Default behavior is identity. Internal-coordinate systems can override
        this to enforce periodic wrapping or other holonomic constraints after
        optimizer steps.
        """
        return coordinate


class ForceSystem(BaseSystem):
    def __init__(self, topology: Topology, potential_: Potential):
        super().__init__()
        self.topology = topology
        self.potential_ = potential_
        self.nbList = None

    def potential(
        self, beta: float = 1.0, *, coord: Optional[JaxArray] = None
    ) -> JaxArray:
        """Calculate potential energy using the potential function."""
        coord = self.topology.coordinate if coord is None else coord
        if coord is None:
            raise ValueError("Topology coordinate is None")
        return self.potential_(coord, beta)

    def grad_potential(
        self, coord: Optional[JaxArray] = None, *, beta: float = 1.0
    ) -> JaxArray:
        """Calculate forces using the potential's force method."""
        return self.potential_.force(self.topology)

    @property
    def coordinate(self) -> Optional[JaxArray]:
        """Get coordinate from topology."""
        return self.topology.coordinate

    @coordinate.setter
    def coordinate(self, coordinate: JaxArray) -> None:
        """Set coordinate in topology."""
        self.topology.coordinate = coordinate


class EnergySystem(BaseSystem):
    def __init__(
        self,
        topology: Topology,
        potential_: Potential,
        *,
        minibatch_size: Optional[int] = None,
        assert_divisible_minibatch: bool = False,
        assume_batch_separable: bool = False,
    ):
        super().__init__()
        potential_cls = type(potential_)
        if (
            potential_cls.__name__ == "AmberPotential"
            and potential_cls.__module__.endswith(".potential.Amber")
        ):
            raise TypeError(
                "AmberPotential must be used with AmberSystem, not EnergySystem."
            )
        self.topology = topology
        self.potential_ = potential_
        self.nbList = None
        self.minibatch_size = minibatch_size
        self.assert_divisible_minibatch = assert_divisible_minibatch
        self.assume_batch_separable = bool(assume_batch_separable)
        self._grad_potential_local_jit = jit(
            vmap(
                grad(
                    lambda coord_single, beta: jnp.sum(
                        self.potential_(coord_single[jnp.newaxis, ...], beta)
                    )
                ),
                in_axes=(0, None),
            )
        )
        self._value_and_grad_potential_batch_jit = jit(
            value_and_grad(
                lambda coord_batch, beta: jnp.sum(
                    self.potential_(coord_batch, beta)
                )
            )
        )

    def _grad_potential_batch(self, coord: JaxArray, beta: float) -> JaxArray:
        if self.assume_batch_separable:
            _, grad_out = self._value_and_grad_potential_batch_jit(coord, beta)
            return grad_out
        return self._grad_potential_local_jit(coord, beta)

    @staticmethod
    def _match_sharding_like_input(inp: JaxArray, out: JaxArray) -> JaxArray:
        in_sharding = getattr(inp, "sharding", None)
        if in_sharding is None:
            return out
        return lax.with_sharding_constraint(out, in_sharding)

    @property
    def coordinate(self) -> Optional[JaxArray]:
        """Get coordinate from topology."""
        return self.topology.coordinate

    @coordinate.setter
    def coordinate(self, coordinate: JaxArray) -> None:
        """Set coordinate in topology."""
        self.topology.coordinate = coordinate

    def potential(
        self,
        beta: float = 1.0,
        *,
        coord: Optional[JaxArray] = None,
    ) -> JaxArray:
        """Calculate potential energy."""
        coord = self.coordinate if coord is None else coord
        if coord is None:
            raise ValueError("Coordinate is None")
        minibatch_size = self.minibatch_size
        assert_divisible_minibatch = self.assert_divisible_minibatch
        if coord.ndim != 3 or minibatch_size is None:
            return self.potential_(coord, beta)

        mbs = int(minibatch_size)
        if mbs <= 0:
            raise ValueError("minibatch_size must be a positive integer")
        batch_size = int(coord.shape[0])
        if mbs >= batch_size:
            return self.potential_(coord, beta)
        if assert_divisible_minibatch and (batch_size % mbs != 0):
            raise ValueError(
                f"Local batch size ({batch_size}) must be divisible by minibatch_size ({mbs})."
            )
        n_chunks = batch_size // mbs
        coord_chunks = coord.reshape((n_chunks, mbs) + coord.shape[1:])

        def body(_, chunk):
            e = self.potential_(chunk, beta)
            return None, e

        _, energy_chunks = lax.scan(body, None, coord_chunks)
        return energy_chunks.reshape((n_chunks * mbs,) + energy_chunks.shape[2:])

    def grad_potential(
        self,
        coord: JaxArray,
        *,
        beta: float = 1.0,
    ) -> JaxArray:
        """Calculate per-replica gradient of potential energy without cross-replica reductions.

        Args:
            coord: Coordinates with shape `(B, N, D)` or single-frame shape `(N, D)`.
            beta: Inverse temperature.
            Uses constructor-level `minibatch_size` and
            `assert_divisible_minibatch` settings.

        When ``assume_batch_separable`` is enabled, three-dimensional batches
        use one reverse pass through the summed batch energy.  This is faster
        for replica-separable potentials, but is deliberately opt-in because a
        batch-coupled potential has different derivatives under that reduction.
        """
        minibatch_size = self.minibatch_size
        assert_divisible_minibatch = self.assert_divisible_minibatch
        if coord.ndim != 3 or minibatch_size is None:
            grad_out = (
                self._grad_potential_batch(coord, beta)
                if coord.ndim == 3
                else self._grad_potential_local_jit(coord, beta)
            )
            if coord.ndim == 3:
                return self._match_sharding_like_input(coord, grad_out)
            return grad_out

        mbs = int(minibatch_size)
        if mbs <= 0:
            raise ValueError("minibatch_size must be a positive integer")
        batch_size = int(coord.shape[0])
        if mbs >= batch_size:
            grad_out = self._grad_potential_batch(coord, beta)
            return self._match_sharding_like_input(coord, grad_out)

        if assert_divisible_minibatch and (batch_size % mbs != 0):
            raise ValueError(
                f"Local batch size ({batch_size}) must be divisible by minibatch_size ({mbs})."
            )
        n_chunks = batch_size // mbs
        coord_chunks = coord.reshape((n_chunks, mbs) + coord.shape[1:])

        def body(_, chunk):
            g = self._grad_potential_batch(chunk, beta)
            return None, g

        _, grad_chunks = lax.scan(body, None, coord_chunks)
        grad_out = grad_chunks.reshape((n_chunks * mbs,) + coord.shape[1:])
        return self._match_sharding_like_input(coord, grad_out)
