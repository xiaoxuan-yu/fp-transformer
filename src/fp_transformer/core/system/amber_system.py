from typing import Any

from jax import grad, jit, lax, vmap
from jax import tree_util
from jax import numpy as jnp

from ...potential import AmberPotential
from ...type import JaxArray, Optional
from ..topology import CoordTopology
from .system import BaseSystem


class AmberSystem(BaseSystem):
    """A System wrapper for AmberPotential with coordinate/topology and neighbor list management.

    Typical usage:
      sys = AmberSystem(prmtop, inpcrd, periodic=True, pme=True)
            E = sys.potential()                  # energy at current coordinates
            G = sys.grad_potential()             # gradient of potential w.r.t. coordinates
            F = sys.forces()                     # forces = -gradient
            sys.update_neighbor_list()           # update neighbor list when coordinates changed
    """

    def __init__(
        self,
        prmtop_file: str,
        inpcrd_file: str,
        *,
        topology: Optional[CoordTopology] = None,
        periodic: bool = False,
        pme: bool = False,
        cutoff: float = 0.5,
        grid_points: int = 96,
        dr_threshold: float = 0.2,
        use_vmap: bool = True,
        minibatch_size: Optional[int] = None,
        assert_divisible_minibatch: bool = False,
    ):
        super().__init__()
        # Construct underlying potential and use its loaded positions as initial coordinates
        self.potential_: AmberPotential = AmberPotential(
            prmtop_file,
            inpcrd_file,
            periodic=periodic,
            pme=pme,
            cutoff=cutoff,
            grid_points=grid_points,
            dr_threshold=dr_threshold,
            use_vmap=use_vmap,
        )
        # Topology holds coordinates initialized from potential's default positions
        self.topology = (
            topology
            if topology is not None
            else CoordTopology(self.potential_.get_default_positions())
        )
        self.minibatch_size = minibatch_size
        self.assert_divisible_minibatch = assert_divisible_minibatch

        # Cache for single-frame neighbor list；批量场景请通过方法参数传入
        # 非周期系统邻居表可选。
        self.nbList = self.allocate_neighbor_list(self.coordinate) if periodic else None

        self.update_neighbor_list_jit = jit(self.potential_.update_neighbor_list)
        self._grad_single_no_nbr = grad(self._energy_single_no_nbr, argnums=0)
        self._grad_single_with_nbr = grad(self._energy_single_with_nbr, argnums=0)
        self._grad_single_no_nbr_jit = jit(self._grad_single_no_nbr)
        self._grad_single_with_nbr_jit = jit(self._grad_single_with_nbr)
        self._grad_batched_no_nbr_jit = jit(
            vmap(self._grad_single_no_nbr, in_axes=(0, None))
        )
        self._grad_batched_broadcast_nbr_jit = jit(
            vmap(self._grad_single_with_nbr, in_axes=(0, None, None))
        )
        self._grad_batched_batched_nbr_jit = jit(
            vmap(self._grad_single_with_nbr, in_axes=(0, None, 0))
        )

    def _energy_single_no_nbr(self, pos: JaxArray, beta: float = 1.0):
        return self.potential_(pos, beta, nbLists=None)

    def _energy_single_with_nbr(self, pos: JaxArray, beta: float = 1.0, nbr=None):
        return self.potential_(pos, beta, nbLists=nbr)

    @staticmethod
    def _is_batched_nbr(nbr, batch_size: int) -> bool:
        from jax import tree_util

        leaves = tree_util.tree_leaves(nbr)
        array_leaves = [
            leaf for leaf in leaves if hasattr(leaf, "shape") and hasattr(leaf, "ndim")
        ]
        if not array_leaves:
            return False
        return all(
            leaf.ndim > 0 and leaf.shape[0] == batch_size for leaf in array_leaves
        )

    @staticmethod
    def _match_sharding_like_input(inp: JaxArray, out: JaxArray) -> JaxArray:
        in_sharding = getattr(inp, "sharding", None)
        if in_sharding is None:
            return out
        return lax.with_sharding_constraint(out, in_sharding)

    # ---------------------------- BaseSystem API ---------------------------- #
    @property
    def coordinate(self) -> Optional[JaxArray]:
        return self.topology.coordinate

    @coordinate.setter
    def coordinate(self, coordinate: JaxArray) -> None:
        self.topology.coordinate = coordinate

    def potential(
        self,
        beta: float = 1.0,
        *,
        coord: Optional[JaxArray] = None,
        nbLists: Optional[Any] = None,
        nbList: Optional[Any] = None,  # deprecated alias; treated the same as nbLists
    ) -> JaxArray:
        """Calculate potential energy at given or current coordinates.

        Args:
            beta: Inverse temperature (forwarded; not used by AmberPotential directly).
            coord: Optional coordinates; if None, uses self.coordinate.
            nbLists: 邻居表参数（单帧传入单个对象；批量传入同批量形状的列表/数组）。
            nbList: 兼容别名；行为与 nbLists 完全一致。
        """
        x = coord if coord is not None else self.coordinate
        if x is None:
            raise ValueError("Coordinate is None")

        # 邻居表优先使用显式传入，其次使用缓存的单帧 nbList
        nb_param = (
            nbLists
            if nbLists is not None
            else (nbList if nbList is not None else self.nbList)
        )
        minibatch_size = self.minibatch_size
        assert_divisible_minibatch = self.assert_divisible_minibatch
        if not (hasattr(x, "ndim") and x.ndim == 3) or minibatch_size is None:
            return self.potential_(x, beta, nbLists=nb_param)

        mbs = int(minibatch_size)
        if mbs <= 0:
            raise ValueError("minibatch_size must be a positive integer")
        batch_size = int(x.shape[0])
        if mbs >= batch_size:
            return self.potential_(x, beta, nbLists=nb_param)
        if assert_divisible_minibatch and (batch_size % mbs != 0):
            raise ValueError(
                f"Local batch size ({batch_size}) must be divisible by minibatch_size ({mbs})."
            )
        n_chunks = batch_size // mbs
        x_chunks = x.reshape((n_chunks, mbs) + x.shape[1:])

        if nb_param is None:
            def body(_, x_chunk):
                e = self.potential_(x_chunk, beta, nbLists=None)
                return None, e

            _, e_chunks = lax.scan(body, None, x_chunks)
            return e_chunks.reshape((n_chunks * mbs,) + e_chunks.shape[2:])

        if self._is_batched_nbr(nb_param, batch_size):
            nb_chunks = tree_util.tree_map(
                lambda leaf: leaf.reshape((n_chunks, mbs) + leaf.shape[1:])
                if (
                    hasattr(leaf, "ndim")
                    and leaf.ndim > 0
                    and int(leaf.shape[0]) == batch_size
                )
                else leaf,
                nb_param,
            )

            def body(_, inputs):
                x_chunk, nb_chunk = inputs
                e = self.potential_(x_chunk, beta, nbLists=nb_chunk)
                return None, e

            _, e_chunks = lax.scan(body, None, (x_chunks, nb_chunks))
            return e_chunks.reshape((n_chunks * mbs,) + e_chunks.shape[2:])

        def body(_, x_chunk):
            e = self.potential_(x_chunk, beta, nbLists=nb_param)
            return None, e

        _, e_chunks = lax.scan(body, None, x_chunks)
        return e_chunks.reshape((n_chunks * mbs,) + e_chunks.shape[2:])

    def grad_potential(
        self,
        coord: Optional[JaxArray] = None,
        *,
        beta: float = 1.0,
        nbLists: Optional[Any] = None,
        nbList: Optional[Any] = None,  # deprecated alias
    ) -> JaxArray:
        """Calculate gradient of potential wrt coordinates.

        单帧：若提供 nbLists 则使用；否则使用缓存的 self.nbList。
        批量：对 (R_i, nb_i) 做 vmap；nbLists 应与批量形状匹配；若为 None，仅对位置做 vmap，
        对周期系统将由底层 Potential 抛错提示需要邻居表。
        """
        x = coord if coord is not None else self.coordinate
        if x is None:
            raise ValueError("Coordinate is None")

        # 选择邻居表参数：批量场景不使用缓存的单帧 nbList，避免形状不匹配
        if hasattr(x, "ndim") and x.ndim == 3:
            nb_param = nbLists if nbLists is not None else nbList
            minibatch_size = self.minibatch_size
            assert_divisible_minibatch = self.assert_divisible_minibatch

            batch_size = int(x.shape[0])
            mbs = None if minibatch_size is None else int(minibatch_size)
            if mbs is not None and mbs <= 0:
                raise ValueError("minibatch_size must be a positive integer")

            use_chunk = mbs is not None and mbs < batch_size

            def _chunked_eval(eval_fn, *, batched_nbr=False):
                if not use_chunk:
                    grad_out = eval_fn(x, nb_param)
                    return self._match_sharding_like_input(x, grad_out)
                if assert_divisible_minibatch and (batch_size % mbs != 0):
                    raise ValueError(
                        f"Local batch size ({batch_size}) must be divisible by minibatch_size ({mbs})."
                    )
                n_chunks = batch_size // mbs
                x_chunks = x.reshape((n_chunks, mbs) + x.shape[1:])

                if batched_nbr:
                    nb_chunks = tree_util.tree_map(
                        lambda leaf: leaf.reshape((n_chunks, mbs) + leaf.shape[1:])
                        if (
                            hasattr(leaf, "ndim")
                            and leaf.ndim > 0
                            and int(leaf.shape[0]) == batch_size
                        )
                        else leaf,
                        nb_param,
                    )
                else:
                    nb_chunks = nb_param

                if batched_nbr:
                    def body(_, inputs):
                        x_chunk, nb_chunk = inputs
                        g = eval_fn(x_chunk, nb_chunk)
                        return None, g

                    _, grad_chunks = lax.scan(body, None, (x_chunks, nb_chunks))
                else:
                    def body(_, x_chunk):
                        g = eval_fn(x_chunk, nb_param)
                        return None, g

                    _, grad_chunks = lax.scan(body, None, x_chunks)
                grad_out = grad_chunks.reshape((n_chunks * mbs,) + x.shape[1:])
                return self._match_sharding_like_input(x, grad_out)

            if nb_param is None:
                if not use_chunk:
                    grad_out = self._grad_batched_no_nbr_jit(x, beta)
                    return self._match_sharding_like_input(x, grad_out)
                if assert_divisible_minibatch and (batch_size % mbs != 0):
                    raise ValueError(
                        f"Local batch size ({batch_size}) must be divisible by minibatch_size ({mbs})."
                    )
                n_chunks = batch_size // mbs
                x_chunks = x.reshape((n_chunks, mbs) + x.shape[1:])

                def body(_, x_chunk):
                    g = self._grad_batched_no_nbr_jit(x_chunk, beta)
                    return None, g

                _, grad_chunks = lax.scan(body, None, x_chunks)
                grad_out = grad_chunks.reshape((n_chunks * mbs,) + x.shape[1:])
                return self._match_sharding_like_input(x, grad_out)

            # If neighbor list is batched, vmap over (pos, nbr). Otherwise broadcast.
            if self._is_batched_nbr(nb_param, x.shape[0]):
                return _chunked_eval(
                    lambda xc, nc: self._grad_batched_batched_nbr_jit(xc, beta, nc),
                    batched_nbr=True,
                )
            return _chunked_eval(
                lambda xc, nc: self._grad_batched_broadcast_nbr_jit(xc, beta, nc),
                batched_nbr=False,
            )

        nb_param = (
            nbLists
            if nbLists is not None
            else (nbList if nbList is not None else self.nbList)
        )

        # Single-frame gradient via cached AD/JIT.
        if nb_param is None:
            return self._grad_single_no_nbr_jit(x, beta)
        return self._grad_single_with_nbr_jit(x, beta, nb_param)

    # ------------------------ Neighbor list management --------------------- #
    def allocate_neighbor_list(
        self, positions: Optional[JaxArray] = None, extra_capacity: int = 0
    ):
        """Allocate and cache a neighbor list for current or provided positions.

        Returns the allocated neighbor list (single) or list batch.
        """
        nbr = self.potential_.allocate_neighbor_list(
            positions, extra_capacity=extra_capacity
        )  # type: ignore
        return nbr

    def update_neighbor_list(
        self,
        positions: Optional[JaxArray] = None,
        nbLists: Optional[Any] = None,
        nbList: Optional[Any] = None,
    ):
        """Update neighbor list for given or current coordinates.

        For periodic PME systems, this should be called when coordinates move significantly.
        """
        pos = positions if positions is not None else self.coordinate
        if pos is None:
            raise ValueError("Coordinate is None")

        if pos.ndim == 3:
            if nbLists is None and nbList is None:
                raise ValueError(
                    "For batched positions, please provide nbLists to update."
                )
            nb_param = nbLists if nbLists is not None else nbList
            nbr = self.potential_.update_neighbor_list(
                pos,
                nbList=nb_param,
                n_times_reallocated=self.potential_.n_times_reallocated,
            )  # type: ignore
            self.nbList = nbr
            return nbr
        else:
            if nbLists is None and nbList is None:
                if self.nbList is None:
                    raise ValueError(
                        "No existing neighbor list to update. Call allocate_neighbor_list() first or pass nbLists/nbList explicitly."
                    )
                nb_param = self.nbList
            else:
                nb_param = nbLists if nbLists is not None else nbList
            nbr = self.potential_.update_neighbor_list(
                pos,
                nbList=nb_param,
                n_times_reallocated=self.potential_.n_times_reallocated,
            )  # type: ignore
            self.nbList = nbr
            return nbr

    # ----------------------------- Utilities ------------------------------ #
    def forces(
        self,
        coord: Optional[JaxArray] = None,
        beta: float = 1.0,
        *,
        nbLists: Optional[Any] = None,
        nbList: Optional[Any] = None,  # deprecated alias
    ) -> JaxArray:
        """Convenience method: compute forces = -grad(potential).

        Mirrors the arguments of grad_potential and flips the sign.
        """
        nb_param = nbLists if nbLists is not None else nbList
        return -self.grad_potential(
            coord=coord,
            beta=beta,
            nbLists=nb_param,
        )

    @property
    def periodic(self) -> bool:
        return getattr(self.potential_, "periodic", False)

    @property
    def pme(self) -> bool:
        return getattr(self.potential_, "pme", False)
