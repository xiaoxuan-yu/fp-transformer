"""Amber forcefields for Potentials (JAX-MD new mm_forcefields API).

This module is a thin wrapper around JAX-MD's OpenMM interoperability:

- jax_md.mm_forcefields.io.openmm.load_amber_system / convert_openmm_system
- jax_md.mm_forcefields.amber.energy.amber_energy

Unit conventions follow JAX-MD forcefield modules:
  - positions: Angstrom
  - energy:    kcal/mol
  - forces:    kcal/(mol*Angstrom)

Design goals (informed by jax-md/tests/amber_test.py):
  - Keep the Potential wrapper "dumb": it does *not* own mutable simulation state.
  - Make neighbor-list allocation/update explicit and functional (allocate cannot be JIT).
  - Support OpenMM nonbonded methods: NoCutoff / CutoffPeriodic / Ewald / PME.
  - Optional virtual-site position projection inside the energy path so grads are correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import logging

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap, tree_util

from jax_md import partition
from jax_md.mm_forcefields.amber import energy as amber_energy
from jax_md.mm_forcefields.io.openmm import (
    convert_openmm_system,
    load_amber_system,
    virtual_site_apply_positions,
)
from jax_md.mm_forcefields.nonbonded import electrostatics

from .base import Potential
from ..type import JaxArray

logger = logging.getLogger(__name__)

try:
    from openmm import app, unit  # type: ignore
except Exception:  # pragma: no cover
    app = None  # type: ignore
    unit = None  # type: ignore


# -----------------------------
# Configuration + helpers
# -----------------------------


@dataclass(frozen=True)
class AmberConfig:
    """Configuration for building an AmberPotential.

    Args:
      prmtop_file: AMBER topology file.
      inpcrd_file: AMBER coordinates file.
      nonbonded_method: OpenMM app.* method (e.g. app.PME, app.Ewald, app.NoCutoff).
      cutoff_angstrom: Nonbonded cutoff in Angstrom. Ignored for NoCutoff.
      dr_threshold_angstrom: Neighbor-list buffer threshold in Angstrom.
      neighbor_format: JAX-MD neighbor list format.
      enable_virtual_sites: If True, project virtual-site positions inside energy
        evaluation (recommended when virtual sites exist).
    """

    prmtop_file: str
    inpcrd_file: str
    nonbonded_method: Any = None
    cutoff_angstrom: float = 8.0
    dr_threshold_angstrom: float = 2.0
    neighbor_format: Any = partition.NeighborListFormat.OrderedSparse
    enable_virtual_sites: bool = True


def _require_openmm() -> None:
    if app is None or unit is None:
        raise ImportError(
            "OpenMM is required for AmberPotential because it uses "
            "jax_md.mm_forcefields.io.openmm.load_amber_system."
        )


def _make_coulomb_handler(
    nb_method: Any,
    r_cut: float,
    recip_alpha: Any,
    recip_grid: Any,
) -> electrostatics.CoulombHandler:
    """Pick a Coulomb handler consistent with how the OpenMM system was built."""
    # NOTE: the returned handler must match the openmm nonbonded method and
    # conversion settings in convert_openmm_system.
    if nb_method == app.PME:
        return electrostatics.PMECoulomb(
            r_cut=r_cut, alpha=recip_alpha, grid_size=recip_grid
        )
    if nb_method == app.Ewald:
        return electrostatics.EwaldCoulomb(r_cut=r_cut)
    # Covers: NoCutoff, CutoffNonPeriodic, CutoffPeriodic.
    #
    # NOTE: JAX-MD's CutoffCoulomb uses `charge_sq / dr` without guarding `dr=0`.
    # In sparse neighbor-list formats there can be masked-out entries that still
    # evaluate `dr=0` (e.g. padding / self pairs), and if masking is applied via
    # multiplication (0 * inf), this can leak NaNs under JAX transformations
    # like `vmap`. Use a numerically safe variant.
    return SafeCutoffCoulomb(r_cut=r_cut)


class SafeCutoffCoulomb(electrostatics.CutoffCoulomb):
    """Cutoff Coulomb with a `dr=0` guard to avoid inf/NaN in masked entries."""

    def prepare_smap(  # type: ignore[override]
        self,
        charges: jnp.ndarray,
        box: Any,
        exc_charge_prod: Optional[jnp.ndarray],
        displacement_fn: Any,
        cutoff_fn: Any,
        fractional_coordinates: bool,
    ):
        from jax_md import smap, space

        def _safe_dr(dr: jnp.ndarray) -> jnp.ndarray:
            # Avoid division-by-zero on masked/padded/self entries.
            return jnp.where(jnp.isclose(dr, 0.0), 1.0, dr)

        def pair_plain_map(dr, charge_sq, **unused_kwargs):
            dr = _safe_dr(dr)
            return electrostatics.COULOMB_CONSTANT * (charge_sq / dr)

        bond_coul_fn = smap.bond(
            pair_plain_map,
            space.canonicalize_displacement_or_metric(displacement_fn),
            charge_sq=None,
        )
        # The current jax-md CutoffCoulomb API expects a single callable here.
        return bond_coul_fn


def _normalize_nonbonded_method(
    *,
    periodic: bool,
    use_pme: bool,
    use_ewald: bool,
) -> Any:
    """Resolve a nonbonded method into an OpenMM app.* enum."""
    _require_openmm()
    if not periodic:
        # NOTE:
        # jax_md.mm_forcefields.io.openmm.convert_openmm_system currently assumes
        # every non-NoCutoff method is periodic and requires `box_vectors`.
        # For AMBER inpcrd files without box info this would crash with:
        #   AttributeError: 'NoneType' object has no attribute 'value_in_unit'
        # Keep non-periodic systems on NoCutoff to avoid that converter bug.
        return app.NoCutoff
    if use_pme:
        return app.PME
    if use_ewald:
        return app.Ewald
    # Default "plain cutoff" periodic electrostatics.
    return app.CutoffPeriodic


# -----------------------------
# Main Potential wrapper
# -----------------------------

NeighborList = Any


class AmberPotential(Potential):
    """Amber forcefield potential built from prmtop/inpcrd via OpenMM -> JAX-MD.

    Key properties:
      - self.mm: output of convert_openmm_system (params/topology/nb_options/etc).
      - self.energy_terms: callable returning a dict of energy terms (incl. 'etotal').
      - self.neighbor_fn: neighbor-list "factory" with allocate()/update().

    This class does not run dynamics; it provides:
      - energy() / __call__ (single or batched)
      - forces() (via JAX autodiff)
      - allocate_neighbor_list() / update_neighbor_list() utilities
    """

    def __init__(
        self,
        prmtop_file: str,
        inpcrd_file: str,
        *,
        positions: Optional[JaxArray] = None,
        periodic: bool = False,
        pme: bool = False,
        ewald: bool = False,
        cutoff: float = 8.0,
        grid_points: Optional[int] = None,
        dr_threshold: float = 2.0,
        enable_virtual_sites: bool = True,
        neighbor_format: Any = partition.NeighborListFormat.OrderedSparse,
        use_vmap: bool = True,
    ):
        """
        Args:
          prmtop_file: Path to the AMBER prmtop file.
          inpcrd_file: Path to the AMBER inpcrd/rst7 file.
          positions: Optional initial coordinates in Angstrom. If None, use file coords.
          periodic: Whether to use periodic boundary conditions.
          pme: Use PME electrostatics (requires periodic=True).
          ewald: Use Ewald electrostatics (requires periodic=True). Ignored if pme=True.
          cutoff: Nonbonded cutoff in nanometer (ignored for NoCutoff).
          grid_points: Optional PME grid override (kept for compatibility).
          dr_threshold: Neighbor-list buffer threshold in Angstrom.
          enable_virtual_sites: Project virtual-site positions inside energy path.
          neighbor_format: Neighbor list format (OrderedSparse recommended for tests).
          use_vmap: Whether to use vmap for batched energy evaluation.
        """
        super().__init__()
        _require_openmm()

        if periodic and (not pme) and (not ewald):
            raise NotImplementedError(
                "periodic=True with pme=False is not supported yet. "
                "Enable pme=True (or ewald=True) or use periodic=False."
            )
        if pme and not periodic:
            raise ValueError("pme=True requires periodic=True.")
        if ewald and not periodic:
            raise ValueError("ewald=True requires periodic=True.")

        self.periodic = bool(periodic)
        self.pme = bool(pme)
        self.ewald = bool(ewald)
        self.grid_points = grid_points
        self.use_vmap = bool(use_vmap)
        self.cutoff = float(cutoff)

        self.config = AmberConfig(
            prmtop_file=prmtop_file,
            inpcrd_file=inpcrd_file,
            nonbonded_method=_normalize_nonbonded_method(
                periodic=periodic, use_pme=pme, use_ewald=ewald
            ),
            cutoff_angstrom=float(cutoff) * 10.0,
            dr_threshold_angstrom=float(dr_threshold),
            neighbor_format=neighbor_format,
            enable_virtual_sites=bool(enable_virtual_sites),
        )

        # --- Load OpenMM system and convert into JAX-MD forcefield structures ---
        load_kwargs: dict[str, Any] = {}
        if self.config.nonbonded_method != app.NoCutoff:
            load_kwargs["nonbondedCutoff"] = self.cutoff * unit.nanometer

        omm_system, omm_topology, omm_positions, omm_box_vectors = load_amber_system(
            self.config.prmtop_file,
            self.config.inpcrd_file,
            self.config.nonbonded_method,
            **load_kwargs,
        )

        r_cut = (
            None
            if self.config.nonbonded_method == app.NoCutoff
            else self.config.cutoff_angstrom
        )
        self.mm = convert_openmm_system(
            omm_system,
            omm_topology,
            omm_positions,
            omm_box_vectors,
            r_cut=r_cut,
            dr_threshold=self.config.dr_threshold_angstrom,
            format=self.config.neighbor_format,
        )

        # Positions/box vectors follow JAX-MD conventions: Angstrom.
        self._default_positions: JaxArray = (
            self.mm.positions if positions is None else jnp.asarray(positions)
        )
        self.box_vectors = self.mm.box_vectors  # may be None for non-periodic systems

        # --- Build energy function ---
        coulomb = _make_coulomb_handler(
            self.config.nonbonded_method,
            self.mm.nb_options.r_cut,
            self.mm.recip_alpha,
            self.mm.recip_grid,
        )

        energy_fn, neighbor_fn, disp_fn, shift_fn = amber_energy(
            self.mm.params,
            self.mm.topology,
            self.mm.box_vectors,
            coulomb_options=coulomb,
            nb_options=self.mm.nb_options,
        )

        self.energy_terms = energy_fn
        self.neighbor_fn = neighbor_fn
        self.disp_fn = disp_fn
        self.shift_fn = shift_fn

        # Keep a "scalar etotal" wrapper for grads/JIT, with virtual-site projection if enabled.
        self._etotal_fn = jit(self._etotal_impl)
        # Pre-compile batched energy paths to avoid constructing `jit(...)` wrappers
        # on every call. This matters a lot in Python-level sampling loops.
        self._etotal_batched_vmap_unbatchednbr = jit(
            vmap(self._etotal_impl, in_axes=(0, None, None, None))
        )
        self._etotal_batched_vmap_batchednbr = jit(
            vmap(self._etotal_impl, in_axes=(0, 0, None, None))
        )
        self._etotal_batched_map_unbatchednbr = jit(
            lambda xb, nbr, params, box: lax.map(
                lambda pos: self._etotal_impl(pos, nbr, params, box),
                xb,
            )
        )
        self._etotal_batched_map_batchednbr = jit(
            lambda xb, nbrb, params, box: lax.map(
                lambda t: self._etotal_impl(t[0], t[1], params, box),
                (xb, nbrb),
            )
        )

        def _edict_impl(pos: JaxArray, nbr: NeighborList, params: Any, box: Any):
            return self.energy_dict(pos, nbr, params=params, box=box)

        self._edict_batched_vmap_unbatchednbr = jit(
            vmap(_edict_impl, in_axes=(0, None, None, None))
        )
        self._edict_batched_vmap_batchednbr = jit(
            vmap(_edict_impl, in_axes=(0, 0, None, None))
        )
        self._edict_batched_map_unbatchednbr = jit(
            lambda xb, nbr, params, box: lax.map(
                lambda pos: self.energy_dict(pos, nbr, params=params, box=box),
                xb,
            )
        )
        self._edict_batched_map_batchednbr = jit(
            lambda xb, nbrb, params, box: lax.map(
                lambda t: self.energy_dict(t[0], t[1], params=params, box=box),
                (xb, nbrb),
            )
        )

        # Optional convenience: cache a neighbor list for non-periodic NoCutoff
        # single-configuration calls (cannot be used safely for arbitrary positions).
        self._default_nbr_list: Optional[NeighborList] = None
        self.n_times_reallocated = 0
        if self.config.nonbonded_method == app.NoCutoff:
            try:
                self._default_nbr_list = self.allocate_neighbor_list(
                    self._default_positions
                )
            except Exception as e:
                logger.debug("Failed to allocate default neighbor list: %s", e)

    # -----------------------------
    # Public: positions / neighbors
    # -----------------------------

    def get_default_positions(self) -> JaxArray:
        """Return the default coordinates (Angstrom) loaded from file or user-provided."""
        return self._default_positions

    def allocate_neighbor_list(
        self,
        positions: Optional[JaxArray],
        *,
        box: Optional[JaxArray] = None,
        extra_capacity: int = 0,
    ) -> NeighborList:
        """Allocate a neighbor list (not JIT-compatible).

        For periodic_general / fractional-coordinate systems, pass `box`.
        """
        if positions is None:
            positions = self._default_positions
        if self.neighbor_fn is None:
            raise RuntimeError("Neighbor function is not available (unexpected).")

        alloc = getattr(self.neighbor_fn, "allocate", None)
        if not callable(alloc):
            raise RuntimeError("neighbor_fn.allocate is not callable.")

        if box is None and getattr(self.mm.nb_options, "use_periodic_general", False):
            box = self.box_vectors
        if positions.ndim == 3:
            # JAX neighbor-list pytrees include function objects in node data,
            # which can differ across allocations and break tree_map stacking.
            # Use a single template neighbor list and broadcast it for batched use.
            pos0 = positions[0]
            if box is None:
                return alloc(pos0, extra_capacity=extra_capacity)
            return alloc(pos0, box=box, extra_capacity=extra_capacity)
        if box is None:
            return alloc(positions, extra_capacity=extra_capacity)
        return alloc(positions, box=box, extra_capacity=extra_capacity)

    def update_neighbor_list(
        self,
        positions: JaxArray,
        nbr_list: Optional[NeighborList] = None,
        *,
        nbList: Optional[NeighborList] = None,
        box: Optional[JaxArray] = None,
        allow_reallocate: bool = True,
        reallocate_growth: int = 2,
        extra_capacity: int = 0,
        n_times_reallocated: int = 0,
    ) -> NeighborList:
        """Update a neighbor list; optionally reallocate if buffer overflows.

        Returns:
          new_nbr_list
        """
        if nbr_list is not None and nbList is not None and nbr_list is not nbList:
            raise ValueError("Provide only one of nbr_list or nbList (alias).")
        if nbr_list is None:
            nbr_list = nbList
        if nbr_list is None:
            if self._default_nbr_list is None:
                raise ValueError(
                    "nbr_list is required (allocate it via allocate_neighbor_list)."
                )
            nbr_list = self._default_nbr_list

        if box is None and getattr(self.mm.nb_options, "use_periodic_general", False):
            box = self.box_vectors

        def _update_one(pos, nbr):
            return nbr.update(pos) if box is None else nbr.update(pos, box=box)

        def _is_batched_nbr(nbr, batch_size: int) -> bool:
            leaves = tree_util.tree_leaves(nbr)
            array_leaves = [
                leaf
                for leaf in leaves
                if hasattr(leaf, "shape") and hasattr(leaf, "ndim")
            ]
            if not array_leaves:
                return False
            return all(
                leaf.ndim > 0 and leaf.shape[0] == batch_size for leaf in array_leaves
            )

        is_batched = positions.ndim == 3

        if not is_batched:
            nbr_new = _update_one(positions, nbr_list)
            overflow = bool(jnp.any(nbr_new.did_buffer_overflow))
            if overflow and allow_reallocate:
                # Reallocate with extra capacity and update again.
                alloc = getattr(self.neighbor_fn, "allocate", None)
                if not callable(alloc):
                    raise RuntimeError(
                        "neighbor_fn.allocate is not callable for reallocation."
                    )
                cap = max(int(extra_capacity), int(reallocate_growth))
                nbr_new = self.allocate_neighbor_list(
                    positions, box=box, extra_capacity=cap
                )
                nbr_new = _update_one(positions, nbr_new)
                overflow = bool(jnp.any(nbr_new.did_buffer_overflow))
                n_times_reallocated += 1
            self.n_times_reallocated = n_times_reallocated
            return nbr_new

        # Batched path: vmap update.
        if _is_batched_nbr(nbr_list, positions.shape[0]):
            update_vmapped = jit(vmap(_update_one, in_axes=(0, 0)))
            nbr_new = update_vmapped(positions, nbr_list)
        else:
            update_vmapped = jit(vmap(_update_one, in_axes=(0, None)))
            nbr_new = update_vmapped(positions, nbr_list)
        overflow = bool(jnp.any(nbr_new.did_buffer_overflow))
        if overflow and allow_reallocate:
            # Allocate from first sample as a template; update all.
            alloc = getattr(self.neighbor_fn, "allocate", None)
            if not callable(alloc):
                raise RuntimeError(
                    "neighbor_fn.allocate is not callable for reallocation (batched)."
                )
            cap = max(int(extra_capacity), int(reallocate_growth))
            template = self.allocate_neighbor_list(
                positions[0], box=box, extra_capacity=cap
            )

            def _update_with_template(pos):
                return _update_one(pos, template)

            nbr_new = jit(vmap(_update_with_template, in_axes=(0,)))(positions)
            overflow = bool(jnp.any(nbr_new.did_buffer_overflow))
            n_times_reallocated += 1
        self.n_times_reallocated = n_times_reallocated
        return nbr_new

    # -----------------------------
    # Energy / forces
    # -----------------------------

    def energy_dict(
        self,
        positions: JaxArray,
        nbr_list: Optional[NeighborList],
        *,
        params: Optional[Any] = None,
        box: Optional[JaxArray] = None,
    ) -> dict[str, jnp.ndarray]:
        """Return the full energy breakdown dict produced by jax-md amber_energy."""
        if nbr_list is None:
            if self._default_nbr_list is None:
                raise ValueError(
                    "nbr_list is required (allocate it via allocate_neighbor_list)."
                )
            nbr_list = self._default_nbr_list
        if params is None:
            params = self.mm.params
        if box is None and getattr(self.mm.nb_options, "use_periodic_general", False):
            box = self.box_vectors

        pos = positions
        if (
            self.config.enable_virtual_sites
            and hasattr(self.mm, "virtual_sites")
            and self.mm.virtual_sites is not None
        ):
            # Project virtual site positions so AD sees the correct dependency.
            pos = virtual_site_apply_positions(
                pos,
                self.mm.virtual_sites,
                displacement_fn=self.disp_fn,
                shift_fn=self.shift_fn,
                box=box,
                use_periodic_general=getattr(
                    self.mm.nb_options, "use_periodic_general", False
                ),
            )

        if box is None:
            return self.energy_terms(pos, nbr_list, params=params)  # type: ignore[misc]
        return self.energy_terms(pos, nbr_list, params=params, box=box)  # type: ignore[misc]

    def energy(
        self,
        positions: JaxArray,
        nbr_list: Optional[NeighborList],
        *,
        params: Optional[Any] = None,
        box: Optional[JaxArray] = None,
        key: str = "etotal",
    ) -> jnp.ndarray:
        """Return a scalar energy (default: total potential energy)."""
        ed = self.energy_dict(positions, nbr_list, params=params, box=box)
        if key in ed:
            return jnp.asarray(ed[key])
        # Backwards compatibility with older dict keys.
        if "total" in ed:
            return jnp.asarray(ed["total"])
        raise KeyError(
            f"Energy dict does not contain '{key}' (available: {sorted(ed.keys())})."
        )

    def _etotal_impl(
        self,
        positions: JaxArray,
        nbr_list: Optional[NeighborList],
        params: Optional[Any] = None,
        box: Optional[JaxArray] = None,
    ) -> jnp.ndarray:
        return self.energy(positions, nbr_list, params=params, box=box, key="etotal")

    def forces(
        self,
        positions: JaxArray,
        nbr_list: Optional[NeighborList],
        *,
        params: Optional[Any] = None,
        box: Optional[JaxArray] = None,
    ) -> JaxArray:
        """Return forces (kcal/(mol*Angstrom)) via -grad(etotal)."""
        grad_fn: Callable[..., JaxArray] = jax.grad(self._etotal_impl)
        return -grad_fn(positions, nbr_list, params, box)

    # -----------------------------
    # Potential interface
    # -----------------------------

    def __call__(
        self,
        x: JaxArray,
        beta: float = 1.0,
        *,
        nbLists: Optional[NeighborList] = None,
        nbList: Optional[NeighborList] = None,
        nbr_list: Optional[NeighborList] = None,
        params: Optional[Any] = None,
        box: Optional[JaxArray] = None,
        return_dict: bool = False,
    ) -> Union[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute potential energy for a single configuration or a batch.

        Args:
          x: Positions in Angstrom, shape (N,3) or (B,N,3).
          beta: Unused (kept for compatibility with the surrounding codebase).
          nbr_list: Neighbor list (single) or pytree with leading batch dimension.
          params: Optional override for forcefield parameters.
          box: Optional box vectors (required for periodic_general / NPT-style workflows).
          return_dict: If True, return the full energy dict instead of a scalar.

        Notes:
          - For periodic systems, you should allocate/update nbr_list externally.
          - For batched calls, nbr_list must be batched in a way that matches x.
        """
        del beta  # not used

        is_batched = x.ndim == 3

        if nbLists is not None and nbList is not None and nbLists is not nbList:
            raise ValueError("Provide only one of nbLists or nbList (alias).")
        if nbLists is None:
            nbLists = nbList
        if nbLists is None:
            nbLists = nbr_list

        if nbLists is None:
            if (not is_batched) and (self._default_nbr_list is not None):
                nbLists = self._default_nbr_list
            elif (
                is_batched
                and self._default_nbr_list is not None
                and self.config.nonbonded_method == app.NoCutoff
            ):
                nbLists = self._default_nbr_list
            else:
                raise ValueError(
                    "nbr_list is required. Allocate it outside jit via "
                    "AmberPotential.allocate_neighbor_list(...) and update each step."
                )

        if not is_batched:
            if return_dict:
                return self.energy_dict(x, nbLists, params=params, box=box)
            return self._etotal_fn(x, nbLists, params, box)

        # Batched path.
        # NOTE: JAX-MD amber_energy can produce NaNs under vmap on some platforms
        # even when single-frame energies are finite. `lax.map` is slower but stable.
        def _is_batched_nbr(nbr, batch_size: int) -> bool:
            leaves = tree_util.tree_leaves(nbr)
            array_leaves = [
                leaf
                for leaf in leaves
                if hasattr(leaf, "shape") and hasattr(leaf, "ndim")
            ]
            if not array_leaves:
                return False
            return all(
                leaf.ndim > 0 and leaf.shape[0] == batch_size for leaf in array_leaves
            )

        is_batched_nbr = _is_batched_nbr(nbLists, x.shape[0])
        if return_dict:
            if self.use_vmap:
                if is_batched_nbr:
                    return self._edict_batched_vmap_batchednbr(x, nbLists, params, box)
                return self._edict_batched_vmap_unbatchednbr(x, nbLists, params, box)

            if is_batched_nbr:
                return self._edict_batched_map_batchednbr(x, nbLists, params, box)
            return self._edict_batched_map_unbatchednbr(x, nbLists, params, box)

        # Scalar energies over batch.
        # If nbr_list is not batched, broadcast it.
        if self.use_vmap:
            if is_batched_nbr:
                return self._etotal_batched_vmap_batchednbr(x, nbLists, params, box)
            return self._etotal_batched_vmap_unbatchednbr(x, nbLists, params, box)

        if is_batched_nbr:
            return self._etotal_batched_map_batchednbr(x, nbLists, params, box)
        return self._etotal_batched_map_unbatchednbr(x, nbLists, params, box)
