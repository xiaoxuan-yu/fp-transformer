"""Differentiable AutoDock Vina 1.1 scoring potential.

This module implements the five pairwise terms from the original Vina paper.
Atom typing is deliberately kept outside the numerical kernel: callers provide
van der Waals radii and boolean chemical-feature masks prepared by their own
PDBQT/RDKit/OpenMM pipeline.
"""

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from ..type import JaxArray
from .base import Potential
from .vina_pallas import (
    VINA_INTRA_PAIR_BLOCK_SIZE,
    vina_interaction_energy,
    vina_intra_energy,
)


@dataclass(frozen=True)
class VinaWeights:
    """AutoDock Vina 1.1 scoring-function weights."""

    gauss1: float = -0.035579
    gauss2: float = -0.005156
    repulsion: float = 0.840245
    hydrophobic: float = -0.035069
    hydrogen_bond: float = -0.587439
    rotatable_bond: float = 0.05846


def _build_torsion_coordinates(
    torsion: JaxArray,
    reference_coordinate: JaxArray,
    rotatable_bonds: JaxArray,
    rotation_masks: JaxArray,
) -> JaxArray:
    """Rodrigues reconstruction used by the primal and reference autodiff."""

    coordinate = jnp.broadcast_to(
        reference_coordinate,
        (torsion.shape[0],) + reference_coordinate.shape,
    )
    for index in range(rotatable_bonds.shape[0]):
        fixed_atom, moving_atom = rotatable_bonds[index]
        origin = coordinate[:, fixed_atom, :]
        axis = coordinate[:, moving_atom, :] - origin
        axis_norm = jnp.linalg.norm(axis, axis=-1, keepdims=True)
        axis = axis / jnp.maximum(
            axis_norm, jnp.finfo(coordinate.dtype).tiny
        )

        relative = coordinate - origin[:, None, :]
        angle = torsion[:, index, None, None]
        cosine = jnp.cos(angle)
        sine = jnp.sin(angle)
        axis_expanded = axis[:, None, :]
        rotated = (
            relative * cosine
            + jnp.cross(axis_expanded, relative, axis=-1) * sine
            + axis_expanded
            * jnp.sum(axis_expanded * relative, axis=-1, keepdims=True)
            * (1.0 - cosine)
            + origin[:, None, :]
        )
        coordinate = jnp.where(
            rotation_masks[index][None, :, None], rotated, coordinate
        )
    return coordinate


@jax.custom_vjp
def _build_torsion_coordinates_with_vjp(
    torsion: JaxArray,
    reference_coordinate: JaxArray,
    rotatable_bonds: JaxArray,
    rotation_masks: JaxArray,
) -> JaxArray:
    """Reconstruct coordinates with an analytic torsion-only pullback."""

    return _build_torsion_coordinates(
        torsion,
        reference_coordinate,
        rotatable_bonds,
        rotation_masks,
    )


def _build_torsion_coordinates_fwd(
    torsion, reference_coordinate, rotatable_bonds, rotation_masks
):
    coordinate = _build_torsion_coordinates(
        torsion,
        reference_coordinate,
        rotatable_bonds,
        rotation_masks,
    )
    return coordinate, (coordinate, rotatable_bonds, rotation_masks)


def _build_torsion_coordinates_bwd(residual, coordinate_cotangent):
    coordinate, rotatable_bonds, rotation_masks = residual
    origin = coordinate[:, rotatable_bonds[:, 0], :]
    axis = coordinate[:, rotatable_bonds[:, 1], :] - origin
    axis = axis / jnp.maximum(
        jnp.linalg.norm(axis, axis=-1, keepdims=True),
        jnp.finfo(coordinate.dtype).tiny,
    )

    relative = coordinate[:, None, :, :] - origin[:, :, None, :]
    torque = jnp.cross(
        relative,
        coordinate_cotangent[:, None, :, :],
        axis=-1,
    )
    torque = jnp.sum(
        jnp.where(rotation_masks[None, :, :, None], torque, 0.0),
        axis=2,
    )
    torsion_cotangent = jnp.sum(axis * torque, axis=-1)
    return torsion_cotangent, None, None, None


_build_torsion_coordinates_with_vjp.defvjp(
    _build_torsion_coordinates_fwd,
    _build_torsion_coordinates_bwd,
)


def build_ligand_intra_pairs(
    num_atoms: int, bonds: JaxArray, rotation_masks: JaxArray
) -> JaxArray:
    """Build torsion-dependent intraligand nonbonded pairs from a bond graph.

    Pairs connected by one, two, or three covalent bonds (1-2, 1-3, and 1-4)
    are excluded, matching ``model::initialize_pairs`` in AutoDock Vina.  A
    1-5 or more distant pair is retained only when at least one torsion places
    its atoms on opposite sides of the rotation mask. The returned rows
    satisfy ``i < j`` and can be passed directly as ``ligand_intra_pairs``.

    This helper runs once during setup, outside JIT. Callers with authoritative
    PDBQT interaction pairs may pass those pairs to :class:`VinaScore` instead.
    """

    if isinstance(num_atoms, bool) or not isinstance(num_atoms, int):
        raise TypeError("num_atoms must be a positive integer")
    if num_atoms <= 0:
        raise ValueError("num_atoms must be positive")
    bonds = np.asarray(bonds, dtype=np.int32)
    if bonds.ndim != 2 or bonds.shape[1] != 2:
        raise ValueError(f"bonds must have shape (num_bonds, 2), got {bonds.shape}")
    if np.any(bonds < 0) or np.any(bonds >= num_atoms):
        raise ValueError("bonds contains an out-of-range atom index")
    if np.any(bonds[:, 0] == bonds[:, 1]):
        raise ValueError("a bond must contain two different atoms")

    rotation_masks = np.asarray(rotation_masks, dtype=bool)
    if rotation_masks.ndim != 2 or rotation_masks.shape[1] != num_atoms:
        raise ValueError(
            "rotation_masks must have shape (num_torsions, num_atoms), "
            f"got {rotation_masks.shape}"
        )

    adjacency = [set() for _ in range(num_atoms)]
    for atom_i, atom_j in bonds:
        adjacency[int(atom_i)].add(int(atom_j))
        adjacency[int(atom_j)].add(int(atom_i))

    graph_distance = np.full((num_atoms, num_atoms), np.inf)
    for start in range(num_atoms):
        graph_distance[start, start] = 0
        frontier = [start]
        while frontier:
            atom = frontier.pop(0)
            next_distance = graph_distance[start, atom] + 1
            for neighbor in adjacency[atom]:
                if graph_distance[start, neighbor] == np.inf:
                    graph_distance[start, neighbor] = next_distance
                    frontier.append(neighbor)

    pairs = []
    for atom_i in range(num_atoms):
        for atom_j in range(atom_i + 1, num_atoms):
            torsion_dependent = np.any(
                rotation_masks[:, atom_i] != rotation_masks[:, atom_j]
            )
            if graph_distance[atom_i, atom_j] > 3 and torsion_dependent:
                pairs.append((atom_i, atom_j))
    return jnp.asarray(pairs, dtype=jnp.int32).reshape((-1, 2))


def vina_pairwise_terms(surface_distance: JaxArray) -> Mapping[str, JaxArray]:
    """Return the five unweighted Vina terms for a surface distance in Angstrom.

    ``surface_distance`` is the interatomic distance minus the sum of the two
    atoms' van der Waals radii. Hydrophobic and hydrogen-bond terms still need
    to be masked according to the atom types by the caller.
    """

    d = jnp.asarray(surface_distance)
    gauss1 = jnp.exp(-jnp.square(d / 0.5))
    gauss2 = jnp.exp(-jnp.square((d - 3.0) / 2.0))
    repulsion = jnp.square(jnp.minimum(d, 0.0))
    hydrophobic = jnp.clip(1.5 - d, 0.0, 1.0)
    hydrogen_bond = jnp.clip(-d / 0.7, 0.0, 1.0)
    return {
        "gauss1": gauss1,
        "gauss2": gauss2,
        "repulsion": repulsion,
        "hydrophobic": hydrophobic,
        "hydrogen_bond": hydrogen_bond,
    }


class VinaScore(Potential):
    """Torsion-space JAX implementation of the AutoDock Vina 1.1 score.

    Parameters are specified for heavy atoms only, matching Vina's united-atom
    model. Coordinates and radii use Angstrom; the returned score is in
    kcal/mol. A hydrogen bond is enabled for a receptor-donor/ligand-acceptor
    or receptor-acceptor/ligand-donor pair. The receptor and the ligand's rigid
    scaffold stay fixed: the only input degrees of freedom are rotations around
    the configured ligand bonds.

    Args:
        receptor_coordinate: Fixed receptor coordinates, shape ``(R, 3)``.
        receptor_vdw_radii: Receptor Vina radii, shape ``(R,)``.
        ligand_vdw_radii: Ligand Vina radii, shape ``(L,)``.
        ligand_reference_coordinate: Ligand coordinates at zero torsion delta,
            shape ``(L, 3)``.
        rotatable_bonds: Directed rotation axes ``(fixed, moving)``, shape
            ``(T, 2)``. Bonds must be ordered from the root towards leaves.
        rotation_masks: Atoms on the moving side of each bond, shape ``(T, L)``.
        ligand_intra_pairs: Unordered ligand atom pairs included in the
            intramolecular score, shape ``(P, 2)``. Covalently excluded pairs
            must not be present, and every row must satisfy ``i < j``.
        receptor_hydrophobic: Hydrophobic receptor atoms, shape ``(R,)``.
        ligand_hydrophobic: Hydrophobic ligand atoms, shape ``(L,)``.
        receptor_hbond_donor: Receptor hydrogen-bond donors, shape ``(R,)``.
        receptor_hbond_acceptor: Receptor acceptors, shape ``(R,)``.
        ligand_hbond_donor: Ligand donors, shape ``(L,)``.
        ligand_hbond_acceptor: Ligand acceptors, shape ``(L,)``.
        cutoff: Center-to-center pair cutoff in Angstrom (Vina uses 8 Angstrom).
        weights: Scoring weights; defaults reproduce AutoDock Vina 1.1.
        interaction_backend: ``"jax"`` uses the portable reference path;
            ``"pallas"`` uses the fused Pallas/Triton score-gradient kernel.
    """

    def __init__(
        self,
        receptor_coordinate: JaxArray,
        receptor_vdw_radii: JaxArray,
        ligand_vdw_radii: JaxArray,
        *,
        ligand_reference_coordinate: JaxArray,
        rotatable_bonds: JaxArray,
        rotation_masks: JaxArray,
        ligand_intra_pairs: JaxArray,
        receptor_hydrophobic: JaxArray | None = None,
        ligand_hydrophobic: JaxArray | None = None,
        receptor_hbond_donor: JaxArray | None = None,
        receptor_hbond_acceptor: JaxArray | None = None,
        ligand_hbond_donor: JaxArray | None = None,
        ligand_hbond_acceptor: JaxArray | None = None,
        cutoff: float = 8.0,
        weights: VinaWeights | None = None,
        interaction_backend: str = "jax",
    ):
        super().__init__()
        receptor_coordinate = jnp.asarray(receptor_coordinate, dtype=float)
        if receptor_coordinate.ndim != 2 or receptor_coordinate.shape[1] != 3:
            raise ValueError(
                "receptor_coordinate must have shape (num_receptor_atoms, 3), "
                f"got {receptor_coordinate.shape}"
            )

        self.receptor_coordinate = receptor_coordinate
        self.receptor_vdw_radii = self._vector(
            "receptor_vdw_radii", receptor_vdw_radii, receptor_coordinate.shape[0]
        )
        ligand_vdw_radii = jnp.asarray(ligand_vdw_radii, dtype=float)
        if ligand_vdw_radii.ndim != 1 or ligand_vdw_radii.shape[0] == 0:
            raise ValueError("ligand_vdw_radii must be a non-empty 1D array")
        self.ligand_vdw_radii = self._vector(
            "ligand_vdw_radii", ligand_vdw_radii, ligand_vdw_radii.shape[0]
        )

        num_receptor_atoms = receptor_coordinate.shape[0]
        num_ligand_atoms = ligand_vdw_radii.shape[0]
        ligand_reference_coordinate = jnp.asarray(
            ligand_reference_coordinate, dtype=receptor_coordinate.dtype
        )
        if ligand_reference_coordinate.shape != (num_ligand_atoms, 3):
            raise ValueError(
                "ligand_reference_coordinate must have shape "
                f"({num_ligand_atoms}, 3), got {ligand_reference_coordinate.shape}"
            )
        self.ligand_reference_coordinate = ligand_reference_coordinate

        rotatable_bonds = jnp.asarray(rotatable_bonds, dtype=jnp.int32)
        if rotatable_bonds.ndim != 2 or rotatable_bonds.shape[1] != 2:
            raise ValueError(
                "rotatable_bonds must have shape (num_torsions, 2), "
                f"got {rotatable_bonds.shape}"
            )
        if bool(jnp.any(rotatable_bonds < 0)) or bool(
            jnp.any(rotatable_bonds >= num_ligand_atoms)
        ):
            raise ValueError("rotatable_bonds contains an out-of-range atom index")
        if bool(jnp.any(rotatable_bonds[:, 0] == rotatable_bonds[:, 1])):
            raise ValueError("a rotatable bond must contain two different atoms")
        bond_vectors = (
            ligand_reference_coordinate[rotatable_bonds[:, 1]]
            - ligand_reference_coordinate[rotatable_bonds[:, 0]]
        )
        if bool(jnp.any(jnp.sum(jnp.square(bond_vectors), axis=-1) == 0)):
            raise ValueError("a rotatable bond cannot have zero length")
        self.rotatable_bonds = rotatable_bonds
        self.num_torsions = rotatable_bonds.shape[0]

        rotation_masks = jnp.asarray(rotation_masks)
        if rotation_masks.shape != (self.num_torsions, num_ligand_atoms):
            raise ValueError(
                "rotation_masks must have shape "
                f"({self.num_torsions}, {num_ligand_atoms}), got {rotation_masks.shape}"
            )
        self.rotation_masks = rotation_masks.astype(bool)
        if bool(jnp.any(~jnp.any(self.rotation_masks, axis=1))):
            raise ValueError("each rotation mask must select at least one atom")

        ligand_intra_pairs = jnp.asarray(ligand_intra_pairs, dtype=jnp.int32)
        if ligand_intra_pairs.ndim != 2 or ligand_intra_pairs.shape[1] != 2:
            raise ValueError(
                "ligand_intra_pairs must have shape (num_pairs, 2), "
                f"got {ligand_intra_pairs.shape}"
            )
        if bool(jnp.any(ligand_intra_pairs < 0)) or bool(
            jnp.any(ligand_intra_pairs >= num_ligand_atoms)
        ):
            raise ValueError("ligand_intra_pairs contains an out-of-range index")
        if bool(jnp.any(ligand_intra_pairs[:, 0] >= ligand_intra_pairs[:, 1])):
            raise ValueError("each ligand_intra_pairs row must satisfy i < j")
        pair_keys = (
            ligand_intra_pairs[:, 0] * num_ligand_atoms
            + ligand_intra_pairs[:, 1]
        )
        if pair_keys.shape[0] != jnp.unique(pair_keys).shape[0]:
            raise ValueError("ligand_intra_pairs must not contain duplicates")
        self.ligand_intra_pairs = ligand_intra_pairs
        self.num_ligand_intra_pairs = ligand_intra_pairs.shape[0]
        self.receptor_hydrophobic = self._mask(
            "receptor_hydrophobic", receptor_hydrophobic, num_receptor_atoms
        )
        self.ligand_hydrophobic = self._mask(
            "ligand_hydrophobic", ligand_hydrophobic, num_ligand_atoms
        )
        self.receptor_hbond_donor = self._mask(
            "receptor_hbond_donor", receptor_hbond_donor, num_receptor_atoms
        )
        self.receptor_hbond_acceptor = self._mask(
            "receptor_hbond_acceptor", receptor_hbond_acceptor, num_receptor_atoms
        )
        self.ligand_hbond_donor = self._mask(
            "ligand_hbond_donor", ligand_hbond_donor, num_ligand_atoms
        )
        self.ligand_hbond_acceptor = self._mask(
            "ligand_hbond_acceptor", ligand_hbond_acceptor, num_ligand_atoms
        )

        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        self.cutoff = float(cutoff)
        self.weights = weights if weights is not None else VinaWeights()
        if interaction_backend not in {"jax", "pallas"}:
            raise ValueError("interaction_backend must be 'jax' or 'pallas'")
        self.interaction_backend = interaction_backend

        self._hydrophobic_pairs = (
            self.receptor_hydrophobic[:, None] & self.ligand_hydrophobic[None, :]
        )
        self._hbond_pairs = (
            self.receptor_hbond_donor[:, None]
            & self.ligand_hbond_acceptor[None, :]
        ) | (
            self.receptor_hbond_acceptor[:, None]
            & self.ligand_hbond_donor[None, :]
        )

        if self.interaction_backend == "pallas":
            padded_receptor_atoms = 1 << max(
                0, (num_receptor_atoms - 1).bit_length()
            )
            padding = padded_receptor_atoms - num_receptor_atoms
            self._pallas_receptor_coordinate = jnp.pad(
                self.receptor_coordinate, ((0, padding), (0, 0))
            )
            self._pallas_receptor_vdw_radii = jnp.pad(
                self.receptor_vdw_radii, (0, padding)
            )
            self._pallas_receptor_hydrophobic = jnp.pad(
                self.receptor_hydrophobic, (0, padding)
            )
            self._pallas_receptor_hbond_donor = jnp.pad(
                self.receptor_hbond_donor, (0, padding)
            )
            self._pallas_receptor_hbond_acceptor = jnp.pad(
                self.receptor_hbond_acceptor, (0, padding)
            )
            self._pallas_receptor_valid = jnp.arange(
                padded_receptor_atoms
            ) < num_receptor_atoms
            self._pallas_weights = jnp.asarray(
                [
                    self.weights.gauss1,
                    self.weights.gauss2,
                    self.weights.repulsion,
                    self.weights.hydrophobic,
                    self.weights.hydrogen_bond,
                ],
                dtype=self.receptor_coordinate.dtype,
            )
            self._pallas_cutoff = jnp.asarray(
                [self.cutoff], dtype=self.receptor_coordinate.dtype
            )
            intra_block_size = VINA_INTRA_PAIR_BLOCK_SIZE
            padded_intra_pairs = max(
                intra_block_size,
                (
                    (self.num_ligand_intra_pairs + intra_block_size - 1)
                    // intra_block_size
                )
                * intra_block_size,
            )
            intra_padding = padded_intra_pairs - self.num_ligand_intra_pairs
            intra_pair_i = self.ligand_intra_pairs[:, 0]
            intra_pair_j = self.ligand_intra_pairs[:, 1]
            intra_hydrophobic = (
                self.ligand_hydrophobic[intra_pair_i]
                & self.ligand_hydrophobic[intra_pair_j]
            )
            intra_hbond = (
                self.ligand_hbond_donor[intra_pair_i]
                & self.ligand_hbond_acceptor[intra_pair_j]
            ) | (
                self.ligand_hbond_acceptor[intra_pair_i]
                & self.ligand_hbond_donor[intra_pair_j]
            )
            self._pallas_intra_pair_i = jnp.pad(
                intra_pair_i, (0, intra_padding)
            )
            self._pallas_intra_pair_j = jnp.pad(
                intra_pair_j, (0, intra_padding)
            )
            self._pallas_intra_pair_radius = jnp.pad(
                self.ligand_vdw_radii[intra_pair_i]
                + self.ligand_vdw_radii[intra_pair_j],
                (0, intra_padding),
            )
            self._pallas_intra_hydrophobic = jnp.pad(
                intra_hydrophobic, (0, intra_padding)
            )
            self._pallas_intra_hbond = jnp.pad(
                intra_hbond, (0, intra_padding)
            )
            self._pallas_intra_valid = jnp.arange(padded_intra_pairs) < (
                self.num_ligand_intra_pairs
            )
            self._pallas_interpret = jax.default_backend() != "gpu"

    @staticmethod
    def _vector(name: str, value: JaxArray, size: int) -> JaxArray:
        value = jnp.asarray(value, dtype=float)
        if value.ndim != 1 or value.shape[0] != size:
            raise ValueError(f"{name} must have shape ({size},), got {value.shape}")
        if bool(jnp.any(value <= 0)):
            raise ValueError(f"{name} must contain only positive radii")
        return value

    @staticmethod
    def _mask(name: str, value: JaxArray | None, size: int) -> JaxArray:
        if value is None:
            return jnp.zeros((size,), dtype=bool)
        value = jnp.asarray(value)
        if value.ndim != 1 or value.shape[0] != size:
            raise ValueError(f"{name} must have shape ({size},), got {value.shape}")
        return value.astype(bool)

    def coordinates(self, torsion: JaxArray) -> JaxArray:
        """Build ligand coordinates from torsion deltas in radians.

        ``torsion`` may have shape ``(B, T)`` or the sampler-native shape
        ``(B, 1, T)``. A zero vector reproduces ``ligand_reference_coordinate``.
        Rotations are composed in ``rotatable_bonds`` order.
        """

        torsion = jnp.asarray(torsion, dtype=self.receptor_coordinate.dtype)
        if torsion.ndim == 3:
            if torsion.shape[1] != 1:
                raise ValueError(
                    "3D torsion input must have shape (num_replicas, 1, num_torsions)"
                )
            torsion = torsion[:, 0, :]
        if torsion.ndim != 2 or torsion.shape[1] != self.num_torsions:
            raise ValueError(
                "torsion must have shape (num_replicas, num_torsions), "
                f"with num_torsions={self.num_torsions}; got {torsion.shape}"
            )

        return _build_torsion_coordinates_with_vjp(
            torsion,
            self.ligand_reference_coordinate,
            self.rotatable_bonds,
            self.rotation_masks,
        )

    def _coordinate_components(
        self, coordinate: JaxArray
    ) -> Mapping[str, JaxArray]:
        """Score receptor-ligand interactions for reconstructed coordinates."""

        coordinate = jnp.asarray(coordinate, dtype=self.receptor_coordinate.dtype)
        if coordinate.ndim != 3 or coordinate.shape[2] != 3:
            raise ValueError(
                "coordinate must have shape (num_replicas, num_ligand_atoms, 3), "
                f"got {coordinate.shape}"
            )
        if coordinate.shape[1] != self.ligand_vdw_radii.shape[0]:
            raise ValueError(
                f"expected {self.ligand_vdw_radii.shape[0]} ligand atoms, "
                f"got {coordinate.shape[1]}"
            )

        displacement = (
            self.receptor_coordinate[None, :, None, :]
            - coordinate[:, None, :, :]
        )
        squared_distance = jnp.sum(jnp.square(displacement), axis=-1)
        tiny = jnp.finfo(squared_distance.dtype).tiny
        distance = jnp.sqrt(jnp.maximum(squared_distance, tiny))
        surface_distance = distance - (
            self.receptor_vdw_radii[:, None]
            + self.ligand_vdw_radii[None, :]
        )
        pair_terms = vina_pairwise_terms(surface_distance)
        within_cutoff = distance < self.cutoff

        def sum_pairs(term: JaxArray, feature_mask: JaxArray | bool = True):
            masked = jnp.where(within_cutoff & feature_mask, term, 0.0)
            return jnp.sum(masked, axis=(1, 2), keepdims=False)[:, None]

        return {
            "gauss1": sum_pairs(pair_terms["gauss1"]),
            "gauss2": sum_pairs(pair_terms["gauss2"]),
            "repulsion": sum_pairs(pair_terms["repulsion"]),
            "hydrophobic": sum_pairs(
                pair_terms["hydrophobic"], self._hydrophobic_pairs
            ),
            "hydrogen_bond": sum_pairs(
                pair_terms["hydrogen_bond"], self._hbond_pairs
            ),
        }

    def _intramolecular_coordinate_components(
        self, coordinate: JaxArray
    ) -> Mapping[str, JaxArray]:
        """Score selected intraligand nonbonded pairs."""

        coordinate = jnp.asarray(coordinate, dtype=self.receptor_coordinate.dtype)
        pair_i = self.ligand_intra_pairs[:, 0]
        pair_j = self.ligand_intra_pairs[:, 1]
        displacement = coordinate[:, pair_i, :] - coordinate[:, pair_j, :]
        squared_distance = jnp.sum(jnp.square(displacement), axis=-1)
        tiny = jnp.finfo(squared_distance.dtype).tiny
        distance = jnp.sqrt(jnp.maximum(squared_distance, tiny))
        surface_distance = distance - (
            self.ligand_vdw_radii[pair_i] + self.ligand_vdw_radii[pair_j]
        )
        pair_terms = vina_pairwise_terms(surface_distance)
        within_cutoff = distance < self.cutoff
        hydrophobic_pairs = (
            self.ligand_hydrophobic[pair_i]
            & self.ligand_hydrophobic[pair_j]
        )
        hbond_pairs = (
            self.ligand_hbond_donor[pair_i]
            & self.ligand_hbond_acceptor[pair_j]
        ) | (
            self.ligand_hbond_acceptor[pair_i]
            & self.ligand_hbond_donor[pair_j]
        )

        def sum_pairs(term: JaxArray, feature_mask: JaxArray | bool = True):
            masked = jnp.where(within_cutoff & feature_mask, term, 0.0)
            return jnp.sum(masked, axis=1, keepdims=True)

        return {
            "gauss1": sum_pairs(pair_terms["gauss1"]),
            "gauss2": sum_pairs(pair_terms["gauss2"]),
            "repulsion": sum_pairs(pair_terms["repulsion"]),
            "hydrophobic": sum_pairs(
                pair_terms["hydrophobic"], hydrophobic_pairs
            ),
            "hydrogen_bond": sum_pairs(
                pair_terms["hydrogen_bond"], hbond_pairs
            ),
        }

    @staticmethod
    def _add_components(
        left: Mapping[str, JaxArray], right: Mapping[str, JaxArray]
    ) -> Mapping[str, JaxArray]:
        return {name: left[name] + right[name] for name in left}

    def intermolecular_components(
        self, torsion: JaxArray
    ) -> Mapping[str, JaxArray]:
        """Return unweighted receptor-ligand Vina term sums."""

        return self._coordinate_components(self.coordinates(torsion))

    def intramolecular_components(
        self, torsion: JaxArray
    ) -> Mapping[str, JaxArray]:
        """Return unweighted intraligand Vina term sums."""

        return self._intramolecular_coordinate_components(self.coordinates(torsion))

    def components(self, torsion: JaxArray) -> Mapping[str, JaxArray]:
        """Return combined unweighted inter- and intramolecular term sums."""

        coordinate = self.coordinates(torsion)
        inter = self._coordinate_components(coordinate)
        intra = self._intramolecular_coordinate_components(coordinate)
        return self._add_components(inter, intra)

    def _weighted_sum(self, terms: Mapping[str, JaxArray]) -> JaxArray:
        weights = self.weights
        return (
            weights.gauss1 * terms["gauss1"]
            + weights.gauss2 * terms["gauss2"]
            + weights.repulsion * terms["repulsion"]
            + weights.hydrophobic * terms["hydrophobic"]
            + weights.hydrogen_bond * terms["hydrogen_bond"]
        )

    def _intermolecular_energy(self, coordinate: JaxArray) -> JaxArray:
        if self.interaction_backend == "jax":
            return self._weighted_sum(self._coordinate_components(coordinate))
        return vina_interaction_energy(
            coordinate,
            self._pallas_receptor_coordinate,
            self._pallas_receptor_vdw_radii,
            self.ligand_vdw_radii,
            self._pallas_receptor_hydrophobic,
            self.ligand_hydrophobic,
            self._pallas_receptor_hbond_donor,
            self._pallas_receptor_hbond_acceptor,
            self.ligand_hbond_donor,
            self.ligand_hbond_acceptor,
            self._pallas_receptor_valid,
            self._pallas_weights,
            self._pallas_cutoff,
            self._pallas_interpret,
        )

    def _intramolecular_energy(self, coordinate: JaxArray) -> JaxArray:
        if self.num_ligand_intra_pairs == 0:
            return jnp.zeros((coordinate.shape[0], 1), dtype=coordinate.dtype)
        if self.interaction_backend == "jax" or self._pallas_interpret:
            return self._weighted_sum(
                self._intramolecular_coordinate_components(coordinate)
            )
        return vina_intra_energy(
            coordinate,
            self._pallas_intra_pair_i,
            self._pallas_intra_pair_j,
            self._pallas_intra_pair_radius,
            self._pallas_intra_hydrophobic,
            self._pallas_intra_hbond,
            self._pallas_intra_valid,
            self._pallas_weights,
            self._pallas_cutoff,
            self._pallas_interpret,
        )

    def energy_components(self, torsion: JaxArray) -> Mapping[str, JaxArray]:
        """Return weighted inter, intra, and total sampling energies."""

        coordinate = self.coordinates(torsion)
        return self._coordinate_energy_components(coordinate)

    def _coordinate_energy_components(
        self, coordinate: JaxArray
    ) -> Mapping[str, JaxArray]:
        """Return weighted energies for an already reconstructed ligand."""

        intermolecular = self._intermolecular_energy(coordinate)
        intramolecular = self._intramolecular_energy(coordinate)
        return {
            "intermolecular": intermolecular,
            "intramolecular": intramolecular,
            "total": intermolecular + intramolecular,
        }

    def total_and_binding_score(
        self, torsion: JaxArray, beta: float = 1.0
    ) -> tuple[JaxArray, JaxArray]:
        """Return sampling and binding scores from one shared forward pass."""

        del beta
        energies = self.energy_components(torsion)
        binding = energies["intermolecular"] / (
            1.0 + self.weights.rotatable_bond * self.num_torsions
        )
        return energies["total"], binding

    def value_and_grad(
        self, torsion: JaxArray, beta: float = 1.0
    ) -> tuple[tuple[JaxArray, JaxArray], JaxArray]:
        """Return total/binding scores and the total-energy torsion gradient.

        ``binding`` is auxiliary output: differentiating the summed total score
        therefore invokes the fused interaction custom VJP only once.
        """

        def objective(value):
            total, binding = self.total_and_binding_score(value, beta)
            return jnp.sum(total), (total, binding)

        (_, scores), gradient = jax.value_and_grad(
            objective, has_aux=True
        )(torsion)
        return scores, gradient

    def binding_score(self, torsion: JaxArray) -> JaxArray:
        """Return the torsion-normalized intermolecular Vina affinity score."""

        coordinate = self.coordinates(torsion)
        intermolecular = self._intermolecular_energy(coordinate)
        return intermolecular / (
            1.0 + self.weights.rotatable_bond * self.num_torsions
        )

    def __call__(self, torsion: JaxArray, beta: float = 1.0) -> JaxArray:
        """Calculate total Vina search energies from ligand torsion deltas.

        ``beta`` is accepted for compatibility with :class:`Potential`; Vina's
        empirical energy itself is temperature independent. The returned value
        is weighted intermolecular plus intramolecular energy. The constant
        torsion normalization used for reported affinity is available through
        :meth:`binding_score` and is intentionally not part of sampling energy.
        """

        return self.total_and_binding_score(torsion, beta)[0]
