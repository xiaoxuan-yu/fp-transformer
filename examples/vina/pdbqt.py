"""Build torsion-space Vina potentials from prepared PDBQT files.

The numerical potential remains torsion-only: the crystallographic ligand
translation and orientation and all receptor atoms are fixed.  PDBQT polar
hydrogens are used for donor typing and then omitted from the united-atom
score, matching Vina's treatment of hydrogen coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from fp_transformer.potential import VinaScore, build_ligand_intra_pairs


# Vina/X-Score radii from AutoDock Vina's atom_constants.h.  PDBQT stores
# AutoDock atom types; donor state is recovered from attached HD atoms.
VINA_RADIUS = {
    "A": 1.9,
    "C": 1.9,
    "N": 1.8,
    "NA": 1.8,
    "O": 1.7,
    "OA": 1.7,
    "S": 2.0,
    "SA": 2.0,
    "P": 2.1,
    "F": 1.5,
    "Cl": 1.8,
    "CL": 1.8,
    "Br": 2.0,
    "BR": 2.0,
    "I": 2.2,
    "Si": 2.2,
    "At": 2.3,
    "Mg": 1.2,
    "Mn": 1.2,
    "Zn": 1.2,
    "Ca": 1.2,
    "Fe": 1.2,
}

COVALENT_RADIUS = {
    "A": 0.77,
    "C": 0.77,
    "N": 0.75,
    "NA": 0.75,
    "O": 0.73,
    "OA": 0.73,
    "S": 1.02,
    "SA": 1.02,
    "P": 1.06,
    "F": 0.71,
    "Cl": 0.99,
    "CL": 0.99,
    "Br": 1.14,
    "BR": 1.14,
    "I": 1.33,
    "Si": 1.11,
    "At": 1.44,
    "Mg": 1.30,
    "Mn": 1.39,
    "Zn": 1.31,
    "Ca": 1.74,
    "Fe": 1.25,
}


@dataclass(frozen=True)
class PDBQTAtom:
    serial: int
    name: str
    coordinate: np.ndarray
    atom_type: str


@dataclass(frozen=True)
class LoadedSystem:
    potential: VinaScore
    receptor_heavy_atoms_total: int
    receptor_pocket_atoms: int
    ligand_heavy_atoms: int
    ligand_polar_hydrogens: int
    inferred_ligand_bonds: int
    pocket_radius: float


def _parse_atom(line: str) -> PDBQTAtom:
    fields = line.split()
    return PDBQTAtom(
        serial=int(fields[1]),
        name=fields[2],
        coordinate=np.asarray(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float32,
        ),
        atom_type=fields[-1],
    )


def parse_pdbqt_atoms(path: Path) -> list[PDBQTAtom]:
    atoms = []
    with path.open() as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")):
                atoms.append(_parse_atom(line))
    if not atoms:
        raise ValueError(f"no atoms found in {path}")
    return atoms


def parse_ligand_tree(
    path: Path,
) -> tuple[list[PDBQTAtom], list[tuple[int, int]], list[set[int]]]:
    """Parse atoms and moving-side membership from nested PDBQT BRANCHes."""

    atoms: list[PDBQTAtom] = []
    branches: list[tuple[int, int]] = []
    branch_members: list[set[int]] = []
    active: list[int] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith("BRANCH"):
                fields = line.split()
                branches.append((int(fields[1]), int(fields[2])))
                branch_members.append(set())
                active.append(len(branches) - 1)
            elif line.startswith("ENDBRANCH"):
                if not active:
                    raise ValueError(f"unbalanced ENDBRANCH in {path}")
                active.pop()
            elif line.startswith(("ATOM  ", "HETATM")):
                atom = _parse_atom(line)
                atoms.append(atom)
                for branch_index in active:
                    branch_members[branch_index].add(atom.serial)
    if active:
        raise ValueError(f"unclosed BRANCH in {path}")
    if not branches:
        raise ValueError(f"no rotatable branches found in {path}")
    return atoms, branches, branch_members


def _is_hydrogen(atom: PDBQTAtom) -> bool:
    return atom.atom_type in {"H", "HD"}


def _infer_bonds(atoms: list[PDBQTAtom], tolerance: float = 0.45) -> np.ndarray:
    coordinate = np.stack([atom.coordinate for atom in atoms])
    radius = np.asarray([COVALENT_RADIUS[atom.atom_type] for atom in atoms])
    distance = np.linalg.norm(
        coordinate[:, None, :] - coordinate[None, :, :], axis=-1
    )
    threshold = radius[:, None] + radius[None, :] + tolerance
    atom_i, atom_j = np.where(
        np.triu((distance > 0.4) & (distance <= threshold), k=1)
    )
    return np.stack([atom_i, atom_j], axis=1).astype(np.int32)


def _chemical_features(
    heavy_atoms: list[PDBQTAtom], polar_hydrogens: list[PDBQTAtom]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinate = np.stack([atom.coordinate for atom in heavy_atoms])
    bonds = _infer_bonds(heavy_atoms)
    hetero = np.asarray(
        [atom.atom_type not in {"A", "C"} for atom in heavy_atoms], dtype=bool
    )
    bonded_to_hetero = np.zeros(len(heavy_atoms), dtype=bool)
    for atom_i, atom_j in bonds:
        if hetero[atom_j]:
            bonded_to_hetero[atom_i] = True
        if hetero[atom_i]:
            bonded_to_hetero[atom_j] = True
    carbon = np.asarray(
        [atom.atom_type in {"A", "C"} for atom in heavy_atoms], dtype=bool
    )
    hydrophobic = carbon & ~bonded_to_hetero
    hydrophobic |= np.asarray(
        [atom.atom_type in {"F", "Cl", "CL", "Br", "BR", "I"}
         for atom in heavy_atoms],
        dtype=bool,
    )
    acceptor = np.asarray(
        [atom.atom_type in {"NA", "OA"} for atom in heavy_atoms], dtype=bool
    )
    donor = np.zeros(len(heavy_atoms), dtype=bool)
    if polar_hydrogens:
        hydrogen_coordinate = np.stack(
            [atom.coordinate for atom in polar_hydrogens]
        )
        donor = np.any(
            np.linalg.norm(
                coordinate[:, None, :] - hydrogen_coordinate[None, :, :],
                axis=-1,
            )
            < 1.35,
            axis=1,
        )
    return hydrophobic, donor, acceptor




def build_pdbqt_potential(
    *,
    receptor_paths: Path | tuple[Path, ...],
    ligand_path: Path,
    pocket_radius: float = 10.0,
    interaction_backend: str = "jax",
) -> LoadedSystem:
    """Construct a torsion-only Vina potential from prepared PDBQT files.

    Receptor heavy atoms farther than ``pocket_radius`` from every heavy atom
    in the input ligand pose are removed once during setup. This is a
    fixed-pocket approximation for conformational sampling, not Vina's grid
    implementation. Multiple receptor files can be supplied to freeze an
    originally flexible receptor at its prepared coordinates.
    """

    if pocket_radius < 8.0:
        raise ValueError("pocket_radius must be at least Vina's 8 A cutoff")

    ligand_atoms, branches, branch_members = parse_ligand_tree(ligand_path)
    ligand_heavy = [atom for atom in ligand_atoms if not _is_hydrogen(atom)]
    ligand_hydrogen = [atom for atom in ligand_atoms if atom.atom_type == "HD"]
    serial_to_index = {
        atom.serial: index for index, atom in enumerate(ligand_heavy)
    }
    try:
        rotatable_bonds = np.asarray(
            [(serial_to_index[a], serial_to_index[b]) for a, b in branches],
            dtype=np.int32,
        )
    except KeyError as error:
        raise ValueError("a PDBQT BRANCH axis contains a hydrogen") from error
    rotation_masks = np.asarray(
        [
            [atom.serial in members for atom in ligand_heavy]
            for members in branch_members
        ],
        dtype=bool,
    )
    ligand_bonds = _infer_bonds(ligand_heavy)
    # Explicitly retain every declared rotatable bond even if rounded PDBQT
    # coordinates put it just beyond the generic covalent-distance tolerance.
    ligand_bond_set = {tuple(sorted(pair)) for pair in ligand_bonds.tolist()}
    ligand_bond_set.update(tuple(sorted(pair)) for pair in rotatable_bonds.tolist())
    ligand_bonds = np.asarray(sorted(ligand_bond_set), dtype=np.int32)
    intra_pairs = build_ligand_intra_pairs(
        len(ligand_heavy), ligand_bonds, rotation_masks
    )
    ligand_hydrophobic, ligand_donor, ligand_acceptor = _chemical_features(
        ligand_heavy, ligand_hydrogen
    )

    if isinstance(receptor_paths, Path):
        receptor_paths = (receptor_paths,)
    receptor_atoms = [
        atom
        for receptor_path in receptor_paths
        for atom in parse_pdbqt_atoms(receptor_path)
    ]
    receptor_heavy = [atom for atom in receptor_atoms if not _is_hydrogen(atom)]
    receptor_hydrogen = [atom for atom in receptor_atoms if atom.atom_type == "HD"]
    ligand_coordinate = np.stack([atom.coordinate for atom in ligand_heavy])
    receptor_coordinate = np.stack([atom.coordinate for atom in receptor_heavy])
    closest = np.min(
        np.linalg.norm(
            receptor_coordinate[:, None, :] - ligand_coordinate[None, :, :],
            axis=-1,
        ),
        axis=1,
    )
    pocket_mask = closest <= pocket_radius
    receptor_pocket = [
        atom for atom, selected in zip(receptor_heavy, pocket_mask) if selected
    ]
    receptor_hydrophobic, receptor_donor, receptor_acceptor = _chemical_features(
        receptor_pocket, receptor_hydrogen
    )

    potential = VinaScore(
        receptor_coordinate=np.stack(
            [atom.coordinate for atom in receptor_pocket]
        ),
        receptor_vdw_radii=np.asarray(
            [VINA_RADIUS[atom.atom_type] for atom in receptor_pocket]
        ),
        ligand_vdw_radii=np.asarray(
            [VINA_RADIUS[atom.atom_type] for atom in ligand_heavy]
        ),
        ligand_reference_coordinate=ligand_coordinate,
        rotatable_bonds=rotatable_bonds,
        rotation_masks=rotation_masks,
        ligand_intra_pairs=intra_pairs,
        receptor_hydrophobic=receptor_hydrophobic,
        ligand_hydrophobic=ligand_hydrophobic,
        receptor_hbond_donor=receptor_donor,
        receptor_hbond_acceptor=receptor_acceptor,
        ligand_hbond_donor=ligand_donor,
        ligand_hbond_acceptor=ligand_acceptor,
        interaction_backend=interaction_backend,
    )
    return LoadedSystem(
        potential=potential,
        receptor_heavy_atoms_total=len(receptor_heavy),
        receptor_pocket_atoms=len(receptor_pocket),
        ligand_heavy_atoms=len(ligand_heavy),
        ligand_polar_hydrogens=len(ligand_hydrogen),
        inferred_ligand_bonds=len(ligand_bonds),
        pocket_radius=float(pocket_radius),
    )


__all__ = [
    "LoadedSystem",
    "build_pdbqt_potential",
    "parse_ligand_tree",
    "parse_pdbqt_atoms",
]
