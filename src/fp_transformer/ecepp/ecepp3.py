from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterable

import numpy as np

from .spec import (
    ECEPPParameters,
    ElectrostaticParameters,
    HydrogenBondParameters,
    LennardJonesParameters,
    TorsionParameters,
)


ECEPP3_ATOM_TYPE_NAMES = (
    "H aliphatic",
    "H aromatic",
    "H on non-sp3 N",
    "H on sp3 N",
    "H on O",
    "H on S",
    "C sp3",
    "C carbonyl/carboxyl",
    "C aromatic",
    "O hydroxyl/ester",
    "O carbonyl",
    "O carboxylate",
    "N aliphatic/charged",
    "N with two H",
    "N other/heterocyclic",
    "S",
    "H delta in Pro/Hyp",
    "C delta in Pro/Hyp",
)

# ECEPP/3 Scheraga parameters from SMMP 3.0.5 init_energy.f. The source lists
# polarizability and electron counts scaled by 100, and Rmin scaled by 100.
_POLARIZABILITY = np.asarray(
    [42, 42, 42, 42, 42, 42, 93, 151, 115, 59, 84, 59, 93, 93, 93, 220, 42, 93],
    dtype=np.float64,
) / 100.0
_EFFECTIVE_ELECTRONS = np.asarray(
    [
        85, 85, 85, 85, 85, 85, 520, 520, 520, 700, 700, 700, 610, 610, 610,
        1480, 85, 520,
    ],
    dtype=np.float64,
) / 100.0
_RMIN = np.asarray(
    [
        292, 293, 268, 268, 283, 293, 412, 374, 370, 324, 312, 324, 351, 351,
        351, 415, 248, 412,
    ],
    dtype=np.float64,
) / 100.0
_SELF_C6 = np.asarray(
    [
        45.5, 45.5, 45.5, 45.5, 45.5, 45.5, 370.5, 766.6, 509.5, 217.2,
        369.0, 217.2, 401.3, 401.3, 401.3, 2274.4, 45.5, 370.5,
    ]
)
_SELF_A12 = np.asarray(
    [
        14090, 14380, 8420, 8420, 11680, 14380, 906100, 1049000, 653600,
        125600, 170200, 125600, 375200, 375200, 375200, 5809000, 5340, 909000,
    ],
    dtype=np.float64,
)

_DONOR_TYPES = (3, 4, 5)
_ACCEPTOR_TYPES = (10, 11, 12, 13, 14, 15)
_HB_D10 = np.asarray(
    [
        [2624, 4014, 2624, 8244, 8244, 8244],
        [2624, 4014, 2624, 8244, 8244, 8244],
        [4610, 5783, 4610, 8244, 8244, 8244],
    ],
    dtype=np.float64,
)
_HB_C12 = np.asarray(
    [
        [5890, 12040, 5890, 32897, 32897, 32897],
        [5890, 12040, 5890, 32897, 32897, 32897],
        [11220, 16583, 11220, 32897, 32897, 32897],
    ],
    dtype=np.float64,
)

_TORSION_BARRIER = np.asarray(
    [20, 0, 0, 2.7, 0.6, 1.8, 1.8, 0, 0, 0, 0, 0, 2, 1.5, 3.5, 8, 18, 20, 15],
    dtype=np.float64,
)
_TORSION_SIGN = np.asarray(
    [-1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, -1, -1, -1, -1, -1],
    dtype=np.float64,
)
_TORSION_PERIODICITY = np.asarray(
    [2, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 3, 3, 2, 2, 2, 2, 2],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ECEPP3Atom:
    name: str
    bond_length: float
    bond_angle_degrees: float
    initial_dihedral_degrees: float
    charge: float
    atom_type: int
    connectivity: tuple[int, int, int, int]
    variable_class: int | None = None
    variable_name: str | None = None


@dataclass(frozen=True)
class ECEPP3Residue:
    name: str
    declared_atom_count: int
    declared_variable_count: int
    atoms: tuple[ECEPP3Atom, ...]

    def __post_init__(self) -> None:
        if len(self.atoms) != self.declared_atom_count:
            raise ValueError(
                f"residue {self.name!r} declares {self.declared_atom_count} atoms "
                f"but contains {len(self.atoms)}"
            )


@dataclass(frozen=True)
class ECEPP3Library:
    residues: dict[str, ECEPP3Residue]
    source: str | None = None

    @classmethod
    def from_smmp(cls, path: str | Path) -> ECEPP3Library:
        path = Path(path)
        return cls.from_smmp_text(path.read_text(), source=str(path))

    @classmethod
    def from_smmp_text(
        cls, text: str, *, source: str | None = None
    ) -> ECEPP3Library:
        residues: dict[str, ECEPP3Residue] = {}
        current_header: tuple[str, int, int] | None = None
        current_atoms: list[ECEPP3Atom] = []

        def finish_residue() -> None:
            nonlocal current_header, current_atoms
            if current_header is None:
                return
            name, atom_count, variable_count = current_header
            if name in residues:
                raise ValueError(f"duplicate ECEPP/3 residue {name!r}")
            residues[name] = ECEPP3Residue(
                name, atom_count, variable_count, tuple(current_atoms)
            )
            current_header = None
            current_atoms = []

        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                finish_residue()
                fields = line[1:].split()
                if len(fields) < 3:
                    raise ValueError(f"invalid residue header on line {line_number}")
                current_header = (fields[0].lower(), int(fields[1]), int(fields[2]))
                continue
            if current_header is None:
                raise ValueError(
                    f"atom record before residue header on line {line_number}"
                )

            fields = line.split()
            has_variable = len(fields) > 3 and "*" in fields[3]
            if len(fields) != (11 if has_variable else 10):
                raise ValueError(
                    f"invalid atom record on line {line_number}: {raw_line}"
                )
            offset = 1 if has_variable else 0
            variable_class = None
            variable_name = None
            if has_variable:
                class_text, variable_name = fields[3].split("*", maxsplit=1)
                variable_class = int(class_text)
            current_atoms.append(
                ECEPP3Atom(
                    name=fields[0].lower(),
                    bond_length=float(fields[1]),
                    bond_angle_degrees=float(fields[2]),
                    initial_dihedral_degrees=float(fields[3 + offset]),
                    charge=float(fields[4 + offset]),
                    atom_type=int(fields[5 + offset]),
                    connectivity=tuple(int(value) for value in fields[6 + offset :]),
                    variable_class=variable_class,
                    variable_name=variable_name,
                )
            )
        finish_residue()
        if not residues:
            raise ValueError("ECEPP/3 library contains no residues")
        return cls(residues, source)


def _as_pairs(value: Iterable[Iterable[int]], name: str) -> np.ndarray:
    array = np.asarray(list(value), dtype=np.int32)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {array.shape}")
    return array


class ECEPP3ParameterSet:
    """The ECEPP/3 atom-type, hydrogen-bond, and torsion parameter tables."""

    dielectric = 2.0
    coulomb_constant = 332.0
    atom_type_names = ECEPP3_ATOM_TYPE_NAMES

    def __init__(self):
        n_types = len(ECEPP3_ATOM_TYPE_NAMES)
        c6 = np.zeros((n_types, n_types), dtype=np.float64)
        a12 = np.zeros_like(c6)
        for i in range(n_types):
            for j in range(i, n_types):
                ai = _POLARIZABILITY[i]
                aj = _POLARIZABILITY[j]
                denominator = np.sqrt(ai / _EFFECTIVE_ELECTRONS[i]) + np.sqrt(
                    aj / _EFFECTIVE_ELECTRONS[j]
                )
                attraction = ai * 362.55 * aj / denominator
                radius = 0.5 * (_RMIN[i] + _RMIN[j])
                repulsion = 0.5 * attraction * radius**6
                c6[i, j] = c6[j, i] = attraction
                a12[i, j] = a12[j, i] = repulsion
        np.fill_diagonal(c6, _SELF_C6)
        np.fill_diagonal(a12, _SELF_A12)
        self.c6 = c6
        self.a12 = a12
        self.a12_one_four = 0.5 * a12

        self.hydrogen_d10 = np.zeros_like(c6)
        self.hydrogen_c12 = np.zeros_like(c6)
        for donor_index, donor_type in enumerate(_DONOR_TYPES):
            for acceptor_index, acceptor_type in enumerate(_ACCEPTOR_TYPES):
                donor = donor_type - 1
                acceptor = acceptor_type - 1
                self.hydrogen_d10[donor, acceptor] = self.hydrogen_d10[
                    acceptor, donor
                ] = _HB_D10[donor_index, acceptor_index]
                self.hydrogen_c12[donor, acceptor] = self.hydrogen_c12[
                    acceptor, donor
                ] = _HB_C12[donor_index, acceptor_index]

    @staticmethod
    def variable_torsion_parameters(classes):
        """Return ECEPP/3 torsion coefficients for internal variables."""
        classes = np.asarray(classes, dtype=np.int32)
        if classes.ndim != 1:
            raise ValueError("variable torsion classes must be one-dimensional")
        if np.any(classes < 1) or np.any(classes > 19):
            raise ValueError("ECEPP/3 torsion classes must be in [1, 19]")
        indices = classes - 1
        barrier_half = 0.5 * _TORSION_BARRIER[indices]
        # Zero-barrier classes still need a valid positive periodicity for a
        # branch-free JAX evaluator.
        periodicity = _TORSION_PERIODICITY[indices].copy()
        periodicity[barrier_half == 0.0] = 1.0
        phase = np.where(_TORSION_SIGN[indices] > 0, 0.0, np.pi)
        return barrier_half, periodicity, phase

    def compile(
        self,
        *,
        charges,
        atom_types,
        nonbonded_pairs: Iterable[Iterable[int]],
        one_four_pairs: Iterable[Iterable[int]] = (),
        torsion_atoms: Iterable[Iterable[int]] = (),
        torsion_classes: Iterable[int] = (),
    ) -> ECEPPParameters:
        charges = np.asarray(charges, dtype=np.float32)
        atom_types = np.asarray(atom_types, dtype=np.int32)
        if charges.ndim != 1 or atom_types.shape != charges.shape:
            raise ValueError(
                "charges and atom_types must be one-dimensional and aligned"
            )
        if np.any(atom_types < 1) or np.any(atom_types > len(self.atom_type_names)):
            raise ValueError("ECEPP/3 atom_types must be in [1, 18]")

        regular = _as_pairs(nonbonded_pairs, "nonbonded_pairs")
        one_four = _as_pairs(one_four_pairs, "one_four_pairs")
        all_pairs = np.concatenate((regular, one_four), axis=0)
        if len(np.unique(np.sort(all_pairs, axis=1), axis=0)) != len(all_pairs):
            raise ValueError("nonbonded_pairs and one_four_pairs must be unique")

        type_indices = atom_types - 1
        lj_pairs: list[list[int]] = []
        lj_a: list[float] = []
        lj_b: list[float] = []
        hb_pairs: list[list[int]] = []
        hb_c: list[float] = []
        hb_d: list[float] = []
        if all_pairs.size and (
            np.any(all_pairs < 0) or np.any(all_pairs >= len(charges))
        ):
            raise ValueError("nonbonded pair index outside the atom range")

        for pair, is_one_four in chain(
            ((pair, False) for pair in regular),
            ((pair, True) for pair in one_four),
        ):
            i, j = pair
            ti, tj = type_indices[i], type_indices[j]
            if self.hydrogen_c12[ti, tj] != 0:
                hb_pairs.append(pair.tolist())
                hb_c.append(self.hydrogen_c12[ti, tj])
                hb_d.append(self.hydrogen_d10[ti, tj])
            else:
                lj_pairs.append(pair.tolist())
                repulsion = self.a12_one_four if is_one_four else self.a12
                lj_a.append(repulsion[ti, tj])
                lj_b.append(self.c6[ti, tj])

        torsion_atoms = np.asarray(list(torsion_atoms), dtype=np.int32)
        torsion_classes = np.asarray(list(torsion_classes), dtype=np.int32)
        if torsion_atoms.size == 0:
            torsion_atoms = np.empty((0, 4), dtype=np.int32)
        if torsion_atoms.shape != (len(torsion_classes), 4):
            raise ValueError("torsion_atoms and torsion_classes must be aligned")
        if np.any(torsion_classes < 1) or np.any(torsion_classes > 19):
            raise ValueError("ECEPP/3 torsion classes must be in [1, 19]")
        active = _TORSION_BARRIER[torsion_classes - 1] != 0
        active_classes = torsion_classes[active] - 1
        phase = np.where(_TORSION_SIGN[active_classes] > 0, 0.0, np.pi)

        return ECEPPParameters(
            n_atoms=len(charges),
            charges=charges,
            electrostatic=ElectrostaticParameters.create(all_pairs),
            lennard_jones=LennardJonesParameters.create(lj_pairs, lj_a, lj_b),
            hydrogen_bond=HydrogenBondParameters.create(hb_pairs, hb_c, hb_d),
            torsion=TorsionParameters.create(
                torsion_atoms[active],
                barrier_half=0.5 * _TORSION_BARRIER[active_classes],
                periodicity=_TORSION_PERIODICITY[active_classes],
                phase=phase,
            ),
            dielectric=self.dielectric,
            coulomb_constant=self.coulomb_constant,
        )
