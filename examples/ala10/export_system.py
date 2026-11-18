"""Export the neutral NH2-(Ala)10-COOH ECEPP/3 system from an external SMMP build.

SMMP is deliberately not vendored because its distribution adds use
restrictions to GPL-2.0. Build its f2py module separately, then pass the module
directory and SMMP data root to this script. The resulting NPZ contains only
the initialized molecule and the numerical gold result needed by the baseline.
The exported file is distributed as data/system.npz.

    python -m examples.ala10.export_system --smmp-module SMMP_F2PY_DIR \
        --smmp-root SMMP_DATA_DIR --output examples/ala10/data/system.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def _name(character_array, index: int) -> str:
    return "".join(
        value.decode() if isinstance(value, bytes) else str(value)
        for value in character_array[:, index]
    ).strip()


def _graph_pairs(parent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_atoms = len(parent)
    distance = np.full((n_atoms, n_atoms), n_atoms + 1, dtype=np.int32)
    np.fill_diagonal(distance, 0)
    for atom in range(1, n_atoms):
        owner = parent[atom]
        distance[atom, owner] = distance[owner, atom] = 1
    for intermediate in range(n_atoms):
        distance = np.minimum(
            distance,
            distance[:, intermediate, None] + distance[None, intermediate, :],
        )
    return (
        np.argwhere(np.triu(distance >= 4, 1)).astype(np.int32),
        np.argwhere(np.triu(distance == 3, 1)).astype(np.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smmp-module", type=Path, required=True)
    parser.add_argument("--smmp-root", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, default=DATA / "neutral_ala10.seq")
    parser.add_argument(
        "--variables", type=Path, default=DATA / "alpha_fixed_sidechain.var"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.smmp_module.resolve()))
    import smmp  # type: ignore[import-not-found]  # external gold implementation

    smmp.epar_l.flex = 0
    smmp.epar_l.sh2 = 0
    smmp.epar_l.epsd = 0
    smmp.epar_l.ientyp = 0
    smmp.isolty.itysol = 0
    smmp.mol_i.ntlml = 0
    smmp.init_energy(f"{args.smmp_root.resolve()}/")
    # SMMP stores paths in a fixed 80-character Fortran field. Use paths
    # relative to the worktree root so a long worktree location is harmless.
    repository = HERE.parents[1]
    sequence = os.path.relpath(args.sequence.resolve(), repository)
    variables = os.path.relpath(args.variables.resolve(), repository)
    os.chdir(repository)
    smmp.init_molecule(
        1,
        "nh2",
        "cooh",
        sequence,
        variables,
    )
    gold_total = float(smmp.energy())

    # ibdat contains every atom index in the initialized bond tree and remains
    # zero beyond the active molecule, so its maximum is the exact atom count.
    n_atoms = int(np.max(smmp.atm_i.ibdat))
    n_variables = int(smmp.mol_i.nvrml[0])
    parent = np.asarray(smmp.atm_i.iowat[:n_atoms], dtype=np.int32) - 1
    parent[0] = 0
    nonbonded, one_four = _graph_pairs(parent)

    fixed = np.asarray(smmp.var_l.fxvr[:n_variables], dtype=bool)
    free_indices = np.flatnonzero(~fixed)
    # The torsion attached to SMMP's third reference atom rotates the complete
    # Cartesian frame and is not an internal molecular degree of freedom.
    frame_atom = int(smmp.mol_i.ixrfpt[2, 0])
    free_indices = free_indices[
        np.asarray(smmp.var_i.iatvr[free_indices]) != frame_atom
    ]
    variable_names = np.asarray(
        [_name(smmp.var_c.nmvr, int(index)) for index in free_indices], dtype="U3"
    )
    variable_classes = np.asarray(
        smmp.var_i.iclvr[free_indices], dtype=np.int32
    )
    initial_variables = np.asarray(
        smmp.var_r.vlvr[free_indices], dtype=np.float64
    )
    effects = np.zeros((n_atoms, len(free_indices)), dtype=np.float64)
    for column, variable in enumerate(free_indices):
        primary = int(smmp.var_i.iatvr[variable]) - 1
        owner_fortran = int(smmp.atm_i.iowat[primary])
        effects[
            np.asarray(smmp.atm_i.iowat[:n_atoms]) == owner_fortran, column
        ] = 1.0

    reference_coordinates = np.stack(
        (
            smmp.atm_r.xat[2 : n_atoms + 2],
            smmp.atm_r.yat[2 : n_atoms + 2],
            smmp.atm_r.zat[2 : n_atoms + 2],
        ),
        axis=1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        parent=parent,
        bond_length=np.asarray(smmp.atm_r.blat[:n_atoms]),
        bond_angle=np.asarray(smmp.atm_r.baat[:n_atoms]),
        base_torsion=np.asarray(smmp.atm_r.toat[:n_atoms]),
        variable_effects=effects,
        initial_variables=initial_variables,
        variable_names=variable_names,
        variable_classes=variable_classes,
        root_position=reference_coordinates[0],
        root_x_axis=np.asarray(
            [smmp.atm_r.xtoat[0], smmp.atm_r.ytoat[0], smmp.atm_r.ztoat[0]]
        ),
        root_z_axis=np.asarray(
            [smmp.atm_r.xbaat[0], smmp.atm_r.ybaat[0], smmp.atm_r.zbaat[0]]
        ),
        charges=np.asarray(smmp.atm_r.cgat[:n_atoms]),
        atom_types=np.asarray(smmp.atm_i.ityat[:n_atoms]),
        nonbonded_pairs=nonbonded,
        one_four_pairs=one_four,
        reference_coordinates=reference_coordinates,
        gold_electrostatic=float(smmp.eny_r.eyel),
        gold_lennard_jones=float(smmp.eny_r.eyvw),
        gold_hydrogen_bond=float(smmp.eny_r.eyhb),
        gold_torsion=float(smmp.eny_r.eyvr),
        gold_etotal=gold_total,
    )
    print(
        f"wrote {args.output}: {n_atoms} atoms, {len(free_indices)} variables, "
        f"E_gold={gold_total:.12f} kcal/mol"
    )


if __name__ == "__main__":
    main()
