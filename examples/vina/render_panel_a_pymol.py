"""Render the real 7ELT_TYM Panel-A structures with a flat PyMOL style."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np


from examples.vina.dataset import load_posebusters_systems
from examples.vina.plot_paper_figures import (
    SCHEMATIC_SYSTEM,
    _real_schematic_data,
)


HERE = Path(__file__).resolve().parent
DEFAULT_ASSET_DIR = HERE / "assets" / "pymol_illustrative"


def _write_ligand_pdb(
    path: Path,
    coordinates: np.ndarray,
    elements: list[str],
    bonds: np.ndarray,
) -> None:
    lines: list[str] = []
    for index, (coordinate, element) in enumerate(zip(coordinates, elements), start=1):
        atom_name = f"{element}{index}"[:4]
        x, y, z = map(float, coordinate)
        lines.append(
            f"HETATM{index:5d} {atom_name:<4s} LIG A   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
        )
    for left, right in bonds:
        lines.append(f"CONECT{int(left) + 1:5d}{int(right) + 1:5d}")
    lines.extend(("TER", "END"))
    path.write_text("\n".join(lines) + "\n")


def _pymol_prelude() -> list[str]:
    return [
        "reinitialize",
        "bg_color white",
        "set ray_opaque_background, off",
        "set orthoscopic, on",
        "set auto_zoom, off",
        "set depth_cue, off",
        "set fog, off",
        "set antialias, 2",
        "set ambient, 0.72",
        "set direct, 0.28",
        "set reflect, 0.06",
        "set specular, 0",
        "set shininess, 0",
        "set ray_shadows, off",
        "set ray_trace_mode, 1",
        "set ray_trace_gain, 0.08",
        "set ray_trace_color, gray60",
        "set_color atom_c, [0.694, 0.655, 0.620]",
        "set_color atom_n, [0.494, 0.576, 0.651]",
        "set_color atom_o, [0.773, 0.561, 0.498]",
        "set_color atom_p, [0.569, 0.639, 0.580]",
        "set_color atom_s, [0.710, 0.616, 0.435]",
        "set_color atom_halogen, [0.569, 0.639, 0.580]",
        # Ice-porcelain landscape low: #DCEBEB.  The asset is composited at
        # full opacity, so its lightness is encoded directly in the RGB value.
        "set_color pocket_surface, [0.862745, 0.921569, 0.921569]",
        "set_color torsion_highlight, [0.600, 0.553, 0.647]",
        "set stick_radius, 0.16",
        "set sphere_scale, 0.26",
        "set stick_quality, 14",
        "set sphere_quality, 2",
    ]


def _color_ligand(selection: str) -> list[str]:
    return [
        f"color atom_c, ({selection}) and elem C",
        f"color atom_n, ({selection}) and elem N",
        f"color atom_o, ({selection}) and elem O",
        f"color atom_p, ({selection}) and elem P",
        f"color atom_s, ({selection}) and elem S",
        f"color atom_halogen, ({selection}) and elem F+Cl+Br+I",
    ]


def _render_assets(
    structure_dir: Path,
    asset_dir: Path,
    protein_path: Path,
    conformer_names: list[str],
    torsion_bond: tuple[int, int],
) -> None:
    commands = _pymol_prelude()
    for name in conformer_names:
        commands.append(f"load {structure_dir / (name + '.pdb')}, {name}")
    commands.extend(_color_ligand("all"))
    commands.extend(
        (
            "orient all",
            "zoom all, 1.8",
            "hide everything, all",
        )
    )
    for name in conformer_names:
        commands.extend(
            (
                f"show sticks, {name}",
                f"show spheres, {name}",
                f"png {asset_dir / (name + '.png')}, 520, 520, 300, ray=1",
                f"hide everything, {name}",
            )
        )

    commands.extend(
        (
            "delete all",
            f"load {protein_path}, protein",
            f"load {structure_dir / 'reference.pdb'}, ligand",
            "remove solvent",
            "select pocket, byres ((polymer.protein and protein) within 9.5 of ligand)",
            "hide everything, all",
            "set ray_trace_mode, 0",
            "set ambient, 0.84",
            "set direct, 0.16",
            "set reflect, 0",
            "set ray_trace_gain, 0",
            "set stick_radius, 0.20, ligand",
            "set sphere_scale, 0.34, ligand",
            "color pocket_surface, pocket",
            "set surface_color, pocket_surface, pocket",
            "set transparency, 0, pocket",
            "set surface_quality, 1",
            "set solvent_radius, 1.4",
            "set surface_mode, 1",
        )
    )
    commands.extend(_color_ligand("ligand"))
    left_id, right_id = torsion_bond
    commands.extend(
        (
            "orient pocket or ligand",
            "zoom pocket or ligand, 0.8",
            "hide everything, all",
            "show surface, pocket",
            f"png {asset_dir / 'system_surface.png'}, 900, 900, 300, ray=1",
            "hide everything, all",
            "show sticks, ligand",
            "show spheres, ligand",
            f"png {asset_dir / 'system_ligand.png'}, 900, 900, 300, ray=1",
            f"create torsion_marker, ligand and id {left_id}+{right_id}",
            "hide everything, all",
            "show sticks, torsion_marker",
            "color torsion_highlight, torsion_marker",
            "set stick_radius, 0.11, torsion_marker",
            f"png {asset_dir / 'system_torsion.png'}, 900, 900, 300, ray=1",
            "quit",
        )
    )
    pml_path = asset_dir / "render.pml"
    pml_path.write_text("\n".join(commands) + "\n")
    subprocess.run(["pymol", "-cq", str(pml_path)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=6300)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    args = parser.parse_args()

    args.asset_dir.mkdir(parents=True, exist_ok=True)
    structure_dir = args.asset_dir / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    schematic = _real_schematic_data(
        {
            "suite": {"protocol": {"seed_base": args.seed}},
            "output_dir": args.output_dir,
        }
    )
    elements = schematic["elements"]
    bonds = schematic["bonds"]
    conformers: dict[str, np.ndarray] = {
        "reference": schematic["reference_coordinates"],
        "fpt_best": schematic["fpt_best_coordinates"],
    }
    conformers.update(
        {
            f"mc_{index}": coordinates
            for index, coordinates in enumerate(schematic["mc_coordinates"])
        }
    )
    conformers.update(
        {
            f"fpt_{time_index}_{particle_index}": coordinates
            for time_index, particles in enumerate(schematic["fpt_coordinates"])
            for particle_index, coordinates in enumerate(particles)
        }
    )
    for name, coordinates in conformers.items():
        _write_ligand_pdb(
            structure_dir / f"{name}.pdb",
            coordinates,
            elements,
            bonds,
        )

    systems = load_posebusters_systems()
    protein_path = (
        HERE
        / systems[SCHEMATIC_SYSTEM]["posebusters_record"]["source_protein"]
    )
    reference = np.asarray(schematic["reference_coordinates"])
    rotatable = np.asarray(schematic["rotatable_bonds"], dtype=int)
    midpoints = np.mean(reference[rotatable], axis=1)
    centroid = np.mean(reference, axis=0)
    midpoint_distance = np.linalg.norm(midpoints - centroid, axis=1)
    highlighted = rotatable[int(np.argmax(midpoint_distance))]
    torsion_bond = (int(highlighted[0]) + 1, int(highlighted[1]) + 1)
    _render_assets(
        structure_dir,
        args.asset_dir,
        protein_path,
        [name for name in conformers if name != "reference"],
        torsion_bond,
    )


if __name__ == "__main__":
    main()
