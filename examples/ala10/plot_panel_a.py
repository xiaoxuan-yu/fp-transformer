#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from sklearn.cluster import KMeans

from examples.common.figure_style import (
    CARTOON_COLOR,
    ELEMENT_N_COLOR,
    ELEMENT_O_COLOR,
    MC_COLOR,
    NEITHER_COLOR,
    REFERENCE_COLOR,
    TEXT_COLOR,
    ULA_COLOR,
    WARM_ACCENT_COLOR,
    configure_matplotlib,
    format_panel_label,
    save_figure,
)


DATA = Path(__file__).resolve().parent / "data"
DEFAULT_SYSTEM = DATA / "system.npz"
DEFAULT_CANONICAL = DATA / "canonical_mc_samples.npz"
DEFAULT_REPRESENTATIVE_OUTPUT = Path("figures") / "ala10_canonical_representatives.npz"
CORE_CARTOON_COLOR = CARTOON_COLOR
TERMINAL_COLORS = (
    WARM_ACCENT_COLOR,
    MC_COLOR,
    ULA_COLOR,
    NEITHER_COLOR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Ala10 coil-to-helical-family experiment panel."
    )
    parser.add_argument("--system", type=Path, default=DEFAULT_SYSTEM)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument(
        "--representative-output",
        type=Path,
        default=DEFAULT_REPRESENTATIVE_OUTPUT,
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("figures") / "ala10_panel_a",
    )
    return parser.parse_args()


def strict_helix_fraction(
    angles: np.ndarray,
    names: np.ndarray,
) -> np.ndarray:
    degrees = np.rad2deg(angles)
    phi = degrees[:, names == "phi"]
    psi = degrees[:, names == "psi"]
    alpha = (
        (phi >= -100.0)
        & (phi <= -30.0)
        & (psi >= -80.0)
        & (psi <= -5.0)
    )
    return np.mean(alpha, axis=1)


def torus_embedding(angles: np.ndarray) -> np.ndarray:
    return np.concatenate((np.cos(angles), np.sin(angles)), axis=1)


def circular_center(angles: np.ndarray) -> np.ndarray:
    return np.angle(np.sum(np.exp(1j * angles), axis=0))


def canonical_cluster_representatives(
    angles: np.ndarray,
    names: np.ndarray,
    *,
    clusters: int = 4,
    seed: int = 20260719,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose the actual canonical-MC sample nearest each cluster center."""
    backbone = np.isin(names, ("phi", "psi"))
    strict_indices = np.flatnonzero(
        strict_helix_fraction(angles, names) >= 0.7
    )
    strict_angles = angles[strict_indices][:, backbone]
    model = KMeans(
        n_clusters=clusters,
        n_init=30,
        random_state=seed,
    ).fit(torus_embedding(strict_angles))

    original_counts = np.bincount(model.labels_, minlength=clusters)
    label_order = np.argsort(original_counts)[::-1]
    remap = np.empty(clusters, dtype=np.int64)
    remap[label_order] = np.arange(clusters)
    labels = remap[model.labels_]
    masses = np.bincount(labels, minlength=clusters) / labels.size

    selected_indices = []
    center_rms_degrees = []
    for cluster in range(clusters):
        local = np.flatnonzero(labels == cluster)
        center = circular_center(strict_angles[local])
        difference = np.angle(
            np.exp(1j * (strict_angles[local] - center))
        )
        distance = np.sqrt(np.mean(difference**2, axis=1))
        nearest = int(local[int(np.argmin(distance))])
        selected_indices.append(int(strict_indices[nearest]))
        center_rms_degrees.append(
            float(np.rad2deg(np.min(distance)))
        )
    selected_indices_array = np.asarray(selected_indices, dtype=np.int64)
    return (
        np.asarray(angles[selected_indices_array], dtype=float),
        np.asarray(masses, dtype=float),
        selected_indices_array,
        np.asarray(center_rms_degrees, dtype=float),
    )


def build_cartesian_coordinates(
    variables: np.ndarray,
    *,
    parent: np.ndarray,
    bond_length: np.ndarray,
    bond_angle: np.ndarray,
    base_torsion: np.ndarray,
    variable_effects: np.ndarray,
    initial_variables: np.ndarray,
    root_position: np.ndarray,
    root_x_axis: np.ndarray,
    root_z_axis: np.ndarray,
) -> np.ndarray:
    """Pure-NumPy form of the ECEPP internal-coordinate reconstruction."""
    variables = np.asarray(variables, dtype=float)
    delta = (
        variables - initial_variables[None, :] + np.pi
    ) % (2.0 * np.pi) - np.pi
    torsion = base_torsion[None, :] + delta @ variable_effects.T
    batch_size = variables.shape[0]
    atom_count = parent.size
    coordinates = np.zeros((batch_size, atom_count, 3), dtype=float)
    x_axes = np.zeros_like(coordinates)
    z_axes = np.zeros_like(coordinates)
    coordinates[:, 0] = root_position
    x_axes[:, 0] = root_x_axis
    z_axes[:, 0] = root_z_axis

    for atom in range(1, atom_count):
        owner = int(parent[atom])
        owner_x = x_axes[:, owner]
        owner_z = z_axes[:, owner]
        y_axis = np.cross(owner_z, owner_x)
        angle = float(bond_angle[atom])
        dihedral = torsion[:, atom, None]
        new_x = (
            -np.cos(angle) * owner_x
            - np.sin(angle) * np.cos(dihedral) * y_axis
            - np.sin(angle) * np.sin(dihedral) * owner_z
        )
        new_x /= np.linalg.norm(new_x, axis=-1, keepdims=True)
        new_z = (
            -np.sin(dihedral) * y_axis
            + np.cos(dihedral) * owner_z
        )
        new_z /= np.linalg.norm(new_z, axis=-1, keepdims=True)
        coordinates[:, atom] = (
            coordinates[:, owner] + bond_length[atom] * new_x
        )
        x_axes[:, atom] = new_x
        z_axes[:, atom] = new_z
    return coordinates


def kabsch_align(
    coordinates: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    moving_center = np.mean(coordinates[mask], axis=0)
    reference_center = np.mean(reference[mask], axis=0)
    moving = coordinates - moving_center
    target = reference - reference_center
    covariance = moving[mask].T @ target[mask]
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    return moving @ rotation


def common_cartoon_view(coordinates: np.ndarray) -> np.ndarray:
    """Align the conserved helix core and expose terminal variation."""
    core = np.asarray(
        [
            atom
            for residue in range(1, 8)
            for atom in (
                10 * residue + 1,
                10 * residue + 3,
                10 * residue + 9,
            )
        ],
        dtype=np.int64,
    )
    reference = coordinates[1]
    aligned = np.stack(
        [
            kabsch_align(
                item,
                reference,
                np.isin(np.arange(item.shape[0]), core),
            )
            for item in coordinates
        ]
    )
    centered_reference = aligned[1, core]
    _, _, right_t = np.linalg.svd(
        centered_reference,
        full_matrices=False,
    )
    vertical = right_t[0]

    terminal_alpha_carbons = np.asarray((3, 83, 93), dtype=np.int64)
    terminal_coordinates = aligned[1:, terminal_alpha_carbons]
    terminal_differences = (
        terminal_coordinates
        - np.mean(terminal_coordinates, axis=0, keepdims=True)
    ).reshape(-1, 3)
    projector = np.eye(3) - np.outer(vertical, vertical)
    terminal_differences = terminal_differences @ projector
    _, _, terminal_right_t = np.linalg.svd(
        terminal_differences,
        full_matrices=False,
    )
    horizontal = terminal_right_t[0]
    horizontal -= vertical * np.dot(horizontal, vertical)
    horizontal /= np.linalg.norm(horizontal)
    depth = np.cross(horizontal, vertical)
    depth /= np.linalg.norm(depth)
    basis = np.column_stack((horizontal, vertical, depth))
    # PyMOL looks down z; x maximizes terminal separation and y follows the core.
    return aligned @ basis


def pdb_atom_indices() -> list[tuple[int, str, int, str]]:
    records: list[tuple[int, str, int, str]] = []
    for residue in range(10):
        nitrogen = 0 if residue == 0 else 10 * residue + 1
        alpha_carbon = 10 * residue + 3
        beta_carbon = 10 * residue + 5
        carbonyl_carbon = 10 * residue + 9
        carbonyl_oxygen = 10 * residue + 10
        records.extend(
            (
                (nitrogen, "N", residue + 1, "N"),
                (alpha_carbon, "CA", residue + 1, "C"),
                (carbonyl_carbon, "C", residue + 1, "C"),
                (carbonyl_oxygen, "O", residue + 1, "O"),
                (beta_carbon, "CB", residue + 1, "C"),
            )
        )
    records.append((101, "OXT", 10, "O"))
    return records


def write_cartoon_pdb(path: Path, coordinates: np.ndarray) -> None:
    lines = []
    for serial, (atom, name, residue, element) in enumerate(
        pdb_atom_indices(),
        start=1,
    ):
        x, y, z = coordinates[atom]
        lines.append(
            f"ATOM  {serial:5d} {name:^4s} ALA A{residue:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
        )
    lines.extend(("TER\n", "END\n"))
    path.write_text("".join(lines))


def color_rgb(hex_color: str) -> tuple[float, float, float]:
    value = hex_color.removeprefix("#")
    return tuple(
        int(value[offset:offset + 2], 16) / 255.0
        for offset in (0, 2, 4)
    )


def render_cartoon_sprites(
    coordinates: np.ndarray,
    directory: Path,
) -> list[np.ndarray]:
    pymol = shutil.which("pymol")
    if pymol is None:
        raise RuntimeError("PyMOL is required to render Ala10 cartoons")

    colors = (REFERENCE_COLOR,) + (CORE_CARTOON_COLOR,) * 4
    commands = [
        "bg_color white",
        "set ray_opaque_background, on",
        "set orthoscopic, on",
        "set auto_zoom, off",
        "set depth_cue, off",
        "set fog, off",
        "set antialias, 2",
        "set ambient, 0.84",
        "set direct, 0.16",
        "set reflect, 0",
        "set specular, 0",
        "set shininess, 0",
        "set ray_shadows, off",
        "set ray_trace_mode, 0",
        "set ray_trace_gain, 0",
        "set cartoon_fancy_helices, on",
        "set cartoon_smooth_loops, on",
        "set cartoon_sampling, 14",
        "set cartoon_loop_radius, 0.32",
        "set stick_quality, 24",
        "set sphere_quality, 4",
        "set stick_radius, 0.13",
        "set sphere_scale, 0.22",
    ]
    nitrogen_red, nitrogen_green, nitrogen_blue = color_rgb(ELEMENT_N_COLOR)
    oxygen_red, oxygen_green, oxygen_blue = color_rgb(ELEMENT_O_COLOR)
    commands.extend(
        (
            "set_color terminal_nitrogen, "
            f"[{nitrogen_red:.6f}, {nitrogen_green:.6f}, "
            f"{nitrogen_blue:.6f}]",
            "set_color terminal_oxygen, "
            f"[{oxygen_red:.6f}, {oxygen_green:.6f}, "
            f"{oxygen_blue:.6f}]",
        )
    )
    png_paths = []
    for index, (item, color) in enumerate(
        zip(coordinates, colors, strict=True)
    ):
        pdb_path = directory / f"ala10_{index}.pdb"
        png_path = directory / f"ala10_{index}.png"
        write_cartoon_pdb(pdb_path, item)
        red, green, blue = color_rgb(color)
        accent = (
            REFERENCE_COLOR
            if index == 0
            else TERMINAL_COLORS[index - 1]
        )
        accent_red, accent_green, accent_blue = color_rgb(accent)
        loop_radius = 0.52 if index == 0 else 0.32
        commands.extend(
            (
                "delete all",
                f"load {pdb_path.as_posix()}, mol",
                "hide everything, mol",
                "alter mol, ss='L'",
                (
                    "alter mol and resi 2-8, ss='H'"
                    if index > 0
                    else "alter mol, ss='L'"
                ),
                "rebuild",
                f"set cartoon_loop_radius, {loop_radius:.2f}",
                f"set_color ala_color_{index}, "
                f"[{red:.6f}, {green:.6f}, {blue:.6f}]",
                f"color ala_color_{index}, mol",
                f"set_color ala_accent_{index}, "
                f"[{accent_red:.6f}, {accent_green:.6f}, "
                f"{accent_blue:.6f}]",
            )
        )
        if index == 0:
            commands.extend(
                (
                    "show cartoon, mol",
                    f"color ala_accent_{index}, mol",
                )
            )
        else:
            terminal = (
                "mol and resi 1+9+10 "
                "and name N+CA+C+O+OXT"
            )
            commands.extend(
                (
                    "show cartoon, mol and resi 2-8",
                    f"show sticks, {terminal}",
                    f"show spheres, {terminal}",
                    f"color ala_accent_{index}, "
                    f"({terminal}) and elem C",
                    f"color terminal_nitrogen, "
                    f"({terminal}) and elem N",
                    f"color terminal_oxygen, "
                    f"({terminal}) and elem O",
                )
            )
        commands.extend(
            (
                "reset",
                "zoom mol, 3.5",
                f"png {png_path.as_posix()}, "
                "width=1400, height=1800, dpi=400, ray=1",
            )
        )
        png_paths.append(png_path)
    commands.append("quit")
    script_path = directory / "render_cartoon.pml"
    script_path.write_text("\n".join(commands) + "\n")
    result = subprocess.run(
        [pymol, "-cq", str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or any(
        not path.exists() for path in png_paths
    ):
        raise RuntimeError(
            "PyMOL cartoon rendering failed:\n"
            + result.stdout
            + result.stderr
        )
    return [trim_sprite(plt.imread(path)) for path in png_paths]


def trim_sprite(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] < 3:
        return image
    rgb = image[:, :, :3]
    occupied = np.max(np.abs(rgb - 1.0), axis=2) > 0.008
    if not np.any(occupied):
        return image
    rows, columns = np.nonzero(occupied)
    padding = 16
    row_start = max(0, int(rows.min()) - padding)
    row_stop = min(image.shape[0], int(rows.max()) + padding + 1)
    column_start = max(0, int(columns.min()) - padding)
    column_stop = min(image.shape[1], int(columns.max()) + padding + 1)
    return image[row_start:row_stop, column_start:column_stop]


def draw_sprite(
    ax,
    image: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> None:
    sprite_ax = ax.inset_axes(bounds)
    sprite_ax.imshow(image, interpolation="hanning")
    sprite_ax.axis("off")


def prepare_schematic_sprites(
    system_path: Path,
    canonical_path: Path,
    representative_output: Path,
) -> tuple[list[np.ndarray], np.ndarray]:
    with np.load(system_path, allow_pickle=False) as data:
        names = data["variable_names"].astype(str)
        initial_variables = np.asarray(data["initial_variables"], dtype=float)
        geometry_data = {
            key: np.asarray(data[key])
            for key in (
                "parent",
                "bond_length",
                "bond_angle",
                "base_torsion",
                "variable_effects",
                "initial_variables",
                "root_position",
                "root_x_axis",
                "root_z_axis",
            )
        }
    with np.load(canonical_path, allow_pickle=False) as data:
        canonical_angles = np.asarray(data["angles"], dtype=float)
    (
        representative_variables,
        canonical_masses,
        canonical_sample_indices,
        center_rms_degrees,
    ) = canonical_cluster_representatives(canonical_angles, names)
    representative_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        representative_output,
        coordinates=representative_variables,
        cluster=np.arange(4, dtype=np.int64),
        canonical_cluster_mass=canonical_masses,
        canonical_sample_index=canonical_sample_indices,
        canonical_center_rms_degrees=center_rms_degrees,
        variable_names=names,
        source_canonical=np.asarray(str(canonical_path)),
        clustering_seed=np.asarray(20260719, dtype=np.int64),
        criterion=np.asarray(
            "strict helix; nearest canonical-MC sample to circular center"
        ),
    )

    coil_variables = initial_variables.copy()
    coil_variables[names == "phi"] = np.deg2rad(-150.0)
    coil_variables[names == "psi"] = np.deg2rad(150.0)
    variables = np.concatenate(
        (coil_variables[None, :], representative_variables)
    )
    cartesian = build_cartesian_coordinates(variables, **geometry_data)
    cartoon_coordinates = common_cartoon_view(cartesian)

    with tempfile.TemporaryDirectory(prefix="ala10-cartoon-") as temporary:
        sprites = render_cartoon_sprites(
            cartoon_coordinates,
            Path(temporary),
        )
    return sprites, canonical_masses


def draw_schematic_panel(
    ax,
    sprites: list[np.ndarray],
    canonical_masses: np.ndarray,
    *,
    panel: str = "A",
    panel_x: float = 0.005,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        panel_x,
        0.90,
        format_panel_label(panel),
        transform=ax.transAxes,
        ha="left",
        va="center",
        color=TEXT_COLOR,
        fontsize=9.5,
        fontweight="bold",
    )
    ax.text(
        0.165,
        0.90,
        "COIL INITIALIZATION",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=TEXT_COLOR,
        fontsize=9.5,
        fontweight="bold",
    )
    ax.text(
        0.725,
        0.90,
        "CANONICAL HELICAL FAMILIES",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=TEXT_COLOR,
        fontsize=9.5,
        fontweight="bold",
    )

    draw_sprite(ax, sprites[0], (0.075, 0.14, 0.18, 0.66))
    ax.add_patch(
        FancyArrowPatch(
            (0.29, 0.46),
            (0.39, 0.46),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.45,
            color=REFERENCE_COLOR,
            shrinkA=0,
            shrinkB=0,
        )
    )

    centers = (0.485, 0.635, 0.785, 0.935)
    for cluster, (center, sprite, mass) in enumerate(
        zip(
            centers,
            sprites[1:],
            canonical_masses,
            strict=True,
        )
    ):
        draw_sprite(ax, sprite, (center - 0.060, 0.19, 0.120, 0.60))
        ax.scatter(
            [center - 0.050],
            [0.105],
            transform=ax.transAxes,
            s=11,
            facecolor=TERMINAL_COLORS[cluster],
            edgecolor="none",
        )
        ax.text(
            center,
            0.105,
            f"H{cluster}  {100.0 * mass:.1f}%",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=TEXT_COLOR,
            fontsize=9.0,
            fontweight="bold",
        )


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    sprites, canonical_masses = prepare_schematic_sprites(
        args.system,
        args.canonical,
        args.representative_output,
    )
    fig, ax = plt.subplots(figsize=(6.69, 1.35))
    fig.subplots_adjust(left=0.025, right=0.99, top=0.97, bottom=0.04)
    draw_schematic_panel(ax, sprites, canonical_masses)
    save_figure(fig, args.output_base)


if __name__ == "__main__":
    main()
