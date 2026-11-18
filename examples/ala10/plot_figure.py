#!/usr/bin/env python3
"""Ala10 main figure: prior-free (blind HDBSCAN) family identification, strict-helix mass,
and canonical-PCA projections of the outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from examples.common.figure_style import (
    FPT_COLOR,
    FPT_FILL_COLOR,
    REFERENCE_COLOR,
    TEXT_COLOR,
    ULA_COLOR,
    ULA_FILL_COLOR,
    clean_axis,
    configure_matplotlib,
    mix_with_white,
    panel_label,
    save_figure,
)
from examples.ala10.plot_panel_a import (
    DEFAULT_CANONICAL,
    DEFAULT_SYSTEM,
    draw_schematic_panel,
    prepare_schematic_sprites,
)


DEFAULT_ANALYSIS = Path("data/ala10/recovery")
DEFAULT_BLIND = Path("data/ala10/identification")
DEFAULT_OUTPUT = Path("figures/ala10_matched_move_main")
LD_RECORDED_FRAMES = 101  # every 200 updates from 0 to 20000
REPLICAS = 4096
DEFAULT_REPRESENTATIVES = Path("figures/ala10_calibrated_recovery_representatives.npz")
METHODS = ("fpt", "ld")
LABELS = {"fpt": "FPT", "ld": "Langevin"}
COLORS = {"fpt": FPT_COLOR, "ld": ULA_COLOR}
PALE_COLORS = {"fpt": FPT_FILL_COLOR, "ld": ULA_FILL_COLOR}
LINESTYLES = {"fpt": "-", "ld": "--"}
MARKERS = {"fpt": "o", "ld": "^"}
CANONICAL_BACKGROUND = mix_with_white(REFERENCE_COLOR, 0.80)
CANONICAL_LINE = mix_with_white(REFERENCE_COLOR, 0.48)
# Opacity transfer per unit of native mass.  Masses are averaged over seeds,
# so accumulated opacity tracks pooled mass independently of the seed count.
MASS_OPACITY_SCALE = 80_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS,
                        help="strict-helix masses, projection, and protocol summary")
    parser.add_argument("--blind-dir", type=Path, default=DEFAULT_BLIND,
                        help="blind HDBSCAN identification audits (audits_*.csv)")
    parser.add_argument("--system", type=Path, default=DEFAULT_SYSTEM)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument(
        "--representative-output",
        type=Path,
        default=DEFAULT_REPRESENTATIVES,
    )
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_method_rows(rows: list[dict]) -> list[dict]:
    integer_fields = (
        "seed",
        "step",
        "repeat",
        "family",
        "left_strict_count",
        "right_strict_count",
        "left_family_count",
        "right_family_count",
        "recovered",
    )
    float_fields = (
        "left_representative_error",
        "right_representative_error",
        "pair_distance",
    )
    for row in rows:
        for field in integer_fields:
            row[field] = int(row[field])
        for field in float_fields:
            row[field] = float(row[field])
    return rows


def parse_reference_rows(rows: list[dict]) -> list[dict]:
    for row in rows:
        row["step"] = int(row["step"])
        row["repeat"] = int(row["repeat"])
        row["family"] = int(row["family"])
        row["recovered"] = int(row["recovered"])
    return rows


def parse_blind_rows(rows: list[dict]) -> list[dict]:
    """Blind-identification audits in the row layout used by the panels."""
    for row in rows:
        row["seed"] = int(row["seed"]) if row["seed"] else None
        for field in ("step", "repeat", "family"):
            row[field] = int(row[field])
        row["recovered"] = int(row["identified"])
    return rows


def parse_mass_rows(rows: list[dict]) -> list[dict]:
    for row in rows:
        row["seed"] = int(row["seed"])
        row["step"] = int(row["step"])
        row["strict_helix_mass"] = float(row["strict_helix_mass"])
    return rows


def subset(rows: list[dict], **conditions) -> list[dict]:
    return [
        row
        for row in rows
        if all(row[field] == value for field, value in conditions.items())
    ]


def recovery_curve(
    rows: list[dict], source: str, seed: int, steps: list[int]
) -> np.ndarray:
    values = []
    for step in steps:
        selected = subset(rows, source=source, seed=seed, step=step)
        values.append(
            4.0 * np.mean([row["recovered"] for row in selected])
        )
    return np.asarray(values)


def plot_recovery_kinetics(
    ax,
    rows: list[dict],
    reference_rows: list[dict],
    summary: dict,
) -> None:
    steps = [int(step) for step in summary["steps"]]
    seeds = [int(seed) for seed in summary["seeds"]]
    for source in METHODS:
        seed_curves = []
        for seed in seeds:
            values = recovery_curve(rows, source, seed, steps)
            seed_curves.append(values)
            ax.plot(
                steps,
                values,
                color=PALE_COLORS[source],
                linewidth=0.85,
                linestyle=LINESTYLES[source],
                alpha=0.82,
                zorder=1,
            )
        mean = np.mean(np.stack(seed_curves), axis=0)
        ax.plot(
            steps,
            mean,
            color=COLORS[source],
            linewidth=1.9,
            linestyle=LINESTYLES[source],
            zorder=4,
        )
        label_y = float(mean[-1])
        if source == "fpt":
            label_y = min(3.72, label_y + 0.12)
        else:
            label_y = max(0.12, label_y + 0.10)
        ax.text(
            19_550,
            label_y,
            LABELS[source],
            ha="right",
            va="bottom",
            color=COLORS[source],
            fontsize=9.0,
        )

    reference_counts = []
    for repeat in sorted({row["repeat"] for row in reference_rows}):
        selected = subset(reference_rows, repeat=repeat)
        reference_counts.append(sum(row["recovered"] for row in selected))
    reference_counts = np.asarray(reference_counts, dtype=float)
    reference_mean = float(np.mean(reference_counts))
    reference_low, reference_high = np.quantile(
        reference_counts, (0.10, 0.90)
    )
    ax.fill_between(
        (0, 20_000),
        reference_low,
        reference_high,
        color=CANONICAL_BACKGROUND,
        alpha=0.55,
        linewidth=0,
        zorder=0,
    )
    ax.axhline(
        reference_mean,
        color=CANONICAL_LINE,
        linewidth=0.9,
        linestyle=":",
        zorder=2,
    )
    ax.text(
        900,
        reference_mean - 0.12,
        "Canonical reference",
        ha="left",
        va="top",
        color=REFERENCE_COLOR,
        fontsize=8.5,
    )
    ax.set_xlim(0, 20_000)
    ax.set_xticks((0, 5_000, 10_000, 15_000, 20_000))
    ax.set_xticklabels(("0", "5k", "10k", "15k", "20k"))
    ax.set_ylim(0, 4.12)
    ax.set_yticks((0, 1, 2, 3, 4))
    ax.set_xlabel(r"$t$ (updates)")
    ax.set_ylabel("Expected identified families")
    ax.set_title("Blind family identification", pad=5)
    clean_axis(ax)


def plot_strict_mass(ax, rows: list[dict], summary: dict) -> None:
    steps = [int(step) for step in summary["steps"]]
    seeds = [int(seed) for seed in summary["seeds"]]
    for source in METHODS:
        curves = []
        for seed in seeds:
            selected = subset(rows, source=source, seed=seed)
            by_step = {row["step"]: row["strict_helix_mass"] for row in selected}
            values = 100.0 * np.asarray([by_step[step] for step in steps])
            curves.append(values)
            ax.plot(
                steps,
                values,
                color=PALE_COLORS[source],
                linewidth=0.8,
                linestyle=LINESTYLES[source],
                alpha=0.75,
                zorder=1,
            )
        mean = np.mean(np.stack(curves), axis=0)
        ax.plot(
            steps,
            mean,
            color=COLORS[source],
            linewidth=1.9,
            linestyle=LINESTYLES[source],
            label=LABELS[source],
            zorder=3,
        )

    canonical_mass = 100.0 * float(summary["canonical"]["strict_mass"])
    ax.axhline(
        canonical_mass,
        color=CANONICAL_LINE,
        linewidth=0.9,
        linestyle=":",
        zorder=0,
    )
    ax.text(
        19_500,
        canonical_mass + 1.15,
        f"Canonical  {canonical_mass:.1f}%",
        ha="right",
        va="bottom",
        color=REFERENCE_COLOR,
        fontsize=8.6,
    )
    ax.set_xlim(0, 20_000)
    ax.set_xticks((0, 5_000, 10_000, 15_000, 20_000))
    ax.set_xticklabels(("0", "5k", "10k", "15k", "20k"))
    ax.set_ylim(0, 66)
    ax.set_yticks((0, 20, 40, 60))
    ax.set_yticklabels(("0%", "20%", "40%", "60%"))
    ax.set_xlabel(r"$t$ (updates)")
    ax.set_ylabel("Strict-helix ensemble mass")
    ax.set_title("Population expansion and retention", pad=5)
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.01, 0.30),
        handlelength=2.0,
    )
    clean_axis(ax)


def endpoint_probability(
    rows: list[dict], source: str, family: int, seed: int | None
) -> float:
    conditions = {"source": source, "family": family, "step": 20_000}
    if seed is not None:
        conditions["seed"] = seed
    selected = subset(rows, **conditions)
    return float(np.mean([row["recovered"] for row in selected]))


def plot_endpoint_recovery(
    ax,
    method_rows: list[dict],
    reference_rows: list[dict],
    summary: dict,
) -> None:
    family_x = np.arange(4, dtype=float)
    seeds = [int(seed) for seed in summary["seeds"]]
    offsets = {"fpt": -0.10, "ld": 0.12}
    styles = {
        "reference": (REFERENCE_COLOR, "s", "Canonical"),
        "fpt": (FPT_COLOR, "o", "FPT"),
        "ld": (ULA_COLOR, "^", "Langevin"),
    }
    pale = {
        "reference": CANONICAL_BACKGROUND,
        "fpt": FPT_FILL_COLOR,
        "ld": ULA_FILL_COLOR,
    }
    # Canonical self-recovery is drawn as a per-family dotted reference level,
    # the same encoding as the canonical lines in panels (b) and (c); the
    # method colors are those of (b) and (c), so the panel needs no key.
    for family in range(4):
        value = endpoint_probability(reference_rows, "reference", family, None)
        ax.plot(
            (family - 0.40, family + 0.40),
            (value, value),
            color=CANONICAL_LINE,
            linewidth=1.1,
            linestyle=":",
            solid_capstyle="butt",
            zorder=1,
        )
    for source in ("fpt", "ld"):
        color, marker, label = styles[source]
        source_rows = method_rows
        if source != "reference":
            for seed_index, seed in enumerate(seeds):
                values = np.asarray(
                    [
                        endpoint_probability(source_rows, source, family, seed)
                        for family in range(4)
                    ]
                )
                jitter = (seed_index - 0.5 * (len(seeds) - 1)) * 0.012
                ax.scatter(
                    family_x + offsets[source] + jitter,
                    values,
                    s=11,
                    marker=marker,
                    facecolor=pale[source],
                    edgecolor=color,
                    linewidth=0.45,
                    alpha=0.88,
                    zorder=2,
                )
        means = np.asarray(
            [
                endpoint_probability(source_rows, source, family, None)
                for family in range(4)
            ]
        )
        ax.scatter(
            family_x + offsets[source],
            means,
            s=30,
            marker=marker,
            facecolor=color,
            edgecolor=color,
            linewidth=0.55,
            label=label,
            zorder=4,
        )

    ax.set_xlim(-0.48, 3.48)
    ax.set_xticks(family_x)
    ax.set_xticklabels(("H0", "H1", "H2", "H3"))
    ax.set_ylim(-0.035, 1.035)
    ax.set_yticks((0.0, 0.25, 0.50, 0.75, 1.0))
    ax.set_yticklabels(("0", ".25", ".50", ".75", "1"))
    ax.set_xlabel("Canonical helical family")
    ax.set_ylabel("Identification probability")
    ax.set_title("Identification at 20k", pad=5)
    # Compact key in the empty band of the H0--H1 columns (0.1--0.45); the
    # canonical entry uses the same dotted reference encoding as (b) and (c).
    ax.legend(
        handles=[
            Line2D([], [], color=CANONICAL_LINE, linewidth=1.1, linestyle=":",
                   label="Canonical"),
            Line2D([], [], linestyle="none", marker="o", markersize=5.2,
                   markerfacecolor=FPT_COLOR, markeredgecolor=FPT_COLOR, label="FPT"),
            Line2D([], [], linestyle="none", marker="^", markersize=5.2,
                   markerfacecolor=ULA_COLOR, markeredgecolor=ULA_COLOR, label="Langevin"),
        ],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.015, 0.085),
        handlelength=1.3,
        handletextpad=0.35,
        labelspacing=0.15,
        borderaxespad=0.0,
        fontsize=8.0,
    )
    clean_axis(ax)


def projection_limits(projection: dict[str, np.ndarray]) -> tuple[tuple, tuple]:
    all_xy = np.vstack(
        (projection["canonical_xy"], projection["fpt_xy"], projection["ld_xy"])
    )
    low = np.min(all_xy, axis=0)
    high = np.max(all_xy, axis=0)
    padding = 0.06 * (high - low)
    return (
        (low[0] - padding[0], high[0] + padding[0]),
        (low[1] - padding[1], high[1] + padding[1]),
    )


def plot_projection(
    ax,
    projection: dict[str, np.ndarray],
    *,
    source: str,
    xlim: tuple,
    ylim: tuple,
    show_ylabel: bool,
    seed_count: int,
    replicas: int,
) -> None:
    canonical_xy = projection["canonical_xy"]
    canonical_label = projection["canonical_label"]
    explained = projection["explained_variance_ratio"]
    ax.scatter(
        canonical_xy[:, 0],
        canonical_xy[:, 1],
        s=1.8,
        color=CANONICAL_BACKGROUND,
        alpha=0.035,
        linewidth=0,
        rasterized=True,
        zorder=1,
    )
    xy = projection[f"{source}_xy"]
    native_mass = projection[f"{source}_weight"]
    color = COLORS[source]
    rgba = np.broadcast_to(
        np.asarray(matplotlib.colors.to_rgba(color)),
        (len(xy), 4),
    ).copy()
    # A common transfer converts native-state mass to opacity in both panels.
    # Because overlaid alpha values compose multiplicatively, this mapping
    # makes local opacity approximately proportional to accumulated mass while
    # remaining bounded in densely populated regions.
    rgba[:, 3] = -np.expm1(-MASS_OPACITY_SCALE * native_mass)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=3.6,
        color=rgba,
        linewidth=0,
        rasterized=True,
        zorder=3,
    )
    for family in range(4):
        center = np.mean(canonical_xy[canonical_label == family], axis=0)
        label = ax.text(
            center[0],
            center[1],
            f"H{family}",
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            color=TEXT_COLOR,
            zorder=6,
        )
        label.set_path_effects(
            [path_effects.withStroke(linewidth=2.1, foreground="white")]
        )
    if source == "fpt":
        title = f"FPT endpoints\nstrict-helix: {len(xy):,} / {seed_count * replicas:,}"
    else:
        total = seed_count * replicas * LD_RECORDED_FRAMES
        title = f"LD pooled archive\nstrict-helix: {len(xy):,} / {total:,}"
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlabel(f"PC1 ({100 * explained[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({100 * explained[1]:.0f}%)" if show_ylabel else "")
    ax.set_title(title, pad=3, fontsize=8.2, linespacing=0.95)
    clean_axis(ax)


def load_projection(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    summary = json.loads(
        (args.analysis_dir / "summary.json").read_text(encoding="utf-8")
    )
    audit_rows = parse_blind_rows(
        [row for path in sorted(args.blind_dir.glob("audits_*.csv")) for row in read_csv(path)]
    )
    reference_rows = [row for row in audit_rows if row["source"] == "reference"]
    method_rows = [row for row in audit_rows if row["source"] in METHODS]
    mass_rows = parse_mass_rows(
        read_csv(args.analysis_dir / "strict_helix_mass.csv")
    )
    projection = load_projection(args.analysis_dir / "projection.npz")
    sprites, canonical_masses = prepare_schematic_sprites(
        args.system,
        args.canonical,
        args.representative_output,
    )

    fig = plt.figure(figsize=(6.69, 6.38))
    outer = fig.add_gridspec(
        3,
        1,
        left=0.095,
        right=0.975,
        bottom=0.075,
        top=0.985,
        height_ratios=(0.65, 1.0, 1.0),
        hspace=0.40,
    )
    upper_results = outer[1].subgridspec(1, 2, wspace=0.30)
    lower_results = outer[2].subgridspec(1, 3, wspace=0.50)
    ax_a = fig.add_subplot(outer[0])
    ax_b = fig.add_subplot(upper_results[0, 0])
    ax_c = fig.add_subplot(upper_results[0, 1])
    ax_d = fig.add_subplot(lower_results[0, 0])
    ax_e = fig.add_subplot(lower_results[0, 1])
    ax_f = fig.add_subplot(lower_results[0, 2])

    draw_schematic_panel(ax_a, sprites, canonical_masses, panel="A", panel_x=-0.07)
    plot_recovery_kinetics(ax_b, method_rows, reference_rows, summary)
    plot_strict_mass(ax_c, mass_rows, summary)
    plot_endpoint_recovery(ax_d, method_rows, reference_rows, summary)
    xlim, ylim = projection_limits(projection)
    plot_projection(
        ax_e,
        projection,
        source="fpt",
        xlim=xlim,
        ylim=ylim,
        show_ylabel=True,
        seed_count=len(summary["seeds"]),
        replicas=REPLICAS,
    )
    plot_projection(
        ax_f,
        projection,
        source="ld",
        xlim=xlim,
        ylim=ylim,
        show_ylabel=False,
        seed_count=len(summary["seeds"]),
        replicas=REPLICAS,
    )
    ax_d.set_box_aspect(1.0)
    for lower_axis in (ax_d, ax_e, ax_f):
        lower_axis.xaxis.set_label_coords(0.5, -0.17)
    panel_label(ax_b, "B")
    panel_label(ax_c, "C")
    panel_label(ax_d, "D", x=-0.23)
    panel_label(ax_e, "E", x=-0.23)
    panel_label(ax_f, "F", x=-0.23)
    save_figure(fig, args.output_base)
    print(args.output_base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
