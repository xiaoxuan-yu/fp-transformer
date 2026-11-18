"""Create publication-ready main and SI figures for the PoseBusters run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path



import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
from scipy.ndimage import gaussian_filter

from examples.vina.pdbqt import build_pdbqt_potential
from examples.vina import statistical_analysis
from examples.vina.dataset import load_posebusters_systems


# White-based publication palette.  Pale fields are explicitly mixed with
# white so their appearance does not depend on alpha compositing or the color
# beneath them.  Landscapes use a separate ice-porcelain teal family.
def _mix_with_white(hex_color: str, white_fraction: float) -> str:
    if not 0.0 <= white_fraction <= 1.0:
        raise ValueError("white_fraction must lie in [0, 1]")
    value = hex_color.removeprefix("#")
    channels = [int(value[offset : offset + 2], 16) for offset in (0, 2, 4)]
    mixed = [
        round(channel * (1.0 - white_fraction) + 255 * white_fraction)
        for channel in channels
    ]
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


CANVAS_COLOR = "#FFFFFF"
MC_COLOR = "#3B5CA8"
FPT_COLOR = "#155B55"
BOTH_COLOR = "#5B5551"
NEITHER_COLOR = "#9A5728"
SAGE_COLOR = "#5D744B"
TEXT_COLOR = "#2F2B29"

LANDSCAPE_COLOR = "#3F8F90"
LANDSCAPE_OUTER_COLOR = _mix_with_white(LANDSCAPE_COLOR, 0.92)
LANDSCAPE_LOW_COLOR = _mix_with_white(LANDSCAPE_COLOR, 0.82)
LANDSCAPE_MID_COLOR = _mix_with_white(LANDSCAPE_COLOR, 0.68)
LANDSCAPE_HIGH_COLOR = _mix_with_white(LANDSCAPE_COLOR, 0.45)
GRID_COLOR = _mix_with_white(LANDSCAPE_COLOR, 0.91)
PALE_FILL = LANDSCAPE_OUTER_COLOR

MC_FILL_COLOR = _mix_with_white(MC_COLOR, 0.72)
FPT_FILL_COLOR = _mix_with_white(FPT_COLOR, 0.72)
SAGE_FILL_COLOR = _mix_with_white(SAGE_COLOR, 0.72)
MC_WASH_COLOR = _mix_with_white(MC_COLOR, 0.88)
FPT_WASH_COLOR = _mix_with_white(FPT_COLOR, 0.88)
MC_MARKER_COLOR = _mix_with_white(MC_COLOR, 0.44)
FPT_MARKER_COLOR = _mix_with_white(FPT_COLOR, 0.20)
BOTH_FILL_COLOR = LANDSCAPE_LOW_COLOR
BOTH_MARKER_COLOR = "#A9A3A0"
NEITHER_FILL_COLOR = _mix_with_white(NEITHER_COLOR, 0.72)
WARM_ACCENT_COLOR = "#A87526"
ELEMENT_N_COLOR = "#4B7186"
ELEMENT_O_COLOR = "#A13F3B"
TORSION_BINS = ("1-2", "3-5", "6-7", "8-12", "13+")
SCHEMATIC_SYSTEM = "7elt_tym"
SCHEMATIC_ASSET = (
    Path(__file__).resolve().parent
    / "assets"
    / "torsion_sampling_pocket_gpt_v2.png"
)
PYMOL_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "pymol_illustrative"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _endpoint_prefix(seed_row: dict[str, str], score: str, level: str) -> str:
    levels = ("easy", "medium", "strict")
    keys = [
        key
        for key in seed_row
        if key.startswith(f"{score}_le_") and key.endswith("_first_hit_step")
    ]
    if len(keys) != len(levels):
        raise RuntimeError(
            f"expected {len(levels)} {score} first-hit columns, got {keys}"
        )
    return keys[levels.index(level)].removesuffix("_first_hit_step")


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Nimbus Sans",
            "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "axes.linewidth": 0.7,
            "axes.edgecolor": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlelocation": "left",
            "axes.titleweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Nimbus Sans",
            "mathtext.it": "Nimbus Sans:italic",
            "mathtext.bf": "Nimbus Sans:bold",
            "savefig.dpi": 600,
            "savefig.bbox": None,
            "figure.facecolor": CANVAS_COLOR,
            "axes.facecolor": CANVAS_COLOR,
            "savefig.facecolor": CANVAS_COLOR,
            "savefig.transparent": False,
        }
    )


def _load_run(output_dir: Path) -> dict:
    suite = json.loads((output_dir / "suite_summary.json").read_text())
    analysis = json.loads((output_dir / "analysis_summary.json").read_text())
    protocol = suite["protocol"]
    strict_rows = [
        row
        for row in suite["rows"]
        if row["score"] == "sampling_energy"
        and row["threshold_level"] == "strict"
    ]
    if len(strict_rows) != len(protocol["systems"]):
        raise RuntimeError("incomplete strict sampling-energy results")

    systems = []
    runs = []
    for row in strict_rows:
        name = row["system"]
        system_dir = output_dir / name
        aggregate = json.loads((system_dir / "aggregate.json").read_text())
        metadata = aggregate["metadata"]
        raw_seed_rows = _read_csv(system_dir / "seed_summary.csv")
        seed_rows = {
            method: tuple(
                sorted(
                    (seed_row for seed_row in raw_seed_rows if seed_row["method"] == method),
                    key=lambda seed_row: int(seed_row["seed"]),
                )
            )
            for method in ("mc", "fpt")
        }
        if any(len(rows) != protocol["num_seeds"] for rows in seed_rows.values()):
            raise RuntimeError(f"{name}: incomplete paired seed summaries")
        if [seed_row["seed"] for seed_row in seed_rows["mc"]] != [
            seed_row["seed"] for seed_row in seed_rows["fpt"]
        ]:
            raise RuntimeError(f"{name}: MC/FPT seed mismatch")
        threshold = float(row["threshold"])
        mc_hit = _optional_float(row["mc_median_hit_step"])
        fpt_hit = _optional_float(row["fpt_median_hit_step"])
        mc_success = float(row["mc_success_rate"]) >= 0.5
        fpt_success = float(row["fpt_success_rate"]) >= 0.5
        if mc_success and fpt_success:
            outcome = "both"
        elif mc_success:
            outcome = "mc_only"
        elif fpt_success:
            outcome = "fpt_only"
        else:
            outcome = "neither"
        trajectories = _read_csv(system_dir / "trajectories.csv")
        paired_rows = list(zip(seed_rows["mc"], seed_rows["fpt"]))
        total_improvements = np.asarray(
            [
                float(mc_row["best_total"]) - float(fpt_row["best_total"])
                for mc_row, fpt_row in paired_rows
            ]
        )
        binding_improvements = np.asarray(
            [
                float(mc_row["best_binding"]) - float(fpt_row["best_binding"])
                for mc_row, fpt_row in paired_rows
            ]
        )
        method_medians = {
            method: {
                metric: float(np.median([float(seed_row[metric]) for seed_row in rows]))
                for metric in ("best_total", "best_binding", "elapsed_seconds")
            }
            for method, rows in seed_rows.items()
        }
        for mc_row, fpt_row in paired_rows:
            prefixes = {
                "mc": _endpoint_prefix(mc_row, "sampling_energy", "strict"),
                "fpt": _endpoint_prefix(fpt_row, "sampling_energy", "strict"),
            }
            if len(set(prefixes.values())) != 1:
                raise RuntimeError(f"{name}: mismatched strict threshold columns")
            prefix = prefixes["mc"]
            run = {
                "system": name,
                "seed": int(mc_row["seed"]),
                "torsions": int(row["torsions"]),
            }
            for method, seed_row in (("mc", mc_row), ("fpt", fpt_row)):
                hit_step = _optional_float(seed_row[f"{prefix}_first_hit_step"])
                if hit_step is not None and hit_step < 0:
                    hit_step = None
                run[f"{method}_hit_step"] = hit_step
                run[f"{method}_hit_seconds"] = _optional_float(
                    seed_row[f"{prefix}_first_hit_seconds_estimate"]
                )
                run[f"{method}_elapsed_seconds"] = float(
                    seed_row["elapsed_seconds"]
                )
            runs.append(run)
        systems.append(
            {
                "name": name,
                "torsions": int(row["torsions"]),
                "threshold": threshold,
                "binding_threshold": float(metadata["binding_thresholds"][2]),
                "mc_hit_step": mc_hit,
                "fpt_hit_step": fpt_hit,
                "mc_hit_seconds": _optional_float(
                    row["mc_median_hit_seconds_estimate"]
                ),
                "fpt_hit_seconds": _optional_float(
                    row["fpt_median_hit_seconds_estimate"]
                ),
                "outcome": outcome,
                "mc_success_rate": float(row["mc_success_rate"]),
                "fpt_success_rate": float(row["fpt_success_rate"]),
                "mc_total_gap": method_medians["mc"]["best_total"] - threshold,
                "fpt_total_gap": method_medians["fpt"]["best_total"] - threshold,
                "mc_binding_gap": method_medians["mc"]["best_binding"]
                - float(metadata["binding_thresholds"][2]),
                "fpt_binding_gap": method_medians["fpt"]["best_binding"]
                - float(metadata["binding_thresholds"][2]),
                "mc_total_best": method_medians["mc"]["best_total"],
                "fpt_total_best": method_medians["fpt"]["best_total"],
                "mc_binding_best": method_medians["mc"]["best_binding"],
                "fpt_binding_best": method_medians["fpt"]["best_binding"],
                "total_improvement": float(np.median(total_improvements)),
                "binding_improvement": float(np.median(binding_improvements)),
                "mc_elapsed_seconds": method_medians["mc"]["elapsed_seconds"],
                "fpt_elapsed_seconds": method_medians["fpt"]["elapsed_seconds"],
                "mc_archive_count": float(row["mc_median_archive_count"]),
                "fpt_archive_count": float(row["fpt_median_archive_count"]),
                "mc_unique_clusters": float(row["mc_median_unique_clusters"]),
                "fpt_unique_clusters": float(row["fpt_median_unique_clusters"]),
                "trajectories": trajectories,
            }
        )
    return {
        "suite": suite,
        "analysis": analysis,
        "systems": systems,
        "runs": runs,
    }


def _clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(CANVAS_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.65, alpha=1.0)
    ax.set_axisbelow(True)


def _format_panel_label(label: str) -> str:
    stripped = label.strip()
    if len(stripped) == 1 and stripped.isalpha():
        return f"({stripped.lower()})"
    return label


def _panel_label(
    ax,
    label: str,
    x: float = -0.18,
    y: float | None = None,
) -> None:
    if y is None:
        title = ax.title
        _, title_y = title.get_position()
        transform = title.get_transform()
        vertical_alignment = title.get_verticalalignment()
        fontsize = title.get_fontsize()
    else:
        title_y = y
        transform = ax.transAxes
        vertical_alignment = "baseline"
        fontsize = 10
    ax.text(
        x,
        title_y,
        _format_panel_label(label),
        transform=transform,
        fontsize=fontsize,
        fontweight="bold",
        va=vertical_alignment,
        ha="left",
        color=TEXT_COLOR,
    )


def _save_figure(fig, output_base: Path) -> None:
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=600)
    fig.savefig(
        output_base.with_suffix(".tif"),
        dpi=600,
        format="tiff",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _plot_anytime_coverage(ax, data: dict) -> None:
    runs = data["runs"]
    steps = int(data["suite"]["protocol"]["steps"])
    grid = np.arange(steps + 1)
    for method, color, label in (
        ("mc", MC_COLOR, "MC"),
        ("fpt", FPT_COLOR, "FPT"),
    ):
        coverage, _, _ = _clustered_coverage_interval(
            runs, f"{method}_hit_step", grid
        )
        ax.step(
            grid,
            coverage,
            where="post",
            color=color,
            linestyle="--" if method == "mc" else "-",
            label=label,
        )
    ax.set_xlim(0, steps)
    ax.set_ylim(0, 103)
    ax.set_xticks(np.arange(0, steps + 1, 250))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{value}%" for value in np.arange(0, 101, 20)])
    ax.set_xlabel(r"$t$ (steps)")
    ax.set_ylabel(r"$P_{\mathrm{hit}}(t)$")
    ax.set_title("Anytime target coverage", pad=6)
    ax.legend(frameon=False, loc="lower right")
    _clean_axis(ax)


def _plot_final_quality(ax, data: dict) -> None:
    systems = data["systems"]
    style = {
        "both": dict(
            label="Both hit",
            marker="o",
            facecolor=BOTH_MARKER_COLOR,
            edgecolor=BOTH_COLOR,
            linewidth=0.65,
            alpha=1.0,
            s=16,
            zorder=2,
        ),
        "fpt_only": dict(
            label="FPT only",
            marker="v",
            facecolor=FPT_COLOR,
            edgecolor=TEXT_COLOR,
            linewidth=0.65,
            alpha=1.0,
            s=34,
            zorder=4,
        ),
        "mc_only": dict(
            label="MC only",
            marker="^",
            facecolor=MC_COLOR,
            edgecolor=TEXT_COLOR,
            linewidth=0.65,
            alpha=1.0,
            s=34,
            zorder=4,
        ),
        "neither": dict(
            label="Neither",
            marker="D",
            facecolor="none",
            edgecolor=NEITHER_COLOR,
            linewidth=1.2,
            alpha=1.0,
            s=32,
            zorder=5,
        ),
    }
    for outcome in ("both", "fpt_only", "mc_only", "neither"):
        selected = [system for system in systems if system["outcome"] == outcome]
        if not selected:
            continue
        ax.scatter(
            [system["mc_total_gap"] for system in selected],
            [system["fpt_total_gap"] for system in selected],
            **style[outcome],
        )
    lower, upper = -3.0, 8.5
    ax.plot([lower, upper], [lower, upper], color=TEXT_COLOR, linestyle="--", linewidth=0.8)
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.axvline(0, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\Delta E_{\mathrm{MC}}$ (kcal/mol)")
    ax.set_ylabel(r"$\Delta E_{\mathrm{FPT}}$ (kcal/mol)")
    ax.set_title("Final quality at equal budget", pad=6)
    ax.legend(
        frameon=False,
        loc="upper left",
        handletextpad=0.4,
        borderaxespad=0.2,
        labelspacing=0.3,
    )
    _clean_axis(ax)


def _strict_torsion_rows(data: dict) -> list[dict]:
    rows = [
        row
        for row in data["analysis"]["torsion_bins"]
        if row["score"] == "sampling_energy"
        and row["threshold_level"] == "strict"
    ]
    indexed = {row["torsion_bin"]: row for row in rows}
    return [indexed[label] for label in TORSION_BINS]


def _plot_complexity(top_ax, bottom_ax, data: dict) -> None:
    rows = _strict_torsion_rows(data)
    x = np.arange(len(rows))
    width = 0.36
    mc = [100 * row["mc_successes"] / row.get("trials", row["systems"]) for row in rows]
    fpt = [100 * row["fpt_successes"] / row.get("trials", row["systems"]) for row in rows]
    top_ax.bar(
        x - width / 2,
        mc,
        width,
        color=MC_FILL_COLOR,
        edgecolor=MC_COLOR,
        linewidth=0.65,
        label="MC",
    )
    top_ax.bar(
        x + width / 2,
        fpt,
        width,
        color=FPT_FILL_COLOR,
        edgecolor=FPT_COLOR,
        linewidth=0.65,
        label="FPT",
    )
    top_ax.set_ylim(0, 110)
    top_ax.set_yticks((0, 50, 100))
    top_ax.set_yticklabels(("0%", "50%", "100%"))
    top_ax.set_xticks([])
    top_ax.set_ylabel(r"$P_{\mathrm{hit}}$")
    top_ax.set_title("Dependence on torsional complexity", pad=6)
    top_ax.legend(frameon=False, loc="lower left", ncol=2, columnspacing=0.8)
    _clean_axis(top_ax)

    step_speedups = [
        row["median_step_speedup_mc_over_fpt_positive_pairs"] for row in rows
    ]
    time_speedups = [row["capped_time_speedup_mc_over_fpt"] for row in rows]
    bottom_ax.plot(
        x,
        step_speedups,
        color=FPT_COLOR,
        marker="s",
        markersize=3.8,
        linestyle="--",
        label="Median step speedup",
    )
    bottom_ax.plot(
        x,
        time_speedups,
        color=NEITHER_COLOR,
        marker="o",
        markersize=4,
        label="Capped-time speedup",
    )
    bottom_ax.axhline(1, color=TEXT_COLOR, linestyle="--", linewidth=0.8)
    bottom_ax.set_yscale("log")
    bottom_ax.set_ylim(0.018, 5.2)
    bottom_ax.set_yticks((0.03, 0.1, 0.3, 1, 3))
    bottom_ax.set_yticklabels(("0.03", "0.1", "0.3", "1", "3"))
    bottom_ax.set_xticks(x)
    bottom_ax.set_xticklabels(TORSION_BINS)
    bottom_ax.set_xlabel(r"$N_{\mathrm{tor}}$")
    bottom_ax.set_ylabel(r"$S_{\mathrm{MC/FPT}}$")
    bottom_ax.legend(
        frameon=False,
        loc="lower right",
        handlelength=1.4,
        handletextpad=0.35,
        labelspacing=0.25,
    )
    _clean_axis(bottom_ax)


def plot_main_figure(data: dict, figure_dir: Path) -> None:
    fig = plt.figure(figsize=(6.69, 2.83), constrained_layout=True)
    outer = fig.add_gridspec(1, 3, width_ratios=(1.08, 1.0, 1.14), wspace=0.22)
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    right = outer[0, 2].subgridspec(2, 1, height_ratios=(1.08, 1), hspace=0.08)
    ax_c_top = fig.add_subplot(right[0, 0])
    ax_c_bottom = fig.add_subplot(right[1, 0])
    _plot_anytime_coverage(ax_a, data)
    _plot_final_quality(ax_b, data)
    _plot_complexity(ax_c_top, ax_c_bottom, data)
    _panel_label(ax_a, "A")
    _panel_label(ax_b, "B")
    _panel_label(ax_c_top, "C", x=-0.16, y=1.38)
    _save_figure(fig, figure_dir / "main_figure_posebusters")


def _plot_anytime_wall(ax, data: dict) -> None:
    runs = data["runs"]
    maximum = max(
        max(run[f"{method}_elapsed_seconds"] for run in runs)
        for method in ("mc", "fpt")
    )
    grid = np.concatenate(([0.0], np.geomspace(1e-6, maximum, 600)))
    for method, color, label in (
        ("mc", MC_COLOR, "MC"),
        ("fpt", FPT_COLOR, "FPT"),
    ):
        coverage, _, _ = _clustered_coverage_interval(
            runs, f"{method}_hit_seconds", grid
        )
        ax.step(
            grid,
            coverage,
            where="post",
            color=color,
            linestyle="--" if method == "mc" else "-",
            label=label,
        )
    ax.set_xscale("symlog", linthresh=1e-4, linscale=0.8)
    ax.set_xlim(0, maximum * 1.08)
    ax.set_ylim(0, 103)
    ticks = (0, 1e-4, 1e-3, 1e-2, 1e-1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(("0", "0.1", "1", "10", "100"))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{value}%" for value in np.arange(0, 101, 20)])
    ax.set_xlabel(r"$t_{\mathrm{wall}}$ (ms; symlog)")
    ax.set_ylabel(r"$P_{\mathrm{hit}}(t)$")
    ax.set_title("Wall-time target coverage", pad=6)
    ax.legend(frameon=False, loc="lower right")
    _clean_axis(ax)


def _torsion_bin_for_count(torsions: int) -> str:
    if torsions <= 2:
        return "1-2"
    if torsions <= 5:
        return "3-5"
    if torsions <= 7:
        return "6-7"
    if torsions <= 12:
        return "8-12"
    return "13+"


def _plot_final_improvement_by_torsions(ax, data: dict) -> None:
    rng = np.random.default_rng(6300)
    grouped = []
    for label in TORSION_BINS:
        grouped.append(
            [
                system["total_improvement"]
                for system in data["systems"]
                if _torsion_bin_for_count(system["torsions"]) == label
            ]
        )
    boxes = ax.boxplot(
        grouped,
        positions=np.arange(len(grouped)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": TEXT_COLOR, "linewidth": 1.2},
        whiskerprops={"color": BOTH_COLOR, "linewidth": 0.8},
        capprops={"color": BOTH_COLOR, "linewidth": 0.8},
        boxprops={"color": BOTH_COLOR, "linewidth": 0.8},
    )
    for box in boxes["boxes"]:
        box.set_facecolor(PALE_FILL)
    for index, values in enumerate(grouped):
        jitter = rng.normal(0, 0.055, len(values))
        ax.scatter(
            index + jitter,
            values,
            color=FPT_MARKER_COLOR,
            alpha=1.0,
            edgecolor=FPT_COLOR,
            linewidth=0.65,
            s=12,
            zorder=2,
        )
    ax.axhline(0, color=TEXT_COLOR, linestyle="--", linewidth=0.8)
    ax.set_xlim(-0.55, len(grouped) - 0.45)
    ax.set_ylim(-0.5, 7.6)
    ax.set_xticks(np.arange(len(grouped)))
    ax.set_xticklabels(TORSION_BINS)
    ax.set_xlabel(r"$N_{\mathrm{tor}}$")
    ax.set_ylabel(r"$E_{\mathrm{MC}}-E_{\mathrm{FPT}}$ (kcal/mol)")
    ax.set_title("Fixed-budget quality gain", pad=6)
    _clean_axis(ax)


def _plot_cluster_diversity(ax, data: dict) -> None:
    joint = [system for system in data["systems"] if system["outcome"] == "both"]
    categories = {
        "FPT more": [
            system
            for system in joint
            if system["fpt_unique_clusters"] > system["mc_unique_clusters"]
        ],
        "MC more": [
            system
            for system in joint
            if system["mc_unique_clusters"] > system["fpt_unique_clusters"]
        ],
        "Tie": [
            system
            for system in joint
            if system["mc_unique_clusters"] == system["fpt_unique_clusters"]
        ],
    }
    styles = {
        "FPT more": (FPT_COLOR, "o"),
        "MC more": (MC_COLOR, "s"),
        "Tie": (BOTH_COLOR, "x"),
    }
    for label, selected in categories.items():
        color, marker = styles[label]
        ax.scatter(
            [system["mc_unique_clusters"] for system in selected],
            [system["fpt_unique_clusters"] for system in selected],
            color=color,
            marker=marker,
            alpha=1.0,
            s=17,
            linewidth=0.7,
            label=label,
        )
    limits = (0.8, 650)
    ax.plot(limits, limits, color=TEXT_COLOR, linestyle="--", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ticks = (1, 4, 16, 64, 256)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(value) for value in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(value) for value in ticks])
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$N_{\mathrm{clust}}^{\mathrm{MC}}$")
    ax.set_ylabel(r"$N_{\mathrm{clust}}^{\mathrm{FPT}}$")
    ax.set_title("Diversity among joint successes", pad=6)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3, labelspacing=0.25)
    _clean_axis(ax)


def _wilson_interval(count: np.ndarray, total: int) -> tuple[np.ndarray, np.ndarray]:
    proportion = count / total
    z = 1.959963984540054
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return 100 * (center - half_width), 100 * (center + half_width)


def _clustered_coverage_interval(
    runs: list[dict], field: str, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    system_names = sorted({run["system"] for run in runs})
    system_rates = []
    for name in system_names:
        hits = np.asarray(
            [
                np.inf if run[field] is None else float(run[field])
                for run in runs
                if run["system"] == name
            ]
        )
        system_rates.append(np.mean(hits[:, None] <= grid[None, :], axis=0))
    system_rates = np.asarray(system_rates)
    coverage = 100 * np.mean(system_rates, axis=0)
    if len(system_rates) <= 1:
        return coverage, coverage, coverage
    radius = 1.96 * 100 * np.std(system_rates, axis=0, ddof=1) / np.sqrt(
        len(system_rates)
    )
    return coverage, np.maximum(0, coverage - radius), np.minimum(100, coverage + radius)


def _publication_panel(ax, label: str, title: str) -> None:
    ax.set_title(title, loc="left", pad=7, fontweight="bold")
    _panel_label(ax, label, x=-0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(CANVAS_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.65, alpha=1.0)
    ax.set_axisbelow(True)


def _schematic_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.63,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax.text(
        x + width / 2,
        y + height * 0.30,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.0,
        color=TEXT_COLOR,
    )


def _schematic_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = BOTH_COLOR,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=color,
            connectionstyle="arc3,rad=0.0",
            shrinkA=2,
            shrinkB=2,
            clip_on=False,
        )
    )


def _schematic_arrow_between(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    padding: float,
) -> None:
    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(delta))
    if length <= 2 * padding:
        return
    direction = delta / length
    padded_start = tuple(np.asarray(start, dtype=float) + padding * direction)
    padded_end = tuple(np.asarray(end, dtype=float) - padding * direction)
    _schematic_arrow(ax, padded_start, padded_end, color=color)


def _schematic_panel_header(ax, reference_ax=None) -> None:
    """Draw the schematic header, optionally aligned to another panel."""
    if reference_ax is None:
        transform = ax.transAxes
        label_x, title_x, header_y = -0.035, 0.035, 1.035
    else:
        figure = ax.get_figure()
        # The square aspect of panel (d) changes its active axes position.  A
        # draw resolves that position before its title transform is reused.
        figure.canvas.draw()
        reference_title = reference_ax._left_title
        reference_title_x, reference_title_y = reference_title.get_position()
        reference_transform = reference_title.get_transform()
        label_display = reference_transform.transform((-0.2, reference_title_y))
        title_display = reference_transform.transform(
            (reference_title_x, reference_title_y)
        )
        header_display_y = ax.transAxes.transform((0.0, 1.035))[1]
        inverse_figure = figure.transFigure.inverted()
        label_x = inverse_figure.transform(label_display)[0]
        title_x = inverse_figure.transform(title_display)[0]
        header_y = inverse_figure.transform((0.0, header_display_y))[1]
        transform = figure.transFigure

    text_kwargs = {
        "transform": transform,
        "fontsize": 9.5,
        "fontweight": "bold",
        "va": "baseline",
        "ha": "left",
        "color": TEXT_COLOR,
    }
    ax.text(label_x, header_y, _format_panel_label("A"), **text_kwargs)
    ax.text(title_x, header_y, "Torsion-only sampling protocol", **text_kwargs)


def _plot_sampling_schematic_vector(ax, *, header_reference_ax=None) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _schematic_panel_header(ax, reference_ax=header_reference_ax)

    receptor = FancyBboxPatch(
        (0.045, 0.655),
        0.235,
        0.245,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=PALE_FILL,
        edgecolor=GRID_COLOR,
        linewidth=0.8,
    )
    ax.add_patch(receptor)
    pocket_x = (0.065, 0.085, 0.105, 0.125, 0.145, 0.165, 0.19, 0.215, 0.245)
    pocket_y = (0.835, 0.805, 0.825, 0.785, 0.765, 0.79, 0.77, 0.805, 0.82)
    ax.plot(
        pocket_x,
        pocket_y,
        transform=ax.transAxes,
        color=SAGE_COLOR,
        linewidth=2.3,
        solid_capstyle="round",
    )
    ligand = ((0.095, 0.745), (0.135, 0.725), (0.175, 0.745), (0.215, 0.715))
    ax.plot(
        [point[0] for point in ligand],
        [point[1] for point in ligand],
        transform=ax.transAxes,
        color=TEXT_COLOR,
        linewidth=1.4,
        zorder=3,
    )
    for index, point in enumerate(ligand):
        ax.add_patch(
            Circle(
                point,
                0.010 if index in (0, 3) else 0.012,
                transform=ax.transAxes,
                facecolor=FPT_COLOR if index in (1, 2) else BOTH_COLOR,
                edgecolor="white",
                linewidth=0.65,
                zorder=4,
            )
        )
    for center in ((0.135, 0.725), (0.175, 0.745)):
        ax.add_patch(
            FancyArrowPatch(
                (center[0] - 0.018, center[1] + 0.025),
                (center[0] + 0.018, center[1] + 0.025),
                transform=ax.transAxes,
                connectionstyle="arc3,rad=-0.65",
                arrowstyle="-|>",
                mutation_scale=6,
                linewidth=0.7,
                color=FPT_COLOR,
            )
        )
    ax.text(
        0.162,
        0.685,
        "Fixed receptor and pose",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.0,
        color=TEXT_COLOR,
    )

    _schematic_box(
        ax,
        0.355,
        0.705,
        0.27,
        0.145,
        "Torsion state",
        "same random angle seeds",
        PALE_FILL,
        GRID_COLOR,
    )
    _schematic_arrow(ax, (0.285, 0.775), (0.35, 0.775))

    _schematic_box(
        ax,
        0.075,
        0.325,
        0.36,
        0.225,
        "Vanilla MC",
        "independent proposal + accept/reject",
        MC_FILL_COLOR,
        MC_COLOR,
    )
    _schematic_box(
        ax,
        0.565,
        0.325,
        0.36,
        0.225,
        "FPT sampler",
        "score gradient + force kernel",
        FPT_FILL_COLOR,
        FPT_COLOR,
    )
    _schematic_arrow(ax, (0.47, 0.70), (0.26, 0.56))
    _schematic_arrow(ax, (0.51, 0.70), (0.74, 0.56))
    ax.text(
        0.49,
        0.615,
        "torsions only",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.0,
        color=BOTH_COLOR,
    )

    for x, y in ((0.13, 0.39), (0.19, 0.45), (0.27, 0.39), (0.35, 0.46)):
        ax.add_patch(
            Circle(
                (x, y),
                0.012,
                transform=ax.transAxes,
                facecolor=MC_COLOR,
                edgecolor="white",
                linewidth=0.65,
                zorder=4,
            )
        )
    for start, end in (
        ((0.13, 0.39), (0.17, 0.42)),
        ((0.27, 0.39), (0.32, 0.43)),
    ):
        _schematic_arrow(ax, start, end)

    fpt_points = ((0.62, 0.40), (0.70, 0.46), (0.79, 0.39), (0.87, 0.46))
    for left, right in zip(fpt_points[:-1], fpt_points[1:]):
        ax.plot(
            (left[0], right[0]),
            (left[1], right[1]),
            transform=ax.transAxes,
            color=FPT_FILL_COLOR,
            linewidth=0.7,
            alpha=1.0,
        )
    for point in fpt_points:
        ax.add_patch(
            Circle(
                point,
                0.012,
                transform=ax.transAxes,
                facecolor=FPT_COLOR,
                edgecolor="white",
                linewidth=0.65,
                zorder=4,
            )
        )

    _schematic_box(
        ax,
        0.285,
        0.065,
        0.43,
        0.145,
        "Shared evaluation",
        "Vina score • equal budget • time-to-target",
        SAGE_FILL_COLOR,
        SAGE_COLOR,
    )
    _schematic_arrow(ax, (0.25, 0.315), (0.40, 0.215))
    _schematic_arrow(ax, (0.75, 0.315), (0.60, 0.215))


def _draw_ligand_glyph(
    ax,
    center: tuple[float, float],
    scale: float,
    torsions: tuple[float, float],
    color: str,
    alpha: float = 1.0,
    zorder: int = 5,
) -> None:
    """Draw one fixed-topology ligand; only the two chain angles vary."""
    cx, cy = center
    ring_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring = np.column_stack(
        (cx + scale * np.cos(ring_angles), cy + scale * np.sin(ring_angles))
    )
    segment = 1.12 * scale
    chain = [ring[0]]
    directions = (0.0, torsions[0], torsions[0] + torsions[1])
    for direction in directions:
        previous = chain[-1]
        chain.append(
            previous
            + segment * np.array((np.cos(direction), np.sin(direction)))
        )
    nodes = np.vstack((ring, np.array(chain[1:])))
    bonds = [(index, (index + 1) % 6) for index in range(6)]
    bonds.extend(((0, 6), (6, 7), (7, 8)))
    for left, right in bonds:
        ax.plot(
            (nodes[left, 0], nodes[right, 0]),
            (nodes[left, 1], nodes[right, 1]),
            color=color,
            alpha=alpha,
            linewidth=max(0.65, 22 * scale),
            solid_capstyle="round",
            zorder=zorder,
        )
    radius = 0.18 * scale
    for x, y in nodes:
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                transform=ax.transData,
                facecolor=color,
                edgecolor="white",
                linewidth=0.65,
                alpha=alpha,
                zorder=zorder + 1,
            )
        )


def _plot_sampling_schematic_abstract(ax, *, header_reference_ax=None) -> None:
    if not SCHEMATIC_ASSET.exists():
        _plot_sampling_schematic_vector(
            ax,
            header_reference_ax=header_reference_ax,
        )
        return
    image = plt.imread(SCHEMATIC_ASSET)
    pocket_crop = image[:720]
    ax.imshow(
        pocket_crop,
        extent=(0.08, 0.92, 0.69, 0.98),
        aspect="auto",
        interpolation="lanczos",
        zorder=0,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _schematic_panel_header(ax, reference_ax=header_reference_ax)
    _draw_ligand_glyph(ax, (0.47, 0.835), 0.036, (0.48, -1.02), TEXT_COLOR)
    ax.text(
        0.50,
        0.705,
        "Fixed receptor + rigid pose  •  torsions only",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.0,
        color=TEXT_COLOR,
    )

    label_box = {
        "boxstyle": "round,pad=0.20",
        "facecolor": "white",
        "linewidth": 0.6,
        "alpha": 0.96,
    }
    ax.text(
        0.225,
        0.615,
        "Vanilla MC\nsingle trajectory",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.0,
        color=MC_COLOR,
        fontweight="bold",
        linespacing=1.15,
        bbox={**label_box, "edgecolor": MC_COLOR},
    )
    ax.text(
        0.775,
        0.615,
        "FPT sampler\nensemble transport",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.0,
        color=FPT_COLOR,
        fontweight="bold",
        linespacing=1.15,
        bbox={**label_box, "edgecolor": FPT_COLOR},
    )

    mc_centers = ((0.15, 0.515), (0.27, 0.425), (0.17, 0.325), (0.27, 0.225))
    mc_torsions = ((0.25, -0.75), (0.75, -1.20), (-0.45, 0.95), (0.40, 0.65))
    _schematic_arrow_between(ax, (0.46, 0.73), mc_centers[0], MC_COLOR, 0.04)
    for start, end in zip(mc_centers[:-1], mc_centers[1:]):
        _schematic_arrow_between(ax, start, end, MC_COLOR, 0.04)
    for center, torsions in zip(mc_centers, mc_torsions):
        _draw_ligand_glyph(ax, center, 0.026, torsions, MC_COLOR)
    ax.text(
        0.23,
        0.555,
        "proposal / accept-reject",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.0,
        color=MC_COLOR,
    )

    ensemble_x = (0.655, 0.775, 0.895)
    ensemble_y = (0.505, 0.365, 0.225)
    ensemble_torsions = (
        ((0.10, -0.70), (0.55, -1.00), (-0.25, 0.85)),
        ((0.20, -0.50), (0.45, -0.72), (0.05, 0.55)),
        ((0.30, -0.35), (0.38, -0.45), (0.20, 0.35)),
    )
    _schematic_arrow_between(ax, (0.54, 0.73), (0.775, 0.55), FPT_COLOR, 0.04)
    for row_index, y in enumerate(ensemble_y):
        ax.plot(
            ensemble_x,
            (y, y, y),
            color=FPT_FILL_COLOR,
            linewidth=0.65,
            linestyle=":",
            alpha=1.0,
            zorder=2,
        )
        for x, torsions in zip(ensemble_x, ensemble_torsions[row_index]):
            _draw_ligand_glyph(ax, (x, y), 0.017, torsions, FPT_COLOR)
    for x in ensemble_x:
        _schematic_arrow(
            ax,
            (x, ensemble_y[0] - 0.025),
            (x, ensemble_y[1] + 0.025),
            color=FPT_COLOR,
        )
        _schematic_arrow(
            ax,
            (x, ensemble_y[1] - 0.025),
            (x, ensemble_y[2] + 0.025),
            color=FPT_COLOR,
        )
    ax.text(
        0.775,
        0.555,
        "score gradient + force kernel",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.0,
        color=FPT_COLOR,
    )

    _schematic_arrow(ax, (mc_centers[-1][0], 0.19), (0.40, 0.15), color=MC_COLOR)
    _schematic_arrow(ax, (0.775, 0.19), (0.60, 0.15), color=FPT_COLOR)
    _schematic_box(
        ax,
        0.29,
        0.045,
        0.42,
        0.105,
        "Shared Vina evaluation",
        "equal budget  •  time-to-target + final quality",
        SAGE_FILL_COLOR,
        SAGE_COLOR,
    )


def _pdbqt_heavy_topology(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    type_to_element = {
        "A": "C",
        "C": "C",
        "N": "N",
        "NA": "N",
        "OA": "O",
        "O": "O",
        "P": "P",
        "S": "S",
        "SA": "S",
        "F": "F",
        "CL": "Cl",
        "BR": "Br",
        "I": "I",
    }
    elements = []
    coordinates = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom_type = line[77:].strip().split()[0].upper()
        if atom_type in {"H", "HD"}:
            continue
        elements.append(type_to_element.get(atom_type, atom_type.capitalize()))
        coordinates.append(
            (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        )
    reference = np.asarray(coordinates, dtype=float)
    covalent_radius = {
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "P": 1.07,
        "S": 1.05,
        "F": 0.57,
        "Cl": 1.02,
        "Br": 1.20,
        "I": 1.39,
    }
    bonds = []
    for left in range(len(elements)):
        for right in range(left + 1, len(elements)):
            distance = float(np.linalg.norm(reference[left] - reference[right]))
            cutoff = 1.23 * (
                covalent_radius.get(elements[left], 0.77)
                + covalent_radius.get(elements[right], 0.77)
            )
            if 0.55 <= distance <= cutoff:
                bonds.append((left, right))
    return elements, reference, np.asarray(bonds, dtype=np.int32)


def _periodic_pca(mc_theta: np.ndarray, fpt_theta: np.ndarray) -> dict:
    pooled_theta = np.concatenate((mc_theta, fpt_theta), axis=0)
    features = np.concatenate(
        (np.cos(pooled_theta), np.sin(pooled_theta)), axis=1
    )
    centered = features - np.mean(features, axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ right[:2].T
    mc_projected = projected[: len(mc_theta)]
    fpt_projected = projected[len(mc_theta) :]

    lower = np.quantile(projected, 0.01, axis=0)
    upper = np.quantile(projected, 0.99, axis=0)
    padding = 0.08 * np.maximum(upper - lower, 1e-6)
    x_edges = np.linspace(lower[0] - padding[0], upper[0] + padding[0], 58)
    y_edges = np.linspace(lower[1] - padding[1], upper[1] + padding[1], 58)
    density, _, _ = np.histogram2d(
        projected[:, 0], projected[:, 1], bins=(x_edges, y_edges)
    )
    density = gaussian_filter(density, sigma=1.25)
    density /= np.max(density)
    kbt = 0.00198720425864083 * 300.0
    free_energy = -kbt * np.log(np.maximum(density, 1e-5))
    free_energy = np.clip(free_energy, 0.0, 5.0)
    return {
        "mc": mc_projected,
        "fpt": fpt_projected,
        "x": 0.5 * (x_edges[:-1] + x_edges[1:]),
        "y": 0.5 * (y_edges[:-1] + y_edges[1:]),
        "free_energy": free_energy,
    }


def _farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    selected = [int(np.argmin(np.sum(np.square(points), axis=1)))]
    minimum_distance = np.sum(
        np.square(points - points[selected[0]][None, :]), axis=1
    )
    for _ in range(1, count):
        candidate = int(np.argmax(minimum_distance))
        selected.append(candidate)
        minimum_distance = np.minimum(
            minimum_distance,
            np.sum(np.square(points - points[candidate][None, :]), axis=1),
        )
    return np.asarray(selected, dtype=np.int32)


def _real_schematic_data(data: dict) -> dict:
    seed = int(data["suite"]["protocol"]["seed_base"])
    archive_path = (
        Path(data["output_dir"])
        / SCHEMATIC_SYSTEM
        / f"archives_seed{seed}.npz"
    )
    with np.load(archive_path) as archive:
        mc_theta = np.asarray(archive["mc_best_total_theta"][:, 0, :])
        fpt_theta = np.asarray(archive["fpt_best_total_theta"][:, 0, :])
        mc_energy = np.asarray(archive["mc_best_total"])
        fpt_energy = np.asarray(archive["fpt_best_total"])
    projection = _periodic_pca(mc_theta, fpt_theta)

    systems = load_posebusters_systems()
    specification = systems[SCHEMATIC_SYSTEM]
    loaded = build_pdbqt_potential(
        receptor_paths=specification["receptor_paths"],
        ligand_path=specification["ligand_path"],
        interaction_backend="jax",
    )
    potential = loaded.potential
    elements, reference, bonds = _pdbqt_heavy_topology(
        Path(specification["ligand_path"])
    )
    rotatable = np.asarray(potential.rotatable_bonds, dtype=np.int32)
    bond_set = {tuple(sorted(map(int, bond))) for bond in bonds}
    bond_set.update(tuple(sorted(map(int, bond))) for bond in rotatable)
    bonds = np.asarray(sorted(bond_set), dtype=np.int32)

    start_index = int(np.argsort(mc_energy)[len(mc_energy) // 2])
    end_index = int(np.argmin(mc_energy))
    fpt_best_index = int(np.argmin(fpt_energy))
    start = mc_theta[start_index]
    delta = np.arctan2(
        np.sin(mc_theta[end_index] - start),
        np.cos(mc_theta[end_index] - start),
    )
    fractions = np.linspace(0.0, 1.0, 4)[:, None]
    mc_path_theta = np.arctan2(
        np.sin(start[None, :] + fractions * delta),
        np.cos(start[None, :] + fractions * delta),
    )

    fpt_indices = _farthest_point_indices(projection["fpt"], 3)
    mc_source_indices = _farthest_point_indices(projection["mc"], 3)
    fpt_target = fpt_theta[fpt_indices]
    fpt_source = mc_theta[mc_source_indices]
    transport_delta = np.arctan2(
        np.sin(fpt_target - fpt_source), np.cos(fpt_target - fpt_source)
    )
    transport = []
    for fraction in (0.0, 0.5, 1.0):
        theta = fpt_source + fraction * transport_delta
        transport.append(np.arctan2(np.sin(theta), np.cos(theta)))
    fpt_transport_theta = np.asarray(transport)

    mc_coordinates = np.asarray(potential.coordinates(mc_path_theta))
    fpt_coordinates = np.asarray(
        potential.coordinates(fpt_transport_theta.reshape(-1, fpt_theta.shape[1]))
    ).reshape(3, 3, len(elements), 3)
    fpt_best_coordinates = np.asarray(
        potential.coordinates(fpt_theta[fpt_best_index][None, :])
    )[0]
    reference_coordinate = np.asarray(potential.ligand_reference_coordinate)
    centered_reference = reference_coordinate - np.mean(
        reference_coordinate, axis=0, keepdims=True
    )
    _, _, right = np.linalg.svd(centered_reference, full_matrices=False)
    return {
        "elements": elements,
        "bonds": bonds,
        "rotatable_bonds": rotatable,
        "projection_basis": right.T,
        "reference_coordinates": reference_coordinate,
        "mc_coordinates": mc_coordinates,
        "fpt_coordinates": fpt_coordinates,
        "fpt_best_coordinates": fpt_best_coordinates,
        "mc_best_projection": projection["mc"][end_index],
        "fpt_best_projection": projection["fpt"][fpt_best_index],
        "mc_best_energy": float(mc_energy[end_index]),
        "fpt_best_energy": float(fpt_energy[fpt_best_index]),
        "fpt_representative_projection": projection["fpt"][fpt_indices],
        "fes": projection,
        "torsions": mc_theta.shape[1],
    }


def _draw_real_molecule(
    ax,
    coordinates: np.ndarray,
    schematic: dict,
    center: tuple[float, float],
    width: float,
    height: float,
    *,
    highlight_torsions: bool = False,
) -> None:
    centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    basis = schematic["projection_basis"]
    projected = centered @ basis[:, :2]
    depth = centered @ basis[:, 2]
    span = np.maximum(np.ptp(projected, axis=0), 1e-6)
    factor = min(width / span[0], height / span[1])
    xy = projected * factor + np.asarray(center)[None, :]

    highlighted = {
        tuple(sorted(map(int, bond)))
        for bond in schematic["rotatable_bonds"]
    }
    bonds = sorted(
        schematic["bonds"],
        key=lambda bond: float(np.mean(depth[np.asarray(bond, dtype=int)])),
    )
    for left, right in bonds:
        key = tuple(sorted((int(left), int(right))))
        color = NEITHER_COLOR if highlight_torsions and key in highlighted else BOTH_COLOR
        linewidth = 1.55 if highlight_torsions and key in highlighted else 0.92
        ax.plot(
            (xy[left, 0], xy[right, 0]),
            (xy[left, 1], xy[right, 1]),
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=4,
        )

    atom_color = {
        "C": "#B7AAA0",
        "N": ELEMENT_N_COLOR,
        "O": ELEMENT_O_COLOR,
        "P": SAGE_COLOR,
        "S": "#B89A5C",
        "F": SAGE_COLOR,
        "Cl": SAGE_COLOR,
        "Br": NEITHER_COLOR,
        "I": NEITHER_COLOR,
    }
    depth_scale = (depth - np.min(depth)) / max(float(np.ptp(depth)), 1e-6)
    base_radius = 0.050 * min(width, height)
    for index in np.argsort(depth):
        element = schematic["elements"][int(index)]
        radius = base_radius * (0.86 + 0.22 * depth_scale[index])
        ax.add_patch(
            Circle(
                xy[index],
                radius,
                facecolor=atom_color.get(element, BOTH_COLOR),
                edgecolor="white",
                linewidth=0.65,
                zorder=5 + int(depth_scale[index] > 0.5),
            )
        )


def _load_pymol_sprite(path: Path) -> np.ndarray:
    image = plt.imread(path)
    if image.ndim == 3 and image.shape[2] == 4:
        occupied = np.argwhere(image[..., 3] > 0.01)
        if occupied.size:
            lower = np.maximum(np.min(occupied, axis=0) - 3, 0)
            upper = np.minimum(np.max(occupied, axis=0) + 4, image.shape[:2])
            image = image[lower[0] : upper[0], lower[1] : upper[1]]
    return image


def _axis_pixel_scales(ax) -> tuple[float, float]:
    bounds = ax.get_window_extent()
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    x_scale = bounds.width / max(abs(x_limits[1] - x_limits[0]), 1e-12)
    y_scale = bounds.height / max(abs(y_limits[1] - y_limits[0]), 1e-12)
    return float(x_scale), float(y_scale)


def _draw_pymol_sprite(
    ax,
    path: Path,
    center: tuple[float, float],
    width: float,
    height: float,
    *,
    zorder: float = 4,
) -> None:
    image = _load_pymol_sprite(path)
    image_height, image_width = image.shape[:2]
    x_scale, y_scale = _axis_pixel_scales(ax)
    scale = min(
        width * x_scale / image_width,
        height * y_scale / image_height,
    )
    display_width = scale * image_width / x_scale
    display_height = scale * image_height / y_scale
    ax.imshow(
        image,
        extent=(
            center[0] - display_width / 2,
            center[0] + display_width / 2,
            center[1] - display_height / 2,
            center[1] + display_height / 2,
        ),
        aspect="auto",
        interpolation="lanczos",
        zorder=zorder,
    )


def _draw_pymol_layer(
    ax,
    path: Path,
    extent: tuple[float, float, float, float],
    *,
    alpha: float,
    zorder: float,
) -> None:
    ax.imshow(
        plt.imread(path),
        extent=extent,
        aspect="auto",
        interpolation="lanczos",
        alpha=alpha,
        zorder=zorder,
    )


def _pymol_layer_centroid(
    path: Path, extent: tuple[float, float, float, float]
) -> tuple[float, float]:
    image = plt.imread(path)
    if image.ndim != 3 or image.shape[2] != 4:
        return ((extent[0] + extent[1]) / 2, (extent[2] + extent[3]) / 2)
    weights = image[..., 3].astype(float)
    total = float(np.sum(weights))
    if total <= 0:
        return ((extent[0] + extent[1]) / 2, (extent[2] + extent[3]) / 2)
    row, column = np.indices(weights.shape)
    mean_column = float(np.sum(column * weights) / total)
    mean_row = float(np.sum(row * weights) / total)
    x = extent[0] + (mean_column / max(image.shape[1] - 1, 1)) * (extent[1] - extent[0])
    y = extent[3] - (mean_row / max(image.shape[0] - 1, 1)) * (extent[3] - extent[2])
    return (x, y)

def _step_header(
    ax, x: float, number: str, title: str, color: str, *, y: float = 0.925
) -> None:
    ax.add_patch(Circle((x, y), 0.021, facecolor=color, edgecolor="none"))
    ax.text(
        x,
        y,
        number,
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color="white",
    )
    ax.text(
        x + 0.03,
        y,
        title,
        ha="left",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=color,
    )


def _plot_sampling_schematic(ax, data: dict, *, header_reference_ax=None) -> None:
    if not SCHEMATIC_ASSET.exists():
        _plot_sampling_schematic_abstract(
            ax,
            header_reference_ax=header_reference_ax,
        )
        return
    schematic = _real_schematic_data(data)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    step3_y_offset = -0.040
    _schematic_panel_header(ax, reference_ax=header_reference_ax)
    _step_header(ax, 0.045, "1", "SYSTEM SETUP", MC_COLOR)
    _step_header(ax, 0.385, "2", "SAMPLING STRATEGY", SAGE_COLOR)
    _step_header(
        ax,
        0.045,
        "3",
        "TORSIONAL OUTPUT · 14D PERIODIC FES",
        WARM_ACCENT_COLOR,
        y=0.465 + step3_y_offset,
    )

    system_surface = PYMOL_ASSET_DIR / "system_surface.png"
    system_ligand_outline = PYMOL_ASSET_DIR / "system_ligand_outline.png"
    system_ligand = PYMOL_ASSET_DIR / "system_ligand.png"
    system_torsion = PYMOL_ASSET_DIR / "system_torsion.png"
    show_rotation_arrow = False
    x_scale, y_scale = _axis_pixel_scales(ax)
    system_center = (0.18, 0.71)
    system_width = 0.29
    system_height = system_width * x_scale / y_scale
    system_extent = (
        system_center[0] - system_width / 2,
        system_center[0] + system_width / 2,
        system_center[1] - system_height / 2,
        system_center[1] + system_height / 2,
    )
    if system_surface.exists() and system_ligand.exists():
        _draw_pymol_layer(
            ax,
            system_surface,
            system_extent,
            alpha=1.0,
            zorder=1,
        )
        if system_ligand_outline.exists():
            _draw_pymol_layer(
                ax,
                system_ligand_outline,
                system_extent,
                alpha=0.92,
                zorder=2,
            )
        _draw_pymol_layer(
            ax,
            system_ligand,
            system_extent,
            alpha=1.0,
            zorder=3,
        )
        centroid = _pymol_layer_centroid(system_ligand, system_extent)
        torsion_anchor = centroid
        if system_torsion.exists():
            _draw_pymol_layer(
                ax,
                system_torsion,
                system_extent,
                alpha=1.0,
                zorder=5,
            )
            torsion_anchor = _pymol_layer_centroid(system_torsion, system_extent)
            show_rotation_arrow = True
    else:
        pocket = plt.imread(SCHEMATIC_ASSET)[:720]
        ax.imshow(
            pocket,
            extent=system_extent,
            aspect="auto",
            interpolation="lanczos",
            zorder=0,
        )
        _draw_real_molecule(
            ax,
            schematic["reference_coordinates"],
            schematic,
            (0.18, 0.71),
            0.18,
            0.18,
            highlight_torsions=True,
        )
        centroid = (0.18, 0.71)
        torsion_anchor = centroid
    if show_rotation_arrow:
        ax.add_patch(
            FancyArrowPatch(
                (torsion_anchor[0] - 0.015, torsion_anchor[1] + 0.010),
                (torsion_anchor[0] + 0.015, torsion_anchor[1] + 0.010),
                connectionstyle="arc3,rad=-0.78",
                arrowstyle="-|>",
                mutation_scale=5.5,
                color=NEITHER_COLOR,
                linewidth=0.65,
                zorder=8,
            )
        )
    _schematic_arrow(ax, (0.325, 0.72), (0.375, 0.72), color=BOTH_COLOR)
    ax.text(
        0.39,
        0.855,
        "MC · isolated trajectories",
        fontsize=9.0,
        color=MC_COLOR,
        fontweight="bold",
    )
    mc_indices = (0, 1, 3)
    mc_centers = ((0.48, 0.79), (0.68, 0.79), (0.88, 0.79))
    for path_index, (index, center) in enumerate(zip(mc_indices, mc_centers)):
        coordinates = schematic["mc_coordinates"][index]
        pymol_asset = PYMOL_ASSET_DIR / f"mc_{index}.png"
        if pymol_asset.exists():
            _draw_pymol_sprite(ax, pymol_asset, center, 0.125, 0.125)
        else:
            _draw_real_molecule(ax, coordinates, schematic, center, 0.11, 0.11)
        if path_index < len(mc_centers) - 1:
            _schematic_arrow(
                ax,
                (center[0] + 0.065, center[1]),
                (mc_centers[path_index + 1][0] - 0.065, center[1]),
                color=MC_COLOR,
            )

    ax.plot((0.38, 0.97), (0.72, 0.72), color=GRID_COLOR, linewidth=0.65)
    ax.text(
        0.39,
        0.685,
        "FPT · ensemble transport",
        fontsize=9.0,
        color=FPT_COLOR,
        fontweight="bold",
    )
    columns = (0.48, 0.68, 0.88)
    rows = (0.64, 0.535)
    for time_index, x in enumerate(columns):
        for particle_index, y in enumerate(rows):
            pymol_asset = PYMOL_ASSET_DIR / f"fpt_{time_index}_{particle_index}.png"
            if pymol_asset.exists():
                _draw_pymol_sprite(ax, pymol_asset, (x, y), 0.112, 0.095)
            else:
                _draw_real_molecule(
                    ax,
                    schematic["fpt_coordinates"][time_index, particle_index],
                    schematic,
                    (x, y),
                    0.10,
                    0.085,
                )
            if time_index < len(columns) - 1:
                _schematic_arrow(
                    ax,
                    (x + 0.060, y),
                    (columns[time_index + 1] - 0.060, y),
                    color=FPT_COLOR,
                )
        ax.plot(
            (x, x),
            (rows[-1], rows[0]),
            color=FPT_FILL_COLOR,
            linewidth=0.65,
            linestyle=":",
            alpha=1.0,
        )

    ax.add_patch(
        FancyArrowPatch(
            (0.675, 0.515),
            (0.675, 0.482 + step3_y_offset),
            arrowstyle="-|>",
            mutation_scale=7,
            color=BOTH_COLOR,
            linewidth=0.85,
            zorder=9,
        )
    )

    fes = schematic["fes"]
    palette = LinearSegmentedColormap.from_list(
        "ice_porcelain_fes",
        (
            LANDSCAPE_HIGH_COLOR,
            LANDSCAPE_MID_COLOR,
            LANDSCAPE_LOW_COLOR,
            LANDSCAPE_OUTER_COLOR,
            CANVAS_COLOR,
        ),
    )
    levels = np.linspace(0.0, 5.0, 11)
    for (
        fes_ax,
        method,
        color,
        title,
        best_asset,
        best_point,
        best_energy,
        structure_center,
    ) in (
        (
            ax.inset_axes((0.085, 0.075 + step3_y_offset, 0.31, 0.32)),
            "mc",
            MC_COLOR,
            "MC trajectory",
            PYMOL_ASSET_DIR / "mc_3.png",
            schematic["mc_best_projection"],
            schematic["mc_best_energy"],
            (0.455, 0.22 + step3_y_offset),
        ),
        (
            ax.inset_axes((0.55, 0.075 + step3_y_offset, 0.31, 0.32)),
            "fpt",
            FPT_COLOR,
            "FPT ensemble",
            PYMOL_ASSET_DIR / "fpt_best.png",
            schematic["fpt_best_projection"],
            schematic["fpt_best_energy"],
            (0.930, 0.22 + step3_y_offset),
        ),
    ):
        fes_ax.contourf(
            fes["x"],
            fes["y"],
            fes["free_energy"].T,
            levels=levels,
            cmap=palette,
            extend="max",
        )
        fes_ax.scatter(
            fes[method][::4, 0],
            fes[method][::4, 1],
            s=5.2,
            color=color,
            alpha=1.0,
            edgecolors="white",
            linewidths=0.65,
            zorder=4,
        )
        fes_ax.scatter(
            [best_point[0]],
            [best_point[1]],
            s=46,
            marker="*",
            facecolor=color,
            edgecolor=TEXT_COLOR,
            linewidth=0.65,
            zorder=7,
        )
        fes_ax.set_xticks(())
        fes_ax.set_yticks(())
        fes_ax.set_xlabel(r"$z_1$", fontsize=9.0, labelpad=1)
        fes_ax.set_title(title, fontsize=9.0, color=color, pad=2, fontweight="bold")
        for spine in ("top", "right"):
            fes_ax.spines[spine].set_visible(False)
        fes_ax.spines["left"].set_linewidth(0.65)
        fes_ax.spines["bottom"].set_linewidth(0.65)
        ax.scatter(
            [structure_center[0] - 0.018],
            [0.365 + step3_y_offset],
            s=21,
            marker="*",
            facecolor=color,
            edgecolor=TEXT_COLOR,
            linewidth=0.65,
            zorder=7,
        )
        ax.text(
            structure_center[0] + 0.003,
            0.365 + step3_y_offset,
            "best",
            fontsize=9.0,
            color=color,
            va="center",
            fontweight="bold",
        )
        if best_asset.exists():
            _draw_pymol_sprite(
                ax,
                best_asset,
                structure_center,
                0.110,
                0.165,
                zorder=6,
            )
        ax.text(
            structure_center[0],
            0.105 + step3_y_offset,
            rf"$E_{{\mathrm{{best}}}}={best_energy:.1f}$"
            "\n"
            r"$\mathrm{kcal\,mol^{-1}}$",
            fontsize=9.0,
            color=TEXT_COLOR,
            ha="center",
            va="center",
            linespacing=1.0,
            zorder=10,
            clip_on=False,
        )
    ax.text(
        0.065,
        0.24 + step3_y_offset,
        r"$z_2$",
        fontsize=9.0,
        rotation=90,
        ha="center",
        va="center",
        color=TEXT_COLOR,
    )

def _plot_publication_step_coverage(ax, data: dict) -> None:
    runs = data["runs"]
    maximum = int(data["suite"]["protocol"]["steps"])
    grid = np.arange(maximum + 1)
    for method, color, fill_color, label in (
        ("mc", MC_COLOR, MC_WASH_COLOR, "MC"),
        ("fpt", FPT_COLOR, FPT_WASH_COLOR, "FPT"),
    ):
        coverage, low, high = _clustered_coverage_interval(
            runs, f"{method}_hit_step", grid
        )
        ax.fill_between(
            grid,
            low,
            high,
            step="post",
            color=fill_color,
            alpha=1.0,
            linewidth=0,
        )
        ax.step(
            grid,
            coverage,
            where="post",
            color=color,
            linestyle="--" if method == "mc" else "-",
            label=label,
        )
    ax.set_xlim(0, maximum)
    ax.set_ylim(0, 103)
    ax.set_xticks((0, 250, 500, 1000))
    # Keep the terminal label inside the tight right-hand column.
    ax.get_xticklabels()[-1].set_horizontalalignment("right")
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.set_yticklabels(("0%", "25%", "50%", "75%", "100%"))
    ax.set_xlabel(r"$t$ (steps)")
    ax.set_ylabel(r"$P_{\mathrm{hit}}(t)$")
    ax.legend(frameon=False, loc="lower right", ncol=2, columnspacing=0.8)
    _publication_panel(ax, "B", "Sampling efficiency")


def _plot_publication_time_coverage(ax, data: dict) -> None:
    runs = data["runs"]
    maximum = max(
        max(run[f"{method}_elapsed_seconds"] for run in runs)
        for method in ("mc", "fpt")
    )
    grid = np.concatenate(([0.0], np.geomspace(1e-6, maximum, 600)))
    for method, color, fill_color, label in (
        ("mc", MC_COLOR, MC_WASH_COLOR, "MC"),
        ("fpt", FPT_COLOR, FPT_WASH_COLOR, "FPT"),
    ):
        coverage, low, high = _clustered_coverage_interval(
            runs, f"{method}_hit_seconds", grid
        )
        ax.fill_between(
            grid,
            low,
            high,
            step="post",
            color=fill_color,
            alpha=1.0,
            linewidth=0,
        )
        ax.step(
            grid,
            coverage,
            where="post",
            color=color,
            linestyle="--" if method == "mc" else "-",
            label=label,
        )
    ax.set_xscale("symlog", linthresh=1e-4, linscale=0.8)
    ax.set_xlim(0, maximum * 1.08)
    ax.set_ylim(0, 103)
    ax.set_xticks((0, 1e-4, 1e-3, 1e-2, 1e-1))
    ax.set_xticklabels(("0", "0.1", "1", "10", "100"))
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.set_yticklabels(("0%", "25%", "50%", "75%", "100%"))
    ax.set_xlabel(r"$t_{\mathrm{wall}}$ (ms; symlog)")
    ax.set_ylabel(r"$P_{\mathrm{hit}}(t)$")
    ax.legend(frameon=False, loc="lower right", ncol=2, columnspacing=0.8)
    _publication_panel(ax, "C", "Wall-time efficiency")


def _plot_publication_complex_time_coverage(ax, data: dict) -> None:
    runs = [run for run in data["runs"] if run["torsions"] >= 8]
    if not runs:
        raise RuntimeError("no systems with at least eight ligand torsions")
    maximum = max(
        max(run[f"{method}_elapsed_seconds"] for run in runs)
        for method in ("mc", "fpt")
    )
    grid = np.concatenate(([0.0], np.geomspace(1e-6, maximum, 600)))
    for method, color, fill_color, label in (
        ("mc", MC_COLOR, MC_WASH_COLOR, "MC"),
        ("fpt", FPT_COLOR, FPT_WASH_COLOR, "FPT"),
    ):
        coverage, low, high = _clustered_coverage_interval(
            runs, f"{method}_hit_seconds", grid
        )
        ax.fill_between(
            grid,
            low,
            high,
            step="post",
            color=fill_color,
            alpha=1.0,
            linewidth=0,
        )
        ax.step(
            grid,
            coverage,
            where="post",
            color=color,
            linestyle="--" if method == "mc" else "-",
            label=label,
        )
    ax.set_xscale("symlog", linthresh=1e-4, linscale=0.8)
    ax.set_xlim(0, maximum * 1.08)
    ax.set_ylim(0, 103)
    ax.set_xticks((0, 1e-4, 1e-3, 1e-2, 1e-1))
    ax.set_xticklabels(("0", "0.1", "1", "10", "100"))
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.set_yticklabels(("0%", "25%", "50%", "75%", "100%"))
    ax.set_xlabel(r"$t_{\mathrm{wall}}$ (ms)")
    ax.set_ylabel(r"$P_{\mathrm{hit}}(t)$")
    ax.legend(frameon=False, loc="lower right", ncol=2, columnspacing=0.8)
    _publication_panel(ax, "C", r"Wall time · $N_{\mathrm{tor}}\geq 8$")


def _plot_publication_final_quality(ax, data: dict) -> None:
    systems = data["systems"]
    styles = {
        "both": (BOTH_MARKER_COLOR, BOTH_COLOR, "o", "Both hit", 15),
        "fpt_only": (FPT_COLOR, TEXT_COLOR, "v", "FPT only", 32),
        "mc_only": (MC_COLOR, TEXT_COLOR, "^", "MC only", 32),
        "neither": ("none", NEITHER_COLOR, "D", "Neither", 31),
    }
    for outcome in ("both", "fpt_only", "mc_only", "neither"):
        selected = [system for system in systems if system["outcome"] == outcome]
        if not selected:
            continue
        facecolor, edgecolor, marker, label, size = styles[outcome]
        ax.scatter(
            [system["mc_total_gap"] for system in selected],
            [system["fpt_total_gap"] for system in selected],
            s=size,
            marker=marker,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.0 if outcome == "neither" else 0.65,
            alpha=1.0,
            label=label,
            zorder=4 if outcome != "both" else 2,
        )
    lower, upper = -3.0, 8.5
    ax.plot((lower, upper), (lower, upper), color=TEXT_COLOR, linestyle="--", linewidth=0.8)
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8)
    ax.axvline(0, color=GRID_COLOR, linewidth=0.8)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\Delta E_{\mathrm{MC}}$ (kcal/mol)")
    ax.set_ylabel(r"$\Delta E_{\mathrm{FPT}}$ (kcal/mol)")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3, labelspacing=0.25)
    _publication_panel(ax, "D", "Fixed-budget quality")


def _plot_publication_energy_effect(ax, data: dict) -> None:
    rng = np.random.default_rng(6300)
    statistics = {row["torsion_bin"]: row for row in data["statistics"]["torsion_bins"]}
    x = np.arange(len(TORSION_BINS))
    for index, label in enumerate(TORSION_BINS):
        raw = np.array(
            [
                system["total_improvement"]
                for system in data["systems"]
                if _torsion_bin_for_count(system["torsions"]) == label
            ]
        )
        violin = ax.violinplot(
            [raw],
            positions=[index],
            widths=0.64,
            points=100,
            bw_method=0.35,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in violin["bodies"]:
            vertices = body.get_paths()[0].vertices
            vertices[:, 0] = np.minimum(vertices[:, 0], index)
            body.set_facecolor(FPT_FILL_COLOR)
            body.set_edgecolor(FPT_COLOR)
            body.set_linewidth(0.65)
            body.set_alpha(1.0)
            body.set_zorder(1)
        ax.scatter(
            index + rng.uniform(0.045, 0.19, len(raw)),
            raw,
            color=FPT_MARKER_COLOR,
            alpha=1.0,
            s=7,
            edgecolor=FPT_COLOR,
            linewidth=0.65,
            zorder=3,
        )
        row = statistics[label]
        center = row["median_final_energy_improvement"]
        low = row["median_final_energy_improvement_bootstrap_ci_low"]
        high = row["median_final_energy_improvement_bootstrap_ci_high"]
        ax.errorbar(
            index,
            center,
            yerr=[[center - low], [high - center]],
            color=TEXT_COLOR,
            ecolor=TEXT_COLOR,
            marker="D",
            markerfacecolor=FPT_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.65,
            markersize=5,
            capsize=2.5,
            linewidth=1.0,
            zorder=5,
        )
    ax.axhline(0, color=TEXT_COLOR, linestyle="--", linewidth=0.8)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 7.6)
    ax.set_xticks(x)
    ax.set_xticklabels(TORSION_BINS)
    ax.set_xlabel(r"$N_{\mathrm{tor}}$")
    ax.set_ylabel(r"$E_{\mathrm{MC}}-E_{\mathrm{FPT}}$ (kcal/mol)")
    _publication_panel(ax, "E", "Quality gain by torsions")


def _plot_publication_speedups(ax, data: dict) -> None:
    rows = {row["torsion_bin"]: row for row in data["statistics"]["torsion_bins"]}
    x = np.arange(len(TORSION_BINS))
    for key, low_key, high_key, color, marker, linestyle, label in (
        (
            "median_step_speedup_mc_over_fpt",
            "median_step_speedup_bootstrap_ci_low",
            "median_step_speedup_bootstrap_ci_high",
            FPT_COLOR,
            "s",
            "--",
            "Median step speedup",
        ),
        (
            "aggregate_capped_time_speedup_mc_over_fpt",
            "aggregate_capped_time_speedup_bootstrap_ci_low",
            "aggregate_capped_time_speedup_bootstrap_ci_high",
            NEITHER_COLOR,
            "o",
            "-",
            "Capped-time speedup",
        ),
    ):
        centers = np.array([rows[label_][key] for label_ in TORSION_BINS])
        lows = np.array([rows[label_][low_key] for label_ in TORSION_BINS])
        highs = np.array([rows[label_][high_key] for label_ in TORSION_BINS])
        ax.plot(
            x,
            centers,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=4,
            label=label,
        )
        ax.errorbar(
            x,
            centers,
            yerr=np.vstack((centers - lows, highs - centers)),
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            capsize=2.2,
            alpha=1.0,
        )
    ax.axhline(1, color=TEXT_COLOR, linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylim(0.015, 8.0)
    ax.set_yticks((0.03, 0.1, 0.3, 1, 3))
    ax.set_yticklabels(("0.03", "0.1", "0.3", "1", "3"))
    ax.set_xticks(x)
    ax.set_xticklabels(TORSION_BINS)
    ax.set_xlabel(r"$N_{\mathrm{tor}}$")
    ax.set_ylabel(r"$S_{\mathrm{MC/FPT}}$ (log)")
    ax.legend(frameon=False, loc="lower right", handlelength=1.5, labelspacing=0.3)
    _publication_panel(ax, "F", "Efficiency by complexity")


def _plot_publication_speedup_effects(ax, data: dict) -> None:
    rows = {row["torsion_bin"]: row for row in data["statistics"]["torsion_bins"]}
    x = np.arange(len(TORSION_BINS))
    series = (
        (
            "median_step_speedup_mc_over_fpt",
            "median_step_speedup_bootstrap_ci_low",
            "median_step_speedup_bootstrap_ci_high",
            FPT_COLOR,
            "s",
            -0.085,
            "Updates",
        ),
        (
            "aggregate_capped_time_speedup_mc_over_fpt",
            "aggregate_capped_time_speedup_bootstrap_ci_low",
            "aggregate_capped_time_speedup_bootstrap_ci_high",
            NEITHER_COLOR,
            "o",
            0.085,
            "Capped time",
        ),
    )
    for key, low_key, high_key, color, marker, offset, label in series:
        centers = np.array([rows[label_][key] for label_ in TORSION_BINS], dtype=float)
        lows = np.array([rows[label_][low_key] for label_ in TORSION_BINS], dtype=float)
        highs = np.array([rows[label_][high_key] for label_ in TORSION_BINS], dtype=float)
        ax.errorbar(
            x + offset,
            centers,
            yerr=np.vstack((centers - lows, highs - centers)),
            color=color,
            ecolor=color,
            fmt=marker,
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.65,
            markersize=5,
            elinewidth=0.9,
            capsize=2.3,
            label=label,
            zorder=3,
        )
    # Unity marks equal efficiency; values above unity favor FPT.
    ax.axhline(1, color=TEXT_COLOR, linestyle="--", linewidth=0.8)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.3, 7.2)
    ax.set_yticks((0, 1, 2, 4, 6))
    ax.set_xticks(x)
    ax.set_xticklabels(TORSION_BINS)
    ax.set_xlabel(r"$N_{\mathrm{tor}}$")
    ax.set_ylabel(r"$S_{\mathrm{MC/FPT}}$")
    handles = [
        Line2D([], [], color=color, marker=marker, linestyle="none", markersize=5,
               markeredgecolor="white", markeredgewidth=0.65, label=label)
        for _, _, _, color, marker, _, label in series
    ]
    legend = ax.legend(
        handles=handles,
        frameon=True,
        loc="upper left",
        borderaxespad=0.3,
        handletextpad=0.35,
        labelspacing=0.18,
    )
    legend.get_frame().set(facecolor="white", edgecolor="none", alpha=0.92)
    _publication_panel(ax, "F", "Efficiency by torsions")


def _plot_publication_diversity(ax, data: dict) -> None:
    joint = [system for system in data["systems"] if system["outcome"] == "both"]
    for relation, color, edgecolor, marker, label in (
        ("fpt", FPT_MARKER_COLOR, FPT_COLOR, "o", "FPT more"),
        ("mc", MC_MARKER_COLOR, MC_COLOR, "s", "MC more"),
        ("tie", TEXT_COLOR, TEXT_COLOR, "x", "Tie"),
    ):
        if relation == "fpt":
            selected = [s for s in joint if s["fpt_unique_clusters"] > s["mc_unique_clusters"]]
        elif relation == "mc":
            selected = [s for s in joint if s["mc_unique_clusters"] > s["fpt_unique_clusters"]]
        else:
            selected = [s for s in joint if s["mc_unique_clusters"] == s["fpt_unique_clusters"]]
        scatter_style = {
            "color": color,
            "marker": marker,
            "alpha": 1.0,
            "s": 22 if relation == "tie" else 15,
            "linewidth": 1.0 if relation == "tie" else 0.7,
            "label": label,
        }
        if relation != "tie":
            scatter_style["edgecolor"] = edgecolor
        ax.scatter(
            [s["mc_unique_clusters"] for s in selected],
            [s["fpt_unique_clusters"] for s in selected],
            **scatter_style,
        )
    limits = (0.8, 650)
    ax.plot(limits, limits, color=TEXT_COLOR, linestyle="--", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ticks = (1, 4, 16, 64, 256)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(value) for value in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(value) for value in ticks])
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(r"$N_{\mathrm{clust}}^{\mathrm{MC}}$")
    ax.set_ylabel(r"$N_{\mathrm{clust}}^{\mathrm{FPT}}$")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3, labelspacing=0.25)
    ax.set_title(
        "Thresholded conformational diversity",
        loc="left",
        pad=7,
        fontweight="bold",
    )


def plot_detailed_main_figure(data: dict, figure_dir: Path) -> None:
    fig = plt.figure(figsize=(6.69, 7.15))
    grid = fig.add_gridspec(
        3,
        3,
        left=0.075,
        right=0.985,
        bottom=0.075,
        top=0.96,
        wspace=0.48,
        hspace=0.58,
        height_ratios=(1.08, 1.08, 1.0),
    )
    ax_a = fig.add_subplot(grid[0:2, 0:2])
    panel_a_position = ax_a.get_position()
    ax_a.set_position(
        (
            # Shift only the schematic body for optical alignment with panel
            # (d); its figure-anchored header remains exactly aligned.
            panel_a_position.x0 - 0.023,
            panel_a_position.y0 - 0.020,
            panel_a_position.width + 0.040,
            panel_a_position.height + 0.020,
        )
    )
    ax_b = fig.add_subplot(grid[0, 2])
    ax_c = fig.add_subplot(grid[1, 2])
    ax_d = fig.add_subplot(grid[2, 0])
    ax_e = fig.add_subplot(grid[2, 1])
    ax_f = fig.add_subplot(grid[2, 2])
    _plot_publication_step_coverage(ax_b, data)
    _plot_publication_time_coverage(ax_c, data)
    _plot_publication_final_quality(ax_d, data)
    _plot_publication_energy_effect(ax_e, data)
    _plot_publication_speedup_effects(ax_f, data)
    _plot_sampling_schematic(ax_a, data, header_reference_ax=ax_d)
    _save_figure(fig, figure_dir / "main_figure_posebusters_detailed")


def plot_complexity_focused_main_figure(data: dict, figure_dir: Path) -> None:
    fig = plt.figure(figsize=(6.69, 7.15))
    grid = fig.add_gridspec(
        3,
        3,
        left=0.075,
        right=0.985,
        bottom=0.075,
        top=0.96,
        wspace=0.48,
        hspace=0.58,
        height_ratios=(1.08, 1.08, 1.0),
    )
    ax_a = fig.add_subplot(grid[0:2, 0:2])
    panel_a_position = ax_a.get_position()
    ax_a.set_position(
        (
            # Keep the optical body alignment identical to the detailed main
            # figure while preserving the shared figure-level header anchor.
            panel_a_position.x0 - 0.023,
            panel_a_position.y0 - 0.020,
            panel_a_position.width + 0.040,
            panel_a_position.height + 0.020,
        )
    )
    ax_b = fig.add_subplot(grid[0, 2])
    ax_c = fig.add_subplot(grid[1, 2])
    ax_d = fig.add_subplot(grid[2, 0])
    ax_e = fig.add_subplot(grid[2, 1])
    ax_f = fig.add_subplot(grid[2, 2])
    _plot_publication_step_coverage(ax_b, data)
    _plot_publication_complex_time_coverage(ax_c, data)
    _plot_publication_final_quality(ax_d, data)
    _plot_publication_energy_effect(ax_e, data)
    _plot_publication_speedup_effects(ax_f, data)
    _plot_sampling_schematic(ax_a, data, header_reference_ax=ax_d)
    _save_figure(fig, figure_dir / "main_figure_posebusters_complexity_focused")


def plot_si_diversity(data: dict, figure_dir: Path) -> None:
    # Match the 0.82\textwidth SI placement so the raster remains at the
    # requested 600 PPI after inclusion rather than being enlarged from a
    # single-column canvas.
    fig, ax = plt.subplots(figsize=(5.33, 4.80))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.15, top=0.91)
    _plot_publication_diversity(ax, data)
    _save_figure(fig, figure_dir / "si_diversity_validation")


def _plot_gap_dumbbell(ax, systems: list[dict], metric: str, title: str) -> None:
    y = np.arange(len(systems))
    mc = np.array([system[f"mc_{metric}_gap"] for system in systems])
    fpt = np.array([system[f"fpt_{metric}_gap"] for system in systems])
    for position, left, right in zip(y, mc, fpt):
        ax.plot([left, right], [position, position], color=GRID_COLOR, linewidth=1.1, zorder=1)
    ax.scatter(mc, y, color=MC_COLOR, marker="o", s=28, label="MC", zorder=3)
    ax.scatter(fpt, y, color=FPT_COLOR, marker="s", s=26, label="FPT", zorder=3)
    ax.axvline(0, color=TEXT_COLOR, linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([system["name"] for system in systems])
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta E$ (kcal/mol)")
    ax.set_title(title, pad=6)
    _clean_axis(ax)


def _plot_bad_trajectory(ax, system: dict, show_y: bool, show_x: bool) -> None:
    threshold = system["threshold"]
    maximum_gap = 0.0
    for method, color, label in (
        ("mc", MC_COLOR, "MC"),
        ("fpt", FPT_COLOR, "FPT"),
    ):
        rows = [row for row in system["trajectories"] if row["method"] == method]
        steps = np.array(sorted({int(row["step"]) for row in rows}))
        gaps = np.array(
            [
                np.median(
                    [
                        float(row["best_total"]) - threshold
                        for row in rows
                        if int(row["step"]) == step
                    ]
                )
                for step in steps
            ]
        )
        if np.any(gaps < -1e-8):
            raise RuntimeError(f"{system['name']} unexpectedly hits strict target")
        maximum_gap = max(maximum_gap, float(gaps.max()))
        ax.plot(
            steps,
            np.log10(1 + np.maximum(gaps, 0)),
            color=color,
            linestyle="--" if method == "mc" else "-",
            marker="o",
            markersize=2.3,
            label=label,
        )
    gap_ticks = np.array([0, 0.25, 1, 2, 5, 10, 25, 50, 100])
    ax.set_yticks(np.log10(1 + gap_ticks))
    ax.set_yticklabels([f"{value:g}" for value in gap_ticks] if show_y else [])
    ax.set_ylim(-0.03, np.log10(1 + 115))
    ax.set_xlim(-15, 1015)
    ax.set_xticks((0, 500, 1000))
    if not show_x:
        ax.set_xticklabels([])
    ax.set_title(f"{system['name']}  ({system['torsions']} torsions)", pad=4)
    if show_y:
        ax.set_ylabel(r"$\Delta E$ (kcal/mol; log)")
    if show_x:
        ax.set_xlabel(r"$t$ (steps)")
    _clean_axis(ax)


def plot_si_figure(data: dict, figure_dir: Path) -> None:
    neither = [system for system in data["systems"] if system["outcome"] == "neither"]
    neither.sort(key=lambda system: system["fpt_total_gap"], reverse=True)
    neither = neither[:6]
    if not neither:
        return

    fig = plt.figure(figsize=(6.50, 6.14))
    grid = fig.add_gridspec(
        3,
        6,
        left=0.085,
        right=0.985,
        bottom=0.075,
        top=0.96,
        wspace=0.5,
        hspace=0.42,
        height_ratios=(0.82, 1.0, 1.0),
    )
    ax_a = fig.add_subplot(grid[0, 0:3])
    ax_b = fig.add_subplot(grid[0, 3:6])
    _plot_gap_dumbbell(ax_a, neither, "total", "Total-energy final quality")
    _plot_gap_dumbbell(ax_b, neither, "binding", "Binding-score final quality")
    ax_b.set_yticklabels([])
    legend_handles = [
        Line2D([], [], color=MC_COLOR, marker="o", linestyle="none", label="MC"),
        Line2D([], [], color=FPT_COLOR, marker="s", linestyle="none", label="FPT"),
    ]
    ax_a.legend(handles=legend_handles, frameon=False, loc="lower right", ncol=2)
    _panel_label(ax_a, "A", x=-0.13)
    _panel_label(ax_b, "B", x=-0.1)

    trajectory_axes = []
    for index, system in enumerate(neither):
        row, column = divmod(index, 3)
        ax = fig.add_subplot(grid[row + 1, 2 * column : 2 * column + 2])
        _plot_bad_trajectory(ax, system, show_y=column == 0, show_x=row == 1)
        trajectory_axes.append(ax)
        _panel_label(ax, chr(ord("C") + index), x=-0.15, y=1.12)
    trajectory_axes[0].legend(frameon=False, loc="upper right")
    _save_figure(fig, figure_dir / "si_neither_hit_cases")


def _plot_bin_anytime(ax, runs: list[dict], title: str, show_y: bool, show_x: bool) -> None:
    steps = 1000
    grid = np.arange(steps + 1)
    for method, color, label in (
        ("mc", MC_COLOR, "MC"),
        ("fpt", FPT_COLOR, "FPT"),
    ):
        coverage, _, _ = _clustered_coverage_interval(
            runs, f"{method}_hit_step", grid
        )
        ax.step(
            grid,
            coverage,
            where="post",
            color=color,
            linestyle="--" if method == "mc" else "-",
            label=label,
        )
    ax.set_xlim(0, steps)
    ax.set_ylim(0, 103)
    ax.set_xticks((0, 500, 1000))
    ax.set_yticks((0, 50, 100))
    ax.set_yticklabels(("0%", "50%", "100%") if show_y else [])
    if not show_x:
        ax.set_xticklabels([])
    if show_y:
        ax.set_ylabel(r"$P_{\mathrm{hit}}(t)$")
    if show_x:
        ax.set_xlabel(r"$t$ (steps)")
    ax.set_title(title, pad=5)
    _clean_axis(ax)


def _plot_outcomes_by_torsions(ax, data: dict) -> None:
    rows = _strict_torsion_rows(data)
    x = np.arange(len(rows))
    both = np.array([row["outcomes"]["both"] for row in rows], dtype=float)
    mc_only = np.array([row["outcomes"]["mc_only"] for row in rows], dtype=float)
    fpt_only = np.array([row["outcomes"]["fpt_only"] for row in rows], dtype=float)
    neither = np.array([row["outcomes"]["neither"] for row in rows], dtype=float)
    totals = np.array(
        [row.get("trials", row["systems"]) for row in rows], dtype=float
    )
    ax.bar(
        x,
        100 * both / totals,
        color=BOTH_FILL_COLOR,
        edgecolor=BOTH_COLOR,
        linewidth=0.65,
        label="Both hit",
    )
    ax.bar(
        x,
        100 * mc_only / totals,
        bottom=100 * both / totals,
        color=MC_FILL_COLOR,
        edgecolor=MC_COLOR,
        linewidth=0.65,
        hatch="//",
        label="MC only",
    )
    ax.bar(
        x,
        100 * fpt_only / totals,
        bottom=100 * (both + mc_only) / totals,
        color=FPT_FILL_COLOR,
        edgecolor=FPT_COLOR,
        linewidth=0.65,
        hatch="\\\\",
        label="FPT only",
    )
    ax.bar(
        x,
        100 * neither / totals,
        bottom=100 * (both + mc_only + fpt_only) / totals,
        color=NEITHER_FILL_COLOR,
        edgecolor=NEITHER_COLOR,
        linewidth=0.65,
        hatch="xx",
        label="Neither",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(TORSION_BINS)
    ax.set_ylim(0, 128)
    ax.set_yticks((0, 50, 100))
    ax.set_yticklabels(("0%", "50%", "100%"))
    ax.set_xlabel(r"$N_{\mathrm{tor}}$")
    ax.set_ylabel(r"$P(\mathrm{outcome})$")
    ax.set_title("Final outcomes", pad=5)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        borderaxespad=0,
        columnspacing=0.65,
        handlelength=1.1,
        handletextpad=0.3,
        labelspacing=0.2,
    )
    _clean_axis(ax)


def plot_si_complexity(data: dict, figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(6.50, 4.15), constrained_layout=True)
    for index, label in enumerate(TORSION_BINS):
        row, column = divmod(index, 3)
        selected = [
            run
            for run in data["runs"]
            if _torsion_bin_for_count(run["torsions"]) == label
        ]
        _plot_bin_anytime(
            axes[row, column],
            selected,
            f"{label} torsions",
            show_y=column == 0,
            show_x=row == 1,
        )
        _panel_label(axes[row, column], chr(ord("A") + index), x=-0.18)
    _plot_outcomes_by_torsions(axes[1, 2], data)
    _panel_label(axes[1, 2], "F", x=-0.18)
    axes[0, 0].legend(frameon=False, loc="lower right")
    _save_figure(fig, figure_dir / "si_complexity_anytime")


def _endpoint_rows(data: dict, score: str) -> list[tuple[str, dict]]:
    return [
        (level.capitalize(), data["analysis"]["endpoints"][f"{score}/{level}"])
        for level in ("easy", "medium", "strict")
    ]


def _plot_endpoint_coverage(ax, data: dict, score: str, title: str) -> None:
    rows = _endpoint_rows(data, score)
    y = np.arange(len(rows))
    mc = np.array(
        [
            100 * row[1]["mc_successes"] / row[1].get("trials", row[1]["systems"])
            for row in rows
        ]
    )
    fpt = np.array(
        [
            100 * row[1]["fpt_successes"] / row[1].get("trials", row[1]["systems"])
            for row in rows
        ]
    )
    for position, left, right in zip(y, mc, fpt):
        ax.plot([left, right], [position, position], color=GRID_COLOR, linewidth=1.2)
    ax.scatter(mc, y, color=MC_COLOR, marker="o", s=28, label="MC", zorder=3)
    ax.scatter(fpt, y, color=FPT_COLOR, marker="s", s=26, label="FPT", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(90, 102)
    ax.set_xlabel(r"$P_{\mathrm{hit}}$")
    ax.set_xticks((90, 95, 100))
    ax.set_xticklabels(("90%", "95%", "100%"))
    ax.set_title(title, pad=5)
    _clean_axis(ax)


def _plot_endpoint_speedups(ax, data: dict) -> None:
    y = np.arange(6)
    labels = []
    values = []
    colors = []
    markers = []
    for score, prefix, color, marker in (
        ("sampling_energy", "Total", FPT_COLOR, "o"),
        ("binding_score", "Binding", MC_COLOR, "s"),
    ):
        for level, row in _endpoint_rows(data, score):
            labels.append(f"{prefix} {level.lower()}")
            values.append(row["capped_time_speedup_mc_over_fpt"])
            colors.append(color)
            markers.append(marker)
    statistics = {
        (row["score"], row["threshold_level"]): row
        for row in data["statistics"]["endpoints"]
    }
    statistic_rows = []
    for score in ("sampling_energy", "binding_score"):
        for level in ("easy", "medium", "strict"):
            statistic_rows.append(statistics[(score, level)])
    for position, value, color, marker, statistic in zip(
        y, values, colors, markers, statistic_rows
    ):
        low = statistic["aggregate_capped_time_speedup_bootstrap_ci_low"]
        high = statistic["aggregate_capped_time_speedup_bootstrap_ci_high"]
        ax.errorbar(
            value,
            position,
            xerr=[[value - low], [high - value]],
            color=color,
            ecolor=color,
            marker=marker,
            markersize=5,
            linewidth=0.9,
            capsize=2.5,
            zorder=3,
        )
    ax.axvline(1, color=TEXT_COLOR, linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.9, 2.7)
    ax.set_xlabel(r"$S_{\mathrm{MC/FPT}}^{\mathrm{cap}}$")
    ax.set_title("Threshold robustness of time-to-target", pad=5)
    _clean_axis(ax)


def _plot_total_binding_improvement(ax, data: dict) -> None:
    ordinary = [system for system in data["systems"] if system["outcome"] != "neither"]
    neither = [system for system in data["systems"] if system["outcome"] == "neither"]
    for selected, color, edgecolor, marker, label, size in (
        (ordinary, BOTH_MARKER_COLOR, BOTH_COLOR, "o", "Other systems", 15),
        (neither, NEITHER_FILL_COLOR, NEITHER_COLOR, "D", "Neither-hit", 35),
    ):
        ax.scatter(
            [system["mc_total_best"] - system["fpt_total_best"] for system in selected],
            [
                system["mc_binding_best"] - system["fpt_binding_best"]
                for system in selected
            ],
            color=color,
            edgecolor=edgecolor,
            marker=marker,
            alpha=1.0,
            s=size,
            linewidth=0.65,
            label=label,
        )
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8)
    ax.axvline(0, color=GRID_COLOR, linewidth=0.8)
    ax.set_xlim(-0.35, 7.45)
    ax.set_ylim(-0.3, 2.3)
    ax.set_xlabel(
        r"$E_{\mathrm{MC}}^{\mathrm{tot}}-E_{\mathrm{FPT}}^{\mathrm{tot}}$ (kcal/mol)"
    )
    ax.set_ylabel(
        r"$E_{\mathrm{MC}}^{\mathrm{bind}}-E_{\mathrm{FPT}}^{\mathrm{bind}}$ (kcal/mol)"
    )
    ax.set_title("Where final-score gains arise", pad=5)
    ax.legend(frameon=False, loc="upper left")
    _clean_axis(ax)


def plot_si_threshold_sensitivity(data: dict, figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.50, 4.52), constrained_layout=True)
    _plot_endpoint_coverage(
        axes[0, 0], data, "sampling_energy", "Sampling-energy coverage"
    )
    _plot_endpoint_coverage(axes[0, 1], data, "binding_score", "Binding-score coverage")
    axes[0, 0].legend(frameon=False, loc="lower right")
    _plot_endpoint_speedups(axes[1, 0], data)
    _plot_total_binding_improvement(axes[1, 1], data)
    for label, ax in zip("ABCD", axes.flat):
        _panel_label(ax, label, x=-0.16)
    _save_figure(fig, figure_dir / "si_threshold_sensitivity")


def _write_captions(data: dict, figure_dir: Path) -> None:
    protocol = data["suite"]["protocol"]
    strict = data["analysis"]["endpoints"]["sampling_energy/strict"]
    captions = f"""# Draft figure captions

## Main figure

**FPT improves target-directed torsional sampling on the PoseBusters benchmark.**
(a) Mean paired-trial fraction across {len(data['systems'])} systems reaching the strict sampling-energy
target as a function of sampling step. Four systems satisfy the target at step
zero after bounded hard-start rejection. (b) Paired final energy gaps after
{protocol['steps']} steps with {protocol['replicas']} particles; negative values
indicate successful strict-target hits, and points below the diagonal favor FPT.
(c) Strict-target coverage and paired fixed-budget quality stratified by ligand
torsion count. A positive final-energy improvement favors FPT. (d) Per-system
fixed-budget energy improvement across torsion-count strata. (e) Median step
and timeout-capped sampling-time speedups by torsion count; a speedup above one
favors FPT. Failures are assigned the full
{protocol['steps']}-step sampling time. Results use {protocol['num_seeds']} paired
seeds beginning at {protocol['seed_base']} per system.

## Supplementary figure

**Final quality and optimization dynamics for representative systems missed by both
methods.** (a-b) Paired final gaps to the strict total-energy and binding-score
targets after {protocol['steps']} steps. Zero denotes target attainment; lower
values are better; $\Delta E=E_\mathrm{{best}}-E_\mathrm{{strict}}$.
(c-h) Running-best total-energy gap for each neither-hit
system. The energy-gap axis uses a log1p transform while retaining tick labels in
kcal/mol. FPT ends closer to the strict total-energy target in all six cases.

Primary strict coverage: MC {strict['mc_successes']}/{strict.get('trials', strict['systems'])}; FPT
{strict['fpt_successes']}/{strict.get('trials', strict['systems'])}.

## Detailed main figure

**FPT improves both target-reaching efficiency and fixed-budget quality, with
the largest gains in high-dimensional torsional searches.** (a) Torsion-only
sampling protocol, illustrated with the 14-torsion PoseBusters system
`7ELT_TYM`. The receptor and ligand rigid-body pose are fixed; MC and FPT
start from the same random torsion seeds and are evaluated with the same Vina
score and sampling budget. MC evolves independent trajectories without
inter-replica communication, whereas FPT
transports an interacting conformational ensemble through the shared torsional
state space. Molecular conformations are reconstructed from the real ligand
topology and torsion archives and rendered with a flat illustrative PyMOL style;
the receptor view is the actual local protein surface. The projected landscape
is an empirical archive-density landscape, $-k_\\mathrm{{B}}T\\log p$, after
periodic-feature PCA of the pooled MC/FPT archive. Stars mark each method's
lowest-total-score archive member; the adjacent structures and labels show those
conformers and their total Vina scores in kcal/mol;
the update arrows are schematic rather than stored time-resolved frames.
(b-c) Strict sampling-energy coverage versus sampling step and
estimated sampling wall time; shading denotes pointwise 95% intervals clustered
over molecular systems. Wall-time first passage is estimated by linearly scaling each
method's measured full-budget sampling time by its hit step. (d) Paired final gap
to the strict target, $\Delta E=E_\mathrm{{best}}-E_\mathrm{{strict}}$.
(e) Per-system fixed-budget energy improvement stratified
by torsion count; half-violins show kernel-density estimates, points show
individual systems, and diamonds with whiskers show median effects and 95%
system-bootstrap intervals. (f) Point-range estimates of median step and
timeout-capped wall-time speedups $S_\mathrm{{MC/FPT}}$, with 95%
system-bootstrap intervals. Unity denotes equal efficiency and values above
unity favor FPT. Each system-level effect summarizes {protocol['num_seeds']}
paired seeds beginning at {protocol['seed_base']}.

## Supplementary complexity figure

**Torsional complexity determines the relative sampling dynamics of MC and
FPT.** (a-e) Strict-target anytime coverage, $P_\mathrm{{hit}}(t)$, in five
torsion-count strata defined by $N_\mathrm{{tor}}$. (f)
Composition of both-hit, FPT-only, and neither-hit outcomes at the
{protocol['steps']}-step budget. FPT's advantage emerges primarily beyond seven
ligand torsions.

## Supplementary threshold-sensitivity figure

**The FPT result is robust across score definitions and target strictness.**
(a-b) Final system coverage for easy, medium, and strict total-energy and
binding-score targets. (c) Aggregate timeout-capped time-to-target speedup for
all six endpoints, $S_\mathrm{{MC/FPT}}^\mathrm{{cap}}$; values
above one favor FPT. (d) Paired fixed-budget total-energy and binding-score
improvements. Representative systems missing the strict total-energy target with
both methods are highlighted. Best-total and best-binding values may arise from
different sampled conformations.

## Supplementary diversity figure

**FPT does not obtain its target-reaching advantage by collapsing conformational
diversity.** Paired numbers of unique strict-threshold clusters among the
{sum(system['outcome'] == 'both' for system in data['systems'])} systems reaching the strict total-energy target with
both methods; $N_\mathrm{{clust}}$ denotes the number of unique clusters.
"""
    (figure_dir / "FIGURE_CAPTIONS.md").write_text(captions)


def _write_detailed_analysis(data: dict, figure_dir: Path) -> None:
    systems = data["systems"]
    runs = data["runs"]
    strict_stat = next(
        row
        for row in data["statistics"]["endpoints"]
        if row["score"] == "sampling_energy" and row["threshold_level"] == "strict"
    )
    steps = (0, 100, 250, 500, 750, 1000)
    times = (0.0001, 0.001, 0.005, 0.01, 0.05, 0.1)
    lines = [
        "# Detailed PoseBusters figure analysis",
        "",
        "## Strict sampling-energy anytime coverage",
        "",
        "| Budget | MC | FPT |",
        "|---:|---:|---:|",
    ]
    for step in steps:
        counts = [
            sum(
                run[f"{method}_hit_step"] is not None
                and run[f"{method}_hit_step"] <= step
                for run in runs
            )
            for method in ("mc", "fpt")
        ]
        lines.append(
            f"| {step} steps | {counts[0]}/{len(runs)} | {counts[1]}/{len(runs)} |"
        )
    lines.extend(
        [
            "",
            "| Estimated sampling time | MC | FPT |",
            "|---:|---:|---:|",
        ]
    )
    for elapsed in times:
        counts = [
            sum(
                run[f"{method}_hit_seconds"] is not None
                and run[f"{method}_hit_seconds"] <= elapsed
                for run in runs
            )
            for method in ("mc", "fpt")
        ]
        lines.append(
            f"| {elapsed * 1000:g} ms | {counts[0]}/{len(runs)} | "
            f"{counts[1]}/{len(runs)} |"
        )
    lines.extend(
        [
            "",
            "MC resolves more very cheap trials at short wall times, whereas FPT "
            "wins the harder tail and reaches "
            f"{strict_stat['fpt_successes']} rather than "
            f"{strict_stat['mc_successes']} of {strict_stat['trials']} paired trials.",
            "",
            "## Fixed-budget total-energy improvement by torsion count",
            "",
            "Positive values are `MC best - FPT best` and favor FPT.",
            "",
            "| Torsions | N | Median improvement | FPT better |",
            "|---:|---:|---:|---:|",
        ]
    )
    for label in TORSION_BINS:
        values = [
            system["total_improvement"]
            for system in systems
            if _torsion_bin_for_count(system["torsions"]) == label
        ]
        lines.append(
            f"| {label} | {len(values)} | {np.median(values):.3f} kcal/mol | "
            f"{sum(value > 0 for value in values)}/{len(values)} |"
        )
    joint = [system for system in systems if system["outcome"] == "both"]
    mc_clusters = [system["mc_unique_clusters"] for system in joint]
    fpt_clusters = [system["fpt_unique_clusters"] for system in joint]
    lines.extend(
        [
            "",
            "The stratified fixed-budget effects summarize the median paired-seed "
            "quality gain within each molecular system.",
            "",
            "## Diversity among jointly successful systems",
            "",
            f"Among {len(joint)} joint strict hits, median unique-cluster counts are "
            f"MC {np.median(mc_clusters):.1f} and FPT {np.median(fpt_clusters):.1f}. "
            f"FPT has more clusters in {sum(f > m for m, f in zip(mc_clusters, fpt_clusters))} "
            f"systems, MC in {sum(m > f for m, f in zip(mc_clusters, fpt_clusters))}, "
            f"with {sum(m == f for m, f in zip(mc_clusters, fpt_clusters))} ties. "
            "Thus there is no median cluster-count loss, although FPT has fewer "
            "clusters in a similar number of individual systems.",
            "",
            "## Interpretation boundary",
            "",
            f"These are paired results from {data['suite']['protocol']['num_seeds']} "
            "random seeds per system. Confidence intervals resample molecular "
            "systems after reducing paired seeds within each system, avoiding "
            "pseudoreplication of the seed trials.",
            "",
        ]
    )
    (figure_dir / "DETAILED_ANALYSIS.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--figure-dir",
        type=Path,
        help="default: <output-dir>/paper_figures",
    )
    parser.add_argument(
        "--complexity-focused-only",
        action="store_true",
        help="write only the alternative complexity-focused main figure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure_dir = args.figure_dir or args.output_dir / "paper_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()
    data = _load_run(args.output_dir)
    data["output_dir"] = args.output_dir
    data["statistics"] = statistical_analysis.analyze(args.output_dir, figure_dir)
    if args.complexity_focused_only:
        plot_complexity_focused_main_figure(data, figure_dir)
        print(f"wrote complexity-focused main figure to {figure_dir}")
        return
    plot_main_figure(data, figure_dir)
    plot_detailed_main_figure(data, figure_dir)
    plot_si_figure(data, figure_dir)
    plot_si_complexity(data, figure_dir)
    plot_si_threshold_sensitivity(data, figure_dir)
    plot_si_diversity(data, figure_dir)
    _write_captions(data, figure_dir)
    _write_detailed_analysis(data, figure_dir)
    print(f"wrote paper figures to {figure_dir}")


if __name__ == "__main__":
    main()
