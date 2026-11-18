#!/usr/bin/env python3
"""Plot the held-out PeriodicSin4 endpoint-kernel MH continuation audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from examples.common.figure_style import (
    CANVAS_COLOR,
    FPT_COLOR,
    FPT_WASH_COLOR,
    REFERENCE_FILL_COLOR,
    clean_axis,
    configure_matplotlib,
    panel_label,
    save_figure,
)


DATA_ROOT = Path("data")  # simulation outputs written by the run commands
FIGURE_ROOT = Path("figures")
DEFAULT_DATA = (
    DATA_ROOT
    / "numerical_periodic_endpoint_kernel"
    / "n3_shared_euler_bb"
)
DEFAULT_OUTPUT = FIGURE_ROOT / "periodic_endpoint_continuation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_metrics(path: Path) -> list[dict[str, float | int | str]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            rows.append(
                {
                    "seed": int(raw["seed"]),
                    "schedule": raw["schedule"],
                    "continuation_steps": int(raw["continuation_steps"]),
                    "occupancy_entropy": float(raw["occupancy_entropy"]),
                    "basin_tv": float(raw["basin_tv"]),
                    "energy_ks": float(raw["energy_ks"]),
                }
            )
    return rows


def metric_matrix(
    rows: list[dict[str, float | int | str]],
    *,
    schedule: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["schedule"] == schedule]
    checkpoints = np.asarray(
        sorted({int(row["continuation_steps"]) for row in selected}), dtype=float
    )
    seeds = np.asarray(sorted({int(row["seed"]) for row in selected}), dtype=int)
    matrix = np.empty((seeds.size, checkpoints.size), dtype=float)
    for seed_index, seed in enumerate(seeds):
        by_step = {
            int(row["continuation_steps"]): float(row[metric])
            for row in selected
            if int(row["seed"]) == seed
        }
        matrix[seed_index] = [by_step[int(step)] for step in checkpoints]
    return checkpoints, seeds, matrix


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    rows = load_metrics(args.data_dir / "metrics.csv")
    summary = json.loads((args.data_dir / "summary.json").read_text(encoding="utf-8"))
    reference = summary["finite_iid_reference"]

    panels = (
        (
            "occupancy_entropy",
            "Occupancy entropy",
            r"$H_{\rm occ}$",
            (0.993, 1.0008),
        ),
        (
            "basin_tv",
            "Basin-mass total variation",
            "TV from uniform basin mass",
            (-0.002, 0.07),
        ),
        (
            "energy_ks",
            "Energy-distribution KS",
            "KS to target energy CDF",
            (-0.008, 0.22),
        ),
    )
    schedule_styles = {
        "fixed": {
            "color": FPT_COLOR,
            "wash": FPT_WASH_COLOR,
            "linestyle": "-",
            "marker": "o",
            "filled": True,
            "label": "Shared-bandwidth FPT endpoint",
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(6.69, 2.25))
    for panel_index, (ax, panel) in enumerate(zip(axes, panels)):
        metric, title, ylabel, ylim = panel
        band = reference[metric]
        ax.axhspan(
            float(band["lower_95"]),
            float(band["upper_95"]),
            color=REFERENCE_FILL_COLOR,
            zorder=0,
        )
        for schedule in schedule_styles:
            checkpoints, _seeds, matrix = metric_matrix(
                rows, schedule=schedule, metric=metric
            )
            style = schedule_styles[schedule]
            for values in matrix:
                ax.plot(
                    checkpoints,
                    values,
                    color=style["wash"],
                    linewidth=0.75,
                    linestyle=style["linestyle"],
                    zorder=1,
                )
            ax.plot(
                checkpoints,
                matrix.mean(axis=0),
                color=style["color"],
                linewidth=1.45,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=style["color"] if style["filled"] else CANVAS_COLOR,
                markeredgecolor=style["color"],
                markeredgewidth=0.65,
                markersize=3.2,
                zorder=2,
            )
        ax.set_xscale("symlog", base=10, linthresh=1, linscale=0.7)
        ax.set(
            title=title,
            xlabel="MH continuation steps",
            ylabel=ylabel,
            xlim=(0, 215),
            ylim=ylim,
            xticks=(0, 1, 5, 25, 200),
        )
        ax.set_xticklabels(("0", "1", "5", "25", "200"))
        if metric == "occupancy_entropy":
            ax.set_yticks((0.994, 0.997, 1.000))
        clean_axis(ax)
        panel_label(ax, chr(ord("a") + panel_index), x=-0.18)

    legend_handles = [
        Line2D(
            [],
            [],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor=style["color"] if style["filled"] else CANVAS_COLOR,
            markeredgecolor=style["color"],
            markeredgewidth=0.65,
            markersize=3.6,
            linewidth=1.45,
            label=style["label"],
        )
        for style in schedule_styles.values()
    ]
    legend_handles.append(
        Patch(
            facecolor=REFERENCE_FILL_COLOR,
            edgecolor="none",
            label="Finite-i.i.d. 95% range",
        )
    )
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.985),
        ncol=2,
        columnspacing=1.0,
        handlelength=2.0,
        handletextpad=0.4,
        fontsize=9.0,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.225,
        top=0.76,
        wspace=0.50,
    )
    save_figure(fig, args.output_base)


if __name__ == "__main__":
    main()
