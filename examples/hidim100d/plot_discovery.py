#!/usr/bin/env python3
"""Main-text stable-basin discovery in the 100-dimensional periodic landscape.

Split-half identification of the 101 analytically stable basins at beta = 5:
four shared-bandwidth FPT realizations and their mean, against Langevin means
labeled by step size; the dotted line is the value for an equilibrium ensemble
of the same size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import patheffects
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from examples.common.figure_style import (
    CANVAS_COLOR,
    FPT_COLOR,
    FPT_FILL_COLOR,
    TEXT_COLOR,
    ULA_COLOR,
    clean_axis,
    configure_matplotlib,
    mix_with_white,
    save_figure,
)
from examples.common.identification import identification, split_half_probability

DATA_ROOT = Path("data")  # simulation outputs written by the run commands
FIGURE_ROOT = Path("figures")
DEFAULT_OUTPUT = FIGURE_ROOT / "hidim_discovery"
DEFAULT_100D_DATA = DATA_ROOT / "numerical_hidim_periodic_discovery" / "tuned"


HUNDRED_D_BETA = 5.0
HUNDRED_D_FPT_RUN = "beta5_fpt_fused_row_k20_40_p1_visits"
HUNDRED_D_BASELINE_RUN = "beta5_ld_noscore_visits"
HUNDRED_D_EQUILIBRIUM_RUN = "../equilibrium_beta5/ld_split_half"


def hundred_d_equilibrium(path: Path) -> float:
    """Expected identified fraction of the equilibrium Langevin ensembles in 100D."""
    summary = json.loads((path / HUNDRED_D_EQUILIBRIUM_RUN / "summary.json").read_text())
    replicas = int(summary["protocol"]["replicas"])
    return float(np.mean([split_half_probability(np.asarray(r["trace"]["quench_counts"]),
                                                 replicas).mean() for r in summary["runs"]]))


def load_hundred_d(path: Path) -> dict:
    """Quench-basin identification for the 100D runs, keyed by method."""
    series = {}
    reference = hundred_d_equilibrium(path)
    for run_dir in (HUNDRED_D_FPT_RUN, HUNDRED_D_BASELINE_RUN):
        summary = json.loads((path / run_dir / "summary.json").read_text(encoding="utf-8"))
        if float(summary["protocol"]["beta"]) != HUNDRED_D_BETA:
            raise ValueError(f"{run_dir} is not a beta = {HUNDRED_D_BETA:g} run")
        for run in summary["runs"]:
            key = ("fpt", None) if run["method"].startswith("fpt") else (
                run["method"], float(run["param"]) if run["method"] == "ld" else None)
            entry = series.setdefault(key, {"steps": None, "coverage": [],
                                            "identified": [], "t_id": []})
            entry["steps"] = np.asarray(run["trace"]["step"], dtype=float)
            entry["coverage"].append(np.asarray(run["trace"]["quench_cum"], dtype=float) / 101.0)
            source = "quench_counts" if run["method"].startswith("fpt") else "distinct_visitors"
            fraction, first = identification(entry["steps"], run["trace"][source],
                                             int(summary["protocol"]["replicas"]), reference)
            entry["identified"].append(fraction)
            entry["t_id"].append(first)
    for entry in series.values():
        entry["coverage"] = np.asarray(entry["coverage"])
        entry["identified"] = np.asarray(entry["identified"])
    series[("equilibrium", None)] = reference
    return series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hundred-d-data", type=Path, default=DEFAULT_100D_DATA)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    hundred = load_hundred_d(args.hundred_d_data)

    fig, ax = plt.subplots(figsize=(6.69, 2.25))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.96)
    ld_steps = sorted(step for (method, step) in hundred if method == "ld")
    ld_shades = np.linspace(0.0, 0.62, len(ld_steps))[::-1]
    for step, shade in zip(ld_steps, ld_shades):
        entry = hundred[("ld", step)]
        ax.step(entry["steps"][1:], entry["identified"].mean(axis=0)[1:], where="post",
                color=mix_with_white(ULA_COLOR, float(shade)), linestyle="--", linewidth=1.25,
                zorder=3)
    fpt = hundred[("fpt", None)]
    for trajectory in fpt["identified"]:
        ax.step(fpt["steps"][1:], trajectory[1:], where="post", color=FPT_FILL_COLOR,
                linewidth=0.8, zorder=4)
    ax.step(fpt["steps"][1:], fpt["identified"].mean(axis=0)[1:], where="post", color=FPT_COLOR,
            linewidth=1.85, zorder=5)
    ax.axhline(hundred[("equilibrium", None)], color=TEXT_COLOR, linestyle=":", linewidth=1.0,
               zorder=2)
    ax.set_xscale("log")
    ax.set(xlabel="Updates", ylabel="Basins identified", xlim=(5, 20000), ylim=(-0.02, 1.04),
           yticks=(0.0, 0.25, 0.5, 0.75, 1.0))
    clean_axis(ax)
    for step, shade in zip(ld_steps, ld_shades):
        entry = hundred[("ld", step)]
        index = int(np.flatnonzero(entry["identified"].mean(axis=0) >= 0.70)[0])
        ax.annotate(f"{step:g}", xy=(entry["steps"][index], 0.70), xytext=(-4.0, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8.2,
                    color=mix_with_white(ULA_COLOR, float(min(shade, 0.40))),
                    path_effects=[patheffects.withStroke(linewidth=2.4, foreground=CANVAS_COLOR)],
                    zorder=6)
    ax.legend(
        handles=[
            Line2D([], [], color=FPT_COLOR, linewidth=1.85, label="FPT, shared bandwidth"),
            Line2D([], [], color=ULA_COLOR, linestyle="--", linewidth=1.25,
                   label=r"Langevin (step $\epsilon$)"),
            Line2D([], [], color=TEXT_COLOR, linestyle=":", linewidth=1.0,
                   label="Equilibrium ensemble"),
        ],
        frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.98), handlelength=1.8,
        handletextpad=0.35, labelspacing=0.18, borderaxespad=0.0, fontsize=8.6,
    )
    save_figure(fig, args.output_base)


if __name__ == "__main__":
    main()
