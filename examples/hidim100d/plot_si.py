#!/usr/bin/env python3
"""Supplementary figure for the 100-dimensional periodic landscape.

(a) Energy per coordinate of the FPT ensembles (RMSprop 0.14, shared
    bandwidth refreshed every update) at beta = 5, four seeds (thin) and their median (thick), with the
    Langevin (step 0.003) reference value; the shaded span marks the range of t_id.
(b) Stationary energy per coordinate of exact per-particle row-mass FPT,
    relative to Langevin, versus the prescribed row mass K for D = 5, 10, 20
    and 100 (started from equilibrated Langevin ensembles; N = 5120).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from examples.common.identification import identification
from examples.hidim100d.plot_discovery import hundred_d_equilibrium
from examples.common.figure_style import (
    CANVAS_COLOR,
    FPT_COLOR,
    NEITHER_COLOR,
    TEXT_COLOR,
    clean_axis,
    configure_matplotlib,
    mix_with_white,
    panel_header,
    save_figure,
)

DATA_ROOT = Path("data")  # simulation outputs written by the run commands
FIGURE_ROOT = Path("figures")
DATA = DATA_ROOT / "numerical_hidim_periodic_discovery"
DEFAULT_OUTPUT = FIGURE_ROOT / "hidim_periodic_si"
FPT_RUN = "tuned/beta{beta}_fpt_fused_row_k20_40_p1_visits"
LD_RUN = "tuned/beta{beta}_ld_noscore_visits"
DIMSCAN = (
    ("5", "equilibrium_dimscan/D5_beta3.53"),
    ("10", "equilibrium_dimscan/D10_beta4.3"),
    ("20", "equilibrium_dimscan/D20_beta5.35"),
    ("100", "equilibrium_beta5"),
)
BETA_COLORS = {5: FPT_COLOR, 8: NEITHER_COLOR}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def runs(path: Path) -> list[dict]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["runs"]


def trace(run: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(run["trace"]["step"], dtype=float), np.asarray(run["trace"][key], dtype=float)


def plot_energy(ax, data: Path) -> None:
    reference = hundred_d_equilibrium(data / "tuned")
    for beta in (5,):
        color = BETA_COLORS[beta]
        fpt = runs(data / FPT_RUN.format(beta=beta))
        replicas = int(json.loads((data / FPT_RUN.format(beta=beta) / "summary.json")
                                  .read_text(encoding="utf-8"))["protocol"]["replicas"])
        first = [identification(r["trace"]["step"], r["trace"]["quench_counts"], replicas,
                                reference)[1]
                 for r in fpt]
        ax.axvspan(min(first), max(first), color=mix_with_white(color, 0.80), linewidth=0, zorder=0)
        for run in fpt:
            x, energy = trace(run, "energy_per_dim")
            ax.plot(x[1:], energy[1:], color=mix_with_white(color, 0.55), linewidth=0.7, zorder=2)
        x, _ = trace(fpt[0], "energy_per_dim")
        median = np.median([trace(r, "energy_per_dim")[1] for r in fpt], axis=0)
        ax.plot(x[1:], median[1:], color=color, linewidth=1.6, zorder=3)
        reference = float(np.mean([r["final_energy_per_dim"] for r in runs(data / LD_RUN.format(beta=beta))
                                   if r["method"] == "ld" and float(r["param"]) == 0.003]))
        ax.axhline(reference, color=color, linestyle=":", linewidth=1.1, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=(1, 20000), ylim=(1e-4, 3.0), xlabel="Updates", ylabel="Energy per coordinate")
    ax.legend(handles=[
        Line2D([], [], color=BETA_COLORS[5], linewidth=1.6, label=r"FPT, $\beta=5$"),
        Line2D([], [], color=TEXT_COLOR, linestyle=":", linewidth=1.1,
               label=r"Langevin, $\epsilon=0.003$"),
        plt.Rectangle((0, 0), 1, 1, color=mix_with_white(TEXT_COLOR, 0.82), linewidth=0,
                      label=r"Basins identified ($t_{\rm id}$)"),
    ], frameon=False, loc="lower left", ncol=2, handlelength=1.6, handletextpad=0.3,
        columnspacing=0.8, labelspacing=0.12, borderaxespad=0.15, fontsize=8.0)
    clean_axis(ax)
    panel_header(ax, "a", "Thermal state of the ensemble", x=-0.22, pad=4.0)


def plot_dimension_scan(ax, data: Path) -> None:
    shades = (0.62, 0.42, 0.2, 0.0)
    for (dim, folder), shade in zip(DIMSCAN, shades):
        base = data / folder
        ld = [r for r in runs(base / "ld_continue") if float(r["param"]) == 0.003][0]
        reference = float(ld["final_energy_per_dim"])
        points = {}
        for sub in ("perparticle_euler_smallK", "perparticle_euler"):
            if not (base / sub / "summary.json").exists():
                continue
            for run in runs(base / sub):
                if float(run["dt"]) == 0.03:
                    points[float(run["row_mass"])] = float(run["final_energy_per_dim"]) / reference
        masses = np.asarray(sorted(points))
        color = mix_with_white(FPT_COLOR, shade)
        ax.plot(masses, [points[m] for m in masses], color=color, marker="o", markersize=3.6,
                markeredgecolor=CANVAS_COLOR, markeredgewidth=0.5, linewidth=1.4,
                label=rf"$D={dim}$", zorder=3)
    ax.axhline(1.0, color=TEXT_COLOR, linestyle=":", linewidth=1.0, zorder=1)
    ax.axhline(0.0, color=TEXT_COLOR, linewidth=0.6, zorder=1)
    ax.set_xscale("log")
    ax.set(xlim=(3, 500), ylim=(-0.08, 1.05), xlabel=r"Row mass $K$",
           ylabel=r"$\langle U\rangle/D$ relative to Langevin")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.93), handlelength=1.8,
              handletextpad=0.35, labelspacing=0.18, borderaxespad=0.1, fontsize=8.6)
    clean_axis(ax)
    panel_header(ax, "b", "Stationary energy versus dimension", x=-0.22, pad=4.0)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(6.69, 2.65))
    plot_energy(axes[0], args.data)
    plot_dimension_scan(axes[1], args.data)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.17, top=0.87, wspace=0.38)
    save_figure(fig, args.output_base)


if __name__ == "__main__":
    main()
