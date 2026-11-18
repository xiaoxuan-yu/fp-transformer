#!/usr/bin/env python3
"""SI diagnostic: basin masses of the 3D eight-well target against the exact Fokker--Planck solution.

beta = 3 beta_0, start from the Boltzmann law restricted to the initial well for all curves.
Wells are grouped by the number of coordinates flipped relative to the
initial well (1, 2, 3; equilibrium masses 3/8, 3/8, 1/8).  The grid oracle
(oracle_periodic_fp3d.py) is placed on the update axis with the Langevin step
(t = 0.008 n).  Langevin, shared-bandwidth FPT with explicit Euler at the
Langevin step, and the RMSprop shared-bandwidth protocol are means over eight
seeds; the exact-row-mass estimator (diag_3d_perparticle_rowmass.py, explicit
Euler with the Langevin step) at K = 3, the midpoint of the shared row-mass
band, is the mean over its four seeds.  Masses are shown relative to
their equilibrium values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np

from examples.common.figure_style import (
    FPT_COLOR,
    MC_COLOR,
    REFERENCE_COLOR,
    TEXT_COLOR,
    ULA_COLOR,
    clean_axis,
    configure_matplotlib,
    mix_with_white,
    panel_header,
    save_figure,
)

DATA_ROOT = Path("data")  # released data, relative to the repository root
FIGURE_ROOT = Path("figures")
DATA = DATA_ROOT / "numerical_lowdim"
DEFAULT_OUTPUT = FIGURE_ROOT / "periodic_oracle_basin_mass"
EPSILON = 0.008
FP_REFERENCE_COLOR = mix_with_white(TEXT_COLOR, 0.62)
FP_REFERENCE_WIDTH = 3.2
CLASSES = np.array([bin(b).count("1") for b in range(8)])
EXACT_ESTIMATOR_COLOR = MC_COLOR
RMSPROP_COLOR = mix_with_white(FPT_COLOR, 0.55)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, default=DATA / "oracle_n3_bb/oracle.json")
    parser.add_argument("--main", type=Path, default=DATA / "periodic_n3_rms_main_bb_visits/summary.json",
                        help="Langevin baseline and RMSprop shared-bandwidth protocol")
    parser.add_argument("--shared", type=Path,
                        default=DATA / "periodic_n3_shared_euler_bb_visits/summary.json",
                        help="shared-bandwidth FPT with explicit Euler")
    parser.add_argument("--exact", type=Path, nargs="*",
                        default=sorted((DATA / "oracle_n3_exact_k3_bb_visits").glob("seed*.json")))
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def grouped(masses: np.ndarray) -> np.ndarray:
    masses = np.asarray(masses, dtype=float)
    return np.stack([masses[..., CLASSES == d].sum(axis=-1) for d in (1, 2, 3)], axis=-1)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    oracle = json.loads(args.oracle.read_text())["records"]
    o_updates = np.array([r["update"] for r in oracle], dtype=float)
    o_mass = grouped(np.array([r["basin_mass"] for r in oracle]))

    main_runs = json.loads(args.main.read_text())["runs"]
    shared_runs = json.loads(args.shared.read_text())["runs"]
    curves = {}
    for key, runs, method in (("ula_fixed", main_runs, "ula_fixed"),
                              ("fpt_rmsprop", main_runs, "fpt_fixed"),
                              ("fpt_fixed", shared_runs, "fpt_fixed")):
        steps = np.asarray(next(iter(runs.values()))[method]["checkpoint"], dtype=float)
        counts = np.asarray([runs[s][method]["counts"] for s in runs], dtype=float)
        frac = counts / counts.sum(axis=-1, keepdims=True)
        curves[key] = (steps, grouped(frac).mean(axis=0))
    exact = None
    if args.exact:
        series = [json.loads(p.read_text())["records"] for p in args.exact]
        steps = np.array([r["update"] for r in series[0]], dtype=float)
        frac = np.asarray([[np.asarray(r["counts"]) / sum(r["counts"]) for r in s] for s in series])
        exact = (steps, grouped(frac).mean(axis=0))

    fig, axes = plt.subplots(1, 3, figsize=(6.69, 2.65), sharey=True)
    equilibrium = np.array([3.0, 3.0, 1.0]) / 8.0
    for k, ax in enumerate(axes):
        scale = equilibrium[k]
        ax.axhline(1.0, color=REFERENCE_COLOR, linestyle=":", linewidth=0.9, zorder=1)
        ax.plot(o_updates, o_mass[:, k] / scale, color=FP_REFERENCE_COLOR,
                linewidth=FP_REFERENCE_WIDTH, solid_capstyle="round", zorder=2)
        steps, mass = curves["ula_fixed"]
        ax.plot(steps, mass[:, k] / scale, color=ULA_COLOR, linestyle="--", linewidth=1.3,
                zorder=3)
        if exact is not None:
            steps, mass = exact
            ax.plot(steps, mass[:, k] / scale, color=EXACT_ESTIMATOR_COLOR, linestyle="-.",
                    linewidth=1.3, zorder=3)
        steps, mass = curves["fpt_rmsprop"]
        ax.plot(steps, mass[:, k] / scale, color=RMSPROP_COLOR, linewidth=1.1, zorder=4)
        steps, mass = curves["fpt_fixed"]
        ax.plot(steps, mass[:, k] / scale, color=FPT_COLOR, linewidth=1.6, zorder=5)
        ax.set_xscale("symlog", linthresh=1.0, linscale=0.5)
        ax.xaxis.set_minor_locator(matplotlib.ticker.SymmetricalLogLocator(
            base=10, linthresh=1.0, subs=np.arange(2, 10)))
        ax.set(xlim=(0, 13500), xticks=(0, 1e2, 1e4), ylim=(-0.03, 1.3),
               yticks=(0.0, 0.5, 1.0), xlabel="Updates")
        clean_axis(ax)
        panel_header(ax, "abc"[k], f"{k + 1} coordinate{'s' if k else ''} flipped", x=-0.12, pad=4.0)
    axes[0].set_ylabel("Mass / equilibrium")
    handles = [
        Line2D([], [], color=FP_REFERENCE_COLOR, linewidth=FP_REFERENCE_WIDTH,
               label="Exact Fokker–Planck"),
        Line2D([], [], color=ULA_COLOR, linestyle="--", linewidth=1.3, label="Langevin"),
        Line2D([], [], color=FPT_COLOR, linewidth=1.6, label="FPT (shared bandwidth)"),
        Line2D([], [], color=RMSPROP_COLOR, linewidth=1.1,
               label="FPT (shared bandwidth, RMSprop)"),
    ]
    if exact is not None:
        handles.append(Line2D([], [], color=EXACT_ESTIMATOR_COLOR, linestyle="-.", linewidth=1.3,
                              label=r"FPT (exact row mass, $K=3$)"))
    handles.append(Line2D([], [], color=REFERENCE_COLOR, linestyle=":", linewidth=0.9,
                          label="Equilibrium"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=8.2, handlelength=1.8, columnspacing=1.2,
               labelspacing=0.2)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.17, top=0.74, wspace=0.12)
    save_figure(fig, args.output_base)


if __name__ == "__main__":
    main()
