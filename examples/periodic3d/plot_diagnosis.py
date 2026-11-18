#!/usr/bin/env python3
"""Main-text diagnosis of the shared-bandwidth approximation on the 3D eight-well target.

beta = 3 beta_0, start from the Boltzmann law restricted to the initial well for every curve.
Every particle method advances by explicit Euler with the Langevin step
dt = 0.008, so updates map to Fokker--Planck time as t = 0.008 n for all curves.
(a) fixed-temperature shared-bandwidth FPT endpoint of the representative seed;
(b) split-half well identification and normalized occupancy entropy; (c) basin mass
in the one- and three-flip well groups against the grid Fokker--Planck
solution; (d) beta<U> against the target value.  Shared-bandwidth FPT
(row-mass band [2, 4]), no-score, and Langevin curves are means over eight
seeds; the exact row-mass estimator at the band midpoint K = 3 is the mean
over four seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from examples.common.figure_style import (
    CANVAS_COLOR,
    FPT_COLOR,
    FPT_MARKER_COLOR,
    FPT_WASH_COLOR,
    GRID_COLOR,
    LANDSCAPE_HIGH_COLOR,
    LANDSCAPE_LOW_COLOR,
    MC_COLOR,
    PANE_EDGE_COLOR,
    REFERENCE_COLOR,
    TEXT_COLOR,
    ULA_COLOR,
    ULA_WASH_COLOR,
    WARM_ACCENT_COLOR,
    clean_axis,
    configure_matplotlib,
    mix_with_white,
    panel_header,
    save_figure,
)
from examples.common.identification import equilibrium_identification, identification

DATA_ROOT = Path("data")  # simulation outputs written by the run commands
FIGURE_ROOT = Path("figures")
DATA = DATA_ROOT / "numerical_lowdim"
DEFAULT_OUTPUT = FIGURE_ROOT / "periodic_diagnosis"
FP_REFERENCE_COLOR = mix_with_white(TEXT_COLOR, 0.62)
FP_REFERENCE_WIDTH = 3.2
CLASSES = np.array([bin(b).count("1") for b in range(8)])
EXACT3_COLOR = MC_COLOR
EXACT_STYLES = (
    ("k3", EXACT3_COLOR, "-.", r"Exact row mass, $K=3$"),
)


def periodic_sin4_centers() -> np.ndarray:
    values = np.asarray((-math.pi / 2.0, math.pi / 2.0))
    mesh = np.meshgrid(values, values, values, indexing="ij")
    return np.stack(mesh, axis=-1).reshape(-1, 3)


def probability_energy_threshold(
    beta: float,
    *,
    probability: float = 0.5,
    resolution: int = 84,
) -> float:
    grid = np.linspace(-math.pi, math.pi, resolution)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    energy = np.exp(
        3.0
        * (
            3.0
            - np.sin(x) ** 4
            - np.sin(y) ** 4
            - np.sin(z) ** 4
        )
    ) - 1.0
    weight = np.exp(-beta * energy)
    order = np.argsort(energy, axis=None)
    energy_sorted = energy.ravel()[order]
    cumulative = np.cumsum(weight.ravel()[order])
    cumulative /= cumulative[-1]
    return float(energy_sorted[np.searchsorted(cumulative, probability)])


def surface_branches(
    energy_level: float,
    *,
    resolution: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    target_sum = 3.0 - math.log(energy_level + 1.0) / 3.0
    values = np.linspace(-math.pi, math.pi, resolution)
    x_grid, y_grid = np.meshgrid(values, values)
    rhs = target_sum - np.sin(x_grid) ** 4 - np.sin(y_grid) ** 4
    valid = (rhs >= 0.0) & (rhs <= 1.0)
    arcs = np.arcsin(np.clip(rhs, 0.0, 1.0) ** 0.25)
    return x_grid, y_grid, [
        np.where(valid, arcs, np.nan),
        np.where(valid, math.pi - arcs, np.nan),
        np.where(valid, -arcs, np.nan),
        np.where(valid, -math.pi + arcs, np.nan),
    ]


def t_interval(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean and two-sided 95% Student-t half width over independent seeds."""
    from scipy import stats

    values = np.asarray(values, dtype=float)
    count = values.shape[axis]
    mean = values.mean(axis=axis)
    half = stats.t.ppf(0.975, count - 1) * values.std(axis=axis, ddof=1) / math.sqrt(count)
    return mean, half


def load_three_d_seeds(path: Path) -> dict:
    """Split-half well identification and occupancy entropy per protocol and seed.

    Langevin output is its trajectory history, so the replicas that have visited a well
    (``distinct_visitors``) supply it; FPT and no-score output is the current ensemble,
    so the particles currently in the well (``counts``) supply it.
    """
    runs = json.loads((path / "summary.json").read_text(encoding="utf-8"))["runs"]
    first_run = next(iter(runs.values()))
    out = {}
    for protocol in first_run:
        checkpoints = np.asarray(first_run[protocol]["checkpoint"], dtype=float)
        entropy = np.asarray([runs[seed][protocol]["entropy"] for seed in runs], dtype=float)
        source = "distinct_visitors" if protocol.startswith("ula") else "counts"
        replicas = int(sum(first_run[protocol]["counts"][0]))
        reference = equilibrium_identification(np.full(8, 1 / 8), replicas)
        pairs = [identification(checkpoints, runs[seed][protocol][source], replicas, reference)
                 for seed in runs]
        out[protocol] = {"checkpoints": checkpoints, "entropy": entropy,
                         "identified": np.asarray([p[0] for p in pairs]),
                         "t_id": [p[1] for p in pairs]}
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, default=DATA / "periodic_n3_rms_main_bb_visits",
                        help="source of the Langevin baseline (Euler step 0.008)")
    parser.add_argument("--shared", type=Path, default=DATA / "periodic_n3_shared_euler_bb_visits",
                        help="shared-bandwidth FPT and no-score control, explicit Euler")
    parser.add_argument("--oracle", type=Path, default=DATA / "oracle_n3_bb/oracle.json")
    parser.add_argument("--exact-k3", type=Path, default=DATA / "oracle_n3_exact_k3_bb_visits")
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def grouped(masses: np.ndarray) -> np.ndarray:
    masses = np.asarray(masses, dtype=float)
    return np.stack([masses[..., CLASSES == d].sum(axis=-1) for d in (1, 2, 3)], axis=-1)


def occupancy_entropy(frac: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(frac > 0, frac * np.log(frac), 0.0)
    return -terms.sum(axis=-1) / math.log(8.0)


def load_exact(path: Path) -> dict:
    series = [json.loads(p.read_text())["records"] for p in sorted(path.glob("seed*.json"))]
    steps = np.array([r["update"] for r in series[0]], dtype=float)
    frac = np.asarray([[np.asarray(r["counts"], dtype=float) / sum(r["counts"]) for r in s]
                       for s in series])
    return {
        "steps": steps,
        "mass": grouped(frac).mean(axis=0),
        "coverage": np.asarray([[r["cumulative_covered"] for r in s] for s in series]) / 8.0,
        "identified": np.asarray([identification(steps, [r["counts"] for r in s],
                                                 int(sum(s[0]["counts"])), 1.0)[0] for s in series]),
        "entropy": occupancy_entropy(frac),
        "energy": np.asarray([[r["beta_mean_energy"] for r in s] for s in series]).mean(axis=0),
    }


def log_binned(steps: np.ndarray, values: np.ndarray, bins: int = 70):
    """Average a noisy checkpoint series over logarithmically spaced update bins."""
    edges = np.geomspace(steps[1], steps[-1] + 1, bins + 1)
    index = np.digitize(steps, edges)
    centers, means = [steps[0]], [values[0]]
    for k in range(1, bins + 1):
        mask = index == k
        if mask.any():
            centers.append(np.exp(np.log(steps[mask]).mean()))
            means.append(values[mask].mean())
    return np.asarray(centers), np.asarray(means)


def symlog_updates(ax, ticks=(0, 1e2, 1e4)) -> None:
    """Update axis that is linear on [0, 1] and logarithmic above, so update 0 is drawn."""
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.5)
    ax.set_xlim(0, 13500)
    ax.set_xticks(ticks)
    ax.xaxis.set_minor_locator(matplotlib.ticker.SymmetricalLogLocator(
        base=10, linthresh=1.0, subs=np.arange(2, 10)))


def positive(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0, values, np.nan)


def draw_landscape(fig, cell, beta: float, endpoint: np.ndarray) -> None:
    ax = fig.add_subplot(cell, projection="3d")
    ax.computed_zorder = False
    level = probability_energy_threshold(beta, probability=0.5)
    x_grid, y_grid, branches = surface_branches(level)
    for branch in branches:
        surface = ax.plot_surface(x_grid, y_grid, branch, color=LANDSCAPE_LOW_COLOR, alpha=1.0,
                                  linewidth=0, antialiased=True, shade=False, zorder=1)
        surface.set_rasterized(True)
    ax.scatter(endpoint[:, 0], endpoint[:, 1], endpoint[:, 2], s=2.4, color=FPT_MARKER_COLOR,
               linewidth=0, depthshade=False, rasterized=True, zorder=3)
    centers = periodic_sin4_centers()
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=22, facecolor="white",
               edgecolor=TEXT_COLOR, linewidth=0.75, depthshade=False, zorder=4)
    start = np.full(3, -math.pi / 2.0)
    ax.scatter(*start, s=52, marker="*", color=WARM_ACCENT_COLOR, edgecolor="white",
               linewidth=0.65, depthshade=False, zorder=5)
    ticks = (-math.pi / 2.0, math.pi / 2.0)
    labels = (r"$-\pi/2$", r"$\pi/2$")
    ax.set(xlim=(-math.pi, math.pi), ylim=(-math.pi, math.pi), zlim=(-math.pi, math.pi),
           xticks=ticks, yticks=ticks, zticks=ticks,
           xlabel=r"$\theta_1$", ylabel=r"$\theta_2$", zlabel=r"$\theta_3$")
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_zticklabels(labels)
    ax.set_box_aspect((1, 1, 1), zoom=1.0)
    ax.view_init(elev=21, azim=42)
    ax.tick_params(pad=0.2, labelsize=8.2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(CANVAS_COLOR)
        axis.pane.set_edgecolor(PANE_EDGE_COLOR)
        axis.pane.set_alpha(1.0)
        axis.label.set_size(8.6)
        axis.labelpad = -2.0
        axis._axinfo["grid"].update({"color": GRID_COLOR, "linewidth": 0.65, "linestyle": "-"})
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=FPT_COLOR,
                   markeredgecolor="none", markersize=3.8, label="Shared-bandwidth endpoint"),
            Line2D([], [], marker="s", linestyle="none", markerfacecolor=LANDSCAPE_LOW_COLOR,
                   markeredgecolor=LANDSCAPE_HIGH_COLOR, markeredgewidth=0.65, markersize=3.8,
                   label="50% mass"),
            Line2D([], [], marker="*", linestyle="none", markerfacecolor=WARM_ACCENT_COLOR,
                   markeredgecolor="white", markersize=5.6, label="Initial well"),
        ],
        frameon=False, loc="upper center", bbox_to_anchor=(0.50, 1.02), ncol=2,
        columnspacing=0.6, handletextpad=0.25, labelspacing=0.2, fontsize=8.2,
        borderaxespad=0.0,
    )


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    summary = json.loads((args.main / "summary.json").read_text())
    shared = json.loads((args.shared / "summary.json").read_text())
    beta = float(summary["protocol"]["beta_target"])
    if not math.isclose(beta, float(shared["protocol"]["beta_target"])):
        raise ValueError("shared-bandwidth control uses a different target beta")
    seeds = load_three_d_seeds(args.main)
    seeds.update({key: value for key, value in load_three_d_seeds(args.shared).items()
                  if key in ("fpt_fixed", "noscore_fixed")})
    with np.load(sorted(args.shared.glob("endpoints_seed*.npz"))[0]) as archive:
        endpoint = np.asarray(archive["fpt_fixed"])

    particle = {}
    for method, runs in (("fpt_fixed", shared["runs"]), ("ula_fixed", summary["runs"])):
        steps = np.asarray(next(iter(runs.values()))[method]["checkpoint"], dtype=float)
        counts = np.asarray([runs[s][method]["counts"] for s in runs], dtype=float)
        frac = counts / counts.sum(axis=-1, keepdims=True)
        energy = np.asarray([runs[s][method]["beta_target_mean_energy"] for s in runs], dtype=float)
        particle[method] = {"steps": steps, "mass": grouped(frac).mean(axis=0),
                            "energy": energy}
    exact = {"k3": load_exact(args.exact_k3)}

    oracle_json = json.loads(args.oracle.read_text())
    oracle = oracle_json["records"]
    o_steps = np.array([r["update"] for r in oracle], dtype=float)
    o_mass = grouped(np.array([r["basin_mass"] for r in oracle]))
    o_energy = np.array([r["beta_mean_energy"] for r in oracle], dtype=float)
    target_energy = float(oracle_json["protocol"]["equilibrium_beta_mean_energy"])

    fig = plt.figure(figsize=(6.69, 5.55))
    outer = fig.add_gridspec(2, 1, height_ratios=(1.25, 1.0), left=0.085, right=0.985,
                             bottom=0.165, top=0.92, hspace=0.36)
    top = outer[0].subgridspec(1, 2, width_ratios=(0.92, 1.0), wspace=0.22)
    right = top[0, 1].subgridspec(2, 1, hspace=0.30)
    bottom = outer[1].subgridspec(1, 3, width_ratios=(1.0, 1.0, 1.08), wspace=0.36)

    draw_landscape(fig, top[0, 0], beta, endpoint)
    fig.text(0.015, 0.94, "(a)", ha="left", va="baseline", color=TEXT_COLOR, fontsize=9.5,
             fontweight="bold")
    fig.text(0.065, 0.94, "Eight-well landscape", ha="left", va="baseline", color=TEXT_COLOR,
             fontsize=9.5, fontweight="bold")

    # (b) discovery and redistribution
    ax_cov = fig.add_subplot(right[0, 0])
    ax_ent = fig.add_subplot(right[1, 0], sharex=ax_cov, sharey=ax_cov)
    for ax, key, level in ((ax_cov, "identified", 1.0), (ax_ent, "entropy", 0.98)):
        noscore = seeds["noscore_fixed"]
        ax.step(noscore["checkpoints"], noscore[key].mean(axis=0), where="post",
                color=REFERENCE_COLOR, linestyle=":", linewidth=1.15, zorder=2)
        if key == "entropy":
            ax.axhline(level, color=TEXT_COLOR, linestyle=":", linewidth=0.7, zorder=1)
        for name, (color, wash, linestyle) in {
            "ula": (ULA_COLOR, ULA_WASH_COLOR, "--"),
            "fpt": (FPT_COLOR, FPT_WASH_COLOR, "-"),
        }.items():
            record = seeds[f"{name}_fixed"]
            mean, half = t_interval(record[key])
            ax.fill_between(record["checkpoints"], np.clip(mean - half, 0, 1),
                            np.clip(mean + half, 0, 1), step="post", color=wash,
                            linewidth=0, zorder=1)
            ax.step(record["checkpoints"], mean, where="post", color=color,
                    linestyle=linestyle, linewidth=1.25, zorder=4)
        for name, color, linestyle, _ in EXACT_STYLES:
            record = exact[name]
            ax.step(record["steps"], record[key].mean(axis=0), where="post",
                    color=color, linestyle=linestyle, linewidth=1.2, zorder=3)
        symlog_updates(ax, ticks=(0, 10, 1e2, 1e3, 1e4))
        ax.set(ylim=(-0.03, 1.08), yticks=(0.0, 0.5, 1.0))
        clean_axis(ax)
    ax_cov.tick_params(axis="x", labelbottom=False)
    ax_cov.set_ylabel("Wells identified", labelpad=2.0)
    ax_ent.set_ylabel(r"$H_{\rm occ}$", labelpad=2.0)
    ax_ent.set_xlabel("Updates", labelpad=1.0)
    panel_header(ax_cov, "b", "Discovery and redistribution", x=-0.16, pad=4.0)

    # (c) basin mass against the continuum solution, relative to equilibrium
    equilibrium = np.array([3.0, 3.0, 1.0]) / 8.0
    mass_axes = [fig.add_subplot(bottom[0, 0])]
    mass_axes.append(fig.add_subplot(bottom[0, 1], sharey=mass_axes[0]))
    for ax, group in zip(mass_axes, (0, 2)):
        scale = equilibrium[group]
        ax.axhline(1.0, color=TEXT_COLOR, linestyle=":", linewidth=0.7, zorder=1)
        ax.plot(o_steps, o_mass[:, group] / scale, color=FP_REFERENCE_COLOR,
                linewidth=FP_REFERENCE_WIDTH, solid_capstyle="round", zorder=2)
        record = particle["ula_fixed"]
        ax.plot(record["steps"], record["mass"][:, group] / scale, color=ULA_COLOR,
                linestyle="--", linewidth=1.2, zorder=3)
        for name, color, linestyle, _ in EXACT_STYLES:
            record = exact[name]
            ax.plot(record["steps"], record["mass"][:, group] / scale, color=color,
                    linestyle=linestyle, linewidth=1.2, zorder=3)
        record = particle["fpt_fixed"]
        ax.plot(record["steps"], record["mass"][:, group] / scale, color=FPT_COLOR,
                linewidth=1.5, zorder=6)
        symlog_updates(ax)
        ax.set(ylim=(-0.03, 1.3), yticks=(0.0, 0.5, 1.0))
        ax.set_xlabel("Updates", labelpad=1.0)
        clean_axis(ax)
    mass_axes[0].set_ylabel("Mass / equilibrium", labelpad=2.0)
    mass_axes[1].tick_params(axis="y", labelleft=False)
    panel_header(mass_axes[0], "c", "One-flip wells", x=-0.30, pad=4.0)
    mass_axes[1].set_title("Three-flip wells", pad=4.0)

    # (d) thermal state
    ax_u = fig.add_subplot(bottom[0, 2])
    ax_u.axhline(target_energy, color=TEXT_COLOR, linestyle=":", linewidth=0.8, zorder=1)
    ax_u.plot(o_steps, o_energy, color=FP_REFERENCE_COLOR, linewidth=FP_REFERENCE_WIDTH,
              solid_capstyle="round", zorder=2)
    record = particle["ula_fixed"]
    ax_u.plot(*log_binned(record["steps"], record["energy"].mean(axis=0)), color=ULA_COLOR,
              linestyle="--", linewidth=1.2, zorder=3)
    for name, color, linestyle, _ in EXACT_STYLES:
        record = exact[name]
        ax_u.plot(*log_binned(record["steps"], record["energy"]), color=color,
                  linestyle=linestyle, linewidth=1.2, zorder=3)
    record = particle["fpt_fixed"]
    ax_u.plot(*log_binned(record["steps"], record["energy"].mean(axis=0)), color=FPT_COLOR,
              linewidth=1.5, zorder=6)
    ax_u.annotate("target", xy=(1.3, target_energy), xytext=(0, 1.5),
                  textcoords="offset points", ha="left", va="bottom", fontsize=8.0,
                  color=TEXT_COLOR)
    symlog_updates(ax_u)
    ax_u.set_yscale("log")
    ax_u.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax_u.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_u.set(ylim=(0.7, 6.5), yticks=(1, 2, 4),
             xlabel="Updates", ylabel=r"$\beta\langle U\rangle$")
    ax_u.xaxis.labelpad = 1.0
    ax_u.yaxis.labelpad = 2.0
    clean_axis(ax_u)
    panel_header(ax_u, "d", "Thermal state", x=-0.34, pad=4.0)

    handles = [
        Line2D([], [], color=FP_REFERENCE_COLOR, linewidth=FP_REFERENCE_WIDTH,
               label="Exact Fokker–Planck"),
        Line2D([], [], color=FPT_COLOR, linewidth=1.5, label="FPT, shared bandwidth"),
        *[Line2D([], [], color=color, linestyle=linestyle, linewidth=1.2, label=label)
          for _, color, linestyle, label in EXACT_STYLES],
        Line2D([], [], color=ULA_COLOR, linestyle="--", linewidth=1.2, label="Langevin"),
        Line2D([], [], color=REFERENCE_COLOR, linestyle=":", linewidth=1.15,
               label="No score"),
    ]
    fig.legend(handles=handles, loc="center", ncol=3, frameon=False,
               bbox_to_anchor=(0.53, 0.045), fontsize=8.2, handlelength=2.0,
               columnspacing=1.1, labelspacing=0.25)

    args.output_base.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, args.output_base)


if __name__ == "__main__":
    main()
