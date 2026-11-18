#!/usr/bin/env python3
"""Plot the complete exact-row-mass transport experiment for the paper and SI."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

from examples.common.figure_style import (
    CANVAS_COLOR,
    FPT_COLOR,
    FPT_MARKER_COLOR,
    GRID_COLOR,
    LANDSCAPE_HIGH_COLOR,
    LANDSCAPE_LOW_COLOR,
    LANDSCAPE_MID_COLOR,
    MC_COLOR,
    MC_FILL_COLOR,
    NEITHER_COLOR,
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


DEFAULT_DATA = Path("data/tb_transport/figure_data")
DEFAULT_MAIN = Path("figures/tb_complete_transport")
DEFAULT_SI = Path("figures/tb_complete_transport_diagnostics")
# Reuse the manuscript palette throughout.  Basin and particle-size identities
# avoid the adjacent dark-blue/dark-green pair; distinct markers provide a
# second encoding when curves overlap or the figure is printed in grayscale.
# The third series uses a green separated from the muted blue in both hue and
# luminance; marker shapes remain the secondary encoding for reduced-color use.
THIRD_SERIES_COLOR = "#2E7D32"
BASIN_COLORS = (ULA_COLOR, MC_COLOR, THIRD_SERIES_COLOR)
BASIN_LABELS = ("A", "B", "C")
# Keep this ordering identical to TB_CENTERS in the analysis pipeline:
# A is the initialized left well, B the upper-right well, and C the lower-right well.
BASIN_CENTERS = np.asarray(((-0.5, 0.5), (0.8, 1.2), (0.5, -0.3)))
SIZE_COLORS = (ULA_COLOR, MC_COLOR, THIRD_SERIES_COLOR)
SIZE_MARKERS = ("^", "s", "o")
ORACLE_COLORS = (FPT_COLOR, MC_COLOR, ULA_COLOR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--main-output-base", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--si-output-base", type=Path, default=DEFAULT_SI)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    standard_error = (
        float(np.std(array, ddof=1) / np.sqrt(array.size))
        if array.size > 1
        else 0.0
    )
    return float(np.mean(array)), standard_error


def grouped_trace(
    rows: list[dict[str, str]],
    *,
    n_particles: int,
    field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = [row for row in rows if int(row["n_particles"]) == n_particles]
    times = sorted(
        {
            float(row["time"])
            for row in subset
            if np.isfinite(float(row[field]))
        }
    )
    means, errors = [], []
    for time_value in times:
        values = [
            float(row[field])
            for row in subset
            if np.isclose(float(row["time"]), time_value)
            and np.isfinite(float(row[field]))
        ]
        mean, error = mean_se(values)
        means.append(mean)
        errors.append(error)
    return np.asarray(times), np.asarray(means), np.asarray(errors)


def grouped_endpoint(
    rows: list[dict[str, str]],
    *,
    sizes: list[int],
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    means, errors = [], []
    for size in sizes:
        values = [
            float(row[field]) for row in rows if int(row["n_particles"]) == size
        ]
        mean, error = mean_se(values)
        means.append(mean)
        errors.append(error)
    return np.asarray(means), np.asarray(errors)


def content_threshold(mass: np.ndarray, fraction: float) -> float:
    values = np.sort(np.asarray(mass, dtype=float).ravel())[::-1]
    cumulative = np.cumsum(values)
    index = min(int(np.searchsorted(cumulative, fraction)), values.size - 1)
    return float(values[index])


def add_landscape(ax, x: np.ndarray, y: np.ndarray, energy: np.ndarray) -> None:
    xx, yy = np.meshgrid(x, y)
    reduced = 1.25 * (energy - float(np.min(energy)))
    cmap = LinearSegmentedColormap.from_list(
        "tb_complete_landscape",
        (
            CANVAS_COLOR,
            mix_with_white(LANDSCAPE_LOW_COLOR, 0.62),
            LANDSCAPE_LOW_COLOR,
            LANDSCAPE_MID_COLOR,
        ),
    ).reversed()
    ax.contourf(
        xx,
        yy,
        reduced,
        levels=np.arange(0.0, 6.01, 0.75),
        cmap=cmap,
        extend="max",
        zorder=0,
    )
    ax.contour(
        xx,
        yy,
        reduced,
        levels=(1.5, 3.0, 4.5),
        colors=LANDSCAPE_HIGH_COLOR,
        linewidths=0.65,
        zorder=1,
    )


def add_reference_mass(ax, x: np.ndarray, y: np.ndarray, mass: np.ndarray) -> None:
    xx, yy = np.meshgrid(x, y)
    threshold_90 = content_threshold(mass, 0.90)
    threshold_50 = content_threshold(mass, 0.50)
    maximum = float(np.max(mass))
    levels = np.asarray((threshold_90, threshold_50, maximum * (1.0 + 1.0e-9)))
    if not np.all(np.diff(levels) > 0.0):
        return
    ax.contourf(
        xx,
        yy,
        mass,
        levels=levels,
        colors=(mix_with_white(MC_COLOR, 0.82), mix_with_white(MC_COLOR, 0.62)),
        zorder=3,
    )
    ax.contour(
        xx,
        yy,
        mass,
        levels=levels[:2],
        colors=MC_COLOR,
        linewidths=(0.65, 0.85),
        zorder=4,
    )


def style_map(ax, *, show_y: bool) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.45, 1.55)
    ax.set_ylim(-1.15, 1.85)
    ax.set_xticks((-1.0, 0.0, 1.0))
    ax.set_yticks((-1.0, 0.0, 1.0))
    ax.set_xlabel(r"$x_1$", labelpad=1)
    if show_y:
        ax.set_ylabel(r"$x_2$", labelpad=1)
    else:
        ax.tick_params(labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(pad=1.4)


def plot_overview(axes, data_dir: Path) -> tuple[Line2D, Line2D]:
    with np.load(data_dir / "overview_snapshots.npz") as archive:
        x = np.asarray(archive["x"], dtype=float)
        y = np.asarray(archive["y"], dtype=float)
        energy = np.asarray(archive["energy"], dtype=float)
        times = np.asarray(archive["times"], dtype=float)
        reference_mass = np.asarray(archive["reference_mass"], dtype=float)
        particle_samples = np.asarray(archive["particle_samples"], dtype=float)
        n_particles = int(np.asarray(archive["n_particles"]).item())

    selected_times = (0.0, 2.0, 10.0, 20.0)
    titles = (
        r"Localized start, $t=0$",
        r"Early transfer, $t=2$",
        r"Three-basin state, $t=10$",
        r"Near equilibrium, $t=20$",
    )
    for panel_index, (ax, time_value, title) in enumerate(
        zip(axes, selected_times, titles, strict=True)
    ):
        matches = np.flatnonzero(np.isclose(times, time_value))
        if matches.size != 1:
            raise ValueError(f"Overview archive lacks t={time_value:g}")
        index = int(matches[0])
        add_landscape(ax, x, y, energy)
        add_reference_mass(ax, x, y, reference_mass[index])
        samples = particle_samples[index]
        stride = max(samples.shape[0] // 620, 1)
        ax.scatter(
            samples[::stride, 0],
            samples[::stride, 1],
            s=2.0,
            color=FPT_MARKER_COLOR,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.86,
            rasterized=True,
            zorder=5,
        )
        for label, center, color in zip(
            BASIN_LABELS, BASIN_CENTERS, BASIN_COLORS, strict=True
        ):
            ax.scatter(
                [center[0]],
                [center[1]],
                s=20,
                facecolor=CANVAS_COLOR,
                edgecolor=color,
                linewidth=1.0,
                zorder=8,
            )
            label_artist = ax.annotate(
                label,
                xy=(center[0], center[1]),
                xytext=(3.6, 3.2),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=9.0,
                color=color,
                fontweight="bold",
                zorder=9,
            )
            label_artist.set_path_effects(
                [
                    path_effects.Stroke(linewidth=1.8, foreground=CANVAS_COLOR),
                    path_effects.Normal(),
                ]
            )
        ax.set_title(title, pad=3)
        style_map(ax, show_y=panel_index == 0)

    return (
        Line2D(
            (0,),
            (0,),
            color=MC_COLOR,
            linewidth=4.0,
            label=r"Time-matched reference $\rho_t$",
        ),
        Line2D(
            (0,),
            (0,),
            color=FPT_MARKER_COLOR,
            marker="o",
            linestyle="none",
            markersize=3.0,
            label=rf"Row-mass particles, $N=${n_particles:,}",
        ),
    )


def plot_basin_transport(
    ax,
    traces: list[dict[str, str]],
    summary: dict[str, object],
) -> None:
    n_particles = max(int(row["n_particles"]) for row in traces)
    target = np.asarray(summary["reference"]["target_basin_mass"], dtype=float)
    for basin_index, color in enumerate(BASIN_COLORS):
        time, particle, particle_se = grouped_trace(
            traces,
            n_particles=n_particles,
            field=f"particle_basin_{basin_index}",
        )
        reference_rows = {
            float(row["time"]): float(row[f"reference_basin_{basin_index}"])
            for row in traces
            if int(row["n_particles"]) == n_particles
        }
        reference = np.asarray([reference_rows[value] for value in time])
        ax.fill_between(
            time,
            particle - particle_se,
            particle + particle_se,
            color=mix_with_white(color, 0.78),
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            time,
            particle,
            color=color,
            marker="o",
            markevery=5,
            markersize=2.5,
            linewidth=1.45,
            zorder=3,
        )
        ax.plot(time, reference, color=color, linestyle="--", linewidth=0.9, zorder=2)
        ax.scatter(
            [20.45],
            [target[basin_index]],
            marker="x",
            s=18,
            color=color,
            linewidth=1.0,
            clip_on=False,
            zorder=4,
        )
    basin_handles = [
        Line2D(
            (0,),
            (0,),
            color=color,
            marker="o",
            linestyle="none",
            markersize=3.5,
            label=f"Basin {label}",
        )
        for label, color in zip(BASIN_LABELS, BASIN_COLORS, strict=True)
    ]
    ax.legend(
        handles=basin_handles,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.965),
        ncol=1,
        fontsize=9.0,
        handletextpad=0.25,
        labelspacing=0.28,
        borderaxespad=0.0,
    )
    ax.set_title("Basin redistribution")
    ax.set_xlabel("Dimensionless time")
    ax.set_ylabel("Basin probability")
    ax.set_xlim(0.0, 20.8)
    ax.set_ylim(0.0, 1.02)
    clean_axis(ax)


def plot_target_approach(
    ax,
    traces: list[dict[str, str]],
    sizes: list[int],
) -> None:
    upper_values: list[float] = []
    for size, color, marker in zip(
        sizes,
        SIZE_COLORS[: len(sizes)],
        SIZE_MARKERS[: len(sizes)],
        strict=True,
    ):
        time, mean, error = grouped_trace(
            traces,
            n_particles=size,
            field="particle_target_w2",
        )
        ax.fill_between(
            time,
            mean - error,
            mean + error,
            color=mix_with_white(color, 0.82),
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            time,
            mean,
            color=color,
            marker=marker,
            markevery=5,
            markerfacecolor=CANVAS_COLOR,
            markeredgewidth=0.8,
            markersize=3.4,
            label=rf"$N=${size:,}",
            zorder=3,
        )
        upper_values.extend((mean + error).tolist())
    reference_rows = {
        float(row["time"]): float(row["reference_target_w2"])
        for row in traces
        if np.isfinite(float(row["reference_target_w2"]))
    }
    reference_time = np.asarray(sorted(reference_rows))
    reference_values = np.asarray([reference_rows[value] for value in reference_time])
    ax.plot(
        reference_time,
        reference_values,
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=1.25,
        label=r"Continuum $\rho_t$",
        zorder=2,
    )
    upper_values.extend(reference_values.tolist())
    ax.set_title("Relaxation to equilibrium")
    ax.set_xlabel("Dimensionless time")
    ax.set_ylabel(r"Distance to $\pi$, $W_2$")
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(0.0, 1.08 * max(upper_values))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=[handles[-1], *handles[:-1]],
        labels=[labels[-1], *labels[:-1]],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.965),
        fontsize=9.0,
        ncol=1,
        handlelength=1.45,
        handletextpad=0.35,
        labelspacing=0.22,
        borderaxespad=0.0,
    )
    clean_axis(ax)


def plot_main(data_dir: Path, output_base: Path) -> None:
    configure_matplotlib()
    traces = read_csv(data_dir / "time_series.csv")
    endpoints = read_csv(data_dir / "endpoint_runs.csv")
    oracle_rows = read_csv(data_dir / "oracle_flow_runs.csv")
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    sizes = sorted({int(row["n_particles"]) for row in endpoints})

    fig = plt.figure(figsize=(6.69, 4.75))
    outer = fig.add_gridspec(
        2,
        3,
        left=0.092,
        right=0.986,
        bottom=0.105,
        top=0.925,
        height_ratios=(0.92, 1.08),
        wspace=0.44,
        hspace=0.46,
    )
    map_grid = outer[0, :].subgridspec(1, 4, wspace=0.18)
    map_axes = tuple(fig.add_subplot(map_grid[0, index]) for index in range(4))
    ax_b = fig.add_subplot(outer[1, 0])
    ax_c = fig.add_subplot(outer[1, 1])
    ax_d = fig.add_subplot(outer[1, 2])

    overview_handles = plot_overview(map_axes, data_dir)
    plot_basin_transport(ax_b, traces, summary)
    plot_target_approach(ax_c, traces, sizes)
    plot_pathwise_tracking(ax_d, oracle_rows, max(sizes))

    fig.legend(
        handles=overview_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.993),
        ncol=2,
        fontsize=9.0,
        handlelength=1.3,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    panel_label(map_axes[0], "a", x=-0.28)
    for label, ax in zip("bcd", (ax_b, ax_c, ax_d), strict=True):
        panel_label(ax, label, x=-0.18)
    save_figure(fig, output_base)


def plot_target_diagnostics(ax, traces: list[dict[str, str]], n_particles: int) -> None:
    series = (
        ("particle_target_density_js", NEITHER_COLOR, "o", "Density JS"),
        ("particle_target_energy_ks", MC_COLOR, "s", "Energy KS"),
        ("particle_target_basin_tv", ULA_COLOR, "^", "Basin TV"),
    )
    upper_values: list[float] = []
    for field, color, marker, label in series:
        time, mean, error = grouped_trace(
            traces, n_particles=n_particles, field=field
        )
        ax.fill_between(
            time,
            mean - error,
            mean + error,
            color=mix_with_white(color, 0.82),
            linewidth=0.0,
        )
        ax.plot(
            time,
            mean,
            color=color,
            marker=marker,
            markevery=5,
            markerfacecolor=CANVAS_COLOR,
            markeredgewidth=0.7,
            markersize=3.2,
            label=label,
        )
        upper_values.extend((mean + error).tolist())
    ax.set_title(rf"Target diagnostics, $N=${n_particles:,}")
    ax.set_xlabel("Dimensionless time")
    ax.set_ylabel("Distribution discrepancy")
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(0.0, 1.08 * max(upper_values))
    ax.legend(frameon=False, loc="upper right", fontsize=9.0, handlelength=1.4)
    clean_axis(ax)


def plot_pathwise_tracking(
    ax,
    oracle_rows: list[dict[str, str]],
    n_particles: int,
) -> None:
    """Show the time-resolved same-initial oracle decomposition."""
    upper_values: list[float] = []
    series = (
        (
            "total_w2",
            ORACLE_COLORS[0],
            "o",
            "-",
            r"$E_{\rm tot}$",
        ),
        (
            "dynamic_w2",
            ORACLE_COLORS[1],
            "s",
            "--",
            r"$E_{\rm dyn}$",
        ),
        (
            "oracle_sampling_w2",
            ORACLE_COLORS[2],
            "D",
            ":",
            r"$E_{\rm init}$",
        ),
    )
    for field, color, marker, linestyle, label in series:
        time, mean, error = grouped_trace(
            oracle_rows,
            n_particles=n_particles,
            field=field,
        )
        ax.fill_between(
            time,
            mean - error,
            mean + error,
            color=mix_with_white(color, 0.82),
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            time,
            mean,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markerfacecolor=CANVAS_COLOR,
            markeredgewidth=0.8,
            markersize=3.5,
            label=label,
            zorder=3,
        )
        upper_values.extend((mean + error).tolist())
    ax.set_title(r"Finite-$N$ transport error")
    ax.set_xlabel("Dimensionless time")
    ax.set_ylabel(r"Time-matched $W_2$")
    ax.set_xlim(0.0, 20.0)
    ax.set_xticks((0.0, 5.0, 10.0, 15.0, 20.0))
    # Reserve a narrow data-free band for the legend.  The three trajectories
    # reach their maximum during inter-basin redistribution, so an ordinary
    # upper-corner legend would hide precisely the feature this panel reports.
    ax.set_ylim(0.0, 1.28 * max(upper_values))
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        fontsize=9.0,
        ncol=3,
        handlelength=0.90,
        handletextpad=0.22,
        columnspacing=0.55,
        borderaxespad=0.0,
    )
    clean_axis(ax)


def plot_dynamic_gap_across_sizes(
    ax,
    oracle_rows: list[dict[str, str]],
    sizes: list[int],
) -> None:
    """Compare the same-initial dynamical gap over the stored horizon."""
    upper_values: list[float] = []
    for size, color, marker in zip(
        sizes,
        SIZE_COLORS[: len(sizes)],
        SIZE_MARKERS[: len(sizes)],
        strict=True,
    ):
        time, mean, error = grouped_trace(
            oracle_rows,
            n_particles=size,
            field="dynamic_w2",
        )
        ax.fill_between(
            time,
            mean - error,
            mean + error,
            color=mix_with_white(color, 0.82),
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            time,
            mean,
            color=color,
            marker=marker,
            markerfacecolor=CANVAS_COLOR,
            markeredgewidth=0.8,
            markersize=3.5,
            linewidth=1.25,
            label=rf"$N=${size:,}",
            zorder=3,
        )
        upper_values.extend((mean + error).tolist())
    ax.set_title(r"Dynamical gap across resolutions")
    ax.set_xlabel("Dimensionless time")
    ax.set_ylabel(r"Oracle gap $W_2(\mu_N,\bar\mu_N)$")
    ax.set_xlim(0.0, 20.0)
    ax.set_xticks((0.0, 5.0, 10.0, 15.0, 20.0))
    ax.set_ylim(0.0, 1.10 * max(upper_values))
    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9.0,
        handlelength=1.35,
        handletextpad=0.35,
        labelspacing=0.22,
    )
    clean_axis(ax)


def plot_weak_flux(
    ax,
    endpoints: list[dict[str, str]],
    sizes: list[int],
    horizon: float,
) -> None:
    series = (
        ("cumulative_weak_linear_residual", ULA_COLOR, "^", "Linear"),
        ("cumulative_weak_quadratic_residual", MC_COLOR, "s", "Quadratic"),
        ("cumulative_weak_bump_residual", NEITHER_COLOR, "o", "Gaussian bumps"),
    )
    x = np.arange(len(sizes), dtype=float)
    plotted_values: list[float] = []
    for field, color, marker, label in series:
        mean, error = grouped_endpoint(endpoints, sizes=sizes, field=field)
        ax.errorbar(
            x,
            mean / horizon,
            yerr=error / horizon,
            color=color,
            marker=marker,
            markerfacecolor=CANVAS_COLOR,
            markeredgewidth=0.8,
            markersize=3.9,
            capsize=2.0,
            label=label,
        )
        plotted_values.extend((mean / horizon).tolist())
        ax.text(
            x[-1] + 0.10,
            mean[-1] / horizon,
            label,
            color=color,
            fontsize=9.0,
            ha="left",
            va="center",
        )
    ax.set_yscale("log")
    ax.set_title("Time-averaged weak-flux defects")
    ax.set_xlabel(r"Particles, $N$")
    ax.set_ylabel(r"$t_{\mathrm{f}}^{-1}\sum_n\Delta t\,|\mathcal{R}_N^\phi|$")
    ax.set_xticks(x, labels=[f"{size:,}" for size in sizes])
    ax.set_xlim(-0.10, x[-1] + 0.60)
    positive = np.asarray([value for value in plotted_values if value > 0.0])
    ax.set_ylim(0.70 * float(np.min(positive)), 1.40 * float(np.max(positive)))
    clean_axis(ax)


def plot_path_controls(
    ax,
    endpoints: list[dict[str, str]],
    sizes: list[int],
    audit: dict[str, object],
) -> None:
    tolerance = float(audit["selector"]["row_mass_relative_tolerance"])
    time_step = float(audit["time_step"]["primary_time_step"])
    x = np.arange(len(sizes), dtype=float)
    action, action_se = grouped_endpoint(
        endpoints, sizes=sizes, field="cumulative_action"
    )
    root, root_se = grouped_endpoint(
        endpoints, sizes=sizes, field="maximum_row_mass_relative_error"
    )
    euler_scale = time_step * action
    euler_scale_se = time_step * action_se
    action_artist = ax.errorbar(
        x,
        action,
        yerr=action_se,
        color=FPT_COLOR,
        marker="o",
        markerfacecolor=CANVAS_COLOR,
        markeredgewidth=0.8,
        markersize=3.9,
        capsize=2.0,
        label=r"Action $\mathcal{A}_{N,t_{\mathrm{f}}}$",
    )
    control_ax = ax.twinx()
    euler_artist = control_ax.errorbar(
        x,
        euler_scale,
        yerr=euler_scale_se,
        color=MC_COLOR,
        marker="s",
        markerfacecolor=CANVAS_COLOR,
        markeredgewidth=0.8,
        markersize=3.9,
        capsize=2.0,
        label=r"Euler scale $\Delta t\,\mathcal{A}_{N,t_{\mathrm{f}}}$",
    )
    root_artist = control_ax.errorbar(
        x,
        root,
        yerr=root_se,
        color=ULA_COLOR,
        marker="^",
        markerfacecolor=CANVAS_COLOR,
        markeredgewidth=0.8,
        markersize=3.9,
        capsize=2.0,
        label="Row-mass root error",
    )
    control_ax.axhline(
        tolerance,
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=0.8,
    )
    control_ax.text(
        0.02,
        0.98,
        "dashed: row-mass tolerance",
        transform=control_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        color=REFERENCE_COLOR,
        bbox={
            "facecolor": CANVAS_COLOR,
            "edgecolor": "none",
            "pad": 0.4,
        },
    )
    ax.set_title("Path and numerical controls")
    ax.set_xlabel(r"Particles, $N$")
    ax.set_ylabel(r"Action $\mathcal{A}_{N,t_{\mathrm{f}}}$", color=FPT_COLOR)
    control_ax.set_ylabel(r"Numerical-error scale ($10^{-3}$)")
    ax.set_xticks(x, labels=[f"{size:,}" for size in sizes])
    ax.set_ylim(0.0, 1.16 * float(np.max(action + action_se)))
    control_upper = max(
        tolerance,
        float(np.max(euler_scale + euler_scale_se)),
        float(np.max(root + root_se)),
    )
    control_ax.set_ylim(0.0, 1.10 * control_upper)
    control_ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{1.0e3 * value:.1f}")
    )
    control_ax.spines["top"].set_visible(False)
    control_ax.spines["right"].set_color(TEXT_COLOR)
    control_ax.tick_params(axis="y", colors=TEXT_COLOR)
    control_ax.grid(False)
    ax.legend(
        handles=(action_artist, euler_artist, root_artist),
        frameon=False,
        loc="center right",
        fontsize=9.0,
        handlelength=1.35,
        labelspacing=0.25,
    )
    clean_axis(ax)


def plot_si(data_dir: Path, output_base: Path) -> None:
    configure_matplotlib()
    endpoints = read_csv(data_dir / "endpoint_runs.csv")
    oracle_rows = read_csv(data_dir / "oracle_flow_runs.csv")
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((data_dir / "numerical_audit.json").read_text(encoding="utf-8"))
    sizes = sorted({int(row["n_particles"]) for row in endpoints})
    horizon = float(summary["protocol"]["horizon"])

    fig = plt.figure(figsize=(6.69, 4.30))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.91,
        bottom=0.105,
        top=0.94,
        height_ratios=(0.82, 1.0),
        wspace=0.42,
        hspace=0.50,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    plot_dynamic_gap_across_sizes(ax_a, oracle_rows, sizes)
    plot_weak_flux(ax_b, endpoints, sizes, horizon)
    plot_path_controls(ax_c, endpoints, sizes, audit)
    panel_label(ax_a, "a", x=-0.08)
    panel_label(ax_b, "b", x=-0.19)
    panel_label(ax_c, "c", x=-0.19)
    save_figure(fig, output_base)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    plot_main(data_dir, args.main_output_base.resolve())
    plot_si(data_dir, args.si_output_base.resolve())
    print(
        "Saved complete-transport main and SI figures to "
        f"{args.main_output_base.resolve()}.* and {args.si_output_base.resolve()}.*"
    )


if __name__ == "__main__":
    main()
