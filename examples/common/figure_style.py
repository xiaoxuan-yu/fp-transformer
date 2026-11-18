from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


# White-based publication palette.  Method inks stay fixed, while every pale
# scientific fill is pre-mixed with white instead of relying on renderer-
# dependent alpha compositing.  Landscapes use a separate ice-porcelain teal
# family so they remain distinct from all method identities.
def mix_with_white(hex_color: str, white_fraction: float) -> str:
    if not 0.0 <= white_fraction <= 1.0:
        raise ValueError("white_fraction must lie in [0, 1]")
    value = hex_color.removeprefix("#")
    if len(value) != 6:
        raise ValueError(f"expected a six-digit hex color, got {hex_color!r}")
    channels = [int(value[offset : offset + 2], 16) for offset in (0, 2, 4)]
    mixed = [
        round(channel * (1.0 - white_fraction) + 255 * white_fraction)
        for channel in channels
    ]
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


CANVAS_COLOR = "#FFFFFF"
MC_COLOR = "#3F6F8A"
FPT_COLOR = "#155B55"
ULA_COLOR = "#9A5728"
REFERENCE_COLOR = "#766E67"
NEITHER_COLOR = "#6A679A"
TEXT_COLOR = "#2F2B29"

LANDSCAPE_COLOR = "#3F8F90"
LANDSCAPE_OUTER_COLOR = mix_with_white(LANDSCAPE_COLOR, 0.92)
LANDSCAPE_LOW_COLOR = mix_with_white(LANDSCAPE_COLOR, 0.82)
LANDSCAPE_MID_COLOR = mix_with_white(LANDSCAPE_COLOR, 0.68)
LANDSCAPE_HIGH_COLOR = mix_with_white(LANDSCAPE_COLOR, 0.45)
LANDSCAPE_COLORS = (
    CANVAS_COLOR,
    LANDSCAPE_OUTER_COLOR,
    LANDSCAPE_LOW_COLOR,
    LANDSCAPE_MID_COLOR,
    LANDSCAPE_HIGH_COLOR,
)
GRID_COLOR = mix_with_white(LANDSCAPE_COLOR, 0.91)
PANE_EDGE_COLOR = mix_with_white(LANDSCAPE_COLOR, 0.94)
PALE_FILL = LANDSCAPE_OUTER_COLOR

MC_FILL_COLOR = mix_with_white(MC_COLOR, 0.72)
FPT_FILL_COLOR = mix_with_white(FPT_COLOR, 0.72)
ULA_FILL_COLOR = mix_with_white(ULA_COLOR, 0.72)
MC_WASH_COLOR = mix_with_white(MC_COLOR, 0.88)
FPT_WASH_COLOR = mix_with_white(FPT_COLOR, 0.88)
ULA_WASH_COLOR = mix_with_white(ULA_COLOR, 0.88)
MC_MARKER_COLOR = mix_with_white(MC_COLOR, 0.44)
FPT_MARKER_COLOR = mix_with_white(FPT_COLOR, 0.20)
ULA_MARKER_COLOR = mix_with_white(ULA_COLOR, 0.44)
REFERENCE_FILL_COLOR = LANDSCAPE_LOW_COLOR
REFERENCE_MARKER_COLOR = LANDSCAPE_HIGH_COLOR
DATA_FILL_COLOR = mix_with_white(FPT_COLOR, 0.55)
HEATMAP_LOW_COLOR = CANVAS_COLOR
HEATMAP_MID_COLOR = mix_with_white(FPT_COLOR, 0.55)
HEATMAP_HIGH_COLOR = FPT_COLOR
CARTOON_COLOR = LANDSCAPE_LOW_COLOR
WARM_ACCENT_COLOR = "#A87526"
ELEMENT_N_COLOR = "#4B7186"
ELEMENT_O_COLOR = "#A13F3B"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Nimbus Sans",
            "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial"],
            # Figures are occasionally reduced to about 91% of their source
            # width in the SI and Chinese reading copy.  A 9 pt source minimum
            # therefore keeps all figure text above 8 pt at final size.
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


def clean_axis(ax, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(CANVAS_COLOR)
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.65, alpha=1.0)
    ax.set_axisbelow(True)


def format_panel_label(label: str) -> str:
    stripped = label.strip()
    if len(stripped) == 1 and stripped.isalpha():
        return f"({stripped.lower()})"
    return label


def panel_label(
    ax,
    label: str,
    *,
    x: float = -0.16,
    y: float | None = None,
) -> None:
    """Place a panel identifier on the same baseline as the axes title.

    Matplotlib applies title padding in physical units, so a separately placed
    label at a fixed axes-relative ``y`` drifts vertically whenever title
    padding or panel height changes. Sharing the title transform and baseline
    keeps the identifier and title aligned throughout the manuscript figures.
    ``y`` remains available for axes without a conventional title.
    """
    text_method = getattr(ax, "text2D", ax.text)
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
        fontsize = 10.5
    text_method(
        x,
        title_y,
        format_panel_label(label),
        transform=transform,
        ha="left",
        va=vertical_alignment,
        color=TEXT_COLOR,
        fontsize=fontsize,
        fontweight="bold",
    )


def panel_header(
    ax,
    label: str,
    title: str,
    *,
    x: float = -0.16,
    **title_kwargs,
) -> None:
    """Set an axes title and align its panel identifier to the same baseline."""
    ax.set_title(title, **title_kwargs)
    panel_label(ax, label, x=x)


def save_figure(fig, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=600)
    fig.savefig(
        output_base.with_suffix(".tif"),
        dpi=600,
        format="tiff",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
