"""Shared matplotlib style for the paper's figures.

Colour choices follow a validated categorical order rather than matplotlib's
default cycle, and are used by *role*: categorical slots carry identity,
a single-hue ramp carries magnitude, and a blue-red pair with a neutral grey
midpoint carries polarity.  Text never wears a series colour.

Two rules do real work in a paper rather than being house style:

* **No dual axes, ever.**  Discrimination and calibration live on different
  scales and the whole claim is about their dissociation; putting them on one
  plot with two y-axes would let the reader's eye invent whatever relationship
  the scale alignment implies.  They get adjacent panels sharing an x-axis.
* **Identity by position or facet once there are more than three categories.**
  Six labels shown as six scatter colours cannot be told apart under common
  colour-vision deficiency; faceting is the fix, not a bigger palette.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical order (light mode).  Assigned in fixed order, never cycled.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Single-hue sequential ramp, light -> dark, for magnitude.
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# Diverging: warm/cool poles, neutral grey midpoint.
DIVERGING_LO, DIVERGING_MID, DIVERGING_HI = "#2a78d6", "#f0efec", "#e34948"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e6e5e1"

# Stable per-site colours, so a site keeps its hue across every figure.
SITE_ORDER = ["mimic_like", "georgia_like", "ptbxl_like", "chapman_like", "code15_like", "korea_like"]
SITE_LABEL = {
    "mimic_like": "US · MIMIC-IV",
    "georgia_like": "US · Georgia",
    "ptbxl_like": "DE · PTB-XL",
    "chapman_like": "CN · Chapman/Ningbo",
    "code15_like": "BR · CODE-15",
    "korea_like": "KR · screening",
}


def site_color(site: str) -> str:
    """Colour follows the entity, so a site keeps its hue when others are filtered out."""
    try:
        return CATEGORICAL[SITE_ORDER.index(site) % len(CATEGORICAL)]
    except ValueError:
        return INK_MUTED


def sequential_cmap(name: str = "cliff_blue"):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, SEQUENTIAL)


def diverging_cmap(name: str = "cliff_div"):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, [DIVERGING_LO, DIVERGING_MID, DIVERGING_HI])


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 8.5,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",        # solid hairlines: dashes read as thresholds
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.color": INK,
    })


def despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def title(ax, text: str, subtitle: str | None = None) -> None:
    ax.set_title(text, color=INK, pad=10 if subtitle else 6)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=7.8,
                color=INK_MUTED, va="bottom", ha="left")
