"""Shared chart style so every figure in the project reads as one system.

Colours follow the entity (a model keeps its colour in every chart).
Categorical order: blue, orange, aqua, yellow, magenta, green, violet, red.
Buy & Hold (the benchmark) is always neutral dark grey.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BENCHMARK = "#52514e"

MODEL_COLORS = {
    "Dummy": "#898781",
    "Always predict Buy": "#c3c2b7",
    "K-nearest neighbours": CATEGORICAL[4],
    "Logistic regression": CATEGORICAL[0],
    "Decision tree": CATEGORICAL[3],
    "Random forest": CATEGORICAL[2],
    "HistGradientBoosting": CATEGORICAL[6],
    "XGBoost": CATEGORICAL[1],
    "Perceptron": CATEGORICAL[5],
    "MLP": CATEGORICAL[7],
    "Buy & Hold": BENCHMARK,
}

CLASS_COLORS = {"Sell": "#e34948", "Hold": "#898781", "Buy": "#2a78d6"}

# diverging blue (positive) <-> red (negative), grey midpoint
DIVERGING = LinearSegmentedColormap.from_list("red_grey_blue", ["#c43c3b", "#f0efec", "#256abf"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.size": 10,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    })


def save(fig, name: str):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png")


def pct(ax, axis="y", decimals=0):
    fmt = mpl.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
