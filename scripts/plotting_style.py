from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHOD_STYLES = {
    "Masked Pair Encoder": {
        "label": "Masked Pair Encoder",
        "color": "#2563EB",
        "linestyle": "-",
        "linewidth": 3.0,
        "alpha": 1.0,
        "zorder": 10,
    },
    "GPT-2": {
        "label": "GPT-2 causal",
        "color": "#D95F02",
        "linestyle": "-",
        "linewidth": 3.0,
        "alpha": 1.0,
        "zorder": 11,
    },
    "OLS": {
        "label": "OLS",
        "color": "#111827",
        "linestyle": "--",
        "linewidth": 3.0,
        "alpha": 1.0,
        "zorder": 12,
    },
    "3NN": {
        "label": "3NN",
        "color": "#7C3AED",
        "linestyle": "-",
        "linewidth": 3.0,
        "alpha": 1.0,
        "zorder": 4,
    },
    "Averaging": {
        "label": "Averaging",
        "color": "#A855F7",
        "linestyle": "--",
        "linewidth": 2.7,
        "alpha": 0.9,
        "zorder": 3,
    },
    "2-layer NN, Adam": {
        "label": "2-layer NN, GD",
        "color": "#059669",
        "linestyle": "--",
        "linewidth": 2.8,
        "alpha": 0.95,
        "zorder": 6,
    },
    "Greedy": {
        "label": "Greedy",
        "color": "#6B7280",
        "linestyle": "--",
        "linewidth": 2.5,
        "alpha": 0.85,
        "zorder": 2,
    },
    "Greedy (sign)": {
        "label": "Greedy (sign)",
        "color": "#8A5A44",
        "linestyle": "--",
        "linewidth": 2.5,
        "alpha": 0.85,
        "zorder": 3,
    },
    "XGB (sign)": {
        "label": "XGB (sign)",
        "color": "#C00000",
        "linestyle": "-.",
        "linewidth": 2.5,
        "alpha": 0.9,
        "zorder": 5,
    },
}


def configure_matplotlib(context: str = "single") -> None:
    if context == "panel":
        font_size = 13
        title_size = 17
        label_size = 14
        tick_size = 12
        legend_size = 11
    else:
        font_size = 22
        title_size = 30
        label_size = 27
        tick_size = 24
        legend_size = 21

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.titlesize": title_size,
            "axes.labelsize": label_size,
            "legend.fontsize": legend_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "axes.linewidth": 1.4,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#CBD5E1",
            "grid.linewidth": 1.0,
            "grid.alpha": 0.55,
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(
    ax,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    ylim: tuple[float, float],
    yticks: list[float],
    panel_label: str | None = None,
    xlim: tuple[float, float] = (0, 40),
    xticks: list[float] | None = None,
) -> None:
    ax.set_title(title, pad=6, fontweight="semibold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if xticks is None:
        xticks = [0, 10, 20, 30, 40]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.grid(True, axis="y")
    ax.grid(True, axis="x", alpha=0.35)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.4)
    ax.tick_params(length=8, width=1.4, color="black", labelcolor="black", pad=7)

    if panel_label is not None:
        ax.text(
            -0.16,
            1.08,
            panel_label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
            color="#111827",
        )


def mark_dimension(ax, *, x: float = 20.0, y: float | None = None, label: str = "d=20") -> None:
    ax.axvline(x, color="#4B5563", linestyle=":", linewidth=2.0, alpha=0.82, zorder=1)
    if y is None:
        y = ax.get_ylim()[1] * 0.92
    ax.text(x + 0.5, y, label, color="#4B5563", va="top", ha="left")


def plot_method(
    ax,
    x,
    y,
    method: str,
    *,
    clip_y: float | None = None,
    label: str | None = None,
):
    style = METHOD_STYLES[method]
    x_arr = np.asarray(x)
    y_arr = np.asarray(y, dtype=float)
    clipped = np.zeros_like(y_arr, dtype=bool)
    y_plot = y_arr

    if clip_y is not None:
        clipped = y_arr > clip_y
        y_plot = y_arr.copy()
        y_plot[clipped] = clip_y

    line_kwargs = {
        "color": style["color"],
        "linestyle": style["linestyle"],
        "linewidth": style["linewidth"],
        "alpha": style["alpha"],
        "label": label if label is not None else style["label"],
        "solid_capstyle": "round",
        "dash_capstyle": "round",
        "zorder": style["zorder"],
    }
    (line,) = ax.plot(
        x_arr,
        y_plot,
        **line_kwargs,
    )
    if clipped.any() and clip_y is not None:
        ax.scatter(
            x_arr[clipped],
            np.full(int(clipped.sum()), clip_y),
            marker="^",
            s=22,
            color=style["color"],
            edgecolor="white",
            linewidth=0.45,
            zorder=style["zorder"] + 1,
            clip_on=False,
        )
    return line


def add_compact_legend(
    ax,
    *,
    ncol: int = 1,
    loc: str = "upper right",
    fontsize: float | None = None,
    handlelength: float = 2.1,
) -> None:
    legend = ax.legend(
        loc=loc,
        ncol=ncol,
        frameon=False,
        fancybox=False,
        borderpad=0.25,
        handlelength=handlelength,
        handletextpad=0.65,
        columnspacing=1.0,
        labelspacing=0.42,
        fontsize=fontsize,
    )
    for line in legend.get_lines():
        line.set_linewidth(3.0)


def save_figure(fig, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=0.12)
    return png_path
