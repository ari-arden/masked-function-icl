import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotting_style import (
    METHOD_STYLES,
    configure_matplotlib,
    save_figure,
    style_axis,
)

MODEL_COLUMNS = [
    ("Masked Pair Encoder", "Masked Pair Encoder"),
    ("GPT-2 causal", "GPT-2"),
    ("2-layer NN, Adam", "2-layer NN, Adam"),
    ("3NN", "3NN"),
]


def _plot_curve(
    ax,
    points,
    values,
    method: str,
    *,
    clip_above: float | None = None,
    linewidth: float | None = None,
    alpha: float | None = None,
    zorder: int | None = None,
    marker: str | None = None,
):
    style = METHOD_STYLES[method]
    y = np.asarray(values, dtype=float)
    x = np.asarray(points, dtype=float)
    clipped = np.zeros_like(y, dtype=bool)
    if clip_above is not None:
        clipped = y > clip_above
        y = y.copy()
        y[clipped] = np.nan

    line = ax.plot(
        x,
        y,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=linewidth if linewidth is not None else style["linewidth"],
        alpha=alpha if alpha is not None else style["alpha"],
        label=style["label"],
        solid_capstyle="round",
        dash_capstyle="round",
        marker=marker,
        markersize=4.6 if marker else None,
        markerfacecolor="white" if marker else None,
        markeredgewidth=1.0 if marker else None,
        zorder=zorder if zorder is not None else style["zorder"],
    )[0]
    return line


def plot(curve_csv: Path, out_dir: Path, stem: str) -> Path:
    curves = pd.read_csv(curve_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    points = curves["point"].to_numpy()

    handles = {}
    handles["3NN"] = _plot_curve(
        ax,
        points,
        curves["3NN"].to_numpy(),
        "3NN",
        linewidth=3.0,
        zorder=4,
    )
    handles["2-layer NN, Adam"] = _plot_curve(
        ax,
        points,
        curves["2-layer NN, Adam"].to_numpy(),
        "2-layer NN, Adam",
        linewidth=3.5,
        zorder=7,
    )
    handles["GPT-2"] = _plot_curve(
        ax,
        points,
        curves["GPT-2 causal"].to_numpy(),
        "GPT-2",
        linewidth=3.4,
        zorder=11,
    )
    handles["Masked Pair Encoder"] = _plot_curve(
        ax,
        points,
        curves["Masked Pair Encoder"].to_numpy(),
        "Masked Pair Encoder",
        linewidth=3.4,
        zorder=12,
    )

    style_axis(
        ax,
        title="Two-Layer ReLU Network ICL",
        xlabel="Number of in-context examples n",
        ylabel="MSE / d",
        ylim=(0, 1.12),
        yticks=[0.0, 0.25, 0.50, 0.75, 1.00],
        xlim=(0, 100),
        xticks=[0, 20, 40, 60, 80, 100],
    )
    legend_order = [
        "Masked Pair Encoder",
        "GPT-2",
        "2-layer NN, Adam",
        "3NN",
    ]
    legend = ax.legend(
        handles=[handles[key] for key in legend_order],
        loc="upper right",
        ncol=2,
        frameon=False,
        fancybox=False,
        borderpad=0.25,
        handlelength=1.9,
        handletextpad=0.65,
        columnspacing=1.0,
        labelspacing=0.42,
        fontsize=15,
    )
    for line in legend.get_lines():
        line.set_linewidth(3.0)

    fig.subplots_adjust(left=0.13, right=0.995, top=0.88, bottom=0.17)
    png_path = save_figure(fig, out_dir, stem)
    plt.close(fig)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curve-csv",
        type=Path,
        default=Path("results/relu_2nn_regression/curves/relu_2nn_icl.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/relu_2nn_regression/figures"),
    )
    parser.add_argument("--stem", default="relu_2nn_icl")
    args = parser.parse_args()
    print(plot(args.curve_csv, args.out_dir, args.stem))


if __name__ == "__main__":
    main()
