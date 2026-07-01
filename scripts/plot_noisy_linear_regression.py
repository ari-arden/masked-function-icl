import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from plotting_style import (
    add_compact_legend,
    configure_matplotlib,
    mark_dimension,
    plot_method,
    save_figure,
    style_axis,
)

MODEL_COLUMNS = [
    ("Masked Pair Encoder", "Masked Pair Encoder"),
    ("GPT-2 causal", "GPT-2"),
    ("OLS", "OLS"),
    ("3NN", "3NN"),
    ("Averaging", "Averaging"),
]


def plot(curve_csv: Path, out_dir: Path, stem: str) -> Path:
    curves = pd.read_csv(curve_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()

    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    points = curves["point"].to_numpy()

    for column, method in MODEL_COLUMNS:
        plot_method(
            ax,
            points,
            curves[column].to_numpy(),
            method,
            clip_y=1.48 if method == "OLS" else None,
        )

    style_axis(
        ax,
        title="Noisy Linear Regression ICL",
        xlabel="Number of in-context examples n",
        ylabel="MSE / d",
        ylim=(0, 1.5),
        yticks=[0.0, 0.5, 1.0, 1.5],
    )
    mark_dimension(ax, y=0.18)
    ax.annotate(
        "OLS spike clipped",
        xy=(20, 1.48),
        xytext=(8.0, 1.36),
        arrowprops={
            "arrowstyle": "-",
            "lw": 0.75,
            "color": "#4B5563",
            "shrinkA": 1,
            "shrinkB": 3,
        },
        fontsize=14,
        color="#4B5563",
    )
    add_compact_legend(ax, ncol=1, loc="upper right")

    fig.subplots_adjust(left=0.14, right=0.995, top=0.88, bottom=0.18)

    png_path = save_figure(fig, out_dir, stem)
    plt.close(fig)
    return png_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curve-csv",
        type=Path,
        default=Path("results/noisy_linear_regression/curves/noisy_linear_regression_icl.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/noisy_linear_regression/figures"),
    )
    parser.add_argument("--stem", default="noisy_linear_regression_icl")
    args = parser.parse_args()
    print(plot(args.curve_csv, args.out_dir, args.stem))


if __name__ == "__main__":
    main()
