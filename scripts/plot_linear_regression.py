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
    ("masked_pair_encoder", "Masked Pair Encoder"),
    ("gpt2_causal_mean", "GPT-2"),
    ("ols", "OLS"),
    ("nn3", "3NN"),
]


def plot(curve_csv: Path, out_dir: Path):
    summary = pd.read_csv(curve_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    points = summary["point"].to_numpy()

    for column, method in MODEL_COLUMNS:
        plot_method(
            ax,
            points,
            summary[column].to_numpy(),
            method,
        )

    style_axis(
        ax,
        title="Linear Regression ICL",
        xlabel="Number of in-context examples n",
        ylabel="MSE / d",
        ylim=(0, 1.2),
        yticks=[0.0, 0.4, 0.8, 1.2],
    )
    mark_dimension(ax, y=1.08)
    add_compact_legend(ax, ncol=1, loc="upper right")

    fig.subplots_adjust(left=0.14, right=0.995, top=0.88, bottom=0.18)
    png_path = save_figure(fig, out_dir, "linear_regression_icl")
    plt.close(fig)
    return png_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curve_csv",
        type=Path,
        default=Path("results/linear_regression/curves/linear_regression_icl.csv"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/linear_regression/figures"),
    )
    args = parser.parse_args()
    print(plot(args.curve_csv, args.out_dir))


if __name__ == "__main__":
    main()
