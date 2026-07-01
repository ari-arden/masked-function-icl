import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from plotting_style import (
    add_compact_legend,
    configure_matplotlib,
    plot_method,
    save_figure,
    style_axis,
)

MODEL_COLUMNS = [
    ("Masked Pair Encoder", "Masked Pair Encoder"),
    ("GPT-2 causal", "GPT-2"),
    ("3NN", "3NN"),
    ("Greedy tree", "Greedy"),
    ("Greedy tree (sign)", "Greedy (sign)"),
    ("XGBoost (sign)", "XGB (sign)"),
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
        )

    style_axis(
        ax,
        title="Decision Tree ICL",
        xlabel="Number of in-context examples n",
        ylabel="MSE",
        ylim=(0, 2.22),
        yticks=[0.0, 0.5, 1.0, 1.5, 2.0],
    )
    add_compact_legend(ax, ncol=2, loc="upper right", fontsize=16, handlelength=1.8)

    fig.subplots_adjust(left=0.14, right=0.995, top=0.88, bottom=0.18)

    png_path = save_figure(fig, out_dir, stem)
    plt.close(fig)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curve-csv",
        type=Path,
        default=Path("results/decision_tree/curves/decision_tree_icl.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/decision_tree/figures"),
    )
    parser.add_argument(
        "--stem",
        default="decision_tree_icl",
    )
    args = parser.parse_args()
    print(plot(args.curve_csv, args.out_dir, args.stem))


if __name__ == "__main__":
    main()
