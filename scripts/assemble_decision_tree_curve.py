import argparse
import csv
from pathlib import Path

NEURAL_COLUMNS = ["Masked Pair Encoder", "GPT-2 causal"]
BASELINE_COLUMNS = ["3NN", "Greedy tree", "Greedy tree (sign)", "XGBoost (sign)"]


def read_by_point(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {int(row["point"]): row for row in reader}


def assemble(args) -> Path:
    neural = read_by_point(args.neural_csv)
    baselines = read_by_point(args.baseline_csv)
    points = sorted(set(neural).intersection(baselines))
    if not points:
        raise ValueError("No overlapping points between neural and baseline CSVs")

    rows = []
    for point in points:
        row = {"point": point}
        for column in NEURAL_COLUMNS:
            row[column] = neural[point][column]
        for column in BASELINE_COLUMNS:
            row[column] = baselines[point][column]
        rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        fieldnames = ["point", *NEURAL_COLUMNS, *BASELINE_COLUMNS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return args.out_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--neural-csv",
        type=Path,
        required=True,
        help="CSV containing point, Masked Pair Encoder, and GPT-2 causal columns.",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("results/decision_tree/raw/decision_tree_baselines.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/decision_tree/curves/decision_tree_icl.csv"),
    )
    args = parser.parse_args()
    print(assemble(args))


if __name__ == "__main__":
    main()
