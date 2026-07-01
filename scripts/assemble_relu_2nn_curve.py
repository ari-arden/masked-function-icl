import argparse
import csv
from pathlib import Path

NEURAL_COLUMNS = [
    ("Masked Pair Encoder", "Masked Pair Encoder"),
    ("GPT-2 causal", "GPT-2 causal"),
]
BASELINE_COLUMNS = ["OLS", "3NN", "Averaging", "2-layer NN, Adam"]


def read_by_point(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {int(row["point"]): row for row in reader}


def _model_value(row: dict[str, str]) -> str:
    if "model" in row:
        return row["model"]
    if "Masked Pair Encoder" in row:
        return row["Masked Pair Encoder"]
    if "GPT-2 causal" in row:
        return row["GPT-2 causal"]
    raise KeyError("Neural CSV must contain a model column")


def _maybe_scale(value: str, scale: float) -> float:
    return float(value) / scale


def assemble(args) -> Path:
    masked = read_by_point(args.masked_csv)
    gpt2 = read_by_point(args.gpt2_csv)
    baselines = read_by_point(args.baseline_csv)
    points = sorted(set(masked).intersection(gpt2).intersection(baselines))
    if not points:
        raise ValueError("No overlapping points between neural and baseline CSVs")

    scale = 1.0
    if getattr(args, "normalize_by_zero", False):
        first_point = points[0]
        scale = max(float(baselines[first_point]["OLS"]), 1e-12)
    elif getattr(args, "normalize_by_d", False):
        scale = float(args.n_dims)

    rows = []
    for point in points:
        row = {"point": point}
        row["Masked Pair Encoder"] = _maybe_scale(_model_value(masked[point]), scale)
        row["GPT-2 causal"] = _maybe_scale(_model_value(gpt2[point]), scale)
        for column in BASELINE_COLUMNS:
            row[column] = _maybe_scale(baselines[point][column], scale)
        rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        fieldnames = ["point", *[name for name, _ in NEURAL_COLUMNS], *BASELINE_COLUMNS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return args.out_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--masked-csv",
        type=Path,
        required=True,
        help="CSV containing point and model columns for the masked model.",
    )
    parser.add_argument(
        "--gpt2-csv",
        type=Path,
        required=True,
        help="CSV containing point and model columns for the GPT-2 model.",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("results/relu_2nn_regression/raw/relu_2nn_baselines.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/relu_2nn_regression/curves/relu_2nn_icl.csv"),
    )
    parser.add_argument("--normalize-by-d", action="store_true")
    parser.add_argument("--normalize-by-zero", action="store_true")
    parser.add_argument("--n-dims", type=int, default=20)
    args = parser.parse_args()
    if args.normalize_by_d and args.normalize_by_zero:
        raise ValueError("Choose at most one normalization mode")
    print(assemble(args))


if __name__ == "__main__":
    main()
