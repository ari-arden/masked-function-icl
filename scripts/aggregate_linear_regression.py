import argparse
import csv
from pathlib import Path
from statistics import stdev


def read_rows(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            int(row["point"]): {
                key: float(value)
                for key, value in row.items()
                if key != "point" and value != ""
            }
            for row in reader
        }


def aggregate(args) -> Path:
    masked_path = args.raw_dir / args.masked_file
    gpt2_paths = sorted(args.raw_dir.glob(args.gpt2_glob))
    if not masked_path.exists():
        raise FileNotFoundError(masked_path)
    if not gpt2_paths:
        raise FileNotFoundError(f"No GPT-2 seed CSVs matched {args.gpt2_glob}")

    masked = read_rows(masked_path)
    gpt2_runs = [read_rows(path) for path in gpt2_paths]
    scale = args.n_dims if args.normalize_by_d else 1.0

    rows = []
    for point in sorted(masked):
        gpt2_values = [run[point]["model"] / scale for run in gpt2_runs]
        rows.append(
            {
                "point": point,
                "masked_pair_encoder": masked[point]["model"] / scale,
                "gpt2_causal_mean": sum(gpt2_values) / len(gpt2_values),
                "gpt2_causal_std": stdev(gpt2_values)
                if len(gpt2_values) > 1
                else 0.0,
                "ols": masked[point]["ols"] / scale,
                "nn3": masked[point]["nn3"] / scale,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        fieldnames = [
            "point",
            "masked_pair_encoder",
            "gpt2_causal_mean",
            "gpt2_causal_std",
            "ols",
            "nn3",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return args.out_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("results/linear_regression/raw"),
    )
    parser.add_argument("--masked-file", default="masked_pair_encoder.csv")
    parser.add_argument("--gpt2-glob", default="gpt2_seed*.csv")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/linear_regression/curves/linear_regression_icl.csv"),
    )
    parser.add_argument("--n-dims", type=int, default=20)
    parser.add_argument(
        "--normalize-by-d",
        action="store_true",
        help="Divide all squared-error columns by d before writing the summary.",
    )
    args = parser.parse_args()
    print(aggregate(args))


if __name__ == "__main__":
    main()
