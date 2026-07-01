import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from aggregate_linear_regression import aggregate  # noqa: E402
from assemble_decision_tree_curve import assemble  # noqa: E402
from eval_decision_tree_baselines import (  # noqa: E402
    sign_preprocess,
    xgb,
    xgboost_sign_predictions,
)
from train_synthetic_icl import parse_monitor_points  # noqa: E402

from tasks import LinearRegression, NoisyLinearRegression  # noqa: E402


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_single_row(path):
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle))


def test_noisy_population_normalization_avoids_empirical_rescaling_when_noise_zero():
    xs = torch.randn(2, 4, 5)
    seeds = [11, 12]
    linear = LinearRegression(5, 2, seeds=seeds)
    noisy = NoisyLinearRegression(
        5,
        2,
        seeds=seeds,
        noise_std=0.0,
        renormalization="population",
    )

    assert torch.allclose(noisy.evaluate(xs), linear.evaluate(xs))


def test_aggregate_linear_regression_can_write_mse_divided_by_d(tmp_path):
    raw_dir = tmp_path / "raw"
    write_csv(
        raw_dir / "masked_pair_encoder.csv",
        ["point", "model", "ols", "nn3", "avg"],
        [{"point": 0, "model": 20.0, "ols": 20.0, "nn3": 40.0, "avg": 0.0}],
    )
    write_csv(
        raw_dir / "gpt2_seed1.csv",
        ["point", "model", "ols", "nn3"],
        [{"point": 0, "model": 10.0, "ols": 20.0, "nn3": 40.0}],
    )
    write_csv(
        raw_dir / "gpt2_seed2.csv",
        ["point", "model", "ols", "nn3"],
        [{"point": 0, "model": 30.0, "ols": 20.0, "nn3": 40.0}],
    )
    out_csv = tmp_path / "summary.csv"

    aggregate(
        SimpleNamespace(
            raw_dir=raw_dir,
            masked_file="masked_pair_encoder.csv",
            gpt2_glob="gpt2_seed*.csv",
            out_csv=out_csv,
            n_dims=20,
            normalize_by_d=True,
        )
    )

    row = read_single_row(out_csv)
    assert float(row["masked_pair_encoder"]) == 1.0
    assert float(row["gpt2_causal_mean"]) == 1.0
    assert float(row["ols"]) == 1.0
    assert float(row["nn3"]) == 2.0


def test_sign_preprocess_modes():
    xs = torch.tensor([[[-2.0, 0.0, 3.0]]])

    assert sign_preprocess(xs, "sign_only").tolist() == [[[-1.0, 0.0, 1.0]]]
    assert sign_preprocess(xs, "concat").shape[-1] == 6


def test_xgboost_sign_predictions_returns_full_prompt_width():
    if xgb is None:
        return

    xs = torch.tensor(
        [
            [[-1.0], [1.0], [-2.0], [2.0]],
            [[1.0], [-1.0], [2.0], [-2.0]],
        ]
    )
    ys = torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    args = SimpleNamespace(
        min_point=1,
        max_point=2,
        xgb_n_estimators=2,
        xgb_max_depth=1,
        xgb_learning_rate=0.1,
        xgb_reg_lambda=1.0,
        xgb_n_jobs=1,
        seed=3,
    )

    pred = xgboost_sign_predictions(xs, ys, args)

    assert pred.shape == ys.shape
    assert torch.equal(pred[:, 0], torch.zeros_like(ys[:, 0]))


def test_assemble_decision_tree_curve_combines_neural_and_baselines(tmp_path):
    neural_csv = tmp_path / "neural.csv"
    baseline_csv = tmp_path / "baselines.csv"
    out_csv = tmp_path / "curve.csv"
    write_csv(
        neural_csv,
        ["point", "Masked Pair Encoder", "GPT-2 causal"],
        [{"point": 1, "Masked Pair Encoder": 0.5, "GPT-2 causal": 0.6}],
    )
    write_csv(
        baseline_csv,
        ["point", "3NN", "Greedy tree", "Greedy tree (sign)", "XGBoost (sign)"],
        [
            {
                "point": 1,
                "3NN": 1.0,
                "Greedy tree": 1.1,
                "Greedy tree (sign)": 1.2,
                "XGBoost (sign)": 1.3,
            }
        ],
    )

    assemble(
        SimpleNamespace(
            neural_csv=neural_csv,
            baseline_csv=baseline_csv,
            out_csv=out_csv,
        )
    )

    row = read_single_row(out_csv)
    assert row["Masked Pair Encoder"] == "0.5"
    assert row["XGBoost (sign)"] == "1.3"


def test_parse_monitor_points_accepts_in_range_values():
    assert parse_monitor_points("5,20,100", n_points=101) == [5, 20, 100]


def test_parse_monitor_points_rejects_out_of_range_values():
    try:
        parse_monitor_points("5,101", n_points=101)
    except ValueError as exc:
        assert "smaller than n_points" in str(exc)
    else:
        raise AssertionError("Expected ValueError for monitor point 101")


def test_assemble_relu_2nn_curve_combines_neural_and_baselines(tmp_path):
    from assemble_relu_2nn_curve import assemble

    masked_csv = tmp_path / "masked.csv"
    gpt2_csv = tmp_path / "gpt2.csv"
    baseline_csv = tmp_path / "baselines.csv"
    out_csv = tmp_path / "curve.csv"
    write_csv(masked_csv, ["point", "model"], [{"point": 5, "model": 0.7}])
    write_csv(gpt2_csv, ["point", "model"], [{"point": 5, "model": 0.8}])
    write_csv(
        baseline_csv,
        ["point", "OLS", "3NN", "Averaging", "2-layer NN, Adam"],
        [
            {
                "point": 5,
                "OLS": 1.1,
                "3NN": 1.2,
                "Averaging": 1.3,
                "2-layer NN, Adam": 0.9,
            }
        ],
    )

    assemble(
        SimpleNamespace(
            masked_csv=masked_csv,
            gpt2_csv=gpt2_csv,
            baseline_csv=baseline_csv,
            out_csv=out_csv,
            normalize_by_zero=False,
        )
    )

    row = read_single_row(out_csv)
    assert row["Masked Pair Encoder"] == "0.7"
    assert row["GPT-2 causal"] == "0.8"
    assert row["2-layer NN, Adam"] == "0.9"
