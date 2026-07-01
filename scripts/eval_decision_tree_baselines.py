import argparse
import csv
import sys
from pathlib import Path

import torch
from tqdm import tqdm

try:
    import xgboost as xgb
except ModuleNotFoundError:
    xgb = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models import DecisionTreeModel, NNModel  # noqa: E402
from samplers import get_data_sampler  # noqa: E402
from tasks import get_task_sampler  # noqa: E402
from train import set_global_seed  # noqa: E402


def sign_preprocess(xs: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "sign_only":
        return torch.sign(xs)
    if mode == "concat":
        return torch.cat([xs, torch.sign(xs)], dim=-1)
    raise ValueError("mode must be 'sign_only' or 'concat'")


def squared_losses(pred: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    return (pred - ys).square()


def xgboost_sign_predictions(xs_sign: torch.Tensor, ys: torch.Tensor, args):
    if xgb is None:
        raise ImportError("xgboost is required for the sign-preprocessed XGBoost baseline")

    xs_cpu = xs_sign.cpu()
    ys_cpu = ys.cpu()
    preds = torch.zeros_like(ys_cpu)
    points = range(args.min_point, args.max_point + 1)

    for point in tqdm(points, desc="xgboost-sign", leave=False):
        pred = torch.zeros_like(ys_cpu[:, 0])
        for row_idx in range(ys_cpu.shape[0]):
            train_xs = xs_cpu[row_idx, :point].numpy()
            train_ys = ys_cpu[row_idx, :point].numpy()
            test_x = xs_cpu[row_idx, point : point + 1].numpy()
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=args.xgb_n_estimators,
                max_depth=args.xgb_max_depth,
                learning_rate=args.xgb_learning_rate,
                subsample=1.0,
                colsample_bytree=1.0,
                reg_lambda=args.xgb_reg_lambda,
                n_jobs=args.xgb_n_jobs,
                random_state=args.seed + point * 100_000 + row_idx,
                verbosity=0,
            )
            model.fit(train_xs, train_ys)
            pred[row_idx] = float(model.predict(test_x)[0])
        preds[:, point] = pred
    return preds


def evaluate(args) -> Path:
    set_global_seed(args.seed)
    data_sampler = get_data_sampler("gaussian", n_dims=args.n_dims)
    task_sampler = get_task_sampler(
        "decision_tree",
        args.n_dims,
        args.batch_size,
        depth=args.tree_depth,
    )
    points = list(range(args.min_point, args.max_point + 1))
    accum = {
        "3NN": {point: [] for point in points},
        "Greedy tree": {point: [] for point in points},
        "Greedy tree (sign)": {point: [] for point in points},
        "XGBoost (sign)": {point: [] for point in points},
    }

    nn3 = NNModel(3)
    greedy = DecisionTreeModel(max_depth=args.tree_depth)
    greedy_sign = DecisionTreeModel(max_depth=args.tree_depth)

    for batch_idx in range(args.eval_batches):
        xs = data_sampler.sample_xs(args.n_points, args.batch_size)
        task = task_sampler()
        ys = task.evaluate(xs)
        xs_sign = sign_preprocess(xs, args.sign_mode)

        predictions = {
            "3NN": nn3(xs, ys),
            "Greedy tree": greedy(xs, ys),
            "Greedy tree (sign)": greedy_sign(xs_sign, ys),
            "XGBoost (sign)": xgboost_sign_predictions(xs_sign, ys, args),
        }
        for name, pred in predictions.items():
            losses = squared_losses(pred, ys)
            for point in points:
                accum[name][point].append(losses[:, point])

        print(f"evaluated {batch_idx + 1}/{args.eval_batches}", flush=True)

    rows = []
    for point in points:
        row = {"point": point}
        for name in accum:
            row[name] = float(torch.cat(accum[name][point]).mean())
        rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        fieldnames = [
            "point",
            "3NN",
            "Greedy tree",
            "Greedy tree (sign)",
            "XGBoost (sign)",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return args.out_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/decision_tree/raw/decision_tree_baselines.csv"),
    )
    parser.add_argument("--seed", type=int, default=717171)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-dims", type=int, default=20)
    parser.add_argument("--n-points", type=int, default=41)
    parser.add_argument("--tree-depth", type=int, default=4)
    parser.add_argument("--min-point", type=int, default=1)
    parser.add_argument("--max-point", type=int, default=40)
    parser.add_argument(
        "--sign-mode",
        choices=["sign_only", "concat"],
        default="sign_only",
        help="Use sign(x), or concatenate raw x and sign(x), for sign baselines.",
    )
    parser.add_argument("--xgb-n-estimators", type=int, default=100)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--xgb-n-jobs", type=int, default=1)
    args = parser.parse_args()
    if args.max_point >= args.n_points:
        raise ValueError("--max-point must be smaller than --n-points")
    print(evaluate(args))


if __name__ == "__main__":
    main()
