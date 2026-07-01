import argparse
import csv
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from base_models import NeuralNetwork, ParallelNetworks  # noqa: E402
from models import AveragingModel, LeastSquaresModel, NNModel  # noqa: E402
from samplers import get_data_sampler  # noqa: E402
from tasks import get_task_sampler  # noqa: E402
from train import set_global_seed  # noqa: E402


def parse_points(raw_points: str, n_points: int) -> list[int]:
    if raw_points.lower() == "all":
        return list(range(n_points))
    points = []
    for raw in raw_points.split(","):
        raw = raw.strip()
        if not raw:
            continue
        point = int(raw)
        if point < 0 or point >= n_points:
            raise ValueError("points must be in [0, n_points)")
        points.append(point)
    if not points:
        raise ValueError("at least one point is required")
    return sorted(set(points))


def squared_losses(pred: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    return (pred - ys).square()


def adam_relu_predictions(
    xs: torch.Tensor,
    ys: torch.Tensor,
    points: list[int],
    *,
    hidden_size: int,
    lr: float,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    xs = xs.to(device)
    ys = ys.to(device)
    predictions = {}
    for point in tqdm(points, desc="2-layer-nn-adam", leave=False):
        if point == 0:
            predictions[point] = torch.zeros_like(ys[:, 0]).cpu()
            continue

        model = ParallelNetworks(
            ys.shape[0],
            NeuralNetwork,
            in_size=xs.shape[-1],
            hidden_size=hidden_size,
            out_size=1,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_xs = xs[:, :point]
        train_ys = ys[:, :point]
        test_x = xs[:, point : point + 1]
        effective_batch_size = min(batch_size, point)

        for _ in range(steps):
            mask = torch.zeros(point, dtype=torch.bool, device=device)
            perm = torch.randperm(point, device=device)
            mask[perm[:effective_batch_size]] = True
            optimizer.zero_grad()
            pred = model(train_xs[:, mask])[:, :, 0]
            loss = torch.nn.functional.mse_loss(pred, train_ys[:, mask])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions[point] = model(test_x)[:, 0, 0].detach().cpu()
    return predictions


def evaluate(args) -> Path:
    set_global_seed(args.seed)
    points = parse_points(args.points, args.n_points)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_sampler = get_data_sampler("gaussian", n_dims=args.n_dims)
    task_sampler = get_task_sampler(
        "relu_2nn_regression",
        args.n_dims,
        args.batch_size,
        hidden_layer_size=args.hidden_layer_size,
    )

    accum = {
        "OLS": {point: [] for point in points},
        "3NN": {point: [] for point in points},
        "Averaging": {point: [] for point in points},
        "2-layer NN, Adam": {point: [] for point in points},
    }
    ols = LeastSquaresModel()
    nn3 = NNModel(3)
    averaging = AveragingModel()

    for batch_idx in range(args.eval_batches):
        xs = data_sampler.sample_xs(args.n_points, args.batch_size)
        task = task_sampler()
        ys = task.evaluate(xs)
        predictions = {
            "OLS": ols(xs, ys),
            "3NN": nn3(xs, ys),
            "Averaging": averaging(xs, ys),
        }
        adam_predictions = adam_relu_predictions(
            xs,
            ys,
            points,
            hidden_size=args.baseline_hidden_size,
            lr=args.adam_lr,
            steps=args.adam_steps,
            batch_size=args.adam_batch_size,
            device=device,
        )

        for name, pred in predictions.items():
            losses = squared_losses(pred, ys)
            for point in points:
                accum[name][point].append(losses[:, point])
        for point in points:
            adam_loss = (adam_predictions[point] - ys[:, point]).square()
            accum["2-layer NN, Adam"][point].append(adam_loss)

        print(f"evaluated {batch_idx + 1}/{args.eval_batches}", flush=True)

    rows = []
    for point in points:
        row = {"point": point}
        for name in accum:
            row[name] = float(torch.cat(accum[name][point]).mean())
        rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        fieldnames = ["point", "OLS", "3NN", "Averaging", "2-layer NN, Adam"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return args.out_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/relu_2nn_regression/raw/relu_2nn_baselines.csv"),
    )
    parser.add_argument("--seed", type=int, default=818181)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-dims", type=int, default=20)
    parser.add_argument("--n-points", type=int, default=101)
    parser.add_argument("--hidden-layer-size", type=int, default=100)
    parser.add_argument("--points", default="0,5,10,20,30,40,60,80,100")
    parser.add_argument("--baseline-hidden-size", type=int, default=100)
    parser.add_argument("--adam-steps", type=int, default=100)
    parser.add_argument("--adam-lr", type=float, default=5e-3)
    parser.add_argument("--adam-batch-size", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(evaluate(args))


if __name__ == "__main__":
    main()
