import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models import LeastSquaresModel, NNModel, TransformerModel  # noqa: E402
from samplers import get_data_sampler  # noqa: E402
from tasks import get_task_sampler  # noqa: E402
from train import set_global_seed  # noqa: E402


def evaluate(args):
    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TransformerModel(
        n_dims=args.n_dims,
        n_positions=args.n_points,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
    ).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    data_sampler = get_data_sampler("gaussian", n_dims=args.n_dims)
    task_sampler = get_task_sampler(
        "linear_regression",
        args.n_dims,
        args.batch_size,
    )
    model_losses = []
    ols_losses = []
    nn3_losses = []
    with torch.no_grad():
        for batch_idx in range(args.eval_batches):
            xs = data_sampler.sample_xs(args.n_points, args.batch_size)
            task = task_sampler()
            ys = task.evaluate(xs)
            metric = task.get_metric()
            point_losses = []
            for point in range(args.n_points):
                xs_cur = xs[:, : point + 1, :].to(device)
                ys_cur = ys[:, : point + 1].to(device)
                pred = model(xs_cur, ys_cur).cpu()
                point_losses.append(metric(pred, ys[:, : point + 1])[:, -1])
            model_losses.append(torch.stack(point_losses, dim=1))
            ols_losses.append(metric(LeastSquaresModel()(xs, ys), ys))
            nn3_losses.append(metric(NNModel(3)(xs, ys), ys))
            if (batch_idx + 1) % max(args.eval_batches // 4, 1) == 0:
                print(
                    f"evaluated {batch_idx + 1}/{args.eval_batches} batches",
                    flush=True,
                )

    curves = {
        "model": torch.cat(model_losses).mean(dim=0),
        "ols": torch.cat(ols_losses).mean(dim=0),
        "nn3": torch.cat(nn3_losses).mean(dim=0),
    }
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["point", "model", "ols", "nn3"])
        writer.writeheader()
        for point in range(args.n_points):
            writer.writerow(
                {
                    "point": point,
                    "model": float(curves["model"][point]),
                    "ols": float(curves["ols"][point]),
                    "nn3": float(curves["nn3"][point]),
                }
            )
    selected = [20, 25, 30, 40]
    print(
        {
            f"n{point}": float(curves["model"][point])
            for point in selected
        },
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--eval_batches", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_dims", type=int, default=20)
    parser.add_argument("--n_points", type=int, default=41)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=8)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
