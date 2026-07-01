import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from masked_objectives import build_objective_masks  # noqa: E402
from models import MaskedPairEncoderModel, NNModel, TransformerModel  # noqa: E402
from samplers import get_data_sampler  # noqa: E402
from tasks import get_task_sampler  # noqa: E402
from train import set_global_seed  # noqa: E402


def task_kwargs(args):
    if args.task == "decision_tree":
        return {"depth": args.tree_depth}
    if args.task == "sparse_linear_regression":
        return {"sparsity": args.sparsity}
    if args.task == "relu_2nn_regression":
        return {"hidden_layer_size": args.hidden_layer_size}
    return {}


def build_model(args, device):
    if args.model_family == "gpt2":
        model = TransformerModel(
            n_dims=args.n_dims,
            n_positions=args.n_points,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
        )
    else:
        model = MaskedPairEncoderModel(
            n_dims=args.n_dims,
            n_positions=args.n_points,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            use_position_embeddings=not args.set_encoder,
        )
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def parse_points(raw_points, n_points):
    if raw_points.lower() == "all":
        return list(range(n_points))
    return [int(point) for point in raw_points.split(",") if point.strip()]


def evaluate(args):
    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    points = parse_points(args.points, args.n_points)
    data_sampler = get_data_sampler("gaussian", n_dims=args.n_dims)
    task_sampler = get_task_sampler(
        args.task,
        args.n_dims,
        args.batch_size,
        **task_kwargs(args),
    )

    model_losses = {point: [] for point in points}
    nn3_losses = []
    with torch.no_grad():
        for batch_idx in range(args.eval_batches):
            xs = data_sampler.sample_xs(args.n_points, args.batch_size)
            task = task_sampler()
            ys = task.evaluate(xs)
            metric = task.get_metric()
            for point in points:
                xs_cur = xs[:, : point + 1].to(device)
                ys_cur = ys[:, : point + 1].to(device)
                if args.model_family == "masked_pair":
                    observed, target = build_objective_masks(
                        ys_cur, objective="query_only"
                    )
                    pred = model(
                        xs_cur,
                        ys_cur,
                        observed_y_mask=observed.to(device),
                        target_y_mask=target.to(device),
                    ).cpu()
                else:
                    pred = model(xs_cur, ys_cur).cpu()
                model_losses[point].append(metric(pred, ys[:, : point + 1])[:, -1])
            nn3_losses.append(metric(NNModel(3)(xs, ys), ys))
            if (batch_idx + 1) % max(args.eval_batches // 4, 1) == 0:
                print(f"evaluated {batch_idx + 1}/{args.eval_batches}", flush=True)

    nn3_curve = torch.cat(nn3_losses).mean(dim=0)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for point in points:
        rows.append(
            {
                "point": point,
                "model": float(torch.cat(model_losses[point]).mean()),
                "nn3": float(nn3_curve[point]),
            }
        )
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["point", "model", "nn3"])
        writer.writeheader()
        writer.writerows(rows)
    print({f"n{row['point']}": row["model"] for row in rows}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--model_family", choices=["masked_pair", "gpt2"], required=True)
    parser.add_argument("--task", default="decision_tree")
    parser.add_argument("--points", default="20,25,30,40")
    parser.add_argument("--eval_batches", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_dims", type=int, default=20)
    parser.add_argument("--n_points", type=int, default=101)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--tree_depth", type=int, default=4)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--hidden_layer_size", type=int, default=100)
    parser.add_argument("--set_encoder", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
