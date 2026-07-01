import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from masked_objectives import (  # noqa: E402
    build_dense_leave_one_out_batch,
    build_objective_masks,
    masked_regression_loss,
    move_random_query_to_last,
    permute_points,
)
from models import (  # noqa: E402
    LeastSquaresModel,
    MaskedPairEncoderModel,
    NNModel,
    TransformerModel,
)
from samplers import get_data_sampler  # noqa: E402
from tasks import get_task_sampler  # noqa: E402
from train import sample_training_points, set_global_seed  # noqa: E402


def task_kwargs_from_args(args):
    if args.task == "sparse_linear_regression":
        return {"sparsity": args.sparsity}
    if args.task == "decision_tree":
        return {"depth": args.tree_depth}
    if args.task == "relu_2nn_regression":
        return {"hidden_layer_size": args.hidden_layer_size}
    return {}


def parse_monitor_points(raw_points, n_points):
    points = []
    for raw in raw_points.split(","):
        raw = raw.strip()
        if not raw:
            continue
        point = int(raw)
        if point < 0:
            raise ValueError("monitor points must be nonnegative")
        if point >= n_points:
            raise ValueError("monitor points must be smaller than n_points")
        points.append(point)
    if not points:
        raise ValueError("at least one monitor point is required")
    return sorted(set(points))


def sample_task(args, n_dims, batch_size, valid_coords=None):
    sampler = get_task_sampler(
        args.task,
        n_dims,
        batch_size,
        **task_kwargs_from_args(args),
    )
    if args.task == "sparse_linear_regression" and valid_coords is not None:
        return sampler(valid_coords=valid_coords)
    return sampler()


def eval_query_curve(model, args, batch_size, batches, device):
    model.eval()
    data_sampler = get_data_sampler("gaussian", n_dims=args.n_dims)
    model_losses = []
    ols_losses = []
    nn_losses = []
    is_masked_model = getattr(model, "supports_masked_y", False)
    with torch.no_grad():
        for _ in range(batches):
            xs = data_sampler.sample_xs(args.n_points, batch_size)
            task = sample_task(args, args.n_dims, batch_size)
            ys = task.evaluate(xs)
            losses = []
            for i in range(args.n_points):
                xs_cur = xs[:, : i + 1, :]
                ys_cur = task.evaluate(xs_cur)
                if is_masked_model:
                    observed, target = build_objective_masks(
                        ys_cur, objective="query_only"
                    )
                    pred = model(
                        xs_cur.to(device),
                        ys_cur.to(device),
                        observed_y_mask=observed.to(device),
                        target_y_mask=target.to(device),
                    ).cpu()
                else:
                    pred = model(xs_cur.to(device), ys_cur.to(device)).cpu()
                losses.append(task.get_metric()(pred, ys_cur)[:, -1])
            model_losses.append(torch.stack(losses, dim=1))
            ols_losses.append(task.get_metric()(LeastSquaresModel()(xs, ys), ys))
            nn_losses.append(task.get_metric()(NNModel(3)(xs, ys), ys))
    model.train()
    return {
        "model": torch.cat(model_losses).mean(dim=0),
        "ols": torch.cat(ols_losses).mean(dim=0),
        "nn3": torch.cat(nn_losses).mean(dim=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_dims", type=int, default=20)
    parser.add_argument("--n_points", type=int, default=41)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--model_family",
        choices=["masked_pair", "gpt2"],
        default="masked_pair",
    )
    parser.add_argument(
        "--task",
        choices=[
            "linear_regression",
            "sparse_linear_regression",
            "decision_tree",
            "relu_2nn_regression",
            "quadratic_regression",
        ],
        default="linear_regression",
    )
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--tree_depth", type=int, default=4)
    parser.add_argument("--hidden_layer_size", type=int, default=100)
    parser.add_argument("--dim_start", type=int, default=None)
    parser.add_argument("--dim_inc", type=int, default=1)
    parser.add_argument("--dim_interval", type=int, default=2000)
    parser.add_argument("--point_start", type=int, default=None)
    parser.add_argument("--point_inc", type=int, default=2)
    parser.add_argument("--point_interval", type=int, default=2000)
    parser.add_argument("--mask_probability", type=float, default=0.5)
    parser.add_argument(
        "--objective",
        choices=[
            "query_only",
            "permuted_query_only",
            "random_prefix_permuted_query",
            "multi_prefix_permuted_query",
            "variable_dense_leave_one_out",
            "query_forced_random_y_mask",
            "dense_leave_one_out",
        ],
        default="query_forced_random_y_mask",
    )
    parser.add_argument("--loo_targets_per_sequence", type=int, default=None)
    parser.add_argument("--random_prefix_min_points", type=int, default=2)
    parser.add_argument("--prefixes_per_step", type=int, default=4)
    parser.add_argument("--set_encoder", action="store_true")
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--eval_batches", type=int, default=8)
    parser.add_argument("--monitor_points", default="5,10,20,25,30,40")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume_model", default=None)
    parser.add_argument("--step_offset", type=int, default=0)
    parser.add_argument("--save_eval_checkpoints", action="store_true")
    parser.add_argument("--out", default="../runs/synthetic_icl")
    args = parser.parse_args()
    monitor_points = parse_monitor_points(args.monitor_points, args.n_points)

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = (SRC / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"

    if args.model_family == "gpt2":
        model = TransformerModel(
            n_dims=args.n_dims,
            n_positions=args.n_points,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
        ).to(device)
    else:
        model = MaskedPairEncoderModel(
            n_dims=args.n_dims,
            n_positions=args.n_points,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            use_position_embeddings=not args.set_encoder,
        ).to(device)
    if args.resume_model is not None:
        state_dict = torch.load(args.resume_model, map_location=device)
        model.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    data_sampler = get_data_sampler("gaussian", n_dims=args.n_dims)

    fieldnames = [
        "step",
        "train_dims",
        "train_points",
        "train_loss",
    ]
    for point in monitor_points:
        fieldnames.append(f"model_n{point}")
    for point in monitor_points:
        fieldnames.append(f"ols_n{point}")
    for point in monitor_points:
        fieldnames.append(f"nn3_n{point}")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

    for step in range(args.steps + 1):
        if args.dim_start is None:
            train_dims = args.n_dims
        else:
            train_dims = min(
                args.n_dims,
                args.dim_start + (step // args.dim_interval) * args.dim_inc,
            )
        if args.point_start is None:
            train_points = args.n_points
        else:
            train_points = min(
                args.n_points,
                args.point_start
                + (step // args.point_interval) * args.point_inc,
            )
        if args.model_family == "gpt2":
            sampled_train_points = train_points
            xs = data_sampler.sample_xs(
                sampled_train_points, args.batch_size, n_dims_truncated=train_dims
            ).to(device)
            task = sample_task(
                args,
                args.n_dims,
                args.batch_size,
                valid_coords=train_dims,
            )
            ys = task.evaluate(xs.cpu()).to(device)
            optimizer.zero_grad()
            pred = model(xs, ys)
            loss = task.get_training_metric()(pred, ys)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
        elif args.objective == "multi_prefix_permuted_query":
            optimizer.zero_grad()
            losses = []
            sampled_train_points = train_points
            for _ in range(args.prefixes_per_step):
                sampled_train_points = sample_training_points(
                    args.objective,
                    train_points,
                    min_points=args.random_prefix_min_points,
                )
                xs = data_sampler.sample_xs(
                    sampled_train_points,
                    args.batch_size,
                    n_dims_truncated=train_dims,
                ).to(device)
                task = sample_task(
                    args,
                    args.n_dims,
                    args.batch_size,
                    valid_coords=train_dims,
                )
                ys = task.evaluate(xs.cpu()).to(device)
                xs_train, ys_train = move_random_query_to_last(xs, ys)
                observed, target = build_objective_masks(
                    ys_train, objective="query_only"
                )
                pred = model(
                    xs_train,
                    ys_train,
                    observed_y_mask=observed.to(device),
                    target_y_mask=target.to(device),
                )
                loss = masked_regression_loss(
                    pred, ys_train, target.to(device), task.get_metric()
                )
                (loss / args.prefixes_per_step).backward()
                losses.append(float(loss.detach().cpu()))
            optimizer.step()
            loss_value = sum(losses) / len(losses)
        else:
            sampled_train_points = sample_training_points(
                args.objective,
                train_points,
                min_points=args.random_prefix_min_points,
            )
            xs = data_sampler.sample_xs(
                sampled_train_points, args.batch_size, n_dims_truncated=train_dims
            ).to(device)
            task = sample_task(
                args,
                args.n_dims,
                args.batch_size,
                valid_coords=train_dims,
            )
            ys = task.evaluate(xs.cpu()).to(device)
            if args.objective in (
                "dense_leave_one_out",
                "variable_dense_leave_one_out",
            ):
                if args.objective == "variable_dense_leave_one_out":
                    xs, ys = permute_points(xs, ys)
                xs_train, ys_train, observed, target = build_dense_leave_one_out_batch(
                    xs, ys, targets_per_sequence=args.loo_targets_per_sequence
                )
            elif args.objective in (
                "permuted_query_only",
                "random_prefix_permuted_query",
            ):
                xs_train, ys_train = move_random_query_to_last(xs, ys)
                observed, target = build_objective_masks(ys_train, objective="query_only")
            elif args.objective == "query_only":
                xs_train, ys_train = xs, ys
                observed, target = build_objective_masks(ys_train, objective="query_only")
            else:
                xs_train, ys_train = xs, ys
                observed, target = build_objective_masks(
                    ys_train,
                    objective="query_forced_random_y_mask",
                    mask_probability=args.mask_probability,
                )
            optimizer.zero_grad()
            pred = model(
                xs_train,
                ys_train,
                observed_y_mask=observed.to(device),
                target_y_mask=target.to(device),
            )
            loss = masked_regression_loss(
                pred, ys_train, target.to(device), task.get_metric()
            )
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())

        if step % args.eval_every == 0:
            curves = eval_query_curve(
                model,
                args,
                args.batch_size,
                args.eval_batches,
                device,
            )
            row = {
                "step": step + args.step_offset,
                "train_dims": train_dims,
                "train_points": sampled_train_points,
                "train_loss": loss_value,
            }
            for point in monitor_points:
                row[f"model_n{point}"] = float(curves["model"][point])
            for point in monitor_points:
                row[f"ols_n{point}"] = float(curves["ols"][point])
            for point in monitor_points:
                row[f"nn3_n{point}"] = float(curves["nn3"][point])
            with csv_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writerow(row)
            print(row, flush=True)
            if args.save_eval_checkpoints:
                torch.save(
                    model.state_dict(),
                    out_dir / f"model_step_{step + args.step_offset:06d}.pt",
                )

    torch.save(model.state_dict(), out_dir / "model.pt")


if __name__ == "__main__":
    main()
