import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from masked_objectives import build_objective_masks  # noqa: E402
from models import (  # noqa: E402
    AveragingModel,
    LeastSquaresModel,
    MaskedPairEncoderModel,
    NNModel,
    TransformerModel,
)
from samplers import get_data_sampler  # noqa: E402
from tasks import get_task_sampler  # noqa: E402
from train import set_global_seed  # noqa: E402


def parse_paths(raw_paths: str) -> list[Path]:
    return [Path(part.strip()) for part in raw_paths.split(";") if part.strip()]


def build_masked_model(args, device):
    model = MaskedPairEncoderModel(
        n_dims=args.n_dims,
        n_positions=args.n_points,
        n_embd=args.masked_n_embd,
        n_layer=args.masked_n_layer,
        n_head=args.masked_n_head,
        use_position_embeddings=not args.set_encoder,
    )
    state = torch.load(args.masked_model_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def build_gpt2_model(model_path, args, device):
    model = TransformerModel(
        n_dims=args.n_dims,
        n_positions=args.n_points,
        n_embd=args.gpt2_n_embd,
        n_layer=args.gpt2_n_layer,
        n_head=args.gpt2_n_head,
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def masked_query_prediction(model, xs_cur, ys_cur, device):
    observed, target = build_objective_masks(ys_cur, objective="query_only")
    pred = model(
        xs_cur.to(device),
        ys_cur.to(device),
        observed_y_mask=observed.to(device),
        target_y_mask=target.to(device),
    )
    return pred.cpu()[:, -1]


def gpt2_query_prediction(model, xs_cur, ys_cur, device):
    pred = model(xs_cur.to(device), ys_cur.to(device))
    return pred.cpu()[:, -1]


def evaluate(args):
    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metric_normalizer = args.n_dims if args.metric_normalization == "by_d" else 1.0
    data_sampler = get_data_sampler("gaussian", n_dims=args.n_dims)
    task_sampler = get_task_sampler(
        "noisy_linear_regression",
        args.n_dims,
        args.batch_size,
        noise_std=args.noise_std,
        renormalization=args.renormalization,
    )

    masked_model = build_masked_model(args, device)
    gpt2_models = [build_gpt2_model(path, args, device) for path in parse_paths(args.gpt2_model_paths)]
    baselines = {
        "OLS": LeastSquaresModel(),
        "3NN": NNModel(3),
        "Averaging": AveragingModel(),
    }

    points = list(range(args.n_points))
    model_losses = {
        "Masked Pair Encoder": {point: [] for point in points},
        "GPT-2 causal": {point: [] for point in points},
    }
    gpt2_seed_losses = [
        {point: [] for point in points} for _ in range(len(gpt2_models))
    ]
    baseline_losses = {
        name: {point: [] for point in points} for name in baselines
    }

    with torch.no_grad():
        for batch_idx in range(args.eval_batches):
            xs = data_sampler.sample_xs(args.n_points, args.batch_size)
            task = task_sampler()
            ys = task.evaluate(xs)

            for point in points:
                xs_cur = xs[:, : point + 1]
                ys_cur = ys[:, : point + 1]
                target_y = ys_cur[:, -1]

                masked_pred = masked_query_prediction(masked_model, xs_cur, ys_cur, device)
                masked_loss = (masked_pred - target_y).square() / metric_normalizer
                model_losses["Masked Pair Encoder"][point].append(masked_loss)

                seed_point_losses = []
                for seed_idx, gpt2_model in enumerate(gpt2_models):
                    gpt2_pred = gpt2_query_prediction(gpt2_model, xs_cur, ys_cur, device)
                    gpt2_loss = (gpt2_pred - target_y).square() / metric_normalizer
                    gpt2_seed_losses[seed_idx][point].append(gpt2_loss)
                    seed_point_losses.append(gpt2_loss)
                model_losses["GPT-2 causal"][point].append(
                    torch.stack(seed_point_losses).mean(dim=0)
                )

            for name, baseline in baselines.items():
                pred = baseline(xs, ys)
                losses = (pred - ys).square() / metric_normalizer
                for point in points:
                    baseline_losses[name][point].append(losses[:, point])

            if (batch_idx + 1) % max(args.eval_batches // 4, 1) == 0:
                print(f"evaluated {batch_idx + 1}/{args.eval_batches}", flush=True)

    rows = []
    for point in points:
        row = {
            "point": point,
            "Masked Pair Encoder": float(
                torch.cat(model_losses["Masked Pair Encoder"][point]).mean()
            ),
            "GPT-2 causal": float(
                torch.cat(model_losses["GPT-2 causal"][point]).mean()
            ),
            "GPT-2 causal std": float(
                torch.tensor(
                    [
                        torch.cat(seed_losses[point]).mean().item()
                        for seed_losses in gpt2_seed_losses
                    ]
                ).std(unbiased=True)
                if len(gpt2_seed_losses) > 1
                else 0.0
            ),
        }
        for name in baselines:
            row[name] = float(torch.cat(baseline_losses[name][point]).mean())
        rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "point",
        "Masked Pair Encoder",
        "GPT-2 causal",
        "GPT-2 causal std",
        "OLS",
        "3NN",
        "Averaging",
    ]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    selected = {row["point"]: row for row in rows if row["point"] in [5, 10, 20, 25, 30, 40]}
    print({f"n{point}": selected[point]["Masked Pair Encoder"] for point in selected})
    print(out_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masked_model_path", type=Path, required=True)
    parser.add_argument("--gpt2_model_paths", required=True)
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("results/noisy_linear_regression/curves/noisy_linear_regression_icl.csv"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=515151)
    parser.add_argument("--eval_batches", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_dims", type=int, default=20)
    parser.add_argument("--n_points", type=int, default=41)
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument(
        "--renormalization",
        choices=["batch", "population", "none"],
        default="batch",
        help=(
            "'batch' reproduces the legacy noisy curve by using "
            "the empirical std of the generated label tensor; 'population' "
            "uses sqrt(scale^2*d + noise_std^2), avoids batch-label "
            "dependence, and is recommended for new main-paper runs; 'none' "
            "leaves noisy labels unscaled."
        ),
    )
    parser.add_argument(
        "--metric_normalization",
        choices=["by_d", "raw"],
        default="by_d",
        help="Report squared error divided by d, or raw squared error.",
    )
    parser.add_argument("--set_encoder", action="store_true")
    parser.add_argument("--masked_n_embd", type=int, default=256)
    parser.add_argument("--masked_n_layer", type=int, default=6)
    parser.add_argument("--masked_n_head", type=int, default=8)
    parser.add_argument("--gpt2_n_embd", type=int, default=256)
    parser.add_argument("--gpt2_n_layer", type=int, default=12)
    parser.add_argument("--gpt2_n_head", type=int, default=8)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
