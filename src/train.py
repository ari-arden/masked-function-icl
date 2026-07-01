import os
import random
import uuid
from random import randint

import numpy as np
import torch
import yaml
from tqdm import tqdm

from config_utils import parse_config_args, to_plain_dict
from curriculum import Curriculum
from masked_objectives import (
    CAUSAL_OBJECTIVE,
    DENSE_LEAVE_ONE_OUT_OBJECTIVE,
    MULTI_PREFIX_PERMUTED_QUERY_OBJECTIVE,
    PERMUTED_QUERY_ONLY_OBJECTIVE,
    RANDOM_PREFIX_PERMUTED_QUERY_OBJECTIVE,
    VARIABLE_DENSE_LEAVE_ONE_OUT_OBJECTIVE,
    build_dense_leave_one_out_batch,
    build_objective_masks,
    masked_regression_loss,
    move_random_query_to_last,
    permute_points,
)
from models import build_model
from samplers import get_data_sampler
from tasks import get_task_sampler

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

torch.backends.cudnn.benchmark = True


def set_global_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_training_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sample_training_points(objective, current_points, min_points=2):
    if objective not in (
        RANDOM_PREFIX_PERMUTED_QUERY_OBJECTIVE,
        MULTI_PREFIX_PERMUTED_QUERY_OBJECTIVE,
        VARIABLE_DENSE_LEAVE_ONE_OUT_OBJECTIVE,
    ):
        return current_points
    if current_points <= min_points:
        return current_points
    return random.randint(min_points, current_points)


def permuted_query_loss(model, xs, ys, loss_func):
    xs, ys = move_random_query_to_last(xs, ys)
    observed_y_mask, target_y_mask = build_objective_masks(
        ys,
        "query_only",
    )
    output = model(
        xs,
        ys,
        observed_y_mask=observed_y_mask,
        target_y_mask=target_y_mask,
    )
    loss = masked_regression_loss(output, ys, target_y_mask, loss_func)
    return loss, output, ys


def train_step(
    model,
    xs,
    ys,
    optimizer,
    loss_func,
    objective=CAUSAL_OBJECTIVE,
    mask_probability=0.15,
    hybrid_query_probability=0.7,
    loo_targets_per_sequence=None,
):
    optimizer.zero_grad()
    if objective == CAUSAL_OBJECTIVE:
        output = model(xs, ys)
        loss = loss_func(output, ys)
    elif objective in (
        PERMUTED_QUERY_ONLY_OBJECTIVE,
        RANDOM_PREFIX_PERMUTED_QUERY_OBJECTIVE,
        MULTI_PREFIX_PERMUTED_QUERY_OBJECTIVE,
    ):
        loss, output, _ = permuted_query_loss(model, xs, ys, loss_func)
    elif objective in (
        DENSE_LEAVE_ONE_OUT_OBJECTIVE,
        VARIABLE_DENSE_LEAVE_ONE_OUT_OBJECTIVE,
    ):
        xs, ys, observed_y_mask, target_y_mask = build_dense_leave_one_out_batch(
            xs,
            ys,
            targets_per_sequence=loo_targets_per_sequence,
        )
        output = model(
            xs,
            ys,
            observed_y_mask=observed_y_mask,
            target_y_mask=target_y_mask,
        )
        loss = masked_regression_loss(output, ys, target_y_mask, loss_func)
    else:
        observed_y_mask, target_y_mask = build_objective_masks(
            ys,
            objective,
            mask_probability=mask_probability,
            hybrid_query_probability=hybrid_query_probability,
        )
        output = model(
            xs,
            ys,
            observed_y_mask=observed_y_mask,
            target_y_mask=target_y_mask,
        )
        loss = masked_regression_loss(output, ys, target_y_mask, loss_func)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)
    device = get_training_device()

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        objective = getattr(args.training, "objective", CAUSAL_OBJECTIVE)
        training_points = sample_training_points(
            objective,
            curriculum.n_points,
            min_points=getattr(args.training, "random_prefix_min_points", 2),
        )

        if objective == MULTI_PREFIX_PERMUTED_QUERY_OBJECTIVE:
            optimizer.zero_grad()
            losses = []
            output = None
            pointwise_ys = None
            prefixes_per_step = getattr(args.training, "prefixes_per_step", 4)
            for _ in range(prefixes_per_step):
                training_points = sample_training_points(
                    objective,
                    curriculum.n_points,
                    min_points=getattr(args.training, "random_prefix_min_points", 2),
                )
                xs = data_sampler.sample_xs(
                    training_points,
                    bsize,
                    curriculum.n_dims_truncated,
                    **data_sampler_args,
                )
                task = task_sampler(**task_sampler_args)
                ys = task.evaluate(xs)
                loss_func = task.get_metric()
                loss_tensor, output_tensor, permuted_ys = permuted_query_loss(
                    model,
                    xs.to(device),
                    ys.to(device),
                    loss_func,
                )
                (loss_tensor / prefixes_per_step).backward()
                losses.append(loss_tensor.detach().item())
                output = output_tensor.detach()
                pointwise_ys = permuted_ys.detach()
            optimizer.step()
            loss = float(np.mean(losses))
        else:
            xs = data_sampler.sample_xs(
                training_points,
                bsize,
                curriculum.n_dims_truncated,
                **data_sampler_args,
            )
            task = task_sampler(**task_sampler_args)
            ys = task.evaluate(xs)
            if objective == VARIABLE_DENSE_LEAVE_ONE_OUT_OBJECTIVE:
                xs, ys = permute_points(xs, ys)

            loss_func = (
                task.get_metric()
                if objective != CAUSAL_OBJECTIVE
                else task.get_training_metric()
            )

            loss, output = train_step(
                model,
                xs.to(device),
                ys.to(device),
                optimizer,
                loss_func,
                objective=objective,
                mask_probability=getattr(args.training, "mask_probability", 0.15),
                hybrid_query_probability=getattr(
                    args.training, "hybrid_query_probability", 0.7
                ),
                loo_targets_per_sequence=getattr(
                    args.training, "loo_targets_per_sequence", None
                ),
            )
            pointwise_ys = ys.to(device)

        point_wise_tags = list(range(training_points))
        point_wise_loss_func = task.get_metric()
        if objective in (
            DENSE_LEAVE_ONE_OUT_OBJECTIVE,
            VARIABLE_DENSE_LEAVE_ONE_OUT_OBJECTIVE,
        ):
            repeats = output.shape[0] // ys.shape[0]
            pointwise_ys = ys.to(device).repeat_interleave(repeats, dim=0)
        point_wise_loss = point_wise_loss_func(output, pointwise_ys).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if (
            i % args.wandb.log_every_steps == 0
            and not args.test_run
            and wandb is not None
        ):
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "training_points": training_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    set_global_seed(getattr(args.training, "seed", None))
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        if wandb is None:
            print("wandb is not installed; continuing without wandb logging.")
        else:
            wandb.init(
                dir=args.out_dir,
                project=args.wandb.project,
                entity=args.wandb.entity,
                config=args.__dict__,
                notes=args.wandb.notes,
                name=args.wandb.name,
                resume=True,
            )

    model = build_model(args.model)
    model.to(get_training_device())
    model.train()

    train(model, args)

    if not args.test_run:
        from eval import get_run_metrics

        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    args = parse_config_args()
    assert args.model.family in [
        "gpt2",
        "lstm",
        "masked_encoder",
        "masked_pair_encoder",
        "masked_pair_set_encoder",
    ]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(to_plain_dict(args), yaml_file, default_flow_style=False)

    main(args)
