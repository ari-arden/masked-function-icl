import random

import torch

CAUSAL_OBJECTIVE = "causal"
QUERY_ONLY_OBJECTIVE = "query_only"
PERMUTED_QUERY_ONLY_OBJECTIVE = "permuted_query_only"
RANDOM_PREFIX_PERMUTED_QUERY_OBJECTIVE = "random_prefix_permuted_query"
MULTI_PREFIX_PERMUTED_QUERY_OBJECTIVE = "multi_prefix_permuted_query"
LEAVE_ONE_OUT_OBJECTIVE = "leave_one_out"
DENSE_LEAVE_ONE_OUT_OBJECTIVE = "dense_leave_one_out"
VARIABLE_DENSE_LEAVE_ONE_OUT_OBJECTIVE = "variable_dense_leave_one_out"
RANDOM_Y_MASK_OBJECTIVE = "random_y_mask"
QUERY_FORCED_RANDOM_Y_MASK_OBJECTIVE = "query_forced_random_y_mask"
HYBRID_OBJECTIVE = "hybrid"

MASKED_OBJECTIVES = {
    QUERY_ONLY_OBJECTIVE,
    PERMUTED_QUERY_ONLY_OBJECTIVE,
    RANDOM_PREFIX_PERMUTED_QUERY_OBJECTIVE,
    MULTI_PREFIX_PERMUTED_QUERY_OBJECTIVE,
    LEAVE_ONE_OUT_OBJECTIVE,
    DENSE_LEAVE_ONE_OUT_OBJECTIVE,
    VARIABLE_DENSE_LEAVE_ONE_OUT_OBJECTIVE,
    RANDOM_Y_MASK_OBJECTIVE,
    QUERY_FORCED_RANDOM_Y_MASK_OBJECTIVE,
    HYBRID_OBJECTIVE,
}


def build_query_only_mask(ys, query_positions):
    b_size, n_points = ys.shape
    positions = torch.arange(n_points, device=ys.device).unsqueeze(0)
    query_positions = query_positions.to(ys.device).view(b_size, 1)
    observed = positions < query_positions
    target_mask = positions == query_positions
    return observed, target_mask


def build_leave_one_out_mask(ys, mask_positions):
    b_size, n_points = ys.shape
    positions = torch.arange(n_points, device=ys.device).unsqueeze(0)
    mask_positions = mask_positions.to(ys.device).view(b_size, 1)
    target_mask = positions == mask_positions
    observed = ~target_mask
    return observed, target_mask


def move_query_to_last(xs, ys, query_positions):
    b_size, n_points = ys.shape
    query_positions = query_positions.to(ys.device).view(b_size, 1)
    positions = torch.arange(n_points, device=ys.device).expand(b_size, n_points)
    support_positions = positions[positions != query_positions].view(
        b_size, n_points - 1
    )
    order = torch.cat([support_positions, query_positions], dim=1)
    moved_xs = xs.gather(1, order.unsqueeze(-1).expand(-1, -1, xs.shape[-1]))
    moved_ys = ys.gather(1, order)
    return moved_xs, moved_ys


def permute_points(xs, ys, generator=None):
    b_size, n_points = ys.shape
    noise = torch.rand(b_size, n_points, device=ys.device, generator=generator)
    order = noise.argsort(dim=1)
    shuffled_xs = xs.gather(1, order.unsqueeze(-1).expand(-1, -1, xs.shape[-1]))
    shuffled_ys = ys.gather(1, order)
    return shuffled_xs, shuffled_ys


def move_random_query_to_last(xs, ys, generator=None):
    query_positions = torch.randint(
        ys.shape[1],
        size=(ys.shape[0],),
        device=ys.device,
        generator=generator,
    )
    return move_query_to_last(xs, ys, query_positions)


def build_dense_leave_one_out_batch(xs, ys, targets_per_sequence=None, generator=None):
    b_size, n_points = ys.shape
    if targets_per_sequence is None or targets_per_sequence >= n_points:
        targets_per_sequence = n_points
        target_positions = torch.arange(n_points, device=ys.device).expand(
            b_size, n_points
        )
    else:
        if targets_per_sequence <= 0:
            raise ValueError("targets_per_sequence must be positive.")
        target_positions = torch.randint(
            n_points,
            size=(b_size, targets_per_sequence),
            device=ys.device,
            generator=generator,
        )

    expanded_xs = xs.repeat_interleave(targets_per_sequence, dim=0)
    expanded_ys = ys.repeat_interleave(targets_per_sequence, dim=0)
    flat_positions = target_positions.reshape(-1)
    positions = torch.arange(n_points, device=ys.device).unsqueeze(0)
    target_mask = positions == flat_positions.view(-1, 1)
    observed = ~target_mask
    return expanded_xs, expanded_ys, observed, target_mask


def build_random_y_mask(ys, mask_probability, generator=None):
    target_mask = torch.rand(
        ys.shape, device=ys.device, generator=generator
    ) < mask_probability
    missing_rows = target_mask.sum(dim=1) == 0
    if missing_rows.any():
        row_indices = torch.arange(ys.shape[0], device=ys.device)[missing_rows]
        fallback_positions = torch.zeros(
            missing_rows.sum().item(), dtype=torch.long, device=ys.device
        )
        target_mask[row_indices, fallback_positions] = True
    observed = ~target_mask
    return observed, target_mask


def build_query_forced_random_y_mask(ys, mask_probability, generator=None):
    target_mask = torch.rand(
        ys.shape, device=ys.device, generator=generator
    ) < mask_probability
    target_mask[:, -1] = True
    observed = ~target_mask
    return observed, target_mask


def sample_query_positions(ys):
    b_size, n_points = ys.shape
    return torch.randint(n_points, size=(b_size,), device=ys.device)


def last_query_positions(ys):
    b_size, n_points = ys.shape
    return torch.full((b_size,), n_points - 1, dtype=torch.long, device=ys.device)


def build_objective_masks(
    ys,
    objective,
    mask_probability=0.15,
    hybrid_query_probability=0.7,
    generator=None,
):
    if objective == QUERY_ONLY_OBJECTIVE:
        return build_query_only_mask(ys, last_query_positions(ys))
    if objective == LEAVE_ONE_OUT_OBJECTIVE:
        return build_leave_one_out_mask(ys, sample_query_positions(ys))
    if objective == RANDOM_Y_MASK_OBJECTIVE:
        return build_random_y_mask(ys, mask_probability, generator=generator)
    if objective == QUERY_FORCED_RANDOM_Y_MASK_OBJECTIVE:
        return build_query_forced_random_y_mask(
            ys, mask_probability, generator=generator
        )
    if objective == HYBRID_OBJECTIVE:
        if random.random() < hybrid_query_probability:
            return build_query_only_mask(ys, last_query_positions(ys))
        return build_random_y_mask(ys, mask_probability, generator=generator)
    raise ValueError(f"Unknown masked objective: {objective}")


def masked_regression_loss(pred, ys, target_mask, elementwise_loss_func):
    losses = elementwise_loss_func(pred, ys)
    return losses[target_mask].mean()
