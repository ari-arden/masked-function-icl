import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from masked_objectives import (  # noqa: E402
    build_dense_leave_one_out_batch,
    build_leave_one_out_mask,
    build_objective_masks,
    build_query_forced_random_y_mask,
    build_query_only_mask,
    build_random_y_mask,
    masked_regression_loss,
    move_query_to_last,
    permute_points,
)
from models import MaskedEncoderModel, MaskedPairEncoderModel  # noqa: E402
from samplers import get_data_sampler  # noqa: E402
from tasks import (
    get_task_sampler,  # noqa: E402
    squared_error,  # noqa: E402
)
from train import (  # noqa: E402
    get_training_device,
    sample_training_points,
    set_global_seed,
    train_step,
)


def test_query_only_mask_observes_only_prefix_labels():
    ys = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    observed, target_mask = build_query_only_mask(
        ys, query_positions=torch.tensor([2])
    )

    assert observed.tolist() == [[True, True, False, False]]
    assert target_mask.tolist() == [[False, False, True, False]]


def test_leave_one_out_masks_exactly_one_label_per_sequence():
    ys = torch.ones(2, 4)

    observed, target_mask = build_leave_one_out_mask(
        ys, mask_positions=torch.tensor([0, 3])
    )

    assert target_mask.tolist() == [
        [True, False, False, False],
        [False, False, False, True],
    ]
    assert observed.tolist() == [
        [False, True, True, True],
        [True, True, True, False],
    ]


def test_random_y_mask_has_at_least_one_target_per_sequence():
    ys = torch.ones(8, 5)
    generator = torch.Generator().manual_seed(0)

    observed, target_mask = build_random_y_mask(
        ys, mask_probability=0.0, generator=generator
    )

    assert target_mask.sum(dim=1).min().item() == 1
    assert torch.equal(observed, ~target_mask)


def test_query_forced_random_y_mask_always_masks_final_label():
    ys = torch.ones(4, 5)
    generator = torch.Generator().manual_seed(0)

    observed, target_mask = build_query_forced_random_y_mask(
        ys, mask_probability=0.0, generator=generator
    )

    assert target_mask.tolist() == [
        [False, False, False, False, True],
        [False, False, False, False, True],
        [False, False, False, False, True],
        [False, False, False, False, True],
    ]
    assert torch.equal(observed, ~target_mask)


def test_masked_regression_loss_uses_only_targets():
    pred = torch.tensor([[1.0, 10.0, 3.0]])
    ys = torch.tensor([[2.0, 0.0, 1.0]])
    target = torch.tensor([[True, False, True]])

    loss = masked_regression_loss(
        pred, ys, target, torch.nn.MSELoss(reduction="none")
    )

    assert torch.isclose(loss, torch.tensor(2.5))


def test_query_only_objective_uses_last_point_as_query():
    ys = torch.ones(2, 4)

    observed, target_mask = build_objective_masks(ys, objective="query_only")

    assert observed.tolist() == [
        [True, True, True, False],
        [True, True, True, False],
    ]
    assert target_mask.tolist() == [
        [False, False, False, True],
        [False, False, False, True],
    ]


def test_query_forced_random_objective_masks_last_point():
    ys = torch.ones(2, 4)

    observed, target_mask = build_objective_masks(
        ys, objective="query_forced_random_y_mask", mask_probability=0.0
    )

    assert observed.tolist() == [
        [True, True, True, False],
        [True, True, True, False],
    ]
    assert target_mask.tolist() == [
        [False, False, False, True],
        [False, False, False, True],
    ]


def test_dense_leave_one_out_batch_expands_every_target_position():
    xs = torch.randn(2, 3, 4)
    ys = torch.randn(2, 3)

    xs_exp, ys_exp, observed, target = build_dense_leave_one_out_batch(xs, ys)

    assert xs_exp.shape == (6, 3, 4)
    assert ys_exp.shape == (6, 3)
    assert torch.equal(xs_exp[:3], xs[:1].expand(3, -1, -1))
    assert torch.equal(ys_exp[:3], ys[:1].expand(3, -1))
    assert target.tolist() == [
        [True, False, False],
        [False, True, False],
        [False, False, True],
        [True, False, False],
        [False, True, False],
        [False, False, True],
    ]
    assert torch.equal(observed, ~target)


def test_dense_leave_one_out_batch_can_sample_targets_per_sequence():
    xs = torch.randn(2, 5, 4)
    ys = torch.randn(2, 5)
    generator = torch.Generator(device=ys.device).manual_seed(0)

    xs_exp, ys_exp, observed, target = build_dense_leave_one_out_batch(
        xs, ys, targets_per_sequence=2, generator=generator
    )

    assert xs_exp.shape == (4, 5, 4)
    assert ys_exp.shape == (4, 5)
    assert target.sum(dim=1).tolist() == [1, 1, 1, 1]
    assert observed.sum(dim=1).tolist() == [4, 4, 4, 4]


def test_move_query_to_last_reorders_each_sequence():
    xs = torch.tensor(
        [
            [[1.0], [2.0], [3.0], [4.0]],
            [[10.0], [20.0], [30.0], [40.0]],
        ]
    )
    ys = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]])

    moved_xs, moved_ys = move_query_to_last(
        xs, ys, query_positions=torch.tensor([1, 2])
    )

    assert moved_xs[:, :, 0].tolist() == [
        [1.0, 3.0, 4.0, 2.0],
        [10.0, 20.0, 40.0, 30.0],
    ]
    assert moved_ys.tolist() == [
        [1.0, 3.0, 4.0, 2.0],
        [10.0, 20.0, 40.0, 30.0],
    ]


def test_masked_encoder_predicts_only_target_positions():
    model = MaskedEncoderModel(
        n_dims=3, n_positions=4, n_embd=16, n_layer=1, n_head=2
    )
    xs = torch.randn(2, 4, 3)
    ys = torch.randn(2, 4)
    observed = torch.tensor(
        [[True, True, False, False], [True, False, True, False]]
    )
    target = torch.tensor(
        [[False, False, True, False], [False, True, False, False]]
    )

    pred = model(xs, ys, observed_y_mask=observed, target_y_mask=target)

    assert pred.shape == (2, 4)
    assert torch.all(pred[~target] == 0)


def test_masked_pair_encoder_predicts_only_target_positions():
    model = MaskedPairEncoderModel(
        n_dims=3, n_positions=4, n_embd=16, n_layer=1, n_head=2
    )
    xs = torch.randn(2, 4, 3)
    ys = torch.randn(2, 4)
    observed = torch.tensor(
        [[True, True, False, False], [True, False, True, False]]
    )
    target = torch.tensor(
        [[False, False, True, False], [False, True, False, False]]
    )

    pred = model(xs, ys, observed_y_mask=observed, target_y_mask=target)

    assert pred.shape == (2, 4)
    assert torch.all(pred[~target] == 0)


def test_masked_train_step_runs_with_query_only_objective():
    model = MaskedEncoderModel(
        n_dims=3, n_positions=4, n_embd=16, n_layer=1, n_head=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xs = torch.randn(2, 4, 3)
    ys = torch.randn(2, 4)

    loss, output = train_step(
        model,
        xs,
        ys,
        optimizer,
        squared_error,
        objective="query_only",
    )

    assert loss >= 0
    assert output.shape == ys.shape


def test_masked_train_step_runs_with_dense_leave_one_out_objective():
    model = MaskedPairEncoderModel(
        n_dims=3, n_positions=4, n_embd=16, n_layer=1, n_head=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xs = torch.randn(2, 4, 3)
    ys = torch.randn(2, 4)

    loss, output = train_step(
        model,
        xs,
        ys,
        optimizer,
        squared_error,
        objective="dense_leave_one_out",
    )

    assert loss >= 0
    assert output.shape == (8, 4)


def test_masked_train_step_runs_with_permuted_query_only_objective():
    model = MaskedPairEncoderModel(
        n_dims=3, n_positions=4, n_embd=16, n_layer=1, n_head=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xs = torch.randn(2, 4, 3)
    ys = torch.randn(2, 4)

    loss, output = train_step(
        model,
        xs,
        ys,
        optimizer,
        squared_error,
        objective="permuted_query_only",
    )

    assert loss >= 0
    assert output.shape == ys.shape


def test_random_prefix_objective_samples_shorter_training_context(monkeypatch):
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return 7

    monkeypatch.setattr("train.random.randint", fake_randint)

    sampled = sample_training_points(
        "random_prefix_permuted_query",
        current_points=20,
        min_points=3,
    )

    assert sampled == 7
    assert calls == [(3, 20)]


def test_multi_prefix_objective_samples_shorter_training_context(monkeypatch):
    monkeypatch.setattr("train.random.randint", lambda low, high: high)

    sampled = sample_training_points(
        "multi_prefix_permuted_query",
        current_points=20,
        min_points=3,
    )

    assert sampled == 20


def test_variable_dense_leave_one_out_samples_shorter_training_context(monkeypatch):
    calls = []

    def fake_randint(low, high):
        calls.append((low, high))
        return 11

    monkeypatch.setattr("train.random.randint", fake_randint)

    sampled = sample_training_points(
        "variable_dense_leave_one_out",
        current_points=41,
        min_points=3,
    )

    assert sampled == 11
    assert calls == [(3, 41)]


def test_permute_points_shuffles_x_and_y_consistently(monkeypatch):
    xs = torch.tensor([[[1.0], [2.0], [3.0]]])
    ys = torch.tensor([[10.0, 20.0, 30.0]])

    shuffled_xs, shuffled_ys = permute_points(
        xs,
        ys,
        generator=torch.Generator(device=ys.device).manual_seed(0),
    )

    assert sorted(shuffled_xs[0, :, 0].tolist()) == [1.0, 2.0, 3.0]
    assert torch.equal(shuffled_ys / 10, shuffled_xs[:, :, 0])


def test_random_prefix_objective_keeps_tiny_contexts_without_sampling(monkeypatch):
    monkeypatch.setattr(
        "train.random.randint",
        lambda low, high: (_ for _ in ()).throw(AssertionError("not called")),
    )

    sampled = sample_training_points(
        "random_prefix_permuted_query",
        current_points=2,
        min_points=3,
    )

    assert sampled == 2


def test_non_random_prefix_objective_uses_curriculum_points(monkeypatch):
    monkeypatch.setattr(
        "train.random.randint",
        lambda low, high: (_ for _ in ()).throw(AssertionError("not called")),
    )

    sampled = sample_training_points(
        "permuted_query_only",
        current_points=20,
        min_points=3,
    )

    assert sampled == 20


def test_masked_train_step_runs_with_random_prefix_permuted_objective():
    model = MaskedPairEncoderModel(
        n_dims=3, n_positions=4, n_embd=16, n_layer=1, n_head=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xs = torch.randn(2, 4, 3)
    ys = torch.randn(2, 4)

    loss, output = train_step(
        model,
        xs,
        ys,
        optimizer,
        squared_error,
        objective="random_prefix_permuted_query",
    )

    assert loss >= 0
    assert output.shape == ys.shape


def test_masked_train_step_runs_with_variable_dense_loo_objective():
    model = MaskedPairEncoderModel(
        n_dims=3, n_positions=4, n_embd=16, n_layer=1, n_head=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    xs = torch.randn(2, 4, 3)
    ys = torch.randn(2, 4)

    loss, output = train_step(
        model,
        xs,
        ys,
        optimizer,
        squared_error,
        objective="variable_dense_leave_one_out",
        loo_targets_per_sequence=2,
    )

    assert loss >= 0
    assert output.shape == (4, 4)


def test_eval_batch_returns_context_curve_for_masked_encoder():
    from eval import eval_batch

    model = MaskedEncoderModel(
        n_dims=3, n_positions=4, n_embd=16, n_layer=1, n_head=2
    )
    xs = get_data_sampler("gaussian", 3).sample_xs(4, 2)
    task_sampler = get_task_sampler("linear_regression", 3, 2)

    metrics = eval_batch(model, task_sampler, xs)

    assert metrics.shape == (2, 4)


def test_masked_eval_uses_demonstrations_then_masked_query_prompt():
    from eval import eval_batch

    class RecordingMaskedModel:
        name = "masked_encoder_recording"

        def __init__(self):
            self.calls = []

        def parameters(self):
            return iter(())

        def __call__(self, xs, ys, observed_y_mask, target_y_mask):
            self.calls.append(
                (
                    xs.shape,
                    observed_y_mask.detach().cpu().clone(),
                    target_y_mask.detach().cpu().clone(),
                )
            )
            return torch.zeros_like(ys)

    model = RecordingMaskedModel()
    xs = get_data_sampler("gaussian", 3).sample_xs(4, 2)
    task_sampler = get_task_sampler("linear_regression", 3, 2)

    eval_batch(model, task_sampler, xs)

    final_shape, final_observed, final_target = model.calls[-1]
    assert final_shape == (2, 4, 3)
    assert final_observed.tolist() == [
        [True, True, True, False],
        [True, True, True, False],
    ]
    assert final_target.tolist() == [
        [False, False, False, True],
        [False, False, False, True],
    ]


def test_masked_eval_with_xs_p_does_not_include_future_queries():
    from eval import eval_batch

    class RecordingMaskedModel:
        name = "masked_encoder_recording"

        def __init__(self):
            self.calls = []

        def parameters(self):
            return iter(())

        def __call__(self, xs, ys, observed_y_mask, target_y_mask):
            self.calls.append(xs.shape)
            return torch.zeros_like(ys)

    model = RecordingMaskedModel()
    xs = get_data_sampler("gaussian", 3).sample_xs(4, 2)
    xs_p = get_data_sampler("gaussian", 3).sample_xs(4, 2)
    task_sampler = get_task_sampler("linear_regression", 3, 2)

    eval_batch(model, task_sampler, xs, xs_p)

    assert model.calls == [
        (2, 1, 3),
        (2, 2, 3),
        (2, 3, 3),
        (2, 4, 3),
    ]


def test_set_global_seed_makes_torch_sampling_reproducible():
    set_global_seed(123)
    first = torch.randn(4)

    set_global_seed(123)
    second = torch.randn(4)

    assert torch.equal(first, second)


def test_get_training_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert get_training_device().type == "cpu"
