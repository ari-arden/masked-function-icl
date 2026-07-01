import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_synthetic_icl import sample_task  # noqa: E402


def _args(task, **overrides):
    values = {
        "task": task,
        "sparsity": 3,
        "tree_depth": 4,
        "hidden_layer_size": 100,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_sparse_linear_training_task_respects_curriculum_valid_coords():
    task = sample_task(
        _args("sparse_linear_regression", sparsity=3),
        n_dims=20,
        batch_size=8,
        valid_coords=5,
    )

    nonzero = task.w_b.squeeze(-1) != 0

    assert nonzero.sum(dim=1).tolist() == [3] * 8
    assert torch.all(~nonzero[:, 5:])


def test_decision_tree_task_uses_configured_depth():
    task = sample_task(
        _args("decision_tree", tree_depth=2),
        n_dims=20,
        batch_size=8,
    )

    assert task.depth == 2
    assert task.dt_tensor.shape[1] == 2 ** (2 + 1) - 1


def test_relu_2nn_pool_default_matches_garg_hidden_size():
    from tasks import Relu2nnRegression

    pool = Relu2nnRegression.generate_pool_dict(n_dims=20, num_tasks=3)

    assert pool["W1"].shape == (3, 20, 100)
    assert pool["W2"].shape == (3, 100, 1)
