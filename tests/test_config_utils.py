import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import (  # noqa: E402
    apply_defaults,
    load_config,
    parse_value,
    to_plain_dict,
)


def test_load_config_merges_inherited_yaml(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "model:\n  family: gpt2\n  n_layer: 3\ntraining:\n  seed: 1\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "inherit:\n  - base.yaml\nmodel:\n  family: masked_encoder\n",
        encoding="utf-8",
    )

    config = load_config(child)

    assert config.model.family == "masked_encoder"
    assert config.model.n_layer == 3
    assert config.training.seed == 1


def test_parse_value_uses_yaml_scalars():
    assert parse_value("true") is True
    assert parse_value("0.15") == 0.15
    assert parse_value('{"sparsity": 3}') == {"sparsity": 3}


def test_to_plain_dict_removes_attribute_dict_wrappers(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("training:\n  seed: 7\n", encoding="utf-8")

    config = load_config(config_path)

    assert to_plain_dict(config) == {"training": {"seed": 7}}


def test_apply_defaults_adds_schema_defaults_needed_by_training(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("training:\n  task: linear_regression\n", encoding="utf-8")

    config = apply_defaults(load_config(config_path))

    assert config.training.num_tasks is None
    assert config.training.num_training_examples is None
    assert config.training.objective == "causal"
    assert config.training.seed is None
