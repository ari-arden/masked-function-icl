import argparse
from copy import deepcopy
from pathlib import Path

import yaml


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def parse_value(value):
    return yaml.safe_load(value)


def _deep_merge(base, override):
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _to_attr_dict(value):
    if isinstance(value, dict):
        return AttrDict({key: _to_attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_attr_dict(item) for item in value]
    return value


def to_plain_dict(value):
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    return value


def load_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}

    inherited = {}
    for parent in current.pop("inherit", []) or []:
        parent_path = (path.parent / parent).resolve()
        inherited = _deep_merge(inherited, to_plain_dict(load_config(parent_path)))

    return _to_attr_dict(_deep_merge(inherited, current))


def apply_override(config, dotted_key, value):
    parts = dotted_key.split(".")
    target = config
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = AttrDict()
        target = target[part]
    target[parts[-1]] = parse_value(value)


def _set_default(config, dotted_key, value):
    parts = dotted_key.split(".")
    target = config
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = AttrDict()
        target = target[part]
    target.setdefault(parts[-1], value)


def apply_defaults(config):
    defaults = {
        "test_run": False,
        "training.objective": "causal",
        "training.mask_probability": 0.15,
        "training.hybrid_query_probability": 0.7,
        "training.num_tasks": None,
        "training.num_training_examples": None,
        "training.resume_id": None,
        "training.seed": None,
        "wandb.notes": "",
        "wandb.log_every_steps": 10,
    }
    for key, value in defaults.items():
        _set_default(config, key, value)
    return config


def parse_config_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    known, unknown = parser.parse_known_args(argv)
    config = apply_defaults(load_config(known.config))
    config.config = known.config

    if len(unknown) % 2 != 0:
        raise ValueError("Overrides must be passed as --dotted.key value pairs")
    for key, value in zip(unknown[0::2], unknown[1::2]):
        if not key.startswith("--"):
            raise ValueError(f"Invalid override key: {key}")
        apply_override(config, key[2:], value)

    return config
