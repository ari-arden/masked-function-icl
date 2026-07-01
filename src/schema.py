def merge(*items):
    merged = {}
    for item in items:
        merged.update(item)
    return merged


def allowed(values):
    return {"allowed": values}


def default(value):
    return {"default": value}


nullable = {"nullable": True}
required = {"required": True}


def stdict(schema):
    return {"type": "dict", "schema": schema}


tboolean = {"type": "bool"}
tdict = {"type": "dict"}
tfloat = {"type": "float"}
tinteger = {"type": "int"}
tstring = {"type": "str"}

model_schema = {
    "family": merge(
        tstring,
        allowed(
            [
                "gpt2",
                "lstm",
                "masked_encoder",
                "masked_pair_encoder",
                "masked_pair_set_encoder",
            ]
        ),
    ),
    "n_positions": merge(tinteger, required),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, required),
}

curriculum_base_schema = {
    "start": merge(tinteger, required),  # initial parameter
    "end": merge(tinteger, required),  # limit of final value
    "inc": merge(tinteger, required),  # how much to increment each time
    "interval": merge(tinteger, required),  # increment every how many steps
}

curriculum_schema = {
    "dims": stdict(curriculum_base_schema),
    "points": stdict(curriculum_base_schema),
}

TASK_LIST = [
    "linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "decision_tree",
]

training_schema = {
    "task": merge(tstring, allowed(TASK_LIST)),
    "task_kwargs": merge(tdict, required),
    "objective": merge(
        tstring,
        allowed(
            [
                "causal",
                "query_only",
                "permuted_query_only",
                "random_prefix_permuted_query",
                "multi_prefix_permuted_query",
                "leave_one_out",
                "dense_leave_one_out",
                "variable_dense_leave_one_out",
                "random_y_mask",
                "query_forced_random_y_mask",
                "hybrid",
            ]
        ),
        default("causal"),
    ),
    "mask_probability": merge(tfloat, default(0.15)),
    "hybrid_query_probability": merge(tfloat, default(0.7)),
    "loo_targets_per_sequence": merge(tinteger, nullable, default(None)),
    "random_prefix_min_points": merge(tinteger, default(2)),
    "prefixes_per_step": merge(tinteger, default(4)),
    "num_tasks": merge(tinteger, nullable, default(None)),
    "num_training_examples": merge(tinteger, nullable, default(None)),
    "data": merge(tstring, allowed(["gaussian"])),
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(3e-4)),
    "train_steps": merge(tinteger, default(1000)),
    "seed": merge(tinteger, nullable, default(None)),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
    "curriculum": stdict(curriculum_schema),
}

wandb_schema = {
    "project": merge(tstring, default("in-context-training")),
    "entity": merge(tstring, default("in-context")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)),
}
