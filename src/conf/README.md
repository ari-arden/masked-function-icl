# Configuration Files

The public reproduction path uses the explicit PowerShell commands in
`reproduce/` and the unified driver `scripts/train_synthetic_icl.py`.

The YAML files in this directory are compact examples for `src/train.py`. They
are intentionally minimal and mirror the public settings that are useful for
inspection:

- `masked_linear_regression.yaml`: masked pair-set encoder with variable dense
  leave-one-out label masking, matching the main linear-regression setup.
- `linear_regression.yaml`: causal Transformer baseline for noiseless linear
  regression.
- `decision_tree.yaml`: causal Transformer baseline for depth-4 decision trees.

Exploratory ablation YAMLs are intentionally excluded from the release
repository to keep the reproduction path focused.
