# Decision Tree Results

This directory contains the depth-4 decision-tree result used for the nonlinear
masked ICL figure.

## Files

- `curves/decision_tree_icl.csv`
- `figures/decision_tree_icl.png`
- `figures/decision_tree_icl.pdf`
- `figures/decision_tree_icl.svg`

## Protocol

- Task: random depth-4 regression trees
- Dimension: `d = 20`
- Prompt length shown: up to 40 in-context examples
- Inputs: Gaussian
- Metric: mean squared error at the query point
- Neural models: Masked Pair Encoder and causal GPT-2 trained from scratch
- Baselines: 3-nearest-neighbor, greedy tree, sign-preprocessed greedy tree,
  and sign-preprocessed XGBoost
- Baseline regeneration script: `scripts/eval_decision_tree_baselines.py`.
  The sign-preprocessed baselines default to `sign(x)` features; the script
  also exposes a `concat` mode for sensitivity checks.
- Final curve assembly script: `scripts/assemble_decision_tree_curve.py`.

Large checkpoints and full training runs, when available, are kept outside this
compact result subdirectory under the project-level `runs*` artifact folders.
