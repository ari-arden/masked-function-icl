# Decision Tree Raw Curves

This directory is the expected location for intermediate decision-tree CSVs.

- `decision_tree_neural.csv`: columns `point`, `Masked Pair Encoder`, and
  `GPT-2 causal`, produced by evaluating trained checkpoints.
- `decision_tree_baselines.csv`: columns `point`, `3NN`, `Greedy tree`,
  `Greedy tree (sign)`, and `XGBoost (sign)`, produced by
  `scripts/eval_decision_tree_baselines.py`.

Use `scripts/assemble_decision_tree_curve.py` to combine these files into
`results/decision_tree/curves/decision_tree_icl.csv`.

