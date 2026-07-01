# Linear Regression Results

This directory contains the clean linear-regression result used for the main
masked ICL figure.

## Files

- `figures/linear_regression_icl.png`
- `figures/linear_regression_icl.pdf`
- `figures/linear_regression_icl.svg`
- `curves/linear_regression_icl.csv`
- `raw/masked_pair_encoder.csv`
- `raw/gpt2_seed300.csv`
- `raw/gpt2_seed301.csv`

## Protocol

- Task: dense linear regression, `f(x) = w^T x`
- Dimension: `d = 20`
- Prompt length: up to 41 in-context examples
- Inputs: Gaussian
- Metric: normalized mean squared error at the query point, `MSE / d`
- Baselines: OLS, 3-nearest-neighbor, causal GPT-2
- Masked model: Masked Pair Encoder with leave-one-out masked label training
- Raw per-model CSVs are stored in `raw/`; regenerate the summary curve with
  `scripts/aggregate_linear_regression.py --normalize-by-d`.

Only successful linear-regression summary results are included here. Full local
training directories and checkpoints, when available, are kept outside this
compact result subdirectory under the project-level `runs*` artifact folders.
