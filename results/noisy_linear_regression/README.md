# Noisy Linear Regression

This directory contains the noisy linear-regression robustness evaluation. The
neural checkpoints are the same models trained on noiseless `d=20` linear
regression; only the evaluation task changes to `noisy_linear_regression` with
`noise_std=1`.

The main checked-in curve uses `--renormalization population` and reports
squared error divided by `d=20`. Population renormalization rescales noisy labels
by the analytic unconditional noisy-label standard deviation
`sqrt(d + noise_std^2)`, avoiding dependence on evaluation-batch label
statistics.

The earlier batch-normalized result is retained as
`curves/noisy_linear_regression_icl_batch_legacy.csv` for provenance only.

Main artifact:

- `curves/noisy_linear_regression_icl.csv`
- `curves/noisy_linear_regression_icl_batch_legacy.csv`
- `figures/noisy_linear_regression_icl.png`
