$ErrorActionPreference = "Stop"

python scripts/plot_noisy_linear_regression.py `
  --curve-csv results/noisy_linear_regression/curves/noisy_linear_regression_icl.csv `
  --out-dir results/noisy_linear_regression/figures
