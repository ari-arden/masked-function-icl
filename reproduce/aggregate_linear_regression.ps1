Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

python scripts/aggregate_linear_regression.py `
  --raw-dir results/linear_regression/raw `
  --out-csv results/linear_regression/curves/linear_regression_icl.csv `
  --n-dims 20 `
  --normalize-by-d

