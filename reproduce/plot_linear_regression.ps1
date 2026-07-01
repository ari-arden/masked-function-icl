$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

python scripts\plot_linear_regression.py `
  --curve_csv results\linear_regression\curves\linear_regression_icl.csv `
  --out_dir results\linear_regression\figures
