$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

python scripts\plot_relu_2nn.py `
  --curve-csv results\relu_2nn_regression\curves\relu_2nn_icl.csv `
  --out-dir results\relu_2nn_regression\figures
