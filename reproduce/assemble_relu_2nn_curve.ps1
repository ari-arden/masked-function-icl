$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

python scripts\assemble_relu_2nn_curve.py `
  --masked-csv results\relu_2nn_regression\raw\masked_pair_encoder.csv `
  --gpt2-csv results\relu_2nn_regression\raw\gpt2_causal.csv `
  --baseline-csv results\relu_2nn_regression\raw\relu_2nn_baselines.csv `
  --out-csv results\relu_2nn_regression\curves\relu_2nn_icl.csv `
  --normalize-by-d `
  --n-dims 20
