$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$checkpoint = "results\linear_regression\checkpoints\gpt2_causal_linear_seed300.pt"
if (-not (Test-Path $checkpoint)) {
  throw "Missing checkpoint $checkpoint. Download it from the release or train it first."
}

python scripts\eval_gpt2_linear_curve.py `
  --model_path $checkpoint `
  --out_csv results\linear_regression\raw\gpt2_seed300.csv `
  --eval_batches 64 `
  --batch_size 64 `
  --seed 1300
