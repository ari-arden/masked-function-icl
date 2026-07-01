$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

python scripts\eval_relu_2nn_baselines.py `
  --out-csv results\relu_2nn_regression\raw\relu_2nn_baselines.csv `
  --seed 818181 `
  --eval-batches 16 `
  --batch-size 64 `
  --n-dims 20 `
  --n-points 101 `
  --hidden-layer-size 100 `
  --points 0,5,10,20,30,40,60,80,100 `
  --baseline-hidden-size 100 `
  --adam-steps 100 `
  --adam-lr 5e-3 `
  --adam-batch-size 100
