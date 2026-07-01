$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$env:CUDA_VISIBLE_DEVICES = "0"
python scripts\train_synthetic_icl.py `
  --model_family gpt2 `
  --task linear_regression `
  --steps 500000 `
  --batch_size 64 `
  --n_dims 20 `
  --n_points 41 `
  --n_embd 256 `
  --n_layer 12 `
  --n_head 8 `
  --lr 1e-4 `
  --dim_start 5 `
  --dim_inc 1 `
  --dim_interval 2000 `
  --point_start 11 `
  --point_inc 2 `
  --point_interval 2000 `
  --eval_every 5000 `
  --eval_batches 8 `
  --save_eval_checkpoints `
  --seed 300 `
  --out ..\runs\linear_gpt2_causal_l12_b64_lr1e4_seed300
