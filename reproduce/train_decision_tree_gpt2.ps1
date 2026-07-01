$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$env:CUDA_VISIBLE_DEVICES = "0"
python scripts\train_synthetic_icl.py `
  --model_family gpt2 `
  --task decision_tree `
  --tree_depth 4 `
  --steps 200000 `
  --batch_size 64 `
  --n_dims 20 `
  --n_points 101 `
  --n_embd 256 `
  --n_layer 12 `
  --n_head 8 `
  --lr 1e-4 `
  --dim_start 5 `
  --dim_inc 1 `
  --dim_interval 2000 `
  --point_start 26 `
  --point_inc 5 `
  --point_interval 2000 `
  --eval_every 5000 `
  --eval_batches 8 `
  --save_eval_checkpoints `
  --seed 401 `
  --out ..\runs\decision_tree_gpt2_causal_depth4_l12_b64_lr1e4_seed401
