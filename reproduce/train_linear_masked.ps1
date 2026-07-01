$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$env:CUDA_VISIBLE_DEVICES = "0"
python scripts\train_synthetic_icl.py `
  --model_family masked_pair `
  --task linear_regression `
  --objective variable_dense_leave_one_out `
  --loo_targets_per_sequence 8 `
  --set_encoder `
  --steps 200000 `
  --batch_size 64 `
  --n_dims 20 `
  --n_points 41 `
  --n_embd 256 `
  --n_layer 6 `
  --n_head 8 `
  --lr 1e-4 `
  --dim_start 5 `
  --dim_inc 1 `
  --dim_interval 2000 `
  --point_start 11 `
  --point_inc 2 `
  --point_interval 2000 `
  --random_prefix_min_points 2 `
  --eval_every 2500 `
  --eval_batches 8 `
  --save_eval_checkpoints `
  --seed 52 `
  --out ..\runs\linear_masked_pair_set_variable_loo_l6_b64_lr1e4_seed52
