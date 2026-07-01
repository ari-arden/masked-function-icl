$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

python scripts\train_synthetic_icl.py `
  --task relu_2nn_regression `
  --model_family masked_pair `
  --set_encoder `
  --objective variable_dense_leave_one_out `
  --loo_targets_per_sequence 8 `
  --steps 300000 `
  --batch_size 64 `
  --n_dims 20 `
  --n_points 101 `
  --hidden_layer_size 100 `
  --n_embd 256 `
  --n_layer 12 `
  --n_head 8 `
  --lr 1e-4 `
  --dim_start 4 `
  --dim_inc 1 `
  --dim_interval 2000 `
  --point_start 22 `
  --point_inc 4 `
  --point_interval 2000 `
  --eval_every 5000 `
  --eval_batches 4 `
  --monitor_points 5,10,20,40,60,80,100 `
  --seed 500 `
  --save_eval_checkpoints `
  --out ..\runs_relu_2nn_20260522\relu_2nn_d20_masked_pair_set_l12_b64_lr1e4_loo8_seed500
