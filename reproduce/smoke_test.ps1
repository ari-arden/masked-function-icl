$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

python -m py_compile `
  scripts\train_synthetic_icl.py `
  scripts\aggregate_linear_regression.py `
  scripts\eval_gpt2_linear_curve.py `
  scripts\eval_noisy_linear_regression.py `
  scripts\eval_decision_tree_baselines.py `
  scripts\eval_relu_2nn_baselines.py `
  scripts\assemble_decision_tree_curve.py `
  scripts\assemble_relu_2nn_curve.py `
  scripts\plot_linear_regression.py `
  scripts\plot_noisy_linear_regression.py `
  scripts\plot_decision_tree.py `
  scripts\plot_relu_2nn.py `
  scripts\plotting_style.py `
  src\models.py `
  src\masked_objectives.py `
  src\train.py

pytest

python scripts\plot_linear_regression.py `
  --curve_csv results\linear_regression\curves\linear_regression_icl.csv `
  --out_dir results\linear_regression\figures

python scripts\plot_noisy_linear_regression.py `
  --curve-csv results\noisy_linear_regression\curves\noisy_linear_regression_icl.csv `
  --out-dir results\noisy_linear_regression\figures

python scripts\plot_decision_tree.py `
  --curve-csv results\decision_tree\curves\decision_tree_icl.csv `
  --out-dir results\decision_tree\figures

if (Test-Path results\relu_2nn_regression\curves\relu_2nn_icl.csv) {
  python scripts\plot_relu_2nn.py `
    --curve-csv results\relu_2nn_regression\curves\relu_2nn_icl.csv `
    --out-dir results\relu_2nn_regression\figures
}

python scripts\train_synthetic_icl.py `
  --model_family masked_pair `
  --task linear_regression `
  --objective variable_dense_leave_one_out `
  --loo_targets_per_sequence 2 `
  --set_encoder `
  --steps 2 `
  --batch_size 4 `
  --n_dims 4 `
  --n_points 41 `
  --n_embd 16 `
  --n_layer 1 `
  --n_head 2 `
  --lr 1e-4 `
  --eval_every 1 `
  --eval_batches 1 `
  --seed 7 `
  --out ..\runs\smoke\masked_linear

python scripts\train_synthetic_icl.py `
  --model_family gpt2 `
  --task linear_regression `
  --steps 2 `
  --batch_size 4 `
  --n_dims 4 `
  --n_points 41 `
  --n_embd 16 `
  --n_layer 1 `
  --n_head 2 `
  --lr 1e-4 `
  --eval_every 1 `
  --eval_batches 1 `
  --seed 8 `
  --out ..\runs\smoke\gpt2_linear

python scripts\train_synthetic_icl.py `
  --model_family masked_pair `
  --task relu_2nn_regression `
  --objective variable_dense_leave_one_out `
  --loo_targets_per_sequence 2 `
  --set_encoder `
  --steps 2 `
  --batch_size 4 `
  --n_dims 5 `
  --n_points 11 `
  --hidden_layer_size 8 `
  --n_embd 16 `
  --n_layer 1 `
  --n_head 2 `
  --lr 1e-4 `
  --eval_every 1 `
  --eval_batches 1 `
  --monitor_points 1,5,10 `
  --seed 9 `
  --out ..\runs\smoke\masked_relu_2nn
