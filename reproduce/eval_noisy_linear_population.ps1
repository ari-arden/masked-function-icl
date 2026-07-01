Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$masked = "results\linear_regression\checkpoints\masked_pair_encoder_linear.pt"
$gpt2Seed300 = "results\linear_regression\checkpoints\gpt2_causal_linear_seed300.pt"
$gpt2Seed301 = "results\linear_regression\checkpoints\gpt2_causal_linear_seed301.pt"

foreach ($checkpoint in @($masked, $gpt2Seed300, $gpt2Seed301)) {
  if (-not (Test-Path $checkpoint)) {
    throw "Missing checkpoint $checkpoint. Download it from the release or train it first."
  }
}

python scripts\eval_noisy_linear_regression.py `
  --masked_model_path $masked `
  --gpt2_model_paths "$gpt2Seed300;$gpt2Seed301" `
  --out_csv results\noisy_linear_regression\curves\noisy_linear_regression_icl.csv `
  --device cuda:0 `
  --eval_batches 64 `
  --batch_size 64 `
  --n_dims 20 `
  --n_points 41 `
  --noise_std 1 `
  --renormalization population `
  --metric_normalization by_d `
  --set_encoder

