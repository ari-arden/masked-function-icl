$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$masked = "results\relu_2nn_regression\checkpoints\masked_pair_encoder_relu_2nn.pt"
$gpt2 = "results\relu_2nn_regression\checkpoints\gpt2_causal_relu_2nn.pt"
foreach ($checkpoint in @($masked, $gpt2)) {
  if (-not (Test-Path $checkpoint)) {
    throw "Missing checkpoint $checkpoint. Download it from the release or train it first."
  }
}

python scripts\eval_checkpoint_curve.py `
  --model_path $masked `
  --out_csv results\relu_2nn_regression\raw\masked_pair_encoder.csv `
  --model_family masked_pair `
  --task relu_2nn_regression `
  --points 0,5,10,20,30,40,60,80,100 `
  --eval_batches 64 `
  --batch_size 64 `
  --seed 828282 `
  --n_dims 20 `
  --n_points 101 `
  --hidden_layer_size 100 `
  --n_embd 256 `
  --n_layer 12 `
  --n_head 8 `
  --set_encoder

python scripts\eval_checkpoint_curve.py `
  --model_path $gpt2 `
  --out_csv results\relu_2nn_regression\raw\gpt2_causal.csv `
  --model_family gpt2 `
  --task relu_2nn_regression `
  --points 0,5,10,20,30,40,60,80,100 `
  --eval_batches 64 `
  --batch_size 64 `
  --seed 838383 `
  --n_dims 20 `
  --n_points 101 `
  --hidden_layer_size 100 `
  --n_embd 256 `
  --n_layer 12 `
  --n_head 8
