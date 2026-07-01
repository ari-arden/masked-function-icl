Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$neuralCsv = "results/decision_tree/raw/decision_tree_neural.csv"
if (-not (Test-Path $neuralCsv)) {
  throw "Missing $neuralCsv. Export Masked Pair Encoder and GPT-2 causal columns before assembling the final curve."
}

python scripts/assemble_decision_tree_curve.py `
  --neural-csv $neuralCsv `
  --baseline-csv results/decision_tree/raw/decision_tree_baselines.csv `
  --out-csv results/decision_tree/curves/decision_tree_icl.csv

