Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

python scripts/eval_decision_tree_baselines.py `
  --out-csv results/decision_tree/raw/decision_tree_baselines.csv `
  --seed 717171 `
  --eval-batches 64 `
  --batch-size 64 `
  --n-dims 20 `
  --n-points 41 `
  --tree-depth 4 `
  --min-point 1 `
  --max-point 40 `
  --sign-mode sign_only `
  --xgb-n-estimators 100 `
  --xgb-max-depth 4 `
  --xgb-learning-rate 0.1 `
  --xgb-reg-lambda 1.0 `
  --xgb-n-jobs 1

