$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

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
