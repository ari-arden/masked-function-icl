$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

python scripts\plot_decision_tree.py `
  --curve-csv results\decision_tree\curves\decision_tree_icl.csv `
  --out-dir results\decision_tree\figures
