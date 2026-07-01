# Reproducibility and Release Checklist

This repository is organized as a research artifact for synthetic masked
in-context learning experiments. The GitHub release contains compact files for
inspection and figure reproduction; full local training artifacts can be shared
separately when needed.

## Files Included in the GitHub Release

The lightweight release includes:

- `results/`: summary CSVs, raw evaluation CSVs, and final figures;
- `src/`, `scripts/`, and `reproduce/`: model, training, evaluation, assembly,
  and plotting code;
- `tests/`: checks for objectives, tasks, and reproduction scripts;
- `README.md`, `ARTIFACTS.md`, and this file: experiment documentation.

The checked-in CSV curves are enough to regenerate all displayed plots.

## Files Intended for Normal Git Commits

For a lightweight GitHub repository, commit:

- source code in `src/`;
- scripts in `scripts/`;
- reproduction commands in `reproduce/`;
- docs in `README.md`, `ARTIFACTS.md`, and this file;
- tests in `tests/`;
- compact CSV curves and final paper figures under `results/`.

Large checkpoints and full training directories are ignored by default. If the
paper or release requires publishing them, use Git LFS or GitHub Release assets
rather than ordinary Git blobs.

## Optional Full Artifacts

Full local runs may include `runs/`, `runs_*/`, and trained `.pt` checkpoints.
They are useful for audit and re-evaluation, but they are not needed to inspect
the code, run tests, or regenerate the paper figures from the checked-in CSV
curves.

## Main Figure Reproduction

Regenerate all checked-in figures from checked-in CSV curves:

```powershell
.\reproduce\plot_all_figures.ps1
```

This writes:

- `results/linear_regression/figures/linear_regression_icl.*`
- `results/noisy_linear_regression/figures/noisy_linear_regression_icl.*`
- `results/decision_tree/figures/decision_tree_icl.*`
- `results/relu_2nn_regression/figures/relu_2nn_icl.*`

## Training From Scratch

Full neural training is GPU intensive. Main entry points:

- `reproduce/train_linear_masked.ps1`
- `reproduce/train_linear_gpt2.ps1`
- `reproduce/train_decision_tree_masked.ps1`
- `reproduce/train_decision_tree_gpt2.ps1`
- `reproduce/train_relu_2nn_masked.ps1`
- `reproduce/train_relu_2nn_gpt2.ps1`

After training, run the matching evaluation, assembly, and plotting scripts
listed in `README.md`.

## Verification Before Release

Run these commands from the repository root:

```powershell
python -m ruff check .
python -m pytest -q
.\reproduce\plot_all_figures.ps1
```

The GitHub Actions workflow runs the corresponding lint, test, and plotting
checks on CPU.

## Reporting Caveats

- Masked Pair Encoder and GPT-2 causal baselines are protocol-level
  comparisons, not single-factor architecture-matched ablations.
- `K=8` in dense leave-one-out masked training means eight expanded
  single-mask copies per sampled prompt, not eight simultaneously leaked target
  labels.
- The two-layer ReLU reference labeled `2-layer NN, GD` follows prior synthetic
  ICL naming. In this repository it is implemented as a fresh per-prompt
  two-layer ReLU network optimized with Adam for 100 steps.
