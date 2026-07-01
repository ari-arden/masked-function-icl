# Artifact Policy

This GitHub package is the lightweight release version of the synthetic masked
ICL project. It contains source code, tests, reproduction scripts, compact CSV
curves, and publication figures. Full local training logs and intermediate
checkpoints are useful for audit, but they are not required for ordinary GitHub
use.

## Compact Release Artifacts

These are the main files needed to inspect and regenerate paper figures:

- `results/linear_regression/`
- `results/noisy_linear_regression/`
- `results/decision_tree/`
- `results/relu_2nn_regression/`

Each result directory contains compact CSV curves and publication figures. These
files are enough to regenerate the displayed plots.

## Full Local Training Artifacts

Full local training folders may exist in the working copy used to produce the
paper figures:

- `runs/`
- `runs_relu_2nn_20260522/`

They are useful for tracing checkpoint selection and training progress, but
they can be several gigabytes. They are intentionally excluded from the standard
GitHub release. If external reviewers need the full local artifacts, publish
them separately through Git LFS or GitHub Release assets.

## Upload Policy

Recommended release layout:

1. Commit source code, tests, documentation, compact CSV curves, and final
   figures.
2. Keep full local run folders outside the ordinary Git history for research
   audit.
3. Publish large `.pt` checkpoints and full run archives through Git LFS or
   GitHub Releases only if external reviewers need to download them.

This policy keeps the released repository easy to clone while preserving a
clear path to full artifact sharing.
