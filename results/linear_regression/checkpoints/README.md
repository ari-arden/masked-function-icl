# Checkpoints

Large model checkpoint files are intentionally not tracked in this repository.

Recommended release layout:

```text
results/linear_regression/checkpoints/
  masked_pair_encoder_linear.pt
  gpt2_causal_linear_seed300.pt
  gpt2_causal_linear_seed301.pt
```

Publish these files with GitHub Releases or Git LFS if needed. The checked-in
CSV curves are enough to regenerate the paper figure without downloading
checkpoints.

