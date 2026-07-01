# Two-Layer ReLU Regression Checkpoints

This directory may contain the large checkpoint files used for full ReLU-2NN
evaluation. The expected names are:

```text
results/relu_2nn_regression/checkpoints/
  masked_pair_encoder_relu_2nn.pt
  gpt2_causal_relu_2nn.pt
```

They can be produced by the training scripts in `reproduce/`. If publishing the
full artifact on GitHub, use Git LFS or GitHub release assets for these files.
