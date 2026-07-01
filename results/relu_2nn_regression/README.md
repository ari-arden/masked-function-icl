# Two-Layer ReLU Regression Results

This directory contains the Garg-style two-layer ReLU neural-network synthetic
ICL experiment:

```math
f(x)=\sum_{j=1}^{100} a_j \operatorname{ReLU}(w_j^\top x),
```

with `d=20` and 100 hidden units in the sampled latent function. The main
paper-style figure compares:

- Masked Pair Encoder
- GPT-2 causal
- 2-layer NN, GD
- 3NN

The summary CSV also retains OLS and Averaging columns for auditing and
appendix-level checks.

Curves are assembled with `MSE / d` normalization.

For reproducibility, the `2-layer NN, GD` curve is computed by fitting a fresh
two-layer ReLU network independently for each prompt using Adam for 100 steps.
The CSV column keeps the implementation-oriented name `2-layer NN, Adam`, while
the plot legend uses the Garg-style label.

Local checkpoint files may be present under `checkpoints/` for audit and
re-evaluation. They are large artifacts; use Git LFS or release assets if you
want to publish them on GitHub. Use the scripts in `reproduce/` to train,
evaluate, assemble, and plot this experiment.
