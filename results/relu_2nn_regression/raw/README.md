# Raw Two-Layer ReLU Regression Outputs

Expected files after running the reproduction scripts:

- `masked_pair_encoder.csv`: neural curve from the masked checkpoint.
- `gpt2_causal.csv`: neural curve from the GPT-2 causal checkpoint.
- `relu_2nn_baselines.csv`: OLS, 3NN, Averaging, and per-prompt two-layer
  ReLU network trained by Adam. The assembled figure labels this
  gradient-trained reference as `2-layer NN, GD` to match the paper convention.

Run `reproduce/assemble_relu_2nn_curve.ps1` to produce
`results/relu_2nn_regression/curves/relu_2nn_icl.csv`.
