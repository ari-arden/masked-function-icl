# Third-Party Attribution

This project builds on the synthetic in-context learning code structure from
Garg, Tsipras, Liang, and Valiant, "What Can Transformers Learn In-Context? A
Case Study of Simple Function Classes" (2022).

The repository keeps the original MIT license and extends the code with:

- masked encoder and masked pair encoder model families;
- masked objective utilities for query-only, randomized-query, and dense
  leave-one-out label masking;
- linear-regression and decision-tree masked ICL training/evaluation scripts;
- Causal GPT-2 baseline reproduction for synthetic function classes;
- decision-tree baselines including greedy-tree, nearest-neighbor, and
  sign-preprocessed XGBoost comparisons;
- paper-ready result tables and figures.

The causal GPT-2 baseline is trained from scratch on synthetic data. It does not
use pretrained natural-language GPT-2 weights.
