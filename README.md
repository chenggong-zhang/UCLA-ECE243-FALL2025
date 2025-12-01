Brain-to-Text Neural Speech Decoder
===================================

This repo extends the UCLA ECE243 baseline GRU CTC decoder with stronger Transformer options, rotary position embeddings (RoPE), and consistency-regularized CTC (CR-CTC). It also includes SpecAugment-style masking, learnable CNN downsampling, and AdamW support.

Key components
--------------
- Baseline GRU decoder with day-adaptation: `src/neural_decoder/model.py`.
- Transformer decoder with CNN frontend, optional RoPE: `src/neural_decoder/model_transformer.py`.
- CR-CTC (CTC + symmetric KL consistency) training: `src/neural_decoder/neural_decoder_trainer.py` (enabled via `lambda_cr > 0`).
- RoPE implementation: `src/neural_decoder/rope.py`.
- Data preparation notebook (phoneme conversion, z-scoring): `notebooks/formatCompetitionData.ipynb` -> outputs `data/ptDecoder_ctc.pkl`.

Training scripts
----------------
- GRU baseline: `scripts/train_model.py`.
- Transformer + SpecAugment + AdamW + RoPE (example config): `scripts/train_transformer.py` (set paths to your local data/logs).
- Transformer + AdamW without CR-CTC: `scripts/train_transformer_adamw.py`.
- CR-CTC sweep (lambda_cr grid): `scripts/train_cr_ctc.py`.

Important args (all set in script arg dicts)
--------------------------------------------
- Common: `datasetPath`, `outputDir`, `batchSize`, `lrStart`, `lrEnd`, `nUnits` (d_model/hidden size), `nLayers`, `nClasses` (=40), `nInputFeatures` (=256), `dropout`, `whiteNoiseSD`, `constantOffsetSD`, `strideLen`, `kernelLen`, `l2_decay`.
- Transformer-specific: `use_transformer` (bool), `nhead`, `dim_feedforward`, `use_rope` (enable rotary embeddings).
- CR-CTC: `lambda_cr` (set >0 to enable symmetric KL consistency). `scripts/train_cr_ctc.py` can sweep values.
- Optimizer: `optimizer` in args (`adam` or `adamw`).
- Augmentations: `timeMasking`, `featureMasking` (SpecAugment-style), `gaussianSmoothWidth`.

Quickstart examples
-------------------
1) Transformer with RoPE, AdamW (edit paths inside file):
   `python scripts/train_transformer.py`

2) CR-CTC sweep (defaults to lambdas grid; edit paths inside file):
   `python scripts/train_cr_ctc.py --lambdas 0.0 0.05 0.1 0.2`

3) GRU baseline:
   `python scripts/train_model.py`

Notes on improvements vs. baseline
----------------------------------
- Added AdamW optimizer with better defaults for Transformers.
- Added SpecAugment (time/feature masking) and learnable CNN downsampling frontend.
- Added RoPE to Transformer attention (apply to Q/K per head).
- Added CR-CTC loss (CTC on two views + symmetric KL with masking) to improve consistency; tunable via `lambda_cr`.
- Fixed training loop to iterate over full dataloader rather than repeating the first batch.
