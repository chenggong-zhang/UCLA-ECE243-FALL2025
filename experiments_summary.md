# Experiment Summary Log

## Experiment 1: Initial Transformer Implementation
**Date:** Nov 21, 2025
**Settings:**
*   **Model:** `TransformerDecoder` (Vanilla, Post-Norm).
*   **Result:** Did not converge.

---

## Experiment 2: Reduced Learning Rate
**Date:** Nov 21, 2025
**Settings:**
*   **Model:** `TransformerDecoder` (Post-Norm) + Scaled Embeddings.
*   **Result:** CER `~0.51`.

---

## Experiment 3: Pre-Norm & AdamW
**Date:** Nov 21, 2025
**Settings:**
*   **Model:** `TransformerDecoder` (Pre-Norm).
*   **Result:** CER `0.316`.

---

## Experiment 4: SpecAugment (Time Masking)
**Date:** Nov 21, 2025
**Settings:**
*   **Model:** 6 Layers.
*   **Result:** CER `0.283`.

---

## Experiment 5: Learnable CNN Front-End
**Date:** Nov 21, 2025
**Settings:**
*   **Architecture:** 2-layer CNN (Stride 2 each) front-end.
*   **Result:** **CER 0.2195**.

---

## Experiment 6: Scaling Up (Deep & Wide)
**Date:** Nov 21, 2025
**Settings:**
*   **Model:** **8 Layers**, **384 Dim** (d_model).
*   **Dropout:** Increased to `0.3`.
*   **Training:** 12,000 batches.

**Results:**
*   **Final CER**: `0.206`.
*   **Analysis**: Almost reached the 0.20 milestone. The loss was still slowly decreasing. The deeper model helped.

**Gap to SOTA:**
*   SOTA (Conformer) likely benefits from "interleaved" convolutions and better augmentation (Feature Masking).
*   We are missing **Feature Masking** (masking specific electrode channels), which is standard in SpecAugment and BCI to handle channel noise.

---

## Experiment 7: Feature Masking (SpecAugment++)
**Date:** Nov 21, 2025
**Planned Changes:**
1.  **Augmentation**: Add **Feature Masking** (masking random channels/frequencies) alongside Time Masking.
2.  **Goal**: Force the model to not rely on any single electrode, improving robustness and generalization.
3.  **Model**: Keep Exp 6 settings (Large CNN-Transformer).

**Hypothesis:**
*   Adding Feature Masking will provide the final regularization push needed to break `0.20` CER.
