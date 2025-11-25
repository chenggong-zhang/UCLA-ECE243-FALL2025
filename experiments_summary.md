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
*   **Model:** 8 Layers, 384 Dim.
*   **Result:** **CER 0.206**.

---

## Experiment 7: Feature Masking (SpecAugment++)
**Date:** Nov 21, 2025
**Settings:**
*   **Augmentation:** Added Feature Masking.
*   **Model:** 8 Layers, 384 Dim.
*   **Result:** **CER 0.2056**.
*   **Analysis:** Negligible gain (-0.0004). The current architecture (CNN-Transformer) has seemingly plateaued around 0.205.

**Next Steps:**
*   To break the 0.20 barrier, we likely need a structural change to mix local/global features better (e.g., Conformer).
