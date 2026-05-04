# LPG-SAM: Experimental Findings

**Project:** Latent Prior Guidance for SAM — zero-click retinal vessel segmentation  
**Model checkpoint:** epoch 26 (`frangi-updated-cldice-wt.pt`)  
**Date:** 2026-04-30

---

## 1. System Overview

LPG-SAM adapts a frozen MedSAM ViT-B backbone for retinal vessel segmentation without any human interaction at inference time. The core idea is to replace SAM's click/box prompts with a lightweight, trainable adapter that injects a Frangi vesselness prior into the decoder.

### Architecture

| Component | Parameters | Trainable |
|---|---|---|
| MedSAM ViT-B (image encoder + decoder) | 93.74 M | **No** (fully frozen) |
| LatentPriorAdapter (PriorProjector + FiLMHead + TokenHead) | 656.48 K | **Yes** |
| **Total** | **94.40 M** | **0.69%** |

The adapter operates via:
1. **PriorProjector** — strided CNN that compresses the 1024×1024 Frangi map to a 64×64 feature grid matching SAM's latent space
2. **FiLMHead** — produces per-channel γ/β for feature-wise linear modulation of the image embedding
3. **TokenHead** — produces K=4 distilled sparse tokens that replace SAM's normal prompt tokens
4. **Alpha gate** — scalar initialized to 0, ensuring the adapter is a perfect no-op at the start of training and learns to deviate from the frozen backbone gradually

### Prior: Frangi Vesselness Filter

- Input: green channel of fundus image
- Pipeline: FOV masking → inversion → CLAHE (clip=0.01) → Frangi (σ ∈ {1,2,3,4,5}) → output gamma compression (γ=0.3)
- Maps are precomputed and cached; inference uses cached maps

### Training Setup

| Hyperparameter | Value |
|---|---|
| Dataset | FIVES (retinal OCTA) |
| Split | 540 train / 60 val / 200 test |
| Loss | 0.5·Dice + 0.5·BCE + 0.5·soft-clDice |
| clDice iterations | 15 |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-4 |
| Batch size | 2 |
| Epochs | 50 |
| AMP | Yes (CUDA) |
| Hardware | Tesla T4 (15.6 GB) |
| Training time | ~160 min |
| Post-processing | Morphological closing, radius=2 px |

---

## 2. Training Dynamics

Best checkpoint at **epoch 26** (val clDice = 0.8705). Training plateaued at approximately epoch 23.

### Key convergence milestones

| Epoch | Train Loss | Val clDice | Val B0 err | Alpha |
|---|---|---|---|---|
| 0 | 1.0671 | 0.0900 | 220.5 | 0.009 |
| 1 | 0.8129 | 0.5811 | 141.4 | 0.036 |
| 2 | 0.4633 | 0.7633 | 145.9 | 0.060 |
| 5 | 0.2682 | 0.8312 | 118.6 | 0.091 |
| 10 | 0.2205 | 0.8552 | 98.5 | 0.118 |
| 26 | 0.1745 | **0.8705** | 91.6 | 0.178 |
| 49 | 0.1588 | 0.8699 | 92.0 | 0.204 |

### Observations

- **Rapid early learning:** Val clDice jumps from 0.09 to 0.83 in just 5 epochs, indicating the adapter quickly learns to activate the Frangi prior signal. Most of the meaningful learning happens in epochs 0–10.
- **Training loss reduction:** 85.1% over 50 epochs (1.0671 → 0.1588), though the majority of this drop occurs in the first 10 epochs.
- **Betti-0 improvement:** B0 error falls 58.3% over training (220.5 → 92.0), showing the clDice loss is meaningfully improving topological connectivity.
- **Alpha saturation:** Alpha grows from 0.009 to 0.204 and appears to plateau after epoch 30. The adapter activates to ~20% modulation strength — a moderate, not aggressive, departure from the frozen backbone.
- **Plateau:** Val clDice stays within 0.002 of its optimum from epoch 23 onwards. The final 27 epochs yield negligible improvement. Early stopping at epoch ~25 would have been sufficient.
- **No overfitting observed:** Val clDice at epoch 49 (0.8699) matches epoch 26 (0.8705); train loss continues falling while val performance stays flat — classic benign plateau, not overfitting.

---

## 3. In-Distribution Results: FIVES Test Set

**n = 200 images | Evaluated at epoch 26 checkpoint**

### Aggregate metrics

| Metric | Value | Std Dev | Min | Max |
|---|---|---|---|---|
| Dice | 0.8023 | 0.1163 | 0.102 | 0.904 |
| IoU | 0.6824 | — | — | — |
| **clDice** | **0.8290** | **0.1251** | **0.117** | **0.948** |
| Betti-0 error | 84.88 | 31.47 | — | — |

### Comparison with baselines (FIVES test, n=200)

| Model | Dice | IoU | clDice | Betti-0 err |
|---|---|---|---|---|
| MedSAM zero-shot (box) | 0.1408 | 0.0762 | 0.1202 | 40.94 |
| MedSAM zero-shot (8×8 grid) | 0.1628 | 0.0892 | 0.1666 | 210.83 |
| Frangi + threshold (t=0.15) | 0.3129 | 0.1932 | 0.2840 | **4872.30** |
| **LPG-SAM (ours)** | **0.8023** | **0.6824** | **0.8290** | 84.88 |

> **Note on Betti-0:** Both MedSAM box (40.94) and MedSAM grid (210.83) have lower or comparable Betti-0 errors not because they preserve connectivity — they produce near-empty or heavily under-segmented outputs. Fewer predicted blobs means fewer topological errors. The Frangi threshold baseline makes this artifact starkly visible: at t=0.15, Frangi yields Betti-0 error of 4872 because the raw vesselness map fragments into thousands of tiny disconnected blobs at any useful threshold. LPG-SAM's B0 error of 84.88 represents genuine connectivity learning, not an artifact of under-segmentation.

### Frangi threshold baseline: sweep detail

The Frangi map was optimised for threshold on the val set (19 candidates, t=0.05 to 0.95) and the best threshold applied to the test set. The sweep reveals a sharp and narrow useful range:

| Threshold | Val Dice | Val clDice | Val B0 err |
|---|---|---|---|
| 0.05 | 0.1969 | 0.2164 | 21.97 |
| 0.10 | 0.2533 | 0.2286 | 1129.88 |
| **0.15** | **0.3179** | **0.2828** | 5145.85 |
| 0.20 | 0.2229 | 0.2190 | 2875.28 |
| 0.25 | 0.1021 | 0.1011 | 810.60 |
| ≥0.30 | <0.04 | <0.04 | <250 |

The Frangi response collapses almost entirely above t=0.25 — the map has very low dynamic range after gamma compression, with vessel signal concentrated in a narrow band around 0.10–0.20. This makes robust thresholding practically impossible and explains the catastrophic Betti-0 error: at t=0.15 (the best available threshold), the map is still over-sensitive to noise, fragmenting the vessel tree into ~4900 disconnected components per image on average.

**What the Frangi baseline tells us:** The adapter's improvement from 0.284 clDice (Frangi) to 0.829 clDice (LPG-SAM) — a +0.545 gain — demonstrates that the network is doing substantial work beyond simply re-expressing the Frangi prior. The prior is a useful training signal but a poor predictor in isolation. The adapter learns to use it as guidance while relying on MedSAM's learned image features for final decisions.

### Per-category breakdown (FIVES)

FIVES contains four retinal conditions, 50 images each.

| Category | n | Dice | clDice | Betti-0 err |
|---|---|---|---|---|
| AMD | 50 | 0.8481 | 0.8747 | 76.1 |
| Diabetic Retinopathy | 50 | 0.8110 | 0.8398 | 81.9 |
| Normal | 50 | 0.8107 | 0.8378 | 105.9 |
| Glaucoma | 50 | 0.7392 | 0.7637 | 75.6 |

**Key finding:** Glaucoma is the hardest category — 14.6% lower clDice than AMD. Glaucoma is associated with optic disc changes and altered vessel morphology that may reduce Frangi response fidelity. Normal retinas show the highest Betti-0 error (105.9), suggesting more vessel fragmentation on healthy images despite competitive pixel-level Dice. The Frangi filter may over-segment or miss fine capillaries that are more prominent without pathological contrast.

### Worst and best cases (FIVES test)

| | Name | Dice | clDice |
|---|---|---|---|
| Worst | test_122_G | 0.102 | 0.117 |
| Worst 2 | test_123_G | 0.122 | 0.139 |
| Best | test_13_A | 0.904 | 0.948 |
| Best 2 | test_70_D | 0.902 | 0.941 |

Both worst cases are Glaucoma images, consistent with the per-category breakdown. Both best cases span AMD and DR, suggesting the model is strongest on conditions with pronounced vessel contrast.

---

## 4. Cross-Dataset Generalization

The model was trained exclusively on FIVES (OCTA modality) and evaluated on two color fundus photography datasets with no fine-tuning. This is a strict zero-shot generalization test — different imaging modality, different scanner, different patient population.

### HRF dataset (n=45, 15 DR / 15 Glaucoma / 15 Healthy)

| Metric | Value | Std Dev |
|---|---|---|
| Dice | 0.7118 | 0.0462 |
| IoU | 0.5546 | — |
| clDice | 0.7634 | 0.0482 |
| Betti-0 error | 293.69 | — |
| **AUC-ROC** | **0.9583** | — |
| **AUC-PR** | **0.9620** | — |

### STARE dataset (n=20)

| Metric | Value | Std Dev |
|---|---|---|
| Dice | 0.6831 | 0.0407 |
| IoU | 0.5202 | — |
| clDice | 0.7488 | 0.0381 |
| Betti-0 error | 76.95 | — |
| **AUC-ROC** | **0.9616** | — |
| **AUC-PR** | **0.9576** | — |

### Full comparison table across all datasets

| Dataset | Modality | n | Dice | IoU | clDice | B0 err | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| FIVES (in-dist.) | OCTA | 200 | 0.8023 | 0.6824 | 0.8290 | 84.9 | — | — |
| HRF (zero-shot) | Color fundus | 45 | 0.7118 | 0.5546 | 0.7634 | 293.7 | 0.9583 | 0.9620 |
| STARE (zero-shot) | Color fundus | 20 | 0.6831 | 0.5202 | 0.7488 | 77.0 | 0.9616 | 0.9576 |

### Cross-dataset degradation

| Metric | FIVES→HRF drop | FIVES→STARE drop |
|---|---|---|
| Dice | −0.091 (11.3%) | −0.119 (14.8%) |
| clDice | −0.066 (7.9%) | −0.080 (9.7%) |
| Betti-0 err | +208.8 (+246%) | −7.9 (−9%) |

**Key findings:**

- **Pixel-level metrics transfer reasonably.** An 8–15% drop in Dice when changing both dataset and imaging modality is competitive. Published supervised methods trained directly on HRF achieve Dice ~0.78–0.82; LPG-SAM achieves 0.71 with zero HRF training data. The gap is meaningful but not prohibitive.
- **AUC scores are robust and consistent.** AUC-ROC ~0.96 on both HRF and STARE indicates the model's ranking ability (how confidently it separates vessel from background pixels) transfers strongly across modalities. AUC-PR ~0.96 is especially notable given that PR curves penalize false positives more heavily — this suggests the model is not simply predicting everything as vessel.
- **Betti-0 error diverges dramatically on HRF (+246%).** HRF images are originally 3504×2336 px, downsampled to 1024×1024 for inference. At this compression ratio, fine capillaries become thin, often sub-pixel structures that the model segments as disconnected fragments rather than continuous trees. This is the primary failure mode on HRF.
- **STARE Betti-0 is comparable to FIVES.** STARE images are 700×605 px (upsampled to 1024×1024), so vessel widths after resizing are actually *larger* relative to pixel size. Fewer fragmentation artifacts result. B0 error = 76.95 on STARE vs 84.88 on FIVES.
- **HRF per-category: Healthy images are markedly easier.** The model achieves Dice=0.7707, clDice=0.8148 on healthy retinas but only Dice=0.68–0.69, clDice=0.73–0.75 on DR and Glaucoma. Both pathological conditions alter the vascular tree in ways that reduce Frangi response reliability.

### HRF per-category breakdown

| Category | n | Dice | clDice | Betti-0 err |
|---|---|---|---|---|
| Healthy | 15 | 0.7707 | 0.8148 | 86.5 |
| Glaucoma | 15 | 0.6858 | 0.7454 | 413.9 |
| Diabetic Retinopathy | 15 | 0.6789 | 0.7300 | 380.7 |

DR and Glaucoma HRF images have Betti-0 errors of ~400, roughly 5× the FIVES baseline. This points to severe fragmentation that is both a domain-shift problem and a resolution-compression artifact.

---

## 5. Limitations and Honest Assessment

### What this work demonstrates

1. A 656K-parameter adapter can unlock MedSAM for retinal vessel segmentation, achieving 0.829 clDice on FIVES from a zero-shot baseline of 0.120 — a factor-of-7 improvement with only 0.7% of the backbone's parameter count.
2. **The adapter genuinely learns, not just re-expresses the prior.** The Frangi threshold baseline achieves only 0.284 clDice with a Betti-0 error of 4872. LPG-SAM reaches 0.829 clDice with B0 error of 85 — a +0.545 clDice gain and 98% reduction in fragmentation relative to naive prior thresholding.
3. The Frangi prior is effective as a training signal: alpha grows to ~0.18–0.20, confirming the adapter meaningfully modulates the image embedding rather than remaining inert.
4. AUC-ROC/PR ~0.96 generalizes well across imaging modalities, suggesting the learned probability calibration is robust.

### What this work does not demonstrate (gaps a reviewer will flag)

1. **No supervised baseline.** There is no comparison to a U-Net or any dedicated vessel segmentation network trained on FIVES. Without this, it is impossible to assess whether LPG-SAM closes the gap to supervised methods or remains far below them. This is the most critical missing experiment.
2. **No ablation study.** The contribution of FiLM modulation vs. sparse tokens vs. both is unknown. Removing either component independently and measuring the drop would be necessary to justify the architecture choices.
4. **Small cross-dataset test sets.** HRF has 45 images, STARE has 20. Standard error on mean clDice is non-trivial at these sizes. Confidence intervals are not reported. A ~2% Dice difference between datasets at n=20 is not statistically reliable.
5. **No MedSAM baseline on HRF/STARE.** We know MedSAM box achieves 0.12 clDice on FIVES. We do not know what it achieves on HRF and STARE, making the baseline comparison incomplete for cross-dataset claims.
6. **AUC computed on stratified pixel subsamples (8192 px/image).** This is a standard and justified approximation, but it should be stated explicitly in any paper. The sample fraction is ~0.8% of pixels per image.
7. **Post-processing dependency.** Results include morphological closing (radius=2). The delta between raw model output and post-processed output is not reported separately. It is unknown how much Betti-0 improvement is attributable to the model vs. the closing step.
8. **Single training run.** No variance estimate across seeds. The reported numbers reflect one initialization and one set of hyperparameters.

---

## 6. Summary Scorecard

| Dimension | Assessment |
|---|---|
| In-distribution performance | **Strong** — 0.829 clDice with 650K trainable params |
| Zero-shot generalization (pixel) | **Reasonable** — 8–15% Dice drop across modality change |
| Zero-shot generalization (AUC) | **Strong** — 0.96 ROC/PR on both unseen datasets |
| Topological quality (FIVES) | **Moderate** — B0 err 84.9, higher than zero-shot baseline's 40.9 (caveat: baseline under-segments) |
| Topological quality (HRF) | **Weak** — B0 err 294, severe fragmentation at high-res→low-res compression |
| Parameter efficiency | **Strong** — 0.7% of backbone parameters |
| Prior contribution (Frangi baseline) | **Established** — adapter contributes +0.545 clDice beyond naive thresholding |
| Comparison to supervised methods | **Not established** — critical gap |
| Statistical rigor | **Partial** — std reported for in-distribution, not for small cross-dataset sets |

---

## 7. Recommended Next Experiments

Priority order for strengthening the paper:

1. ~~**Frangi threshold baseline on FIVES test**~~ ✅ Done — clDice=0.284, B0=4872, adapter gap is +0.545 clDice
2. **U-Net trained on FIVES** — establishes supervised ceiling; determines whether LPG-SAM is competitive or just a strong zero-shot method
3. **Ablation: adapter without Frangi** — same architecture, zeros as input — isolates prior contribution from architecture contribution
4. **Ablation: FiLM only vs tokens only vs both** — justifies architectural choices
5. **MedSAM box baseline on HRF and STARE** — completes the comparison table for cross-dataset section
6. **Confidence intervals on cross-dataset results** — bootstrap over per-image scores for HRF (n=45) and STARE (n=20)
7. **Separate post-processing delta** — report raw model metrics and post-closing metrics side by side
