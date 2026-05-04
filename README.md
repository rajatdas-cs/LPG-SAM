# LPG-SAM: Latent Prior Guidance for SAM

**Zero-click retinal vessel segmentation via a 656K-parameter adapter over frozen MedSAM.**

> Course project — CS 736: Medical Image Computing, IIT Bombay  
> Sagnik Nandi (23B0905) · Rajat Das (23B0917)

---

## Overview

Segment Anything Model (SAM) is a powerful foundation model for image segmentation, but it requires human interaction (clicks or bounding boxes) at inference time. In clinical settings — particularly retinal imaging — this requirement is impractical.

LPG-SAM replaces those human prompts with a **Latent Prior Guidance adapter** trained to inject a Frangi vesselness prior directly into SAM's latent space. The result is a fully automatic, zero-click segmentation system that trains only **656K parameters** (0.69% of the 93.74M-parameter frozen backbone).

```
Frangi vesselness map  ──►  LatentPriorAdapter  ──►  FiLM modulation
                                                  └──►  Prior tokens (×4)
                                                             │
Fundus image  ──►  Frozen MedSAM encoder  ──►  z_image  ─────┘
                                                             │
                                              Frozen decoder ▼
                                                   Vessel mask
```

---

## Architecture

The adapter has four components, all trained end-to-end while every MedSAM weight stays frozen:

| Component | Role | Params |
|---|---|---|
| **PriorProjector** | Strided CNN: Frangi (1,1024,1024) → features (256,64,64) | ~598K |
| **FiLMHead** | Two 1×1 convs → γ, β for feature-wise linear modulation | ~33K |
| **TokenHead** | Quadrant pooling + MLP → 4 sparse prior tokens | ~25K |
| **α gate** | Scalar gate, init=0 (identity at start of training) | 1 |
| MedSAM ViT-B | Image encoder + mask decoder | 93.74M (**frozen**) |

**FiLM modulation:** `z_mod = z + α·(γ⊙z + β)`

The α gate initialized to zero guarantees the adapter is a perfect identity at epoch 0 — training begins from the frozen MedSAM zero-shot baseline and learns to deviate from it progressively.

**Frangi prior pipeline:**  
Green channel → FOV masking → inversion → CLAHE → multi-scale Frangi (σ∈{1–5}) → gamma compression (γ=0.3) → cached as `.npy`

---

## Results

### FIVES In-Distribution (n=200 test images)

| Model | Dice | IoU | clDice | Betti-0 err |
|---|---|---|---|---|
| MedSAM zero-shot (box prompt) | 0.141 | 0.076 | 0.120 | 40.9 |
| MedSAM zero-shot (8×8 grid) | 0.163 | 0.089 | 0.167 | 210.8 |
| Frangi threshold (best t=0.15) | 0.313 | 0.193 | 0.284 | 4872.3 |
| **LPG-SAM (ours)** | **0.802** | **0.682** | **0.829** | **84.9** |

### Cross-Dataset Zero-Shot Generalization (no fine-tuning)

| Dataset | Modality | n | Dice | IoU | clDice | B0 err | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|---|---|
| FIVES (in-distribution) | OCTA | 200 | 0.802 | 0.682 | 0.829 | 84.9 | — | — |
| HRF (zero-shot) | Color fundus | 45 | 0.712 | 0.555 | 0.763 | 293.7 | **0.958** | **0.962** |
| STARE (zero-shot) | Color fundus | 20 | 0.683 | 0.520 | 0.749 | 77.0 | **0.962** | **0.958** |

### Per-Category Breakdown (FIVES)

| Category | Dice | clDice | Betti-0 err |
|---|---|---|---|
| AMD | 0.848 | 0.875 | 76.1 |
| Diabetic Retinopathy | 0.811 | 0.840 | 81.9 |
| Normal | 0.811 | 0.838 | 105.9 |
| Glaucoma | 0.739 | 0.764 | 75.6 |

---

## Repository Structure

```
lpg-sam/
├── src/
│   ├── adapter.py        # LatentPriorAdapter (PriorProjector, FiLMHead, TokenHead, α)
│   ├── sam_wrapper.py    # FrozenSAM — frozen backbone wrapper with decode loop fix
│   ├── dataset.py        # FundusVesselDataset — joint augmentation + MedSAM normalization
│   ├── frangi.py         # Frangi vesselness pipeline (CPU preprocessing)
│   ├── losses.py         # Dice + BCE + differentiable soft clDice (Shit et al. 2021)
│   ├── metrics.py        # MetricAccumulator (Dice, IoU, clDice, Betti-0, AUC)
│   └── utils.py          # Device, seeds, parameter counting
├── scripts/
│   ├── train.py          # Training loop (AdamW, cosine LR, AMP, checkpointing)
│   ├── eval.py           # Cross-dataset evaluation (HRF / STARE)
│   ├── eval_baseline.py  # MedSAM zero-shot baseline
│   ├── cache_frangi.py   # Precompute Frangi maps → .npy cache (run once)
│   ├── prepare_data.py   # Resize + train/val/test split to 1024×1024
│   ├── frangi_sweep.py   # Hyperparameter sweep over Frangi thresholds/sigmas
│   └── smoke_test.py     # End-to-end sanity check
├── configs/
│   └── config.yaml       # All hyperparameters (paths, model, training, loss, eval)
├── report/
│   ├── report.pdf        # Full technical report
│   └── slides.pdf        # Presentation slides
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes PyTorch, torchvision, scikit-image, and the SAM package from Meta:

```
torch>=2.0
torchvision
numpy
scikit-image
Pillow
pyyaml
tqdm
matplotlib
git+https://github.com/facebookresearch/segment-anything.git
```

### 2. Download MedSAM checkpoint

Download `medsam_vit_b.pth` from the [MedSAM repository](https://github.com/bowang-lab/MedSAM) and place it at:

```
checkpoints/medsam_vit_b.pth
```

### 3. Prepare data

Download the [FIVES dataset](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169) and run:

```bash
# Resize images to 1024×1024 and create train/val/test splits
python -m scripts.prepare_data --data_root data/raw/FIVES --out_root data/processed/FIVES

# Precompute Frangi vesselness maps (run once — takes ~5 min on CPU)
python -m scripts.cache_frangi --data_root data/processed/FIVES
```

Expected directory layout after preparation:

```
data/processed/FIVES/
├── images_1024/     # RGB PNGs at 1024×1024
├── masks_1024/      # Binary mask PNGs at 1024×1024
├── frangi_cache/    # Precomputed Frangi maps as .npy
├── train.txt        # Image stems for training split
├── val.txt
└── test.txt
```

---

## Training

```bash
python -m scripts.train --config configs/config.yaml
```

Key config options in `configs/config.yaml`:

```yaml
train:
  batch_size: 2       # Safe for 16 GB GPU with AMP at 1024×1024
  epochs: 50
  lr: 1.0e-4
  amp: true           # Mixed precision — recommended for CUDA

loss:
  lambda_dice: 0.5
  lambda_bce: 0.5
  lambda_cldice: 0.5
  cldice_iter: 15
```

Training logs are saved to `runs/<timestamp>/`. The best checkpoint (by val clDice) is saved to `checkpoints/best.pt`.

**What to watch during training:**
- `alpha` should climb away from 0. If it stays flat, check that gradients reach the FiLM path.
- Val clDice at epoch 0 (~0.09) is the zero-shot baseline. Meaningful learning starts in epochs 1–5.
- The model typically plateaus around epoch 23–26.

---

## Evaluation

```bash
# FIVES test set
python -m scripts.eval --config configs/config.yaml --checkpoint checkpoints/best.pt --dataset FIVES

# Zero-shot generalization on HRF / STARE
python -m scripts.eval --config configs/config.yaml --checkpoint checkpoints/best.pt --dataset HRF
python -m scripts.eval --config configs/config.yaml --checkpoint checkpoints/best.pt --dataset STARE

# MedSAM zero-shot baseline (no adapter)
python -m scripts.eval_baseline --config configs/config.yaml --prompt box
```

---

## Smoke Test

Verify the full forward pass runs correctly before training:

```bash
python -m scripts.smoke_test
```

This checks: FrozenSAM loads, `encode_image` runs under `no_grad`, `decode_masks` loops correctly, adapter produces the right shapes, and the α=0 identity invariant holds (`|z_mod - z| < 1e-6`).

---

## Loss Function

```
L = 0.5 · Dice  +  0.5 · BCE  +  0.5 · soft-clDice
```

- **Dice** — handles class imbalance (vessels are <10% of pixels; BCE alone biases toward "all background")
- **BCE** — per-pixel gradient signal everywhere, including empty regions
- **Soft clDice** — differentiable centerline Dice (Shit et al., CVPR 2021) that rewards topological connectivity, not just area overlap; implemented via iterative soft erosion (`-maxpool(-x)`)

All losses operate on raw logits; sigmoid is applied internally.

---

## Design Decisions

**Why freeze MedSAM?**  
MedSAM's ViT-B encoder costs ~12 GB of activation memory at 1024×1024 when backprop is enabled. Freezing it (and wrapping `encode_image` in `torch.no_grad`) eliminates this entirely, enabling batch size ≥2 on a 16 GB GPU.

**Why the per-image decode loop?**  
SAM's mask decoder is a per-image API that internally `repeat_interleave`s prompts for one image. Passing a batch of B images causes an internal B² tensor expansion that breaks shapes. The fix loops over the batch one image at a time.

**Why GroupNorm instead of BatchNorm in PriorProjector?**  
BatchNorm running statistics are noisy at batch size 2–4. GroupNorm(8) is stable at any batch size.

**Why strided convolution (not fixed interpolation) in PriorProjector?**  
Learned downsampling lets the network decide which spatial information to preserve, rather than applying a fixed interpolation formula that may discard fine capillary signal.

**Why α gate?**  
If the adapter started with random FiLM modulation, the decoder would see garbage image embeddings from step 0 and the loss would spike. α=0 at init means the adapter starts as a perfect identity; the network learns to activate it gradually.

---

## Citation

If you use this code, please cite the underlying methods:

```bibtex
@article{MedSAM2024,
  title={Segment anything in medical images},
  author={Ma, Jun et al.},
  journal={Nature Communications},
  year={2024}
}

@inproceedings{shit2021cldice,
  title={clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation},
  author={Shit, Suprosanna et al.},
  booktitle={CVPR},
  year={2021}
}
```

---

## License

This project is released for academic use only. MedSAM and SAM are subject to their respective licenses from Meta and the MedSAM authors.
