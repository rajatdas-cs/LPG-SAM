# LPG-SAM: Latent Prior Guidance for SAM

**Zero-click topological guidance for retinal vessel segmentation via Frangi prior injection into a frozen SAM decoder.**

A research project exploring whether a deterministic mathematical prior (Frangi vesselness) can replace human prompts (clicks/boxes) when adapting foundation segmentation models to thin tubular structures.

---

## The problem

Foundation segmentation models like the Segment Anything Model (SAM) and its medical variant MedSAM achieve strong performance on natural images and many medical modalities, but they share two limitations on retinal vessel segmentation:

1. **They need prompts.** SAM is designed around human interaction — a user clicks a point or draws a box, and the model segments whatever is at that location. For dense, branching, multi-scale structures like retinal vessels, you can't realistically click every capillary. Existing automation hacks (grid prompts, full-image boxes) produce mediocre results.

2. **They confuse blobs with vessels.** When a fundus image contains pathological lesions (hemorrhages, exudates, microaneurysms), the model often segments the blob-like lesion as if it were a vascular structure. SAM's training distribution is dominated by blob-shaped objects, so its decoder has a strong inductive bias toward enclosed regions rather than thin tubular continuity.

The clinical cost: any downstream analysis (tortuosity measurement, vessel density estimation, branching pattern analysis) is corrupted by these failure modes.

## The hypothesis

A deterministic prior with the right inductive bias — specifically, the **Frangi vesselness filter**, which uses Hessian eigenvalue analysis to detect tubular structures while suppressing blob-like ones — can replace the human prompt entirely. By injecting this prior directly into SAM's frozen decoder via a small trainable adapter, we get:

- **Zero-click operation**: no user interaction needed.
- **Topological bias**: Frangi mathematically prefers tubes over blobs, so hemorrhages get suppressed.
- **Foundation model preservation**: SAM's image encoder and mask decoder stay frozen, so the experiment cleanly measures the value of the prior, not the value of fine-tuning.

## What we built

LPG-SAM (Latent Prior Guidance for SAM) is a small trainable adapter module that sits between MedSAM's frozen image encoder and frozen mask decoder. It takes the Frangi response of an input fundus image and injects it into SAM's two prompt slots:

- **Dense pathway**: FiLM-style modulation of SAM's image embedding
- **Sparse pathway**: K=4 distilled "prior tokens" replacing human-click tokens

A learnable scalar gate, initialized to zero, ensures that at training step 0 the model exactly reproduces zero-shot SAM. As training proceeds, the gate opens and the prior steers the decoder.

The entire trainable footprint is ~800K parameters (less than 1% of MedSAM). The image encoder (89M params) and mask decoder (4M params) are completely frozen — gradients flow through them during backpropagation, but their weights never update.

---

## Architecture

### Forward pass

```
Data prep time (CPU, once per dataset)
──────────────────────────────────────
image_raw                                  mask_raw
    │                                          │
    ▼  resize 1024x1024                        ▼  resize 1024x1024 (NN)
image_1024 (3, 1024, 1024)                 mask_1024 (1, 1024, 1024)
    │
    ▼  apply FOV mask
    ▼  extract green channel
    ▼  invert (vessels -> bright)
    ▼  re-mask
    ▼  CLAHE (local contrast)
    ▼  re-mask
    ▼  multi-scale Frangi (sigmas=[1,2,3,4,5])
    ▼  re-mask
frangi_full (1, 1024, 1024)


Training time (GPU, per batch)
──────────────────────────────
image_1024                                  frangi_full
    │                                           │
    ▼  [SAM encoder, FROZEN, no_grad]           │
z_image (B, 256, 64, 64)                        │
                                                │
                                                ▼  [PriorProjector, trainable]
                                            prior_features (B, 256, 64, 64)
                                                │
                                                ├──► γ (B, 256, 64, 64)   FiLM
                                                ├──► β (B, 256, 64, 64)   FiLM
                                                └──► sparse_tokens (B, 4, 256)

z_modulated = z_image + α · (γ ⊙ z_image + β)         [α=0 at init]
                                                │
                                                ▼  [SAM decoder, FROZEN, gradients flow]
mask_logits (B, 1, 256, 256)
                                                │
                                                ▼  bilinear upsample
mask_pred (B, 1, 1024, 1024)
                                                │
                                                ▼  vs ground truth
Loss = λ_dice · Dice + λ_bce · BCE + λ_cldice · (1 − soft_clDice)
                                                │
                                                ▼  backward
Updates only: PriorProjector, FiLM γ/β heads, TokenHead, α
```

### The PriorProjector

A strided CNN that converts the Frangi response into SAM-compatible features in a single pass:

```
Input:   (B, 1, 1024, 1024)

Block 1: Conv(1→16,    stride=2) + GroupNorm + GELU   →  (B, 16,  512, 512)
Block 2: Conv(16→32,   stride=2) + GroupNorm + GELU   →  (B, 32,  256, 256)
Block 3: Conv(32→64,   stride=2) + GroupNorm + GELU   →  (B, 64,  128, 128)
Block 4: Conv(64→128,  stride=2) + GroupNorm + GELU   →  (B, 128, 64,  64)
Block 5: Conv(128→256, stride=1) + GroupNorm + GELU   →  (B, 256, 64,  64)

Output:  (B, 256, 64, 64)
```

This combines spatial downsampling (1024 → 64, 16×) with channel expansion (1 → 256) in one network. Each strided block doubles channels and halves spatial dimensions, like a standard ResNet stem. The downsampling is **learned** rather than fixed (bilinear/maxpool), so the network can preserve sparse vessel signals that a fixed downsample would average into background.

Why GroupNorm instead of BatchNorm: training uses batch size 2–4, and BatchNorm with small batches is unstable.

### The FiLM heads

Two parallel 1×1 convs produce per-(channel, location) modulation parameters γ and β from the prior features. SAM's image embedding is then modulated as:

```
z_modulated = z_image + α · (γ ⊙ z_image + β)
```

where α is a single learnable scalar initialized to 0. At α=0, `z_modulated == z_image` exactly, so the model is bit-identical to zero-shot SAM at the start of training. As α grows from zero, the prior begins to steer the decoder.

This is the most important design property: **the adapter cannot catastrophically destroy SAM's pretrained features**, because training starts from the zero-shot baseline and the optimizer decides how far to move.

The FiLM heads are also initialized with very small weights (`std=1e-3`) so that even if α weren't zero, the modulation would start near identity. Belt and suspenders.

### The token head

The same prior features are pooled and projected into K=4 distilled tokens of dimension 256:

```
prior_features (B, 256, 64, 64)
    │
    ▼  AdaptiveAvgPool2d to (B, 256, 2, 2)
    ▼  reshape to (B, 4, 256)
    ▼  per-token MLP (256 → 256 → 256)
sparse_tokens (B, 4, 256)
```

These tokens replace the click/box tokens that SAM normally consumes. The decoder cross-attends to them as if they were human prompts. The 2×2 spatial pooling gives each token a weak quadrant-level spatial interpretation (top-left token sees the upper-left quadrant of the image, etc.).

K=4 is in-distribution for SAM, which was trained with sparse prompts of 1–4 tokens. Ablations: K ∈ {1, 4, 8}.

### Why both pathways

You might ask: if the dense FiLM pathway already injects spatial information, why also use sparse tokens? Three reasons:

1. **They operate at different layers in the decoder.** The dense embedding gets added to the image embedding before the decoder's transformer blocks. Sparse tokens get concatenated into the query stream and participate in cross-attention. Different injection points, different computational influence.

2. **Dense is "where," sparse is "what."** The dense pathway tells the decoder where the prior thinks vessels are. The sparse pathway gives the decoder a learned "vessel query" embedding to attend with. Together they form a complete prompt: a location prior and a content prior.

3. **It mirrors how SAM is normally prompted.** SAM normally receives either a click (sparse) on an unprompted image (dense=0), or a mask prompt (dense) with no click (sparse=0). LPG-SAM fills both slots, giving SAM the richest possible prompt — and crucially, doing nothing the architecture wasn't designed to handle. We are not bending SAM; we are feeding its existing inputs from a different source.

### Parameter budget

| Component                          | Params  | Trainable? |
|------------------------------------|---------|------------|
| SAM image encoder (ViT-B)          | 89 M    | ❌         |
| SAM prompt encoder (bypassed)      | 6 K     | ❌         |
| SAM mask decoder                   | 4 M     | ❌         |
| **PriorProjector**                 | ~250 K  | ✅         |
| **FiLM heads (γ, β)**              | ~130 K  | ✅         |
| **Token head**                     | ~270 K  | ✅         |
| **α (learnable scalar)**           | 1       | ✅         |
| **Total trainable**                | **~650 K** | |

We train less than 1% of the total parameter count.

---

## The Frangi prior

The Frangi vesselness filter (Frangi et al. 1998) is a classical multi-scale tubular structure detector. At each pixel, it computes the eigenvalues λ₁, λ₂ of the local Hessian matrix and uses their ratio to distinguish:

- **Tubular structures** (one large eigenvalue, one small): high response
- **Blob-like structures** (two large eigenvalues): low response, controlled by the `beta` parameter
- **Background noise** (two small eigenvalues): suppressed by the `gamma` parameter

This is exactly the inductive bias we want for retinal vessels: it loves vessels and ignores hemorrhages. Multi-scale analysis (sigmas = [1, 2, 3, 4, 5] pixels at 1024² resolution) captures both fine capillaries and large arcade vessels.

### Preprocessing pipeline

The Frangi response is precomputed once per image at data prep time and cached as `.npy` files on disk. It is **not** recomputed during training. The full pipeline:

1. **Resize** the raw fundus image to 1024×1024
2. **Compute FOV mask** by intensity threshold on the mean luminance
3. **Apply FOV mask to the raw RGB** (critical ordering — see below)
4. **Extract the green channel** (where retinal vessels have the highest contrast against the background, due to hemoglobin's strong green absorption)
5. **Invert** so vessels become bright on a dark background
6. **Re-apply FOV mask** to keep the exterior at zero
7. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to fix the illumination gradient between optic disc and periphery
8. **Re-apply FOV mask** to suppress any halo CLAHE introduced
9. **Multi-scale Frangi** with sigmas = [1, 2, 3, 4, 5]
10. **Final FOV mask** to kill any leakage at the boundary

### Why the FOV mask gets applied four times

This was a real bug discovered during development. The FOV mask must be applied to the **raw RGB image before any other operation**, not just at the end. Here's why:

The original image has black (=0) outside the retina. If we invert before masking, the outside becomes white (=1, maximally bright). CLAHE then sees a huge intensity contrast at the FOV boundary and enhances it into a sharp ridge. Frangi happily detects this ridge as a "vessel" running around the entire retinal disc, which dominates the normalization and squashes the actual vessels into invisibility.

The fix: mask the raw image first, so the outside stays black through inversion and CLAHE. Then re-mask after each step as belt-and-suspenders. The result is a clean Frangi response with no boundary artifact.

### Why the green channel

A fundus image's three channels carry very different signal:

- **Red**: dominated by the reddish-orange choroid background. Vessels are also red, so contrast is poor. Often saturated near the optic disc.
- **Green**: hemoglobin absorbs green light strongly, so vessels appear as the **darkest** structures against a brighter background. Highest vessel contrast of any channel. **This is what every retinal vessel paper has used for 25+ years.**
- **Blue**: very noisy, low SNR because the lens and vitreous absorb blue light. Often nearly black.

Using the green channel exclusively for the Frangi computation is the standard baseline that's hard to beat. The full RGB image still goes into SAM's encoder unchanged.

---

## Training

### Loss

A composite of three terms:

```
L = λ_dice · DiceLoss(pred, gt)
  + λ_bce  · BCEWithLogitsLoss(pred, gt)
  + λ_cldice · (1 − soft_clDice(pred, gt))
```

with default weights `λ_dice = 0.5, λ_bce = 0.5, λ_cldice = 0.3`.

- **BCE** gives a strong per-pixel classification signal but is blind to topology and underestimates the minority (vessel) class.
- **Dice** handles class imbalance via region overlap. Standard for medical segmentation.
- **soft clDice** (Shit et al. CVPR 2021) is a differentiable proxy for centerline Dice. It rewards getting the **skeleton** right, not just the area, which is the topological fidelity we care about. Implemented via iterative soft erosion (min-pool) and soft dilation (max-pool) operations that approximate morphological skeletonization while remaining differentiable.

The composite loss is what makes LPG-SAM care about both pixel accuracy and vessel connectivity. The Frangi prior tells the network *where* tubular structures are; the soft clDice term gives it an explicit gradient toward *producing* tubular outputs.

### Setup

- **Optimizer**: AdamW, lr = 1e-4, weight decay = 1e-4
- **Schedule**: cosine decay with 2 epochs of linear warmup
- **Batch size**: 2–4 (limited by 1024² input on a single 16 GB GPU)
- **Mixed precision (AMP)**: enabled on CUDA, no-op on MPS/CPU
- **Gradient clipping**: norm 1.0
- **Epochs**: 50
- **Augmentation (conservative)**:
  - Horizontal flip (p=0.5)
  - Vertical flip (p=0.5)
  - 90/180/270 degree rotation (p=0.75)
  - **No** color jitter (would corrupt the green channel)
  - **No** elastic deformation (would invalidate alignment)
  - **No** scale/crop (SAM is hardcoded for 1024×1024)

All spatial augmentations are applied **identically** to the image, the Frangi prior, and the GT mask. They're all 90° rotations or flips — exact pixel permutations with no interpolation, so binary masks stay binary.

### What to watch during training

1. **The α gate climbing from zero.** This is the single most important diagnostic. If α stays at exactly 0.000 across multiple epochs, gradients aren't reaching it and something is broken. If it climbs steadily (typically reaches 0.3–0.8 by mid-training), the adapter is learning.

2. **Validation clDice improving over epoch 0.** Epoch 0 with α≈0 is the zero-shot baseline. If clDice doesn't improve from there, the prior isn't carrying useful signal.

3. **Component losses staying balanced.** If one loss term dominates the others by 10×, the lambda weights need retuning.

### Checkpointing

The trainer saves two checkpoints:

- `best.pt` — the epoch with highest validation clDice
- `last.pt` — the most recent epoch

Both contain only the adapter weights (~3 MB), not SAM, since SAM is frozen and identical to the original MedSAM checkpoint.

---

## Evaluation

### Metrics

| Metric              | What it measures              | Why we care                                          |
|---------------------|-------------------------------|------------------------------------------------------|
| **clDice**          | Centerline overlap            | **Headline metric.** Topological fidelity.           |
| **Dice**            | Region overlap                | Sanity check / class-imbalance-aware overlap.        |
| **IoU (Jaccard)**   | Region overlap, stricter      | Reported by convention.                              |
| **Betti-0 error**   | Connected component mismatch  | Catches over-merging and over-fragmentation.         |

clDice (Shit et al. 2021) is the harmonic mean of:

- T_prec = `|skel(pred) ∩ gt|  /  |skel(pred)|`  ("how much of the predicted skeleton lies in GT?")
- T_sens = `|skel(gt)   ∩ pred| /  |skel(gt)|`    ("how much of the GT skeleton lies in pred?")

It's structurally identical to F1/Dice but applied to skeleton-based precision/recall. A model with high Dice but poor connectivity (thick blobs instead of thin connected vessels) gets penalized by clDice in a way standard Dice misses.

We use the **hard** clDice (skimage's `skeletonize`) for evaluation and the **soft** clDice (differentiable) only for training.

### Cross-dataset generalization

The strongest experimental design for this project: train on FIVES, then evaluate on **completely unseen datasets** without any fine-tuning. This makes the comparison to zero-shot SAM genuinely fair — both models see the test datasets for the first time at eval, so any LPG-SAM advantage is attributable to the prior, not to in-distribution training.

The cross-dataset suite:

| Dataset      | Images | Resolution | Notes                                          |
|--------------|--------|------------|------------------------------------------------|
| **FIVES test** | 200    | 2048²      | In-distribution. The "easy" number.            |
| **DRIVE**    | 40     | 565×584    | Classic retinal vessel benchmark.              |
| **STARE**    | 20     | 700×605    | Contains pathology (good for hemorrhage story).|
| **CHASE_DB1**| 28     | 999×960    | Child retinas, different vessel statistics.    |

**Important methodological rule**: cross-dataset eval is run **once at the end**, on locked weights. Looking at DRIVE/STARE/CHASE numbers during development would silently turn them into validation sets and invalidate the generalization claim.

### Final results table (template)

| Method                          | FIVES test | DRIVE | STARE | CHASE | Mean cross-dataset |
|---------------------------------|------------|-------|-------|-------|--------------------|
| Frangi alone (thresholded)      | …          | …     | …     | …     | …                  |
| MedSAM zero-shot (full box)     | …          | …     | …     | …     | …                  |
| **LPG-SAM (ours)**              | …          | …     | …     | …     | …                  |

Each cell reports **Dice / clDice / Betti-0** error. The headline number is mean cross-dataset clDice.

The "Frangi alone" row is the critical control: if a simple threshold on the Frangi response matches LPG-SAM, then SAM is contributing nothing and the project is just an expensive Frangi filter. This row is what proves the adapter is doing real work.

---

## Repository structure

```
lpg-sam/
├── README.md                       # this file
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── config.yaml                 # all hyperparameters in one place
│
├── src/
│   ├── utils.py                    # device helpers, seeding, param counting
│   ├── frangi.py                   # green channel, CLAHE, FOV mask, Frangi
│   ├── dataset.py                  # FundusVesselDataset + JointAugment
│   ├── sam_wrapper.py              # FrozenSAM wrapper around MedSAM
│   ├── adapter.py                  # PriorProjector + FiLM + TokenHead
│   ├── losses.py                   # Dice, BCE, soft-clDice, CompositeLoss
│   └── metrics.py                  # hard Dice, IoU, clDice, Betti-0
│
├── scripts/
│   ├── smoke_test.py               # end-to-end SAM↔adapter handshake test
│   ├── prepare_data.py             # resize raw FIVES → 1024² PNGs + splits
│   ├── cache_frangi.py             # precompute Frangi responses to disk
│   ├── train.py                    # main training loop
│   └── eval.py                     # evaluate checkpoint, write CSV+overlays
│
├── notebooks/
│   ├── frangi_tuning.ipynb         # visual sigma sweep
│   ├── sam_baseline.ipynb          # zero-shot baseline numbers
│   └── ablations.ipynb
│
├── data/                           # gitignored
│   ├── raw/FIVES/
│   └── processed/FIVES/
│       ├── images_1024/
│       ├── masks_1024/
│       ├── frangi_cache/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
│
├── checkpoints/                    # gitignored
│   ├── medsam_vit_b.pth
│   ├── best.pt
│   └── last.pt
│
└── runs/                           # gitignored — training logs + eval outputs
```

---

## Setup and how to run

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`:
```
torch>=2.0
torchvision
numpy
scikit-image
scikit-learn
Pillow
pyyaml
tqdm
matplotlib
git+https://github.com/facebookresearch/segment-anything.git
```

### 2. Smoke tests (no data required, ~1 minute)

```bash
python -m src.adapter      # adapter shapes + α=0 invariant
python -m src.losses       # all loss terms + gradient flow
python -m src.metrics      # all metrics on synthetic cases
python -m src.frangi       # full Frangi pipeline on synthetic image
python -m src.dataset      # creates fake dataset on disk, loads it
```

All five should print `[OK] ... is working correctly.` Each takes a few seconds.

### 3. Get the MedSAM checkpoint

Download `medsam_vit_b.pth` (~370 MB) from the [MedSAM repo](https://github.com/bowang-lab/MedSAM) and place it at `checkpoints/medsam_vit_b.pth`.

### 4. SAM ↔ adapter handshake test

```bash
python -m scripts.smoke_test
```

This is the most important pre-training check. It verifies the entire model side: SAM loads, adapter connects to it, gradients flow through the frozen decoder back to the adapter, and the α=0 invariant holds bit-exactly against real SAM output.

### 5. Get FIVES

Download from [figshare](https://figshare.com/articles/dataset/FIVES_A_Fundus_Image_Dataset_for_Artificial_Intelligence_based_Vessel_Segmentation/19688169) and extract to `data/raw/FIVES/` so that:

```
data/raw/FIVES/
├── train/Original/   (600 images)
├── train/Ground truth/
├── test/Original/    (200 images)
└── test/Ground truth/
```

### 6. Prepare data

```bash
python -m scripts.prepare_data \
    --raw-root data/raw/FIVES \
    --out-root data/processed/FIVES \
    --val-fraction 0.1 \
    --seed 42
```

Resizes everything to 1024×1024, writes `train.txt`/`val.txt`/`test.txt`, and namespaces stems with `train_` or `test_` prefixes to prevent split collisions (FIVES restarts numbering between train and test, so without namespacing the same stem appears in both).

### 7. Cache Frangi

```bash
python -m scripts.cache_frangi --data-root data/processed/FIVES
```

Runs the full Frangi pipeline on every image and saves the responses as `.npy` files. Takes ~20 minutes on CPU. Idempotent — re-running skips files that already exist; use `--force` to recompute.

### 8. Sanity-check the cached priors

Before committing GPU time, visually verify the Frangi cache looks right:

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

stem = "train_1_A"
img = np.array(Image.open(f"data/processed/FIVES/images_1024/{stem}.png"))
frangi = np.load(f"data/processed/FIVES/frangi_cache/{stem}.npy")

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].set_title("image")
ax[1].imshow(frangi, cmap="hot")
ax[1].set_title("frangi")
plt.show()
```

Expected: vessels brightly outlined in the Frangi panel, dark background, **no bright ring at the FOV boundary**. If you see a ring, something is wrong with the mask ordering and you should not train.

### 9. Train

```bash
python -m scripts.train --config configs/config.yaml
```

On a Kaggle T4 with batch size 2 and AMP, 50 epochs takes roughly 1.5–2 hours. Watch the `alpha` value in the progress bar — it must climb from zero.

### 10. Evaluate on FIVES test

```bash
python -m scripts.eval \
    --config configs/config.yaml \
    --checkpoint checkpoints/best.pt \
    --dataset-root data/processed/FIVES \
    --split-file data/processed/FIVES/test.txt \
    --tag fives_test
```

Outputs go to `runs/eval_fives_test_<timestamp>/`:
- `metrics.json` — aggregate scores
- `per_image.csv` — sorted worst-clDice-first for failure inspection
- `overlays/` — 5-panel visualizations (image | frangi | gt | pred | TP-FP-FN)

### 11. Cross-dataset eval (do once at the end)

For each of DRIVE, STARE, CHASE_DB1, prepare the data and run eval the same way. **Only run this after all FIVES-based hyperparameter tuning is done.**

---

## Compute requirements

- **Storage**: ~5 GB total (FIVES raw + processed + Frangi cache + checkpoints)
- **GPU**: 16 GB VRAM is comfortable. Tested on Kaggle T4 (16 GB) and P100 (16 GB).
- **Training time**: ~1.5–2 hours for 50 epochs on T4 with batch size 2 + AMP
- **Frangi caching**: ~20 minutes one-time, CPU only
- **Eval**: ~5 minutes per dataset

The single biggest VRAM optimization is wrapping SAM's image encoder forward in `torch.no_grad()`. The encoder has 89M frozen params; storing its activations for backprop would use ~6 GB. Skipping activation storage (since the encoder is frozen) is what makes batch size 2–4 fit on a 16 GB T4. This is implemented in `FrozenSAM.encode_image()` and is the most important line of memory-management code in the project.

## Development workflow

We use the **Mac as IDE, Kaggle as training cluster** model:

- **On the Mac**: write code, run all smoke tests, run the SAM handshake test, optionally do a 2-step training run for final verification. Apple Silicon MPS works for everything except actually training (slow, memory-tight).
- **On Kaggle**: clone the repo, install deps, link the MedSAM checkpoint and FIVES from Kaggle Datasets, run prep + cache + train + eval. ~3 hours of GPU time per full experiment.

The code is device-agnostic — `get_device()` in `utils.py` picks CUDA → MPS → CPU in priority order, and AMP gracefully no-ops on non-CUDA devices. The same code runs everywhere without modification.

## Bugs we hit and fixed during development

These are documented because they're instructive about what to watch for in any frozen-backbone adapter project:

1. **MedSAM checkpoint loaded with CUDA tensors.** `segment_anything`'s loader doesn't pass `map_location`, so it crashes on CPU/MPS machines. Fixed by loading the state dict ourselves with `torch.load(path, map_location="cpu")` and feeding it to the SAM constructor.

2. **FIVES split contamination from filename collisions.** FIVES restarts numbering inside `train/` and `test/`, so `train/Original/1_A.png` and `test/Original/1_A.png` have the same stem. Our processed data folder uses stems as filenames, so test images silently overwrote train images during prep. ~50 train images turned into duplicate test images, which would have inflated reported test metrics. Fixed by namespacing stems with `train_` / `test_` prefixes and adding a `seen_stems` guard that hard-fails on any duplicate.

3. **Frangi pipeline ordering: CLAHE before FOV masking creates a bright boundary ring.** The original image has black outside the retina; if you invert before masking, the outside becomes white; CLAHE then sees a huge intensity contrast at the boundary and enhances it into a sharp ridge; Frangi detects the ridge as a "vessel" and normalization squashes the actual vessels. Fixed by applying the FOV mask to the **raw RGB image first**, then re-applying it after inversion, after CLAHE, and after Frangi (four times total — paranoid but cheap).

The pattern across all three: silent data corruption is far more dangerous than loud crashes. Adding hard assertions and visual sanity checks at every stage caught all three before they reached the training loop.

## Design choices we explicitly considered and rejected

- **Cross-attention between prior and image features (instead of FiLM).** More expressive but harder to initialize as identity. FiLM with α=0 gives us the bit-exact zero-shot reproduction property; cross-attention does not. We can revisit this as a v2 ablation.

- **Training the decoder lightly (instead of fully frozen).** Would probably get higher Dice, but muddies the experimental claim. We want to show that **a deterministic prior alone**, plumbed into a fully frozen foundation model, improves topology. Training any part of the decoder weakens that claim.

- **Caching SAM image encoder outputs to disk.** Would make training ~10× faster (frozen encoder forward never changes). We didn't because it's incompatible with image augmentation, and 600 images is a small enough training set that augmentation matters. Easy to add if training speed becomes a bottleneck.

- **Per-channel α (256 scalars) instead of single global α.** More expressive but harder to monitor. A single α is the simplest possible gate and gives us a single number to watch in the progress bar. Per-channel α is a future ablation.

- **Multi-resolution prior injection** (also feeding the 64² prior into the decoder's upsampling path where features expand back to 256²). Would recover fine vessel detail lost in the 16× downsample, but requires touching the decoder's intermediate layers, which is more invasive. Saved for v2.

- **Vanilla SAM instead of MedSAM.** Considered, rejected. MedSAM is a stronger and more honest baseline. Beating vanilla SAM on retinal vessels (which it has barely seen) is too easy; beating MedSAM (which has seen related medical structures) is meaningful.

## Known limitations

1. **The 16× spatial downsample from 1024² to 64² is lossy.** Capillaries are 1–2 pixels wide at 1024² and get averaged into background at 64². The strided CNN mitigates this with learned downsampling, but there's an information ceiling we can't exceed with single-resolution injection. The expected gain is probably +5 to +15 clDice points, not +30. Anything more would require multi-resolution injection (v2).

2. **The sparse token pathway is not gated by α.** Unlike the FiLM dense pathway, the sparse tokens are present from training step 0. SAM is robust to noisy sparse prompts in early training, but the fact that this pathway isn't bit-exactly zero at init means epoch 0 of LPG-SAM is **not** identical to zero-shot SAM. The image-embedding pathway is exact at init; the token pathway is approximately the no-op. In practice this is fine but worth noting.

3. **No formal Betti-1 (loop count) error.** The current Betti-0 error (component count mismatch) catches over-merging and over-fragmentation, but doesn't directly measure loop topology. Adding Betti-1 requires `gudhi` or persistent homology code, which we skipped for v1.

4. **FOV mask is auto-computed from intensity threshold.** Every dataset we use ships its own FOV masks, but we don't load them — we recompute via thresholding. The auto-computed mask is good enough for FIVES/DRIVE/STARE/CHASE_DB1 but may be wrong on datasets with unusual lighting or low-contrast borders. If cross-dataset eval shows weird artifacts, the first thing to check is whether the auto FOV mask is correct on those specific images.

## Acknowledgments and references

- **SAM**: Kirillov et al. *Segment Anything*, 2023. [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- **MedSAM**: Ma et al. *Segment Anything in Medical Images*, Nature Communications 2024.
- **Frangi vesselness filter**: Frangi et al. *Multiscale Vessel Enhancement Filtering*, MICCAI 1998.
- **soft clDice**: Shit et al. *clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation*, CVPR 2021.
- **FiLM**: Perez et al. *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018.
- **FIVES**: Jin et al. *FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel Segmentation*, Scientific Data 2022.
- **DRIVE / STARE / CHASE_DB1**: standard retinal vessel benchmarks.
