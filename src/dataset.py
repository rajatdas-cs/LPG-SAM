"""
dataset.py
----------
PyTorch Dataset for FIVES + cross-dataset eval (DRIVE, STARE, CHASE_DB1).

Each sample is a dict with three tensors:
    {
        "image":  (3, 1024, 1024)  float32   -- RGB, normalized for MedSAM
        "frangi": (1, 1024, 1024)  float32   -- Frangi vesselness in [0, 1]
        "mask":   (1, 1024, 1024)  float32   -- binary GT vessel mask
        "name":   str                        -- file stem (for logging/eval)
    }

The dataset assumes data has already been processed by:
    1. scripts/prepare_data.py  -> writes resized images + masks to disk
    2. scripts/cache_frangi.py  -> writes Frangi .npy files to disk

We do NOT recompute Frangi at training time. It's slow and never changes.

MedSAM normalization
--------------------
MedSAM was trained with per-image min-max normalization to [0, 1], NOT
ImageNet mean/std. Forgetting this will silently destroy zero-shot SAM
performance and confuse the entire ablation table. We do it here in
__getitem__ and document it loudly.

Augmentation philosophy
-----------------------
Conservative, as agreed:
    - Horizontal flip (p=0.5)
    - Vertical flip (p=0.5)
    - 90/180/270 degree rotation (p=0.75)
NO color jitter (would corrupt the green channel that Frangi feeds on).
NO elastic deformation (would invalidate the alignment between image,
frangi, and mask without giving us much).
NO scale/crop (SAM is hardcoded for 1024x1024 input).

CRITICAL: every spatial augmentation must be applied IDENTICALLY to the
image, the Frangi prior, and the mask. If they get out of sync, the
prior lies to the model and training silently breaks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def medsam_normalize(image: np.ndarray) -> np.ndarray:
    """
    Per-image min-max normalize to [0, 1], following the MedSAM convention.

    Parameters
    ----------
    image : np.ndarray
        Shape (H, W, 3) or (3, H, W). Any dtype.

    Returns
    -------
    np.ndarray
        Same shape, float32, values in [0, 1].

    Notes
    -----
    Vanilla SAM uses ImageNet mean/std. MedSAM uses min-max. They are NOT
    interchangeable. This function implements the MedSAM convention. If
    you switch to vanilla SAM, swap this for ImageNet normalization.
    """
    img = image.astype(np.float32)
    lo = img.min()
    hi = img.max()
    if hi - lo < 1e-8:
        # Degenerate case: flat image. Return zeros to avoid div-by-zero.
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Joint augmentation (image + frangi + mask must move together)
# ---------------------------------------------------------------------------

class JointAugment:
    """
    Apply identical spatial augmentation to image, frangi, and mask.

    Operates on numpy arrays so we can call it before tensor conversion.
    Expected shapes:
        image  : (H, W, 3)  float32
        frangi : (H, W)     float32
        mask   : (H, W)     float32 in {0, 1}

    Augmentations:
        - hflip (p=0.5)
        - vflip (p=0.5)
        - rotate by k*90 degrees (k uniform in {0, 1, 2, 3})

    All ops are size-preserving and exact (no interpolation), so the
    binary mask stays binary and the FOV mask stays consistent.
    """

    def __init__(self, p_hflip: float = 0.5, p_vflip: float = 0.5, rotate90: bool = True):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.rotate90 = rotate90

    def __call__(
        self,
        image: np.ndarray,
        frangi: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Horizontal flip (along width axis = axis 1 for HW or HWC)
        if rng.random() < self.p_hflip:
            image = np.ascontiguousarray(image[:, ::-1, :])
            frangi = np.ascontiguousarray(frangi[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        # Vertical flip (along height axis = axis 0)
        if rng.random() < self.p_vflip:
            image = np.ascontiguousarray(image[::-1, :, :])
            frangi = np.ascontiguousarray(frangi[::-1, :])
            mask = np.ascontiguousarray(mask[::-1, :])

        # Rotation by k*90 degrees. axes=(0, 1) rotates the spatial plane.
        if self.rotate90:
            k = int(rng.integers(0, 4))  # 0, 1, 2, or 3
            if k != 0:
                image = np.ascontiguousarray(np.rot90(image, k=k, axes=(0, 1)))
                frangi = np.ascontiguousarray(np.rot90(frangi, k=k, axes=(0, 1)))
                mask = np.ascontiguousarray(np.rot90(mask, k=k, axes=(0, 1)))

        return image, frangi, mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FundusVesselDataset(Dataset):
    """
    Dataset for FIVES, DRIVE, STARE, or CHASE_DB1.

    Expects this directory layout (produced by prepare_data.py + cache_frangi.py):

        <data_root>/
            images_1024/      # (H, W, 3) uint8 PNGs at 1024x1024
                img001.png
                img002.png
                ...
            masks_1024/       # (H, W) uint8 PNGs at 1024x1024, values 0 or 255
                img001.png
                img002.png
                ...
            frangi_cache/     # (H, W) float32 .npy at 1024x1024, values in [0, 1]
                img001.npy
                img002.npy
                ...

    The split file is a plain text file with one image stem per line, e.g.:

        img001
        img002
        img007
        ...

    Parameters
    ----------
    data_root : str or Path
        Path to the dataset's processed root (e.g. "data/processed/FIVES").
    split_file : str or Path
        Path to a text file listing image stems for this split.
    augment : bool
        If True, apply JointAugment. Off for val/test.
    seed : int
        Base seed for the augmentation RNG. The actual RNG is per-sample
        derived from (seed, index) so augmentations are reproducible
        across runs but different per sample.
    """

    def __init__(
        self,
        data_root: str | Path,
        split_file: str | Path,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "images_1024"
        self.masks_dir = self.data_root / "masks_1024"
        self.frangi_dir = self.data_root / "frangi_cache"

        for d in (self.images_dir, self.masks_dir, self.frangi_dir):
            if not d.exists():
                raise FileNotFoundError(
                    f"Missing directory: {d}\n"
                    f"Run scripts/prepare_data.py and scripts/cache_frangi.py first."
                )

        # Read the split file. Each line is one image stem (no extension).
        split_file = Path(split_file)
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file, "r") as f:
            self.stems: list[str] = [
                line.strip() for line in f if line.strip()
            ]
        if not self.stems:
            raise ValueError(f"Split file is empty: {split_file}")

        self.augment = augment
        self.augmenter = JointAugment() if augment else None
        self.seed = seed

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]

        # ---- Load image (uint8 RGB PNG, 1024x1024x3) ----
        img_path = self.images_dir / f"{stem}.png"
        image = _load_png_rgb(img_path)  # (H, W, 3) uint8

        # ---- Load mask (uint8 grayscale PNG, 0 or 255) ----
        mask_path = self.masks_dir / f"{stem}.png"
        mask = _load_png_gray(mask_path)  # (H, W) uint8
        mask = (mask > 127).astype(np.float32)  # binarize -> {0.0, 1.0}

        # ---- Load Frangi (float32 .npy in [0, 1]) ----
        frangi_path = self.frangi_dir / f"{stem}.npy"
        frangi = np.load(frangi_path).astype(np.float32)  # (H, W)

        # ---- Sanity: shapes must match ----
        if image.shape[:2] != mask.shape or image.shape[:2] != frangi.shape:
            raise RuntimeError(
                f"Shape mismatch for '{stem}': "
                f"image {image.shape}, mask {mask.shape}, frangi {frangi.shape}"
            )

        # ---- Augmentation (joint, before normalization) ----
        if self.augmenter is not None:
            # Per-sample RNG so augmentations are reproducible per (seed, idx).
            rng = np.random.default_rng(self.seed * 1_000_003 + idx)
            image, frangi, mask = self.augmenter(image, frangi, mask, rng)

        # ---- Normalize image for MedSAM (per-image min-max to [0, 1]) ----
        image = medsam_normalize(image)  # still (H, W, 3), float32

        # ---- Convert to tensors with channel-first layout ----
        image_t = torch.from_numpy(image).permute(2, 0, 1).contiguous()  # (3, H, W)
        frangi_t = torch.from_numpy(frangi).unsqueeze(0).contiguous()    # (1, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0).contiguous()        # (1, H, W)

        return {
            "image": image_t,
            "frangi": frangi_t,
            "mask": mask_t,
            "name": stem,
        }


# ---------------------------------------------------------------------------
# Small helpers (kept here so dataset.py is self-contained)
# ---------------------------------------------------------------------------

def _load_png_rgb(path: Path) -> np.ndarray:
    """Load an RGB PNG as a (H, W, 3) uint8 numpy array."""
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def _load_png_gray(path: Path) -> np.ndarray:
    """Load a grayscale PNG as a (H, W) uint8 numpy array."""
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im.convert("L"))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Standalone test that creates a tiny fake dataset on disk and verifies
    the Dataset class loads it correctly. Does NOT touch real FIVES data.

    Run with:
        python -m src.dataset
    """
    import shutil
    import tempfile

    print("Building tiny fake dataset on disk ...")
    tmp = Path(tempfile.mkdtemp(prefix="lpgsam_dstest_"))
    try:
        (tmp / "images_1024").mkdir(parents=True)
        (tmp / "masks_1024").mkdir(parents=True)
        (tmp / "frangi_cache").mkdir(parents=True)

        from PIL import Image as PILImage

        H = W = 1024
        rng = np.random.default_rng(0)
        stems = ["img001", "img002", "img003"]
        for stem in stems:
            img = (rng.random((H, W, 3)) * 255).astype(np.uint8)
            mask = ((rng.random((H, W)) > 0.7) * 255).astype(np.uint8)
            frangi = rng.random((H, W)).astype(np.float32)

            PILImage.fromarray(img).save(tmp / "images_1024" / f"{stem}.png")
            PILImage.fromarray(mask).save(tmp / "masks_1024" / f"{stem}.png")
            np.save(tmp / "frangi_cache" / f"{stem}.npy", frangi)

        split_file = tmp / "train.txt"
        split_file.write_text("\n".join(stems) + "\n")

        print("Loading via FundusVesselDataset (no augment) ...")
        ds = FundusVesselDataset(tmp, split_file, augment=False)
        assert len(ds) == 3
        sample = ds[0]
        assert sample["image"].shape == (3, H, W)
        assert sample["frangi"].shape == (1, H, W)
        assert sample["mask"].shape == (1, H, W)
        assert sample["image"].dtype == torch.float32
        assert 0.0 <= sample["image"].min().item()
        assert sample["image"].max().item() <= 1.0
        assert set(sample["mask"].unique().tolist()) <= {0.0, 1.0}
        print(f"  ok: sample[0] = {sample['name']}")
        print(f"      image: {tuple(sample['image'].shape)} {sample['image'].dtype}")
        print(f"      frangi: {tuple(sample['frangi'].shape)} {sample['frangi'].dtype}")
        print(f"      mask: {tuple(sample['mask'].shape)} {sample['mask'].dtype}")

        print("Loading with augment=True ...")
        ds_aug = FundusVesselDataset(tmp, split_file, augment=True, seed=42)
        s1 = ds_aug[0]
        s2 = ds_aug[0]  # same idx, same seed -> identical (reproducibility check)
        assert torch.equal(s1["image"], s2["image"]), (
            "Augmentation is not reproducible for the same (seed, idx)"
        )
        # Verify image+mask+frangi stay aligned: the unique mask values
        # should be exactly {0, 1} after any augmentation (no interpolation).
        assert set(s1["mask"].unique().tolist()) <= {0.0, 1.0}
        print("  ok: augmentation is deterministic per (seed, idx) and "
              "preserves binary masks")

        print("\n[OK] FundusVesselDataset works.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
