"""
scripts/prepare_data.py
-----------------------
Convert raw FIVES into the processed format expected by the dataset.

What it does
------------
1. Reads raw FIVES from `data/raw/FIVES/`.
2. Resizes every image to 1024x1024 RGB and saves to
   `data/processed/FIVES/images_1024/<stem>.png`.
3. Resizes every mask to 1024x1024 binary and saves to
   `data/processed/FIVES/masks_1024/<stem>.png`.
4. Writes train/val/test split files at
   `data/processed/FIVES/{train,val,test}.txt`.

It does NOT compute Frangi -- that's `cache_frangi.py`, which runs after
this and can be re-run separately if you want to retune sigmas.

FIVES dataset layout (as released)
----------------------------------
The official FIVES release ships as:

    FIVES/
        train/
            Original/   <- 600 RGB fundus images (.png)
            Ground truth/ <- 600 binary vessel masks (.png), same filenames
        test/
            Original/   <- 200 images
            Ground truth/ <- 200 masks

We use the official train/test split. From the 600 train images we carve
out a small val set (default: 60 images = 10%) for hyperparameter
selection. The remaining 540 are used for training. The 200 test images
are LOCKED -- never look at them until the final eval.

Run with
--------
    python -m scripts.prepare_data \\
        --raw-root data/raw/FIVES \\
        --out-root data/processed/FIVES \\
        --val-fraction 0.1 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


TARGET_SIZE = 1024


# ---------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------

def resize_image_rgb(img: Image.Image, size: int = TARGET_SIZE) -> Image.Image:
    """
    Resize an RGB PIL image to (size, size) using bilinear interpolation.

    Bilinear is appropriate for natural images. We use BICUBIC for
    slightly better quality at the cost of a few extra ms per image.
    """
    return img.convert("RGB").resize((size, size), resample=Image.BICUBIC)


def resize_mask_binary(mask: Image.Image, size: int = TARGET_SIZE) -> Image.Image:
    """
    Resize a binary mask to (size, size) using nearest-neighbor.

    Nearest-neighbor is critical: any interpolation (bilinear, bicubic)
    would produce intermediate gray values that don't correspond to real
    vessels. After NN resize we re-binarize to {0, 255} just to be safe.
    """
    m = mask.convert("L").resize((size, size), resample=Image.NEAREST)
    arr = np.array(m)
    arr = (arr > 127).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


# ---------------------------------------------------------------------------
# FIVES discovery
# ---------------------------------------------------------------------------

def discover_fives(raw_root: Path) -> dict[str, list[tuple[Path, Path]]]:
    """
    Find FIVES image/mask pairs.

    Returns a dict {"train": [(img_path, mask_path), ...], "test": [...]}.

    The official FIVES layout uses "Original" and "Ground truth" subdirs
    with matching filenames. Some mirrors use slightly different folder
    names; we handle a few common variants.
    """
    train_root = raw_root / "train"
    test_root = raw_root / "test"
    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError(
            f"Expected FIVES at {raw_root} with 'train/' and 'test/' subdirs, "
            f"but didn't find them. Got: {raw_root.iterdir() if raw_root.exists() else 'missing'}"
        )

    def find_pairs(split_root: Path) -> list[tuple[Path, Path]]:
        # Try a few common names for the image and mask folders.
        candidates_img = ["Original", "original", "Image", "images", "img"]
        candidates_msk = [
            "Ground truth", "Ground_truth", "GroundTruth",
            "ground truth", "ground_truth", "groundtruth",
            "Mask", "masks", "labels", "Label",
        ]
        img_dir = next((split_root / c for c in candidates_img if (split_root / c).exists()), None)
        msk_dir = next((split_root / c for c in candidates_msk if (split_root / c).exists()), None)
        if img_dir is None or msk_dir is None:
            raise FileNotFoundError(
                f"Couldn't find image or mask subfolder under {split_root}. "
                f"Looked for {candidates_img} and {candidates_msk}."
            )

        img_paths = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        )
        pairs: list[tuple[Path, Path]] = []
        missing: list[str] = []
        for ip in img_paths:
            # Mask filename matches the image stem. Try common extensions.
            mp = None
            for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
                cand = msk_dir / f"{ip.stem}{ext}"
                if cand.exists():
                    mp = cand
                    break
            if mp is None:
                missing.append(ip.name)
                continue
            pairs.append((ip, mp))

        if missing:
            print(f"  WARNING: {len(missing)} images had no matching mask, e.g. {missing[:3]}")
        return pairs

    train_pairs = find_pairs(train_root)
    test_pairs = find_pairs(test_root)
    return {"train": train_pairs, "test": test_pairs}


# ---------------------------------------------------------------------------
# Split file writing
# ---------------------------------------------------------------------------

def write_split_file(stems: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(stems) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare FIVES for LPG-SAM training."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/FIVES"))
    parser.add_argument("--out-root", type=Path, default=Path("data/processed/FIVES"))
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of train split to hold out as val.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=TARGET_SIZE)
    args = parser.parse_args()

    print(f"Raw root: {args.raw_root}")
    print(f"Out root: {args.out_root}")
    print(f"Target size: {args.size}x{args.size}")
    print(f"Val fraction: {args.val_fraction}")
    print()

    # ---------------------------------------------------------
    # Step 1: discover all image/mask pairs
    # ---------------------------------------------------------
    print("Discovering FIVES files ...")
    pairs = discover_fives(args.raw_root)
    n_train_total = len(pairs["train"])
    n_test = len(pairs["test"])
    print(f"  found {n_train_total} train pairs, {n_test} test pairs")
    if n_train_total == 0 or n_test == 0:
        raise RuntimeError("No pairs found. Check the raw FIVES path.")

    # ---------------------------------------------------------
    # Step 2: build the train/val split (deterministic from seed)
    # ---------------------------------------------------------
    rng = random.Random(args.seed)
    shuffled = list(pairs["train"])
    rng.shuffle(shuffled)
    n_val = int(round(n_train_total * args.val_fraction))
    val_pairs = shuffled[:n_val]
    train_pairs = shuffled[n_val:]
    test_pairs = pairs["test"]
    print(f"  split: train={len(train_pairs)}  val={len(val_pairs)}  test={len(test_pairs)}")
    print()

    # ---------------------------------------------------------
    # Step 3: create output directories
    # ---------------------------------------------------------
    images_out = args.out_root / "images_1024"
    masks_out = args.out_root / "masks_1024"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Step 4: resize and save (train + val + test in one pass,
    #         since the images_1024/ folder is shared across splits)
    # ---------------------------------------------------------
    all_pairs = [
        ("train", train_pairs),
        ("val", val_pairs),
        ("test", test_pairs),
    ]
    split_stems: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    total = sum(len(p) for _, p in all_pairs)
    bar = tqdm(total=total, desc="Resizing FIVES", ncols=80)
    seen_stems: set[str] = set()  # safety net to catch any future collisions
    for split_name, pair_list in all_pairs:
        for img_path, mask_path in pair_list:
            # Namespace the stem with the source split so that train and
            # test images with the same original filename (e.g. "1_A.png"
            # appears in BOTH train/Original/ and test/Original/ in FIVES)
            # don't overwrite each other in the shared images_1024/ folder.
            #
            # Note: val samples come from the train split, so they get
            # the "train_" prefix too -- they ARE train images, just
            # held out for validation.
            source_split = "test" if split_name == "test" else "train"
            stem = f"{source_split}_{img_path.stem}"

            # Hard fail if we ever see the same stem twice. This should
            # be impossible after the namespacing above, but a crashed
            # run is infinitely better than silent data contamination.
            if stem in seen_stems:
                raise RuntimeError(
                    f"Duplicate stem '{stem}' encountered. "
                    f"Source: {img_path}. Aborting to prevent silent "
                    f"overwrite of an earlier file."
                )
            seen_stems.add(stem)

            # Skip if already resized (idempotent re-runs).
            out_img = images_out / f"{stem}.png"
            out_msk = masks_out / f"{stem}.png"
            if not out_img.exists():
                with Image.open(img_path) as im:
                    resize_image_rgb(im, args.size).save(out_img)
            if not out_msk.exists():
                with Image.open(mask_path) as mk:
                    resize_mask_binary(mk, args.size).save(out_msk)

            split_stems[split_name].append(stem)
            bar.update(1)
    bar.close()

    # ---------------------------------------------------------
    # Step 5: write split files
    # ---------------------------------------------------------
    print()
    print("Writing split files ...")
    for split_name, stems in split_stems.items():
        out_file = args.out_root / f"{split_name}.txt"
        write_split_file(sorted(stems), out_file)
        print(f"  {out_file}  ({len(stems)} stems)")

    print()
    print("[OK] Data prep complete.")
    print(f"Next step: python -m scripts.cache_frangi --data-root {args.out_root}")


if __name__ == "__main__":
    main()
