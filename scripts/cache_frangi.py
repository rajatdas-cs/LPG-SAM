"""
scripts/cache_frangi.py
-----------------------
Pre-compute Frangi vesselness for every image in the processed dataset
and save the results to disk as float32 .npy files.

We do this once, offline, because:

1. Frangi via skimage runs on CPU and takes ~1-2 seconds per 1024x1024
   image. Multiplied by 800 FIVES images x N epochs, that's hours of
   wasted compute every training run.
2. The Frangi response never changes (it's deterministic given the
   same image and sigmas). Recomputing per epoch is just stupid.
3. We can re-tune sigmas and re-cache without touching anything else
   in the pipeline.

This script is idempotent: re-running skips images whose .npy file
already exists. Use --force to recompute everything.

Run with
--------
    python -m scripts.cache_frangi \\
        --data-root data/processed/FIVES \\
        --sigmas 1 2 3 4 5

For sigma tuning, use notebooks/frangi_tuning.ipynb to pick values
visually first, then re-run this script with the chosen sigmas.

Output
------
Writes .npy files to <data_root>/frangi_cache/<stem>.npy.
Each file is shape (1024, 1024), float32, values in [0, 1].
Total disk usage: ~3 MB per image, ~2.5 GB for FIVES (800 images).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.frangi import fundus_to_frangi, DEFAULT_SIGMAS, DEFAULT_CLAHE_CLIP


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache Frangi responses for a processed dataset."
    )
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="Path to processed dataset root (must contain images_1024/).",
    )
    parser.add_argument(
        "--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMAS),
        help="Frangi sigmas in pixels at 1024x1024 resolution.",
    )
    parser.add_argument(
        "--clahe-clip", type=float, default=DEFAULT_CLAHE_CLIP,
        help="CLAHE clip limit. Lower = more conservative.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if cache files already exist.",
    )
    args = parser.parse_args()

    images_dir = args.data_root / "images_1024"
    cache_dir = args.data_root / "frangi_cache"

    if not images_dir.exists():
        raise FileNotFoundError(
            f"images_1024/ not found at {images_dir}. "
            f"Run scripts/prepare_data.py first."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() == ".png")
    if not image_files:
        raise RuntimeError(f"No PNG images found in {images_dir}")

    print(f"Data root:    {args.data_root}")
    print(f"Images found: {len(image_files)}")
    print(f"Cache dir:    {cache_dir}")
    print(f"Sigmas:       {args.sigmas}")
    print(f"CLAHE clip:   {args.clahe_clip}")
    print(f"Force redo:   {args.force}")
    print()

    n_skipped = 0
    n_processed = 0
    t_start = time.time()

    bar = tqdm(image_files, desc="Frangi cache", ncols=80)
    for img_path in bar:
        out_path = cache_dir / f"{img_path.stem}.npy"

        # Idempotent skip.
        if out_path.exists() and not args.force:
            n_skipped += 1
            bar.set_postfix(processed=n_processed, skipped=n_skipped)
            continue

        # Load RGB image.
        with Image.open(img_path) as im:
            rgb = np.array(im.convert("RGB"))

        # Run pipeline. FOV mask is auto-computed inside fundus_to_frangi.
        frangi = fundus_to_frangi(
            rgb,
            fov_mask=None,
            sigmas=tuple(args.sigmas),
            clahe_clip=args.clahe_clip,
        )

        # Save as float32 .npy.
        np.save(out_path, frangi.astype(np.float32))
        n_processed += 1
        bar.set_postfix(processed=n_processed, skipped=n_skipped)
    bar.close()

    elapsed = time.time() - t_start
    print()
    print(f"[OK] Frangi caching complete.")
    print(f"  processed: {n_processed}")
    print(f"  skipped:   {n_skipped}")
    print(f"  elapsed:   {elapsed:.1f} s "
          f"({elapsed / max(n_processed, 1):.2f} s/image)")
    if n_processed > 0:
        print(f"\nCached files written to: {cache_dir}")


if __name__ == "__main__":
    main()
