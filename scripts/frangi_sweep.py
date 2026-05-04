"""
Frangi parameter sweep for visual inspection.

Runs many variations of the Frangi pipeline on a single fundus image and
writes per-variation preview PNGs to `frangi_check/` so you can eyeball which
combination highlights vessels best.

Usage:
    python -m scripts.frangi_sweep --stem train_1_A
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import exposure, filters
from skimage.filters import frangi, sato, meijering, hessian
from skimage.morphology import binary_closing, disk

from src.frangi import compute_fov_mask, apply_fov_mask


# ---------------------------------------------------------------------------
# Preprocessing variants
# ---------------------------------------------------------------------------

def get_green(rgb: np.ndarray) -> np.ndarray:
    g = rgb[:, :, 1].astype(np.float32)
    if g.max() > 1.5:
        g /= 255.0
    return g


def get_luminance(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32)
    if x.max() > 1.5:
        x /= 255.0
    # ITU-R BT.601
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def preprocess(
    rgb: np.ndarray,
    fov: np.ndarray,
    channel: str = "green",
    clahe_clip: float = 0.01,
    clahe_kernel: int | None = None,
    gamma_correct: float | None = None,
    denoise_sigma: float | None = None,
) -> np.ndarray:
    """Apply one preprocessing recipe and return the single-channel image
    that will be fed to a ridge filter (vessels bright, background dark,
    masked to FOV)."""
    if channel == "green":
        base = get_green(rgb)
    elif channel == "luminance":
        base = get_luminance(rgb)
    elif channel == "green_masked_raw":
        # mask raw RGB FIRST, then extract green (prevents boundary ridge)
        rgb_masked = apply_fov_mask(rgb.astype(np.float32) / 255.0, fov)
        base = rgb_masked[:, :, 1]
    else:
        raise ValueError(channel)

    inverted = 1.0 - base
    inverted = apply_fov_mask(inverted, fov)

    if gamma_correct is not None:
        inverted = np.clip(inverted, 0, 1) ** gamma_correct

    enhanced = exposure.equalize_adapthist(
        np.clip(inverted, 0, 1),
        clip_limit=clahe_clip,
        kernel_size=clahe_kernel,
    ).astype(np.float32)
    enhanced = apply_fov_mask(enhanced, fov)

    if denoise_sigma is not None and denoise_sigma > 0:
        from skimage.filters import gaussian
        enhanced = gaussian(enhanced, sigma=denoise_sigma).astype(np.float32)
        enhanced = apply_fov_mask(enhanced, fov)

    return enhanced


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

def norm01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, None)
    m = x.max()
    return x / m if m > 0 else x


def build_variants() -> list[dict]:
    """Return a list of variant dicts. Each dict has a `name`, a `pre`
    callable (rgb, fov) -> (H,W) float, and a `filter` callable (img) -> response."""
    variants = []

    # --- Baseline: current repo default ---
    variants.append(dict(
        name="00_baseline_default",
        pre=lambda rgb, fov: preprocess(rgb, fov),
        filter=lambda x: frangi(x, sigmas=(1, 2, 3, 4, 5), black_ridges=False,
                                alpha=0.5, beta=0.5, gamma=None),
    ))

    # --- Sigma sweeps ---
    sigma_sets = {
        "01_sigmas_fine": (0.5, 1.0, 1.5, 2.0),
        "02_sigmas_narrow_small": (1, 2, 3),
        "03_sigmas_wide": (1, 2, 3, 4, 5, 6, 7, 8),
        "04_sigmas_very_wide": (1, 3, 5, 7, 9, 11),
        "05_sigmas_log": (1, 1.5, 2.3, 3.4, 5.1, 7.6),
        "06_sigmas_mid": (2, 4, 6, 8),
    }
    for name, sig in sigma_sets.items():
        variants.append(dict(
            name=name,
            pre=lambda rgb, fov: preprocess(rgb, fov),
            filter=(lambda s: lambda x: frangi(x, sigmas=s, black_ridges=False))(sig),
        ))

    # --- Beta / gamma sweeps (blob + background sensitivity) ---
    for beta in (0.3, 0.8, 1.5):
        variants.append(dict(
            name=f"10_beta_{beta}",
            pre=lambda rgb, fov: preprocess(rgb, fov),
            filter=(lambda b: lambda x: frangi(
                x, sigmas=(1, 2, 3, 4, 5), black_ridges=False, beta=b))(beta),
        ))
    for gamma in (5, 15, 30):
        variants.append(dict(
            name=f"11_gamma_{gamma}",
            pre=lambda rgb, fov: preprocess(rgb, fov),
            filter=(lambda g: lambda x: frangi(
                x, sigmas=(1, 2, 3, 4, 5), black_ridges=False, gamma=g))(gamma),
        ))

    # --- CLAHE variants ---
    for clip in (0.003, 0.02, 0.04):
        variants.append(dict(
            name=f"20_clahe_clip_{clip}",
            pre=(lambda c: lambda rgb, fov: preprocess(rgb, fov, clahe_clip=c))(clip),
            filter=lambda x: frangi(x, sigmas=(1, 2, 3, 4, 5), black_ridges=False),
        ))
    for ks in (64, 128, 256):
        variants.append(dict(
            name=f"21_clahe_kernel_{ks}",
            pre=(lambda k: lambda rgb, fov: preprocess(rgb, fov, clahe_kernel=k))(ks),
            filter=lambda x: frangi(x, sigmas=(1, 2, 3, 4, 5), black_ridges=False),
        ))

    # --- Gamma correction (nonlinear contrast pre-Frangi) ---
    for gc in (0.6, 0.8, 1.3):
        variants.append(dict(
            name=f"30_gamma_correct_{gc}",
            pre=(lambda g: lambda rgb, fov: preprocess(rgb, fov, gamma_correct=g))(gc),
            filter=lambda x: frangi(x, sigmas=(1, 2, 3, 4, 5), black_ridges=False),
        ))

    # --- Denoise pre-filter ---
    for ds in (0.5, 1.0):
        variants.append(dict(
            name=f"40_denoise_sigma_{ds}",
            pre=(lambda s: lambda rgb, fov: preprocess(rgb, fov, denoise_sigma=s))(ds),
            filter=lambda x: frangi(x, sigmas=(1, 2, 3, 4, 5), black_ridges=False),
        ))

    # --- Alternative ridge filters ---
    variants.append(dict(
        name="50_sato",
        pre=lambda rgb, fov: preprocess(rgb, fov),
        filter=lambda x: sato(x, sigmas=(1, 2, 3, 4, 5), black_ridges=False),
    ))
    variants.append(dict(
        name="51_meijering",
        pre=lambda rgb, fov: preprocess(rgb, fov),
        filter=lambda x: meijering(x, sigmas=(1, 2, 3, 4, 5), black_ridges=False),
    ))
    variants.append(dict(
        name="52_hessian",
        pre=lambda rgb, fov: preprocess(rgb, fov),
        filter=lambda x: hessian(x, sigmas=(1, 2, 3, 4, 5), black_ridges=False),
    ))

    # --- Channel variants ---
    variants.append(dict(
        name="60_luminance_channel",
        pre=lambda rgb, fov: preprocess(rgb, fov, channel="luminance"),
        filter=lambda x: frangi(x, sigmas=(1, 2, 3, 4, 5), black_ridges=False),
    ))
    variants.append(dict(
        name="61_green_masked_raw",
        pre=lambda rgb, fov: preprocess(rgb, fov, channel="green_masked_raw"),
        filter=lambda x: frangi(x, sigmas=(1, 2, 3, 4, 5), black_ridges=False),
    ))

    # --- Combination: promising stack ---
    variants.append(dict(
        name="70_combo_fine_sigmas_strong_clahe",
        pre=lambda rgb, fov: preprocess(rgb, fov, clahe_clip=0.03),
        filter=lambda x: frangi(x, sigmas=(0.8, 1.2, 1.8, 2.7, 4.0),
                                black_ridges=False, beta=0.5),
    ))
    variants.append(dict(
        name="71_combo_sato_strong_clahe",
        pre=lambda rgb, fov: preprocess(rgb, fov, clahe_clip=0.03),
        filter=lambda x: sato(x, sigmas=(1, 2, 3, 4, 5), black_ridges=False),
    ))
    variants.append(dict(
        name="72_combo_masked_raw_fine",
        pre=lambda rgb, fov: preprocess(rgb, fov, channel="green_masked_raw",
                                        clahe_clip=0.02),
        filter=lambda x: frangi(x, sigmas=(0.8, 1.5, 2.5, 4.0, 6.0),
                                black_ridges=False),
    ))

    # --- Post-processing: gamma compress output for visualization ---
    variants.append(dict(
        name="80_default_post_gamma_0.5",
        pre=lambda rgb, fov: preprocess(rgb, fov),
        filter=lambda x: np.power(
            norm01(frangi(x, sigmas=(1, 2, 3, 4, 5), black_ridges=False)), 0.5),
    ))
    variants.append(dict(
        name="81_default_post_gamma_0.3",
        pre=lambda rgb, fov: preprocess(rgb, fov),
        filter=lambda x: np.power(
            norm01(frangi(x, sigmas=(1, 2, 3, 4, 5), black_ridges=False)), 0.3),
    ))

    return variants


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", default="train_1_A",
                        help="image stem under data/processed/FIVES/images_1024")
    parser.add_argument(
        "--data-root", default="data/processed/FIVES",
        help="processed FIVES root containing images_1024/")
    parser.add_argument("--out-dir", default="frangi_check")
    args = parser.parse_args()

    img_path = Path(args.data_root) / "images_1024" / f"{args.stem}.png"
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    out_dir = Path(args.out_dir) / args.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.array(Image.open(img_path).convert("RGB"))
    fov = compute_fov_mask(rgb)
    print(f"Loaded {img_path}, shape={rgb.shape}, FOV pixels={fov.sum()}")

    # Save the input for reference.
    plt.imsave(out_dir / "_input_rgb.png", rgb)

    variants = build_variants()
    print(f"Running {len(variants)} variants ...")

    for v in variants:
        name = v["name"]
        try:
            pre_img = v["pre"](rgb, fov)
            resp = v["filter"](pre_img)
            resp = apply_fov_mask(resp, fov)
            resp = norm01(resp)
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            continue

        # Per-variant 3-panel figure (input | preprocessed | frangi).
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(rgb)
        ax[0].set_title("image")
        ax[0].axis("off")
        ax[1].imshow(pre_img, cmap="gray")
        ax[1].set_title("preprocessed (into filter)")
        ax[1].axis("off")
        ax[2].imshow(resp, cmap="hot")
        ax[2].set_title(f"{name}\nmin={resp.min():.3f} max={resp.max():.3f}")
        ax[2].axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

        # Also dump the raw response grayscale for crisp inspection.
        plt.imsave(out_dir / f"{name}_raw.png", resp, cmap="gray", vmin=0, vmax=1)
        print(f"  [OK] {name}")

    print(f"\nDone. Open {out_dir}/ and compare.")


if __name__ == "__main__":
    main()
