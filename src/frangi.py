"""
frangi.py
---------
Frangi vesselness prior for retinal fundus images.

This module is purely image processing — no PyTorch, no GPU. It runs on
CPU during data preparation (`scripts/cache_frangi.py`) and the resulting
.npy arrays are loaded by `dataset.py` at training time.

Pipeline for one fundus image:

    RGB image (H, W, 3)
        |
        v   extract green channel (vessels are darkest here)
    green (H, W), uint8 in [0, 255]
        |
        v   normalize to [0, 1] and invert (vessels become BRIGHT)
    inverted (H, W), float32 in [0, 1]
        |
        v   CLAHE — fixes illumination gradient across the retina
    enhanced (H, W), float32 in [0, 1]
        |
        v   apply FOV mask — zero out non-retina pixels
    masked (H, W), float32
        |
        v   multi-scale Frangi (skimage)
    frangi (H, W), float32 in [0, 1]   <- this is what we cache

Why each step matters
---------------------
- Green channel: hemoglobin absorbs green light strongly, so vessels appear
  as the darkest structures in the green channel. Red is dominated by the
  reddish background; blue is too noisy. Standard in every retinal vessel
  paper for the last 25 years.

- Invert: skimage's frangi() detects bright ridges by default
  (`black_ridges=False`). Inverting makes vessels bright on a dark
  background, matching that convention. We could equivalently set
  `black_ridges=True` and skip inversion — same result. Inversion is
  more explicit and matches how Frangi responses are usually visualized.

- CLAHE (Contrast Limited Adaptive Histogram Equalization): fundus images
  have severe illumination gradients (the optic disc is very bright, the
  periphery is dim). Without CLAHE, Frangi gives weak responses in dim
  regions and gets saturated near the disc. CLAHE fixes both by equalizing
  contrast in local tiles.

- FOV (field of view) mask: fundus cameras capture a circular region of
  the retina surrounded by black. The boundary between retina and black
  is a sharp edge that Frangi LOVES — it produces a giant spurious
  response along the entire FOV boundary that looks like a huge vessel.
  We zero out non-FOV pixels BEFORE running Frangi, which prevents the
  edge from existing in the first place.

- Multi-scale Frangi: scikit-image's `frangi` filter takes a `sigmas`
  argument controlling the Gaussian scales at which it analyzes the
  Hessian eigenvalues. For 1024x1024 fundus images, sigmas in
  [1, 2, 3, 4, 5] capture both fine capillaries (small sigma) and
  large arcade vessels (large sigma). Tuning notebook lives in
  `notebooks/frangi_tuning.ipynb`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from skimage import exposure, filters
from skimage.morphology import closing, disk


DEFAULT_SIGMAS: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
DEFAULT_CLAHE_CLIP: float = 0.01
# Gamma applied to the normalized Frangi output. Values < 1 compress the
# dynamic range so dim capillaries become more visible relative to large vessels.
# 0.3 chosen from visual sweep on FIVES (variant 81_default_post_gamma_0.3).
DEFAULT_OUTPUT_GAMMA: float = 0.3


# ---------------------------------------------------------------------------
# Channel extraction and inversion
# ---------------------------------------------------------------------------

def extract_green_channel(rgb_image: np.ndarray) -> np.ndarray:
    """
    Extract the green channel from an RGB image and convert to float [0, 1].

    Parameters
    ----------
    rgb_image : np.ndarray
        Shape (H, W, 3), uint8 (0-255) or float (0-1).

    Returns
    -------
    np.ndarray
        Shape (H, W), float32 in [0, 1].
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(
            f"Expected (H, W, 3) RGB image, got shape {rgb_image.shape}"
        )
    green = rgb_image[:, :, 1].astype(np.float32)

    # Normalize to [0, 1] regardless of input range.
    if green.max() > 1.5:
        green /= 255.0
    return green


def invert(image: np.ndarray) -> np.ndarray:
    """
    Invert a [0, 1] image so that dark vessels become bright.

    Parameters
    ----------
    image : np.ndarray
        Float image in [0, 1].

    Returns
    -------
    np.ndarray
        1.0 - image.
    """
    return 1.0 - image


# ---------------------------------------------------------------------------
# CLAHE — local contrast enhancement
# ---------------------------------------------------------------------------

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = DEFAULT_CLAHE_CLIP,
    kernel_size: int | None = None,
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization.

    Parameters
    ----------
    image : np.ndarray
        Float image in [0, 1], shape (H, W).
    clip_limit : float
        Clipping limit for histogram equalization. Lower = more conservative.
        skimage default is 0.01. Higher values give more aggressive
        contrast enhancement but can amplify noise.
    kernel_size : int or None
        Size of the local equalization tiles. If None, skimage uses
        1/8 of the image dimensions, which is sensible for fundus images.

    Returns
    -------
    np.ndarray
        Enhanced image, float32 in [0, 1].
    """
    return exposure.equalize_adapthist(
        image,
        clip_limit=clip_limit,
        kernel_size=kernel_size,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# FOV (field of view) masks
# ---------------------------------------------------------------------------

def compute_fov_mask(
    rgb_image: np.ndarray,
    threshold: float = 0.05,
    closing_radius: int = 5,
) -> np.ndarray:
    """
    Automatically compute a field-of-view mask for a fundus image.

    Many retinal datasets ship FOV masks alongside the images. For datasets
    that don't (or for sanity-checking the provided masks), we compute one
    by thresholding the image intensity. The retina is much brighter than
    the surrounding black border, so a low threshold works reliably.

    Parameters
    ----------
    rgb_image : np.ndarray
        Shape (H, W, 3), uint8 or float.
    threshold : float
        Intensity threshold (after normalization to [0, 1]) below which a
        pixel is considered outside the retina. 0.05 is conservative and
        works for FIVES, DRIVE, STARE, CHASE_DB1.
    closing_radius : int
        Morphological closing radius (in pixels) to fill small gaps in the
        mask. Removes specks of "outside" inside the retinal region.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (H, W). True = inside retina.
    """
    # Use mean of all channels for thresholding — more robust than green
    # alone since the optic disc is bright in all channels.
    img = rgb_image.astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0
    luminance = img.mean(axis=2)

    fov = luminance > threshold

    # Morphological closing to fill tiny gaps inside the retinal disc.
    if closing_radius > 0:
        fov = closing(fov, footprint=disk(closing_radius))

    return fov.astype(bool)


def apply_fov_mask(image: np.ndarray, fov_mask: np.ndarray) -> np.ndarray:
    """
    Zero out pixels outside the FOV. Works on (H, W) or (H, W, C) images.
    """
    if image.ndim == 2:
        return image * fov_mask
    return image * fov_mask[:, :, None]


# ---------------------------------------------------------------------------
# Multi-scale Frangi filter
# ---------------------------------------------------------------------------

def compute_frangi(
    image: np.ndarray,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
    black_ridges: bool = False,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float | None = None,
    output_gamma: float = DEFAULT_OUTPUT_GAMMA,
) -> np.ndarray:
    """
    Run multi-scale Frangi vesselness on a single-channel image.

    Parameters
    ----------
    image : np.ndarray
        Single-channel float image, shape (H, W).
    sigmas : sequence of float
        Gaussian scales. (1–5) covers capillaries through large arcades at 1024².
    black_ridges : bool
        False = detect bright ridges (use after inversion).
    alpha, beta, gamma : float
        Standard Frangi parameters. beta suppresses blob-like responses
        (higher = more suppression of hemorrhage-like structures).
    output_gamma : float
        Power applied to the normalized response before returning. Values < 1
        compress the dynamic range so dim capillaries become more visible
        relative to bright large vessels. 0.3 chosen from visual sweep.

    Returns
    -------
    np.ndarray
        Vesselness response, float32 in [0, 1], shape (H, W).
    """
    response = filters.frangi(
        image,
        sigmas=sigmas,
        black_ridges=black_ridges,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    response = np.clip(response, 0.0, None)
    max_val = response.max()
    if max_val > 0:
        response = response / max_val

    if output_gamma != 1.0:
        response = np.power(response, output_gamma)

    return response.astype(np.float32)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def fundus_to_frangi(
    rgb_image: np.ndarray,
    fov_mask: np.ndarray | None = None,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
    clahe_clip: float = DEFAULT_CLAHE_CLIP,
) -> np.ndarray:
    """
    Full preprocessing + Frangi pipeline for a fundus image.

    Use this from `cache_frangi.py` to convert each raw fundus image into
    its vesselness response (cached as .npy and fed to the adapter at training time).

    Parameters
    ----------
    rgb_image : np.ndarray
        Shape (H, W, 3), the resized 1024x1024 RGB image.
    fov_mask : np.ndarray or None
        Boolean mask of shape (H, W). If None, will be auto-computed.
    sigmas : sequence of float
        Frangi scales.
    clahe_clip : float
        CLAHE clip limit.

    Returns
    -------
    np.ndarray
        Vesselness response, float32 in [0, 1], shape (H, W).
    """
    # 1. Compute FOV mask if not provided.
    if fov_mask is None:
        fov_mask = compute_fov_mask(rgb_image)

    # 2. Extract green channel and mask BEFORE inversion.
    #    Outside the FOV is black (0) in the raw image. If we invert first,
    #    the exterior becomes 1.0 (white). CLAHE then sees a massive intensity
    #    contrast at the boundary and enhances it into a sharp ring that Frangi
    #    detects as a "vessel." Masking first keeps the exterior at 0 through
    #    inversion (0 stays 0 after the FOV mask re-application below).
    green = extract_green_channel(rgb_image)
    green = apply_fov_mask(green, fov_mask)

    # 3. Invert so vessels become bright. Outside is 1.0 here, but step 4
    #    will kill it before CLAHE ever sees it.
    inverted = invert(green)
    inverted = apply_fov_mask(inverted, fov_mask)  # outside back to 0

    # 4. CLAHE for local contrast — now sees 0 outside, no boundary spike.
    enhanced = apply_clahe(inverted, clip_limit=clahe_clip)
    enhanced = apply_fov_mask(enhanced, fov_mask)  # belt-and-suspenders

    # 5. Run multi-scale Frangi vesselness filter.
    frangi_response = compute_frangi(enhanced, sigmas=sigmas, black_ridges=False)

    # 6. Final mask to kill any residual boundary response.
    frangi_response = apply_fov_mask(frangi_response, fov_mask)

    return frangi_response


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick check that the pipeline runs without errors on a synthetic image.
    Run with:
        python -m src.frangi
    """
    print("Generating synthetic fundus-like image ...")
    H, W = 1024, 1024
    rng = np.random.default_rng(0)

    # Synthetic "retina": a bright circle with a few dark linear streaks
    # to simulate vessels.
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H // 2, W // 2
    radius = min(H, W) // 2 - 20
    retina_mask = (yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2

    img = np.zeros((H, W, 3), dtype=np.float32)
    img[..., 0] = 0.6 * retina_mask  # red dominant
    img[..., 1] = 0.4 * retina_mask  # green
    img[..., 2] = 0.1 * retina_mask  # blue

    # Add some "vessels" (dark lines) to the green channel.
    for _ in range(10):
        y0, x0 = rng.integers(100, H - 100), rng.integers(100, W - 100)
        y1, x1 = y0 + rng.integers(-200, 200), x0 + rng.integers(-200, 200)
        n_pts = 200
        ys = np.linspace(y0, y1, n_pts).astype(int).clip(0, H - 1)
        xs = np.linspace(x0, x1, n_pts).astype(int).clip(0, W - 1)
        img[ys, xs, :] *= 0.3

    img = (img * 255).astype(np.uint8)
    print(f"  image shape: {img.shape}, dtype: {img.dtype}")

    print("Running fundus_to_frangi ...")
    frangi_out = fundus_to_frangi(img)
    print(f"  frangi shape: {frangi_out.shape}, dtype: {frangi_out.dtype}")
    print(f"  frangi range: [{frangi_out.min():.3f}, {frangi_out.max():.3f}]")
    assert frangi_out.shape == (H, W)
    assert frangi_out.dtype == np.float32
    assert 0.0 <= frangi_out.min() and frangi_out.max() <= 1.0
    print("\n[OK] frangi pipeline runs without errors.")
