"""
losses.py
---------
Loss functions for LPG-SAM training.

We use a composite loss with three terms:

    L = lambda_dice * DiceLoss(pred, gt)
      + lambda_bce  * BCEWithLogitsLoss(pred, gt)
      + lambda_cldice * (1 - SoftClDice(pred, gt))

Why three terms?
----------------
- BCE: per-pixel classification signal. Strong gradient everywhere,
  but blind to topology. On its own, BCE-trained models love to
  predict the dominant class (background) and underestimate vessels.

- Dice: region-overlap signal that handles class imbalance gracefully.
  Vessels are <10% of pixels in fundus images, so a per-pixel loss
  alone biases the model toward predicting "no vessel everywhere."
  Dice fixes this by measuring overlap relative to set size, not
  raw pixel count. Standard in medical segmentation since ~2016.

- soft clDice: differentiable proxy for centerline Dice (Shit et al.
  CVPR 2021). Dice overlap can be high while topology is wrong --
  e.g. a model that predicts thick disconnected blobs has good Dice
  but terrible vessel connectivity. Soft clDice rewards getting the
  SKELETON right, not just the area, which is exactly the topological
  fidelity we care about.

The composite loss is what makes LPG-SAM care about both pixel accuracy
AND vessel connectivity. The Frangi prior gives the network a hint about
where tubular structures are; the soft-clDice term gives it an explicit
gradient toward producing tubular outputs.

All losses operate on LOGITS, not probabilities. Sigmoid happens inside
each loss function. This is the standard convention and avoids numerical
issues with `log(0)` from saturated sigmoids.

Tensor conventions
------------------
Throughout this file:
    pred_logits : (B, 1, H, W) raw logits from the SAM decoder
    target      : (B, 1, H, W) binary {0, 1} ground truth
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dice loss
# ---------------------------------------------------------------------------

def dice_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Soft Dice loss on binary segmentation logits.

    Dice coefficient:
        D = 2 * |pred ∩ gt| / (|pred| + |gt|)
    Loss:
        L = 1 - D

    Parameters
    ----------
    pred_logits : torch.Tensor
        Shape (B, 1, H, W). Raw logits.
    target : torch.Tensor
        Shape (B, 1, H, W). Binary {0, 1}.
    smooth : float
        Laplace smoothing in numerator and denominator. Stabilizes the
        loss when both pred and gt are mostly empty (avoids 0/0 -> nan).
    eps : float
        Small constant added to denominator only, as a final safety net.

    Returns
    -------
    torch.Tensor
        Scalar loss in [0, 1]. Lower is better.
    """
    pred = torch.sigmoid(pred_logits)
    # Per-image Dice, then average over batch. Sum over spatial dims only.
    dims = (1, 2, 3)  # (C, H, W)
    intersection = (pred * target).sum(dim=dims)
    cardinality = pred.sum(dim=dims) + target.sum(dim=dims)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth + eps)
    return 1.0 - dice.mean()


# ---------------------------------------------------------------------------
# BCE with logits (just a thin wrapper for symmetry)
# ---------------------------------------------------------------------------

def bce_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy with logits. Mean reduction over all pixels.

    Just a thin wrapper around F.binary_cross_entropy_with_logits so it
    has the same call signature as the other losses.
    """
    return F.binary_cross_entropy_with_logits(pred_logits, target, reduction="mean")


# ---------------------------------------------------------------------------
# Soft skeletonization (differentiable proxy for morphological skeleton)
# ---------------------------------------------------------------------------

def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable approximation of binary erosion.

    Erosion of a binary image keeps only pixels whose entire local
    neighborhood is also "on". For continuous values in [0, 1], we
    approximate this with min-pooling over a small neighborhood.
    Min-pooling is implemented via -max_pool(-x), which is itself
    differentiable.

    We use a 3x3 cross-shaped neighborhood (the union of a 3x1 vertical
    and a 1x3 horizontal pool), which is the original soft-clDice
    formulation from Shit et al. and gives smoother gradients than a
    full 3x3 square.

    Parameters
    ----------
    x : torch.Tensor
        Shape (B, 1, H, W), values in [0, 1].

    Returns
    -------
    torch.Tensor
        Eroded soft mask, same shape, values in [0, 1].
    """
    p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable approximation of binary dilation.

    Dilation grows on-pixels into their neighborhood. For continuous
    values, max-pooling is the natural soft analog. We use a 3x3 square
    here (vs. the cross used for erosion); this small asymmetry is part
    of the original formulation and works well in practice.

    Parameters
    ----------
    x : torch.Tensor
        Shape (B, 1, H, W), values in [0, 1].
    """
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    """Soft morphological opening: erode then dilate."""
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x: torch.Tensor, num_iter: int = 10) -> torch.Tensor:
    """
    Differentiable soft skeletonization.

    Iteratively peels off the outer layer of the foreground while
    preserving the centerline. Each iteration:
        1. Compute the "opening residual": x - open(x). This is the
           outermost layer of the current foreground.
        2. Add it to the running skeleton.
        3. Erode x for the next iteration.

    After `num_iter` iterations, the running skeleton approximates the
    centerline of the original foreground. Larger num_iter handles
    thicker structures but costs more compute. For retinal vessels at
    1024x1024 with vessel thicknesses of 1-15 pixels, num_iter=10 is
    a good default.

    Parameters
    ----------
    x : torch.Tensor
        Shape (B, 1, H, W), values in [0, 1]. Soft binary mask.
    num_iter : int
        Number of erosion iterations. 10 is a sensible default.

    Returns
    -------
    torch.Tensor
        Soft skeleton, same shape, values in [0, 1].

    Reference
    ---------
    Shit et al. "clDice -- A Novel Topology-Preserving Loss Function for
    Tubular Structure Segmentation", CVPR 2021.
    """
    skel = F.relu(x - _soft_open(x))
    for _ in range(num_iter):
        x = _soft_erode(x)
        opened = _soft_open(x)
        delta = F.relu(x - opened)
        # Standard formulation: skel + delta - skel*delta (soft union)
        skel = skel + F.relu(delta - skel * delta)
    return skel


# ---------------------------------------------------------------------------
# Soft clDice loss
# ---------------------------------------------------------------------------

def soft_cldice_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    num_iter: int = 10,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Differentiable centerline Dice loss.

    soft clDice has two halves:

        T_prec = |skel(pred) ∩ target| / |skel(pred)|
            "what fraction of the predicted skeleton lies inside GT?"

        T_sens = |skel(target) ∩ pred| / |skel(target)|
            "what fraction of the GT skeleton lies inside the prediction?"

    soft clDice score:
        clD = 2 * (T_prec * T_sens) / (T_prec + T_sens)

    Loss:
        L = 1 - clD

    The harmonic mean structure is identical to F1 / Dice but applied
    to skeleton-based precision/recall. This rewards predictions that
    are correctly CONNECTED, not just overlapping.

    Parameters
    ----------
    pred_logits : torch.Tensor
        Shape (B, 1, H, W). Raw logits.
    target : torch.Tensor
        Shape (B, 1, H, W). Binary {0, 1}.
    num_iter : int
        Number of soft skeletonization iterations.
    smooth : float
        Laplace smoothing.

    Returns
    -------
    torch.Tensor
        Scalar loss in [0, 1]. Lower = better topology match.
    """
    pred = torch.sigmoid(pred_logits)

    # Soft skeletons of both masks.
    pred_skel = soft_skeletonize(pred, num_iter=num_iter)
    target_skel = soft_skeletonize(target, num_iter=num_iter)

    # Topological precision: how much of pred_skel is covered by target.
    t_prec = (
        (pred_skel * target).sum(dim=(1, 2, 3)) + smooth
    ) / (pred_skel.sum(dim=(1, 2, 3)) + smooth)

    # Topological sensitivity: how much of target_skel is covered by pred.
    t_sens = (
        (target_skel * pred).sum(dim=(1, 2, 3)) + smooth
    ) / (target_skel.sum(dim=(1, 2, 3)) + smooth)

    cldice = 2.0 * (t_prec * t_sens) / (t_prec + t_sens + 1e-7)
    return 1.0 - cldice.mean()


# ---------------------------------------------------------------------------
# Composite loss module
# ---------------------------------------------------------------------------

class CompositeLoss(nn.Module):
    """
    Weighted sum of Dice + BCE + soft-clDice.

    Defaults follow the original soft-clDice paper (Shit et al. 2021):
        lambda_dice  = 1.0
        lambda_bce   = 0.0   (set > 0 if you want pixel-level signal too)
        lambda_cldice = 0.3

    For LPG-SAM I'd start with all three on:
        lambda_dice  = 0.5
        lambda_bce   = 0.5
        lambda_cldice = 0.3
    and tune lambda_cldice in {0.1, 0.3, 0.5} as an ablation.

    The forward pass returns BOTH the total loss and a dict of the
    individual terms, which the trainer logs separately so we can see
    which term is doing what.

    Parameters
    ----------
    lambda_dice : float
    lambda_bce : float
    lambda_cldice : float
    cldice_iter : int
        Soft skeletonization iterations. Higher = handles thicker
        structures, more compute.
    """

    def __init__(
        self,
        lambda_dice: float = 0.5,
        lambda_bce: float = 0.5,
        lambda_cldice: float = 0.3,
        cldice_iter: int = 10,
    ) -> None:
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.lambda_cldice = lambda_cldice
        self.cldice_iter = cldice_iter

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Returns
        -------
        total : torch.Tensor
            Scalar weighted sum of all enabled terms.
        components : dict[str, torch.Tensor]
            {"dice": ..., "bce": ..., "cldice": ..., "total": ...}
            All scalars on the same device as `pred_logits`.
            Useful for logging individual terms during training.
        """
        device = pred_logits.device
        zero = torch.zeros((), device=device)

        dice_term = (
            dice_loss(pred_logits, target)
            if self.lambda_dice > 0
            else zero
        )
        bce_term = (
            bce_loss(pred_logits, target)
            if self.lambda_bce > 0
            else zero
        )
        cldice_term = (
            soft_cldice_loss(pred_logits, target, num_iter=self.cldice_iter)
            if self.lambda_cldice > 0
            else zero
        )

        total = (
            self.lambda_dice * dice_term
            + self.lambda_bce * bce_term
            + self.lambda_cldice * cldice_term
        )

        return total, {
            "dice": dice_term.detach(),
            "bce": bce_term.detach(),
            "cldice": cldice_term.detach(),
            "total": total.detach(),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Verify each loss runs, returns the right shape, and has non-zero
    gradients on a toy input. Run with:
        python -m src.losses
    """
    from src.utils import get_device, device_info, seed_everything

    seed_everything(42)
    device = get_device()
    print(f"Device: {device_info(device)}\n")

    B, H, W = 2, 256, 256

    # Random logits + a synthetic GT mask shaped like a few thin lines.
    pred_logits = torch.randn(B, 1, H, W, device=device, requires_grad=True)
    target = torch.zeros(B, 1, H, W, device=device)
    # Draw a few diagonal "vessels" in each batch item.
    for b in range(B):
        for offset in (-30, 0, 30):
            for i in range(H):
                j = i + offset
                if 0 <= j < W:
                    target[b, 0, i, j] = 1.0
                    if j + 1 < W:
                        target[b, 0, i, j + 1] = 1.0  # 2-pixel-thick lines

    # ----- Individual losses -----
    print("[1/4] dice_loss ...")
    d = dice_loss(pred_logits, target)
    assert d.ndim == 0, "dice_loss should return a scalar"
    assert 0.0 <= d.item() <= 1.0
    print(f"      ok: dice = {d.item():.4f}")

    print("[2/4] bce_loss ...")
    b = bce_loss(pred_logits, target)
    assert b.ndim == 0
    print(f"      ok: bce  = {b.item():.4f}")

    print("[3/4] soft_cldice_loss ...")
    c = soft_cldice_loss(pred_logits, target, num_iter=5)
    assert c.ndim == 0
    assert 0.0 <= c.item() <= 1.0
    print(f"      ok: cldice = {c.item():.4f}")

    # ----- Composite loss + gradient check -----
    print("[4/4] CompositeLoss + gradient flow ...")
    loss_fn = CompositeLoss(
        lambda_dice=0.5, lambda_bce=0.5, lambda_cldice=0.3, cldice_iter=5
    ).to(device)
    total, components = loss_fn(pred_logits, target)
    assert total.ndim == 0
    assert "dice" in components and "bce" in components and "cldice" in components
    print(f"      total = {total.item():.4f}")
    print(f"      components: dice={components['dice'].item():.4f} "
          f"bce={components['bce'].item():.4f} "
          f"cldice={components['cldice'].item():.4f}")

    total.backward()
    assert pred_logits.grad is not None
    assert pred_logits.grad.abs().max().item() > 0, "no gradient on pred_logits"
    print(f"      ok: max |grad| = {pred_logits.grad.abs().max().item():.4e}")

    # ----- Sanity: perfect prediction should give ~zero loss -----
    print("\n[bonus] perfect-prediction sanity check ...")
    perfect_logits = (target * 20.0 - 10.0)  # large positive where GT=1, large negative where GT=0
    with torch.no_grad():
        d_perfect = dice_loss(perfect_logits, target).item()
        c_perfect = soft_cldice_loss(perfect_logits, target, num_iter=5).item()
    print(f"      dice loss (perfect)   = {d_perfect:.4e}")
    print(f"      cldice loss (perfect) = {c_perfect:.4e}")
    assert d_perfect < 0.01, "dice loss should be ~0 on perfect prediction"
    assert c_perfect < 0.05, "cldice loss should be near 0 on perfect prediction"
    print("      ok")

    print("\n[OK] losses.py is working correctly.")
