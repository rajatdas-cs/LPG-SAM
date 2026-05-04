"""
metrics.py
----------
Evaluation metrics for LPG-SAM. These are HARD (non-differentiable)
metrics, computed on binarized predictions for reporting numbers in
the final results table.

For training-time loss functions, see losses.py.

Metrics implemented
-------------------
- Dice coefficient (overlap)
- IoU / Jaccard (overlap, more conservative than Dice)
- clDice  (centerline Dice -- the headline topology metric)
- Betti-0 error  (number of connected components -- proxy for over/under-segmentation)

We do NOT implement HD95 (Hausdorff distance) here because (a) it's
slow on 1024x1024 binary masks and (b) it's less informative than
clDice for tubular structures. Add it later if a reviewer asks.

All metric functions take and return CPU numpy arrays. The trainer's
eval loop is expected to convert tensors to numpy with `.cpu().numpy()`
before calling these. This keeps metric code separate from torch and
makes it trivially testable.

Tensor / array conventions
--------------------------
    pred : (H, W) bool or {0, 1} int -- binarized prediction
    gt   : (H, W) bool or {0, 1} int -- ground truth
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label as cc_label


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def to_bool(x: np.ndarray) -> np.ndarray:
    """Coerce to boolean. Accepts {0, 1} int, {0, 255} uint8, or float in [0, 1]."""
    if x.dtype == bool:
        return x
    if np.issubdtype(x.dtype, np.floating):
        return x > 0.5
    return x > 0


def logits_to_binary(logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert raw logits to a binary mask.

    Parameters
    ----------
    logits : np.ndarray
        Raw logits, any shape. Typically (H, W) or (1, H, W).
    threshold : float
        Probability threshold (after sigmoid). Default 0.5.

    Returns
    -------
    np.ndarray
        Boolean mask, same shape minus any leading singleton channel dim.
    """
    # Drop a leading channel dim if present.
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs > threshold


# ---------------------------------------------------------------------------
# Region overlap metrics
# ---------------------------------------------------------------------------

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Hard Dice coefficient.

        D = 2 * |pred ∩ gt| / (|pred| + |gt|)

    Returns 1.0 if both pred and gt are completely empty (vacuous case).
    Returns 0.0 if exactly one of them is empty.
    """
    p = to_bool(pred)
    g = to_bool(gt)

    p_sum = int(p.sum())
    g_sum = int(g.sum())
    if p_sum == 0 and g_sum == 0:
        return 1.0
    if p_sum == 0 or g_sum == 0:
        return 0.0

    inter = int(np.logical_and(p, g).sum())
    return (2.0 * inter) / (p_sum + g_sum)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Intersection over Union (Jaccard index).

        IoU = |pred ∩ gt| / |pred ∪ gt|

    Always <= Dice. The two carry similar information but IoU is more
    sensitive to small errors. Both are reported by convention.
    """
    p = to_bool(pred)
    g = to_bool(gt)

    p_sum = int(p.sum())
    g_sum = int(g.sum())
    if p_sum == 0 and g_sum == 0:
        return 1.0
    if p_sum == 0 or g_sum == 0:
        return 0.0

    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    return inter / union


# ---------------------------------------------------------------------------
# Centerline Dice (clDice) -- the headline topology metric
# ---------------------------------------------------------------------------

def cldice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Hard centerline Dice score.

    Defined as the harmonic mean of:
        T_prec = |skel(pred) ∩ gt|    / |skel(pred)|
        T_sens = |skel(gt)   ∩ pred|  / |skel(gt)|

        clDice = 2 * (T_prec * T_sens) / (T_prec + T_sens)

    This is the metric the project optimizes for. It rewards correct
    vessel CONNECTIVITY rather than just region overlap.

    Uses scikit-image's `skeletonize` (Lee et al. 1994), which is the
    standard hard skeletonization. The training-time differentiable
    version lives in losses.py and is only an approximation; this is
    the real thing.

    Returns 1.0 if both masks are empty.

    Reference
    ---------
    Shit et al. "clDice -- A Novel Topology-Preserving Loss Function
    for Tubular Structure Segmentation", CVPR 2021.
    """
    p = to_bool(pred)
    g = to_bool(gt)

    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0

    pred_skel = skeletonize(p)
    gt_skel = skeletonize(g)

    pred_skel_sum = int(pred_skel.sum())
    gt_skel_sum = int(gt_skel.sum())

    # Edge case: a tiny mask whose skeleton is empty after thinning.
    # Treat that as "not enough structure to score" -> 0.
    if pred_skel_sum == 0 or gt_skel_sum == 0:
        return 0.0

    t_prec = int(np.logical_and(pred_skel, g).sum()) / pred_skel_sum
    t_sens = int(np.logical_and(gt_skel, p).sum()) / gt_skel_sum

    if t_prec + t_sens == 0:
        return 0.0
    return (2.0 * t_prec * t_sens) / (t_prec + t_sens)


# ---------------------------------------------------------------------------
# Betti-0 error (connected components mismatch)
# ---------------------------------------------------------------------------

def betti0_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Absolute difference in number of connected components.

        b0_error = | #components(pred) - #components(gt) |

    Betti-0 counts connected components (the 0-th Betti number from
    algebraic topology). For vessel segmentation:

        - Too FEW components in pred -> over-merging (vessels glued
          to hemorrhages or to each other across gaps).
        - Too MANY components in pred -> over-fragmentation (vessels
          broken into disconnected pieces).

    Either failure mode hurts downstream clinical use. clDice penalizes
    both somewhat indirectly via the skeleton overlap; Betti-0 error
    measures it directly.

    Note: this is a simplified topology metric. The "real" thing would
    use persistent homology (Betti-1 for loops too), but for retinal
    vessels at this resolution Betti-0 alone is informative enough and
    avoids a dependency on `gudhi`.

    Returns
    -------
    float
        Non-negative integer (returned as float for averaging).
    """
    p = to_bool(pred)
    g = to_bool(gt)

    # 8-connectivity is appropriate for vessels (diagonal pixels
    # belonging to the same vessel should not be split into separate
    # components).
    _, n_p = cc_label(p, connectivity=2, return_num=True)
    _, n_g = cc_label(g, connectivity=2, return_num=True)
    return float(abs(n_p - n_g))


# ---------------------------------------------------------------------------
# Eval-loop accumulator
# ---------------------------------------------------------------------------

@dataclass
class MetricAccumulator:
    """
    Running accumulator for per-image metrics over an eval pass.

    Usage
    -----
    >>> acc = MetricAccumulator()
    >>> for pred, gt in eval_loader:
    ...     acc.update(pred, gt)
    >>> print(acc.summary())   # dict of mean values

    Tracks Dice, IoU, clDice, and Betti-0 error. All metrics are
    computed per-image and averaged at the end. Per-image (rather
    than micro-averaged over all pixels) is the standard convention
    for medical segmentation reporting.

    The accumulator also keeps per-image scores in `self.per_image`
    so the eval script can dump them to a CSV for inspection.
    """

    dice: list[float] = field(default_factory=list)
    iou: list[float] = field(default_factory=list)
    cldice: list[float] = field(default_factory=list)
    betti0: list[float] = field(default_factory=list)
    names: list[str] = field(default_factory=list)

    def update(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        name: str | None = None,
    ) -> None:
        """
        Score one (pred, gt) pair and append the results.

        Parameters
        ----------
        pred : np.ndarray
            Binary or float prediction, shape (H, W). If shape is
            (1, H, W) the channel dim is stripped automatically.
        gt : np.ndarray
            Binary GT, same shape as pred.
        name : str or None
            Optional sample name (e.g. file stem). Stored alongside
            the scores so per-image CSVs are easy to write.
        """
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        if gt.ndim == 3 and gt.shape[0] == 1:
            gt = gt[0]

        self.dice.append(dice_score(pred, gt))
        self.iou.append(iou_score(pred, gt))
        self.cldice.append(cldice_score(pred, gt))
        self.betti0.append(betti0_error(pred, gt))
        self.names.append(name if name is not None else f"sample_{len(self.dice)}")

    def summary(self) -> dict[str, float]:
        """Return mean of each metric across all accumulated samples."""
        if not self.dice:
            return {"dice": 0.0, "iou": 0.0, "cldice": 0.0, "betti0_error": 0.0, "n": 0}
        return {
            "dice": float(np.mean(self.dice)),
            "iou": float(np.mean(self.iou)),
            "cldice": float(np.mean(self.cldice)),
            "betti0_error": float(np.mean(self.betti0)),
            "n": len(self.dice),
        }

    def per_image_table(self) -> list[dict]:
        """
        Return a list of per-image score dicts, one per accumulated sample.
        Useful for writing a CSV in the eval script.
        """
        return [
            {
                "name": self.names[i],
                "dice": self.dice[i],
                "iou": self.iou[i],
                "cldice": self.cldice[i],
                "betti0_error": self.betti0[i],
            }
            for i in range(len(self.dice))
        ]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Verify each metric on synthetic cases with known answers.
    Run with:
        python -m src.metrics
    """
    print("[1/5] Perfect prediction should give Dice=1, IoU=1, clDice=1, Betti=0 ...")
    H, W = 256, 256
    gt = np.zeros((H, W), dtype=bool)
    # Draw a single horizontal line as our "vessel"
    gt[128, 50:200] = True
    gt[129, 50:200] = True  # 2-pixel-thick

    perfect = gt.copy()
    assert abs(dice_score(perfect, gt) - 1.0) < 1e-6
    assert abs(iou_score(perfect, gt) - 1.0) < 1e-6
    assert abs(cldice_score(perfect, gt) - 1.0) < 1e-6
    assert betti0_error(perfect, gt) == 0.0
    print("      ok")

    print("[2/5] Empty pred + non-empty gt should give all zero ...")
    empty = np.zeros_like(gt)
    assert dice_score(empty, gt) == 0.0
    assert iou_score(empty, gt) == 0.0
    assert cldice_score(empty, gt) == 0.0
    print("      ok")

    print("[3/5] Empty pred + empty gt should give vacuous 1 ...")
    assert dice_score(empty, empty) == 1.0
    assert iou_score(empty, empty) == 1.0
    assert cldice_score(empty, empty) == 1.0
    print("      ok")

    print("[4/5] Disconnected prediction should give Betti-0 error > 0 ...")
    # gt has 1 connected component. Pred has 3 disjoint blobs.
    pred_disconnected = np.zeros_like(gt)
    pred_disconnected[20:30, 20:30] = True
    pred_disconnected[100:110, 100:110] = True
    pred_disconnected[200:210, 200:210] = True
    err = betti0_error(pred_disconnected, gt)
    assert err == 2.0, f"expected |3 - 1| = 2, got {err}"
    print(f"      ok: betti0_error = {err}")

    print("[5/5] MetricAccumulator over a few samples ...")
    acc = MetricAccumulator()
    acc.update(perfect, gt, name="perfect")
    acc.update(empty, gt, name="empty")
    acc.update(pred_disconnected, gt, name="disconnected")
    summary = acc.summary()
    print(f"      n          = {summary['n']}")
    print(f"      mean dice  = {summary['dice']:.4f}")
    print(f"      mean iou   = {summary['iou']:.4f}")
    print(f"      mean cld   = {summary['cldice']:.4f}")
    print(f"      mean betti = {summary['betti0_error']:.4f}")
    assert summary["n"] == 3
    assert 0.0 < summary["dice"] < 1.0   # mix of perfect and bad
    print("      ok")

    print("\n[OK] metrics.py is working correctly.")
