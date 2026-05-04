"""
scripts/eval.py
---------------
Evaluate a trained LPG-SAM checkpoint on one or more datasets.

Run with:
    # FIVES test split (in-distribution)
    python -m scripts.eval \\
        --config configs/config.yaml \\
        --checkpoint checkpoints/best.pt \\
        --dataset-root data/processed/FIVES \\
        --split-file data/processed/FIVES/test.txt \\
        --tag fives_test

    # Cross-dataset (run separately for each)
    python -m scripts.eval \\
        --config configs/config.yaml \\
        --checkpoint checkpoints/best.pt \\
        --dataset-root data/processed/DRIVE \\
        --split-file data/processed/DRIVE/all.txt \\
        --tag drive_xdataset

For cross-dataset evals to work, you must first run prepare_data.py and
cache_frangi.py on each dataset using the SAME 1024x1024 resize. Tuning
the model hyperparameters using cross-dataset numbers defeats the
generalization claim -- only run cross-dataset eval ONCE at the end.

Outputs
-------
A folder under <log_dir>/eval_<tag>_<timestamp>/ containing:
    - metrics.json     : aggregate Dice/IoU/clDice/Betti-0 over the split
    - per_image.csv    : per-sample scores (sorted by clDice ascending,
                         so worst cases are at the top -- useful for
                         qualitative inspection)
    - overlays/        : optional PNG visualizations (configured by
                         eval.save_overlays in config)
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from skimage.morphology import binary_closing, disk
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.adapter import LatentPriorAdapter
from src.dataset import FundusVesselDataset
from src.metrics import MetricAccumulator, logits_to_binary
from src.sam_wrapper import FrozenSAM
from src.utils import device_info, get_device, seed_everything


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(
    ckpt_path: Path,
    adapter: LatentPriorAdapter,
    device: torch.device,
) -> dict[str, Any]:
    """
    Load adapter weights from a training checkpoint.

    Returns the full checkpoint dict for inspection (epoch, val_metrics,
    config). SAM weights are NOT in the checkpoint -- we load those
    separately from the original MedSAM .pth.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Forward pass (same as in train.py)
# ---------------------------------------------------------------------------

def forward_pass(
    sam: FrozenSAM,
    adapter: LatentPriorAdapter,
    image: torch.Tensor,
    frangi: torch.Tensor,
) -> torch.Tensor:
    """Frozen encoder -> adapter -> frozen decoder. Returns (B, 1, 256, 256) logits."""
    z_image = sam.encode_image(image)
    z_mod, sparse_tokens = adapter(z_image, frangi)
    dense_default = sam.zero_dense_prompt(image.shape[0], image.device)
    masks_low_res, _ = sam.decode_masks(
        image_embeddings=z_mod,
        sparse_prompt_embeddings=sparse_tokens,
        dense_prompt_embeddings=dense_default,
        multimask_output=False,
    )
    return masks_low_res


def upsample_to(mask_logits: torch.Tensor, size: int = 1024) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        mask_logits, size=(size, size), mode="bilinear", align_corners=False
    )


# ---------------------------------------------------------------------------
# Overlay visualization
# ---------------------------------------------------------------------------

def save_overlay(
    out_path: Path,
    image: np.ndarray,    # (H, W, 3) float in [0, 1]
    frangi: np.ndarray,   # (H, W) float in [0, 1]
    pred: np.ndarray,     # (H, W) bool
    gt: np.ndarray,       # (H, W) bool
) -> None:
    """
    Save a 5-panel side-by-side visualization to disk.

    Panels: image | frangi | gt | pred | overlay (gt vs pred)
    The overlay panel uses:
        green : true positive (correctly predicted vessel)
        red   : false positive (predicted vessel where there isn't one)
        blue  : false negative (missed vessel)

    Uses matplotlib so you don't need any extra deps.
    """
    import matplotlib.pyplot as plt

    H, W = gt.shape
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(image)
    axes[0].set_title("image")

    axes[1].imshow(frangi, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("frangi")

    axes[2].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("ground truth")

    axes[3].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("prediction")

    # Color-coded overlay
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    overlay[..., 1] = np.logical_and(pred, gt).astype(np.float32)        # G: TP
    overlay[..., 0] = np.logical_and(pred, ~gt).astype(np.float32)       # R: FP
    overlay[..., 2] = np.logical_and(~pred, gt).astype(np.float32)       # B: FN
    axes[4].imshow(overlay)
    axes[4].set_title("TP=g  FP=r  FN=b")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    sam: FrozenSAM,
    adapter: LatentPriorAdapter,
    loader: DataLoader,
    device: torch.device,
    binarize_threshold: float,
    overlay_dir: Path | None,
    n_overlays: int,
    closing_radius: int = 0,
) -> MetricAccumulator:
    """
    Run the model over a DataLoader and accumulate metrics.

    Saves up to `n_overlays` prediction overlays into `overlay_dir`.

    Parameters
    ----------
    closing_radius : int
        If > 0, apply morphological closing with this disk radius (pixels)
        to each binarized prediction before scoring. Bridges small gaps
        that cause over-fragmentation without retraining.
    """
    adapter.eval()
    acc = MetricAccumulator()
    n_saved_overlays = 0

    bar = tqdm(loader, desc="evaluating", ncols=100)
    for batch in bar:
        image = batch["image"].to(device, non_blocking=True)
        frangi_t = batch["frangi"].to(device, non_blocking=True)
        target = batch["mask"]
        names = batch["name"]

        mask_logits_low = forward_pass(sam, adapter, image, frangi_t)
        mask_logits = upsample_to(mask_logits_low, size=1024)
        logits_np = mask_logits.detach().cpu().numpy()

        for b in range(logits_np.shape[0]):
            pred_bin = logits_to_binary(logits_np[b], threshold=binarize_threshold)
            if closing_radius > 0:
                pred_bin = binary_closing(pred_bin, footprint=disk(closing_radius))
            gt_bin = target[b, 0].numpy() > 0.5
            acc.update(pred_bin, gt_bin, name=names[b])

            # Save overlays for the first n samples (so we have something
            # to show in figures and qualitative inspection).
            if overlay_dir is not None and n_saved_overlays < n_overlays:
                img_np = image[b].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
                fr_np = frangi_t[b, 0].cpu().numpy()
                save_overlay(
                    out_path=overlay_dir / f"{names[b]}.png",
                    image=img_np,
                    frangi=fr_np,
                    pred=pred_bin,
                    gt=gt_bin,
                )
                n_saved_overlays += 1

        bar.set_postfix(
            cld=f"{(sum(acc.cldice) / max(len(acc.cldice), 1)):.3f}",
            dice=f"{(sum(acc.dice) / max(len(acc.dice), 1)):.3f}",
        )
    bar.close()
    return acc


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_per_image_csv(path: Path, rows: list[dict]) -> None:
    """
    Write per-image scores to CSV, sorted by clDice ascending.

    Sorting worst-first makes it easy to inspect failure cases: open
    the CSV in a spreadsheet and the top entries are the images the
    model struggled with most. Pair with the saved overlays to see why.
    """
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["cldice"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LPG-SAM on a dataset split.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Adapter checkpoint .pt produced by train.py")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Processed dataset root (must have images_1024/, "
                             "masks_1024/, frangi_cache/).")
    parser.add_argument("--split-file", type=Path, required=True,
                        help="Text file listing image stems for this eval.")
    parser.add_argument("--tag", type=str, required=True,
                        help="Short label used in output folder name.")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["train"]["seed"])
    device = get_device()

    print("=" * 70)
    print(" LPG-SAM evaluation")
    print("=" * 70)
    print(f"Config:        {args.config}")
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Dataset root:  {args.dataset_root}")
    print(f"Split:         {args.split_file}")
    print(f"Device:        {device_info(device)}")

    # ---------------------------------------------------------
    # Build output dir
    # ---------------------------------------------------------
    log_root = Path(cfg["paths"]["log_dir"])
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = log_root / f"eval_{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays" if cfg["eval"]["save_overlays"] > 0 else None
    print(f"Output dir:    {out_dir}\n")

    # ---------------------------------------------------------
    # Build model + load weights
    # ---------------------------------------------------------
    print("[1/3] Loading frozen MedSAM ...")
    sam = FrozenSAM(checkpoint_path=cfg["paths"]["sam_checkpoint"]).to(device)

    print("[2/3] Loading adapter checkpoint ...")
    adapter = LatentPriorAdapter(num_tokens=cfg["model"]["num_tokens"]).to(device)
    ckpt = load_checkpoint(args.checkpoint, adapter, device)
    print(f"      checkpoint epoch: {ckpt.get('epoch', '?')}")
    print(f"      checkpoint val:   {ckpt.get('val_metrics', {})}")
    print(f"      adapter alpha:    {adapter.alpha.item():+.4f}")

    # ---------------------------------------------------------
    # Build dataset
    # ---------------------------------------------------------
    print("[3/3] Building dataset ...")
    ds = FundusVesselDataset(
        data_root=args.dataset_root,
        split_file=args.split_file,
        augment=False,
        seed=cfg["train"]["seed"],
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"] and device.type == "cuda",
    )
    print(f"      {len(ds)} samples\n")

    # ---------------------------------------------------------
    # Run evaluation
    # ---------------------------------------------------------
    acc = evaluate(
        sam=sam,
        adapter=adapter,
        loader=loader,
        device=device,
        binarize_threshold=cfg["eval"]["binarize_threshold"],
        overlay_dir=overlay_dir,
        n_overlays=cfg["eval"]["save_overlays"],
        closing_radius=cfg["eval"].get("closing_radius", 0),
    )

    # ---------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------
    summary = acc.summary()
    print("\n" + "=" * 70)
    print(f" Results ({args.tag})")
    print("=" * 70)
    print(f"  n samples:    {summary['n']}")
    print(f"  Dice:         {summary['dice']:.4f}")
    print(f"  IoU:          {summary['iou']:.4f}")
    print(f"  clDice:       {summary['cldice']:.4f}    <- headline metric")
    print(f"  Betti-0 err:  {summary['betti0_error']:.2f}")
    print()

    # metrics.json
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "tag": args.tag,
                "checkpoint": str(args.checkpoint),
                "dataset_root": str(args.dataset_root),
                "split_file": str(args.split_file),
                "summary": summary,
                "checkpoint_epoch": ckpt.get("epoch"),
                "adapter_alpha": float(adapter.alpha.item()),
            },
            f,
            indent=2,
        )
    print(f"  -> {metrics_path}")

    # per_image.csv
    csv_path = out_dir / "per_image.csv"
    write_per_image_csv(csv_path, acc.per_image_table())
    print(f"  -> {csv_path}")

    if overlay_dir is not None:
        n_overlays = len(list(overlay_dir.glob("*.png")))
        print(f"  -> {overlay_dir} ({n_overlays} overlays)")


if __name__ == "__main__":
    main()
