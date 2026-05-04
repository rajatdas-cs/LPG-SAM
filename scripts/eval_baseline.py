"""
scripts/eval_baseline.py
------------------------
Evaluate zero-shot MedSAM (no adapter, no training) on a dataset using
automated prompts.

This is the baseline that LPG-SAM must beat. The comparison is fair
because:
  - MedSAM is fully frozen (same as in LPG-SAM).
  - The prompts are deterministic and code-generated -- no human
    interaction, no manual annotation, fully reproducible.
  - The same images, masks, and metrics are used for both.

Three prompt strategies are supported:

  1. "box" (default): a single bounding box covering the full image.
     This is the primary baseline. It gives MedSAM maximum spatial
     information with zero human knowledge. MedSAM was trained mainly
     with box prompts, so this is in-distribution.

  2. "center": a single foreground point at the image center.
     Weakest baseline -- MedSAM may segment just one structure near
     the center. Useful as a lower bound.

  3. "grid": an NxN grid of foreground points, each run independently,
     results unioned into a single mask. Strongest baseline -- more
     prompts means more coverage. Slow (N^2 forward passes per image).

Run with:
    python -m scripts.eval_baseline \\
        --config configs/config.yaml \\
        --dataset-root data/processed/FIVES \\
        --split-file data/processed/FIVES/test.txt \\
        --prompt-strategy box \\
        --tag fives_baseline_box

For the final results table you should run all three strategies on each
dataset, but the "box" row is the one you compare LPG-SAM against.
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
import torch.nn.functional as F
import yaml
from PIL import Image
from skimage.morphology import binary_closing, disk
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import FundusVesselDataset
from src.metrics import MetricAccumulator, logits_to_binary
from src.sam_wrapper import (
    FrozenSAM,
    SAM_INPUT_SIZE,
    SAM_EMBED_DIM,
    SAM_EMBED_GRID,
    SAM_DECODER_OUTPUT,
)
from src.utils import device_info, get_device, seed_everything


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def make_box_prompt(
    sam: FrozenSAM,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full-image bounding box prompt.

    SAM's PromptEncoder encodes a box as two point tokens (top-left and
    bottom-right corners) plus the learned box-corner embeddings. We
    call the real PromptEncoder here so the decoder receives exactly
    the kind of input it was trained on.

    Returns
    -------
    sparse_embeddings : (B, N, 256)
    dense_embeddings  : (B, 256, 64, 64)
    """
    # Box format: (B, 1, 4) where 4 = [x_min, y_min, x_max, y_max]
    # Full image box: [0, 0, 1023, 1023]
    boxes = torch.tensor(
        [[[0.0, 0.0, SAM_INPUT_SIZE - 1, SAM_INPUT_SIZE - 1]]],
        device=device,
    ).expand(batch_size, -1, -1)  # (B, 1, 4)

    sparse, dense = sam.sam.prompt_encoder(
        points=None,
        boxes=boxes,
        masks=None,
    )
    return sparse, dense


def make_center_point_prompt(
    sam: FrozenSAM,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single foreground point at image center.

    Returns
    -------
    sparse_embeddings : (B, N, 256)
    dense_embeddings  : (B, 256, 64, 64)
    """
    cx = SAM_INPUT_SIZE // 2
    cy = SAM_INPUT_SIZE // 2

    # Points format: (B, N_points, 2) with coordinates in pixel space.
    # Labels: 1 = foreground, 0 = background.
    points = torch.tensor(
        [[[cx, cy]]],
        dtype=torch.float32,
        device=device,
    ).expand(batch_size, -1, -1)  # (B, 1, 2)

    labels = torch.ones(
        batch_size, 1,
        dtype=torch.long,
        device=device,
    )  # (B, 1) all foreground

    sparse, dense = sam.sam.prompt_encoder(
        points=(points, labels),
        boxes=None,
        masks=None,
    )
    return sparse, dense


def make_grid_point_prompts(
    sam: FrozenSAM,
    device: torch.device,
    grid_n: int = 8,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    NxN grid of foreground points. Each point is a SEPARATE prompt
    (SAM runs independently per point). The caller unions the results.

    We return a list of (sparse, dense) pairs, one per grid point.
    Each is batch_size=1 because we process one point at a time per
    image (the outer loop handles batching across images).

    Parameters
    ----------
    grid_n : int
        Number of points along each axis. Default 8 -> 64 total prompts.

    Returns
    -------
    list of (sparse, dense) pairs, length grid_n^2.
    """
    step = SAM_INPUT_SIZE // (grid_n + 1)
    prompts = []

    for i in range(1, grid_n + 1):
        for j in range(1, grid_n + 1):
            x = j * step
            y = i * step
            points = torch.tensor(
                [[[x, y]]],
                dtype=torch.float32,
                device=device,
            )  # (1, 1, 2)
            labels = torch.ones(1, 1, dtype=torch.long, device=device)

            sparse, dense = sam.sam.prompt_encoder(
                points=(points, labels),
                boxes=None,
                masks=None,
            )
            prompts.append((sparse, dense))

    return prompts


# ---------------------------------------------------------------------------
# Baseline forward passes
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_box_or_center(
    sam: FrozenSAM,
    loader: DataLoader,
    device: torch.device,
    prompt_strategy: str,
    binarize_threshold: float,
    closing_radius: int = 0,
) -> MetricAccumulator:
    """
    Evaluate with a single prompt per image (box or center point).
    """
    acc = MetricAccumulator()

    bar = tqdm(loader, desc=f"baseline ({prompt_strategy})", ncols=100)
    for batch in bar:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"]
        names = batch["name"]
        B = image.shape[0]

        # Encode image.
        z_image = sam.encode_image(image)

        # Generate prompt.
        if prompt_strategy == "box":
            sparse, dense = make_box_prompt(sam, B, device)
        elif prompt_strategy == "center":
            sparse, dense = make_center_point_prompt(sam, B, device)
        else:
            raise ValueError(f"Unknown strategy: {prompt_strategy}")

        # Decode.
        mask_logits_low, _ = sam.decode_masks(
            image_embeddings=z_image,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        mask_logits = F.interpolate(
            mask_logits_low,
            size=(SAM_INPUT_SIZE, SAM_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        logits_np = mask_logits.cpu().numpy()
        target_np = target.numpy()
        for b in range(B):
            pred_bin = logits_to_binary(logits_np[b], threshold=binarize_threshold)
            if closing_radius > 0:
                pred_bin = binary_closing(pred_bin, footprint=disk(closing_radius))
            gt_bin = target_np[b, 0] > 0.5
            acc.update(pred_bin, gt_bin, name=names[b])

        bar.set_postfix(
            cld=f"{(sum(acc.cldice) / max(len(acc.cldice), 1)):.3f}",
            dice=f"{(sum(acc.dice) / max(len(acc.dice), 1)):.3f}",
        )
    bar.close()
    return acc


@torch.no_grad()
def eval_grid(
    sam: FrozenSAM,
    loader: DataLoader,
    device: torch.device,
    grid_n: int,
    binarize_threshold: float,
    closing_radius: int = 0,
) -> MetricAccumulator:
    """
    Evaluate with an NxN grid of point prompts, each run independently,
    results unioned per image.

    This is slow (grid_n^2 decoder forward passes per image) but gives
    the strongest possible zero-shot baseline.
    """
    acc = MetricAccumulator()

    # Pre-generate grid prompts (batch_size=1 for each; we process
    # images one at a time for simplicity).
    grid_prompts = make_grid_point_prompts(sam, device, grid_n=grid_n)

    bar = tqdm(loader, desc=f"baseline (grid {grid_n}x{grid_n})", ncols=100)
    for batch in bar:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"]
        names = batch["name"]
        B = image.shape[0]

        z_image = sam.encode_image(image)

        for b in range(B):
            # Run decoder once per grid point and union the masks.
            z_single = z_image[b : b + 1]  # (1, 256, 64, 64)
            union_mask = torch.zeros(
                1, 1, SAM_INPUT_SIZE, SAM_INPUT_SIZE, device=device
            )

            for sparse, dense in grid_prompts:
                mask_low, _ = sam.decode_masks(
                    image_embeddings=z_single,
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                )
                mask_full = F.interpolate(
                    mask_low,
                    size=(SAM_INPUT_SIZE, SAM_INPUT_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
                # Union via max of probabilities. We take the sigmoid
                # first so we're combining probabilities, not logits
                # (max of logits has a different meaning).
                union_mask = torch.max(union_mask, torch.sigmoid(mask_full))

            # Convert unioned probability back to "logits" for the
            # standard binarization path (threshold at 0.5 on probs).
            logits_np = union_mask.cpu().numpy()
            gt_bin = target[b, 0].numpy() > 0.5
            pred_bin = logits_np[0, 0] > binarize_threshold  # already probs
            if closing_radius > 0:
                pred_bin = binary_closing(pred_bin, footprint=disk(closing_radius))
            acc.update(pred_bin, gt_bin, name=names[b])

        bar.set_postfix(
            cld=f"{(sum(acc.cldice) / max(len(acc.cldice), 1)):.3f}",
            dice=f"{(sum(acc.dice) / max(len(acc.dice), 1)):.3f}",
        )
    bar.close()
    return acc


# ---------------------------------------------------------------------------
# Output writing (same as eval.py)
# ---------------------------------------------------------------------------

def write_per_image_csv(path: Path, rows: list[dict]) -> None:
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
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot MedSAM baseline (no adapter)."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument(
        "--prompt-strategy",
        type=str,
        default="box",
        choices=["box", "center", "grid"],
        help="Which automated prompt to use. 'box' is the primary baseline.",
    )
    parser.add_argument(
        "--grid-n", type=int, default=8,
        help="Grid density for 'grid' strategy. grid_n=8 -> 64 prompts/image.",
    )
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["train"]["seed"])
    device = get_device()

    print("=" * 70)
    print(" MedSAM zero-shot baseline evaluation")
    print("=" * 70)
    print(f"Config:           {args.config}")
    print(f"Dataset root:     {args.dataset_root}")
    print(f"Split:            {args.split_file}")
    print(f"Prompt strategy:  {args.prompt_strategy}")
    if args.prompt_strategy == "grid":
        print(f"Grid density:     {args.grid_n}x{args.grid_n} = {args.grid_n**2} prompts/image")
    print(f"Device:           {device_info(device)}")

    # ---------------------------------------------------------
    # Output dir
    # ---------------------------------------------------------
    log_root = Path(cfg["paths"]["log_dir"])
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = log_root / f"baseline_{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir:       {out_dir}\n")

    # ---------------------------------------------------------
    # Load frozen MedSAM (no adapter)
    # ---------------------------------------------------------
    print("[1/2] Loading frozen MedSAM ...")
    sam = FrozenSAM(checkpoint_path=cfg["paths"]["sam_checkpoint"]).to(device)

    # ---------------------------------------------------------
    # Build dataset
    # ---------------------------------------------------------
    print("[2/2] Building dataset ...")
    ds = FundusVesselDataset(
        data_root=args.dataset_root,
        split_file=args.split_file,
        augment=False,
        seed=cfg["train"]["seed"],
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size if args.prompt_strategy != "grid" else 1,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"] and device.type == "cuda",
    )
    print(f"      {len(ds)} samples\n")

    # ---------------------------------------------------------
    # Run evaluation
    # ---------------------------------------------------------
    threshold = cfg["eval"]["binarize_threshold"]
    closing_radius = cfg["eval"].get("closing_radius", 0)

    if args.prompt_strategy in ("box", "center"):
        acc = eval_box_or_center(
            sam=sam,
            loader=loader,
            device=device,
            prompt_strategy=args.prompt_strategy,
            binarize_threshold=threshold,
            closing_radius=closing_radius,
        )
    elif args.prompt_strategy == "grid":
        acc = eval_grid(
            sam=sam,
            loader=loader,
            device=device,
            grid_n=args.grid_n,
            binarize_threshold=threshold,
            closing_radius=closing_radius,
        )
    else:
        raise ValueError(f"Unknown strategy: {args.prompt_strategy}")

    # ---------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------
    summary = acc.summary()
    print("\n" + "=" * 70)
    print(f" Baseline results ({args.tag}, prompt={args.prompt_strategy})")
    print("=" * 70)
    print(f"  n samples:    {summary['n']}")
    print(f"  Dice:         {summary['dice']:.4f}")
    print(f"  IoU:          {summary['iou']:.4f}")
    print(f"  clDice:       {summary['cldice']:.4f}    <- headline metric")
    print(f"  Betti-0 err:  {summary['betti0_error']:.2f}")
    print()

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "tag": args.tag,
                "prompt_strategy": args.prompt_strategy,
                "grid_n": args.grid_n if args.prompt_strategy == "grid" else None,
                "dataset_root": str(args.dataset_root),
                "split_file": str(args.split_file),
                "summary": summary,
            },
            f,
            indent=2,
        )
    print(f"  -> {metrics_path}")

    csv_path = out_dir / "per_image.csv"
    write_per_image_csv(csv_path, acc.per_image_table())
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
