"""
scripts/train.py
----------------
Train the LatentPriorAdapter on top of frozen MedSAM.

Run with:
    python -m scripts.train --config configs/config.yaml

What this script does
---------------------
1. Loads config, sets seeds, picks device.
2. Builds FrozenSAM (encoder + decoder, both frozen).
3. Builds LatentPriorAdapter (the only trainable thing).
4. Builds train/val DataLoaders from the processed FIVES dataset.
5. For each epoch:
     - Train: forward through frozen encoder, adapter, frozen decoder;
       compute composite loss; backprop; update only adapter params.
     - Val:   compute hard Dice / clDice / IoU / Betti-0 on val split.
     - Save: best-val-clDice checkpoint and last checkpoint.
6. Logs everything to a run folder under <log_dir>/<timestamp>/ as a
   plain text file (no wandb/tensorboard dependencies).

Key things to watch during training
-----------------------------------
- adapter.alpha climbing away from zero. If it stays at 0, gradients
  aren't reaching it (something is broken in the FiLM path).
- val clDice improving relative to epoch 0. Epoch 0 with alpha~=0 IS
  the zero-shot baseline (modulo the 4 prior tokens). If clDice doesn't
  improve, the prior isn't carrying useful signal.
- Composite loss components staying balanced. If one term dominates,
  retune lambda weights in the config.

What this script does NOT do
----------------------------
- Cross-dataset eval (DRIVE/STARE/CHASE) -- that's scripts/eval.py.
- Hyperparameter sweeps -- run train.py multiple times with different
  configs.
- Resuming from a crashed run -- if a run dies, restart from scratch.
  We're training for ~1-2 hours, not days, so this is fine.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.adapter import LatentPriorAdapter
from src.dataset import FundusVesselDataset
from src.losses import CompositeLoss
from src.metrics import MetricAccumulator, logits_to_binary
from src.sam_wrapper import FrozenSAM
from src.utils import (
    count_parameters,
    device_info,
    format_param_count,
    get_device,
    seed_everything,
    summarize_trainable,
)


# ---------------------------------------------------------------------------
# Config + run dir
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_run_dir(log_root: Path) -> Path:
    """
    Create a timestamped subdirectory under log_root for this run.

    Returns
    -------
    Path
        e.g. runs/2026-04-11_14-32-07/
    """
    log_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = log_root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_warmup_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr_ratio: float = 0.01,
) -> float:
    """
    Linear warmup followed by cosine decay.

    Parameters
    ----------
    step : int
        Current step (0-indexed).
    total_steps : int
        Total training steps.
    warmup_steps : int
        Number of linear-warmup steps at the start.
    base_lr : float
        Peak learning rate (after warmup).
    min_lr_ratio : float
        Minimum LR as a fraction of base_lr at the end of cosine decay.

    Returns
    -------
    float
        The learning rate for this step.
    """
    import math

    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


# ---------------------------------------------------------------------------
# Forward pass: shared by train and val
# ---------------------------------------------------------------------------

def forward_pass(
    sam: FrozenSAM,
    adapter: LatentPriorAdapter,
    image: torch.Tensor,
    frangi: torch.Tensor,
) -> torch.Tensor:
    """
    Run the full LPG-SAM forward pass.

    Parameters
    ----------
    sam : FrozenSAM
    adapter : LatentPriorAdapter
    image : (B, 3, 1024, 1024) -- normalized RGB
    frangi : (B, 1, 1024, 1024) -- Frangi response in [0, 1]

    Returns
    -------
    torch.Tensor
        Mask logits at SAM's native decoder resolution (B, 1, 256, 256).
        Upsampling to GT size happens in the loss / metric code.
    """
    # Frozen encoder, no_grad to save VRAM.
    z_image = sam.encode_image(image)

    # Adapter (trainable). Produces modulated image embedding + sparse tokens.
    z_mod, sparse_tokens = adapter(z_image, frangi)

    # Frozen decoder. Gradients still flow back through it to the adapter.
    dense_default = sam.zero_dense_prompt(image.shape[0], image.device)
    masks_low_res, _iou = sam.decode_masks(
        image_embeddings=z_mod,
        sparse_prompt_embeddings=sparse_tokens,
        dense_prompt_embeddings=dense_default,
        multimask_output=False,
    )
    return masks_low_res  # (B, 1, 256, 256)


def upsample_to(mask_logits: torch.Tensor, size: int = 1024) -> torch.Tensor:
    """Upsample mask logits to (size, size) with bilinear interpolation."""
    return torch.nn.functional.interpolate(
        mask_logits, size=(size, size), mode="bilinear", align_corners=False
    )


# ---------------------------------------------------------------------------
# Train + validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    epoch: int,
    sam: FrozenSAM,
    adapter: LatentPriorAdapter,
    loss_fn: CompositeLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    global_step: int,
    total_steps: int,
    use_amp: bool,
) -> tuple[dict[str, float], int]:
    """
    Run one training epoch.

    Returns
    -------
    metrics : dict
        Average values of total/dice/bce/cldice losses and current alpha.
    new_global_step : int
    """
    adapter.train()

    running = {"total": 0.0, "dice": 0.0, "bce": 0.0, "cldice": 0.0}
    n_batches = 0

    bar = tqdm(loader, desc=f"epoch {epoch:03d} train", ncols=100, leave=False)
    for batch in bar:
        image = batch["image"].to(device, non_blocking=True)
        frangi = batch["frangi"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)

        # LR schedule (per-step cosine with warmup).
        lr = cosine_warmup_lr(
            step=global_step,
            total_steps=total_steps,
            warmup_steps=cfg["train"]["warmup_epochs"] * len(loader),
            base_lr=cfg["train"]["lr"],
        )
        set_lr(optimizer, lr)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision context. autocast is a no-op on CPU/MPS so this
        # path works on every device; only CUDA actually saves memory.
        autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp)
        with autocast_ctx:
            mask_logits_low = forward_pass(sam, adapter, image, frangi)
            mask_logits = upsample_to(mask_logits_low, size=1024)
            loss, components = loss_fn(mask_logits, target)

        # AMP-aware backward + step.
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                adapter.parameters(), cfg["train"]["grad_clip"]
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                adapter.parameters(), cfg["train"]["grad_clip"]
            )
            optimizer.step()

        # Bookkeeping.
        for k in running:
            running[k] += components[k].item()
        n_batches += 1
        global_step += 1

        # Live progress info -- alpha is the most important number to watch.
        bar.set_postfix(
            loss=f"{components['total'].item():.3f}",
            dice=f"{components['dice'].item():.3f}",
            cld=f"{components['cldice'].item():.3f}",
            alpha=f"{adapter.alpha.item():+.3f}",
            lr=f"{lr:.1e}",
        )
    bar.close()

    avg = {k: v / max(n_batches, 1) for k, v in running.items()}
    avg["alpha"] = float(adapter.alpha.item())
    return avg, global_step


@torch.no_grad()
def validate(
    epoch: int,
    sam: FrozenSAM,
    adapter: LatentPriorAdapter,
    loader: DataLoader,
    device: torch.device,
    binarize_threshold: float,
) -> dict[str, float]:
    """
    Run validation and return averaged metrics (Dice, IoU, clDice, Betti-0).
    """
    adapter.eval()
    acc = MetricAccumulator()

    bar = tqdm(loader, desc=f"epoch {epoch:03d} val  ", ncols=100, leave=False)
    for batch in bar:
        image = batch["image"].to(device, non_blocking=True)
        frangi = batch["frangi"].to(device, non_blocking=True)
        target = batch["mask"]  # keep on CPU
        names = batch["name"]

        mask_logits_low = forward_pass(sam, adapter, image, frangi)
        mask_logits = upsample_to(mask_logits_low, size=1024)

        # Binarize per-image and feed the accumulator.
        logits_np = mask_logits.detach().cpu().numpy()
        target_np = target.numpy()
        for b in range(logits_np.shape[0]):
            pred_bin = logits_to_binary(logits_np[b], threshold=binarize_threshold)
            gt_bin = target_np[b, 0] > 0.5
            acc.update(pred_bin, gt_bin, name=names[b])

        bar.set_postfix(
            cld=f"{(sum(acc.cldice) / max(len(acc.cldice), 1)):.3f}",
            dice=f"{(sum(acc.dice) / max(len(acc.dice), 1)):.3f}",
        )
    bar.close()

    return acc.summary()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    adapter: LatentPriorAdapter,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metrics: dict[str, float],
    cfg: dict[str, Any],
) -> None:
    """
    Save adapter weights + optimizer + epoch + val metrics + config.

    SAM weights are NOT saved (they're frozen and identical to the
    original MedSAM checkpoint). This keeps our .pt files tiny (~3 MB
    instead of ~370 MB).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state_dict": adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_metrics": val_metrics,
            "config": cfg,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train LPG-SAM adapter on FIVES.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Resume from a checkpoint .pt file (e.g. checkpoints/last.pt). "
             "Restores adapter weights, optimizer state, and epoch counter.",
    )
    parser.add_argument(
        "--ckpt-every", type=int, default=5,
        help="Save a numbered checkpoint every N epochs (safety net for Colab crashes).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["train"]["seed"])
    device = get_device()

    print("=" * 70)
    print(" LPG-SAM training")
    print("=" * 70)
    print(f"Config:        {args.config}")
    print(f"Device:        {device_info(device)}")

    # ---------------------------------------------------------
    # Run directory + log file
    # ---------------------------------------------------------
    log_root = Path(cfg["paths"]["log_dir"])
    run_dir = make_run_dir(log_root)
    log_file = run_dir / "train.log"
    cfg_dump = run_dir / "config.yaml"
    with open(cfg_dump, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Run dir:       {run_dir}")
    print(f"Log file:      {log_file}")

    def log(msg: str) -> None:
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    # ---------------------------------------------------------
    # Build model
    # ---------------------------------------------------------
    log("\n[1/4] Loading frozen MedSAM ...")
    sam = FrozenSAM(checkpoint_path=cfg["paths"]["sam_checkpoint"]).to(device)
    summarize_trainable(sam, name="FrozenSAM")

    log("[2/4] Building LatentPriorAdapter ...")
    adapter = LatentPriorAdapter(num_tokens=cfg["model"]["num_tokens"]).to(device)
    summarize_trainable(adapter, name="LatentPriorAdapter")

    n_trainable = count_parameters(adapter, only_trainable=True)
    log(f"      total trainable params: {format_param_count(n_trainable)}")

    # ---------------------------------------------------------
    # Build datasets and loaders
    # ---------------------------------------------------------
    log("[3/4] Building datasets ...")
    fives_root = Path(cfg["paths"]["fives_root"])
    train_ds = FundusVesselDataset(
        data_root=fives_root,
        split_file=fives_root / "train.txt",
        augment=cfg["data"]["augment_train"],
        seed=cfg["train"]["seed"],
    )
    val_ds = FundusVesselDataset(
        data_root=fives_root,
        split_file=fives_root / "val.txt",
        augment=False,
        seed=cfg["train"]["seed"],
    )
    log(f"      train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"] and device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"] and device.type == "cuda",
    )

    # ---------------------------------------------------------
    # Loss, optimizer, scaler
    # ---------------------------------------------------------
    log("[4/4] Building loss + optimizer ...")
    loss_fn = CompositeLoss(
        lambda_dice=cfg["loss"]["lambda_dice"],
        lambda_bce=cfg["loss"]["lambda_bce"],
        lambda_cldice=cfg["loss"]["lambda_cldice"],
        cldice_iter=cfg["loss"]["cldice_iter"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in adapter.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    log(f"      AMP enabled: {use_amp}")

    # ---------------------------------------------------------
    # Resume from checkpoint (optional)
    # ---------------------------------------------------------
    n_epochs = cfg["train"]["epochs"]
    start_epoch = 0
    best_cldice = -1.0
    best_epoch = -1

    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"

    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        log(f"\nResuming from {args.resume} ...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_cldice = ckpt["val_metrics"].get("cldice", -1.0)
        best_epoch = ckpt["epoch"]
        log(f"  Resumed at epoch {start_epoch}  (best clDice so far: {best_cldice:.4f})")

    # ---------------------------------------------------------
    # CSV metrics log — one row per epoch, written incrementally
    # so you can inspect it while training is running.
    # ---------------------------------------------------------
    metrics_csv = run_dir / "metrics.csv"
    csv_header = (
        "epoch,loss,dice,bce,cldice,alpha,"
        "val_dice,val_iou,val_cldice,val_b0err\n"
    )
    if not metrics_csv.exists():
        metrics_csv.write_text(csv_header)

    # ---------------------------------------------------------
    # Train loop
    # ---------------------------------------------------------
    log("\n" + "=" * 70)
    log(" Training")
    log("=" * 70)

    total_steps = n_epochs * len(train_loader)
    # When resuming, fast-forward global_step so the LR schedule is correct.
    global_step = start_epoch * len(train_loader)

    t_start = time.time()
    for epoch in range(start_epoch, n_epochs):
        # ----- Train -----
        train_metrics, global_step = train_one_epoch(
            epoch=epoch,
            sam=sam,
            adapter=adapter,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            loader=train_loader,
            device=device,
            cfg=cfg,
            global_step=global_step,
            total_steps=total_steps,
            use_amp=use_amp,
        )

        # ----- Validate -----
        val_metrics = validate(
            epoch=epoch,
            sam=sam,
            adapter=adapter,
            loader=val_loader,
            device=device,
            binarize_threshold=cfg["eval"]["binarize_threshold"],
        )

        # ----- Log line -----
        log(
            f"epoch {epoch:03d} | "
            f"loss {train_metrics['total']:.3f} "
            f"(dice {train_metrics['dice']:.3f} "
            f"bce {train_metrics['bce']:.3f} "
            f"cld {train_metrics['cldice']:.3f}) | "
            f"alpha {train_metrics['alpha']:+.3f} | "
            f"val: dice {val_metrics['dice']:.3f} "
            f"iou {val_metrics['iou']:.3f} "
            f"cld {val_metrics['cldice']:.3f} "
            f"b0err {val_metrics['betti0_error']:.2f}"
        )

        # ----- CSV log (append one row) -----
        with open(metrics_csv, "a") as f:
            f.write(
                f"{epoch},"
                f"{train_metrics['total']:.4f},"
                f"{train_metrics['dice']:.4f},"
                f"{train_metrics['bce']:.4f},"
                f"{train_metrics['cldice']:.4f},"
                f"{train_metrics['alpha']:.4f},"
                f"{val_metrics['dice']:.4f},"
                f"{val_metrics['iou']:.4f},"
                f"{val_metrics['cldice']:.4f},"
                f"{val_metrics['betti0_error']:.2f}\n"
            )

        # ----- Checkpoint -----
        save_checkpoint(last_ckpt, adapter, optimizer, epoch, val_metrics, cfg)

        if val_metrics["cldice"] > best_cldice:
            best_cldice = val_metrics["cldice"]
            best_epoch = epoch
            save_checkpoint(best_ckpt, adapter, optimizer, epoch, val_metrics, cfg)
            log(f"           ^ new best val clDice = {best_cldice:.4f}")

        # Periodic numbered checkpoint (safety net for Colab crashes).
        if args.ckpt_every > 0 and (epoch + 1) % args.ckpt_every == 0:
            periodic_ckpt = ckpt_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(periodic_ckpt, adapter, optimizer, epoch, val_metrics, cfg)
            log(f"           [saved periodic checkpoint: {periodic_ckpt.name}]")

    elapsed = time.time() - t_start
    log("\n" + "=" * 70)
    log(f" Training complete in {elapsed / 60:.1f} min")
    log(f" Best val clDice: {best_cldice:.4f} at epoch {best_epoch}")
    log(f" Best checkpoint: {best_ckpt}")
    log("=" * 70)

    # Dump final summary as JSON for easy parsing later.
    with open(run_dir / "summary.json", "w") as f:
        json.dump(
            {
                "best_val_cldice": best_cldice,
                "best_epoch": best_epoch,
                "elapsed_min": elapsed / 60,
                "epochs": n_epochs,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
