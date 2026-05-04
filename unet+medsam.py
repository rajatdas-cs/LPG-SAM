"""
Frequency-Decomposed Dual-Branch Retinal Vessel Segmentation on FIVES.

Architecture
------------
Input x is decomposed (GPU-side, differentiable-friendly) into:
    x_low  = G_sigma(x)      (Gaussian low-pass)
    x_high = x - x_low       (Laplacian residual)

Branch 1 (global): MedSAM ViT-B image encoder + lightweight FPN-style decoder,
                   operates on x_low. Encoder is partially frozen (only last 2
                   blocks + decoder are trained) for sample-efficient transfer.

Branch 2 (local):  Compact U-Net (~2M params) operating on x_high.
                   Captures thin vessels, edges, fine boundaries.

Fusion: A spatial gate alpha in [0,1]^{HxW} is predicted from concatenated
        branch features. Final logit:
            logit = alpha * z_high + (1 - alpha) * z_low
        Entropy regularizer on alpha (toward 0.5 prior, weighted) prevents
        gate collapse onto a single branch.

Loss: Dice + BCE + Tversky (alpha=0.3, beta=0.7) for thin-vessel recall,
      + small gate-collapse penalty.

Training:
    - Mixed precision (torch.cuda.amp).
    - Cosine LR with warmup, AdamW.
    - Robust checkpointing: every epoch + every N steps + on SIGTERM.
      Resumes RNG state, optimizer, scheduler, scaler, epoch, best metric.
    - Logging: TensorBoard + JSONL run log + console.

Usage (Colab)
-------------
    !python train_fives.py \
        --data_root /content/FIVES \
        --medsam_ckpt /content/medsam_vit_b.pth \
        --out_dir /content/runs/fives_dualbranch \
        --epochs 80 --batch_size 4 --img_size 512

Everything else is downloaded/created automatically at the end of execution
(final weights, predictions on test set, metrics JSON, training curves PNG).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from PIL import Image

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class Logger:
    """Console + JSONL + TensorBoard logger."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = open(out_dir / "log.jsonl", "a", buffering=1)
        self.tb = SummaryWriter(out_dir / "tb")

    def log(self, payload: Dict[str, Any], step: Optional[int] = None,
            tb_prefix: str = "") -> None:
        record = {"time": time.time(), **payload}
        if step is not None:
            record["step"] = step
        self.jsonl.write(json.dumps(record) + "\n")
        msg = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in payload.items())
        print(msg, flush=True)
        if step is not None:
            for k, v in payload.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    self.tb.add_scalar(f"{tb_prefix}{k}", float(v), step)

    def close(self) -> None:
        self.jsonl.close()
        self.tb.close()


def _find_dir_case_insensitive(parent: Path, candidates: List[str]) -> Path:
    """FIVES has 'Ground truth' / 'Original' but case can vary across mirrors."""
    if not parent.exists():
        raise FileNotFoundError(f"{parent} does not exist")
    lower_map = {p.name.lower(): p for p in parent.iterdir() if p.is_dir()}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise FileNotFoundError(
        f"None of {candidates} found in {parent}. Got: {list(lower_map)}"
    )


class FIVESDataset(Dataset):
    """FIVES retinal vessel dataset.

    Returns
    -------
    image : float tensor (3, H, W) in [0,1] (ImageNet-normalized)
    mask  : float tensor (1, H, W) in {0,1}
    name  : original filename stem
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, root: Path, split: str, img_size: int = 512,
                 augment: bool = False):
        split_dir = root / split
        self.img_dir = _find_dir_case_insensitive(split_dir, ["Original", "original"])
        self.mask_dir = _find_dir_case_insensitive(
            split_dir, ["Ground truth", "ground_truth", "GroundTruth", "groundtruth"]
        )
        self.img_size = img_size
        self.augment = augment

        img_paths = sorted([p for p in self.img_dir.iterdir()
                            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
        if not img_paths:
            raise RuntimeError(f"No images found in {self.img_dir}")
        # Pair with masks (same stem; mask may be .png even if image is .jpg)
        self.pairs: List[Tuple[Path, Path]] = []
        mask_index = {p.stem: p for p in self.mask_dir.iterdir() if p.is_file()}
        for ip in img_paths:
            mp = mask_index.get(ip.stem)
            if mp is None:
                continue
            self.pairs.append((ip, mp))
        if not self.pairs:
            raise RuntimeError(
                f"No matched (image, mask) pairs between {self.img_dir} and {self.mask_dir}"
            )

        self.mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _load_rgb(p: Path) -> np.ndarray:
        with Image.open(p) as im:
            return np.array(im.convert("RGB"))

    @staticmethod
    def _load_mask(p: Path) -> np.ndarray:
        with Image.open(p) as im:
            arr = np.array(im.convert("L"))
        return (arr > 127).astype(np.uint8)

    def _resize(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = self.img_size
        img_pil = Image.fromarray(img).resize((s, s), Image.BILINEAR)
        mask_pil = Image.fromarray(mask * 255).resize((s, s), Image.NEAREST)
        return np.array(img_pil), (np.array(mask_pil) > 127).astype(np.uint8)

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Geometric: hflip / vflip / 90-rot
        if random.random() < 0.5:
            img = img[:, ::-1, :].copy(); mask = mask[:, ::-1].copy()
        if random.random() < 0.5:
            img = img[::-1, :, :].copy(); mask = mask[::-1, :].copy()
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k).copy(); mask = np.rot90(mask, k).copy()
        # Photometric: brightness/contrast jitter (mild — vessels are subtle)
        if random.random() < 0.5:
            img_f = img.astype(np.float32) / 255.0
            brightness = random.uniform(-0.1, 0.1)
            contrast = random.uniform(0.9, 1.1)
            img_f = (img_f - 0.5) * contrast + 0.5 + brightness
            img = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        return img, mask

    def __getitem__(self, idx: int):
        img_p, mask_p = self.pairs[idx]
        img = self._load_rgb(img_p)
        mask = self._load_mask(mask_p)
        img, mask = self._resize(img, mask)
        if self.augment:
            img, mask = self._augment(img, mask)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - self.mean) / self.std
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return img_t, mask_t, img_p.stem


class FrequencyDecomposition(nn.Module):
    """Decompose x = x_low + x_high using a fixed Gaussian low-pass.

    The Gaussian kernel is registered as a (non-trainable) buffer so it moves
    with .to(device) and is included in state_dict for reproducibility.
    """

    def __init__(self, sigma: float = 2.0, kernel_size: Optional[int] = None,
                 channels: int = 3):
        super().__init__()
        if kernel_size is None:
            # 6*sigma rule, force odd
            kernel_size = max(3, int(6 * sigma) | 1)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels

        ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        gauss_1d = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel_2d = torch.outer(gauss_1d, gauss_1d)  # (k, k)
        kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pad = self.kernel_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        x_low = F.conv2d(x_pad, self.kernel, groups=self.channels)
        x_high = x - x_low
        return x_low, x_high



class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int,
                       pad_hw: Tuple[int, int], hw: Tuple[int, int]) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist, mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn: torch.Tensor, q: torch.Tensor,
                           rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor,
                           q_size: Tuple[int, int], k_size: Tuple[int, int]) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn = (attn.view(B, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)
    return attn


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 use_rel_pos: bool = False, input_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if use_rel_pos:
            assert input_size is not None
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, use_rel_pos: bool = False,
                 window_size: int = 0, input_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, int(dim * mlp_ratio))
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, kernel_size=(16, 16), stride=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).permute(0, 2, 3, 1)  # B C H W -> B H W C


class ImageEncoderViT(nn.Module):
    """SAM ViT-B image encoder. Loads MedSAM/SAM checkpoints by key prefix."""

    def __init__(self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, out_chans: int = 256, qkv_bias: bool = True,
                 use_abs_pos: bool = True, use_rel_pos: bool = True,
                 window_size: int = 14,
                 global_attn_indexes: Tuple[int, ...] = (2, 5, 8, 11)):
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans, embed_dim=embed_dim,
        )
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            ))
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            # Handle pos_embed resizing if input size differs from training size
            if x.shape[1] != self.pos_embed.shape[1] or x.shape[2] != self.pos_embed.shape[2]:
                pe = self.pos_embed.permute(0, 3, 1, 2)
                pe = F.interpolate(pe, size=(x.shape[1], x.shape[2]), mode="bicubic", align_corners=False)
                pe = pe.permute(0, 2, 3, 1)
                x = x + pe
            else:
                x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x  # (B, 256, H/16, W/16)


def load_medsam_image_encoder(ckpt_path: Path, img_size: int = 512) -> ImageEncoderViT:
    """Load MedSAM ViT-B image encoder weights, handling pos_embed interpolation."""
    encoder = ImageEncoderViT(img_size=img_size)
    state = torch.load(ckpt_path, map_location="cpu")
    # The MedSAM checkpoint is a flat dict with keys like 'image_encoder.patch_embed.proj.weight'
    enc_state = {}
    prefix = "image_encoder."
    for k, v in state.items():
        if k.startswith(prefix):
            enc_state[k[len(prefix):]] = v
    if not enc_state:
        # Some checkpoints may already be keyed without prefix
        enc_state = {k: v for k, v in state.items()
                     if any(k.startswith(p) for p in ["patch_embed", "blocks", "neck", "pos_embed"])}
    if not enc_state:
        raise RuntimeError(f"Could not find image_encoder weights in {ckpt_path}")

    # Resize pos_embed if model was trained at 1024 but we're using e.g. 512
    if "pos_embed" in enc_state and encoder.pos_embed is not None:
        src = enc_state["pos_embed"]
        dst_shape = encoder.pos_embed.shape
        if src.shape != dst_shape:
            print(f"[MedSAM] Resizing pos_embed {tuple(src.shape)} -> {tuple(dst_shape)}")
            src = src.permute(0, 3, 1, 2)
            src = F.interpolate(src, size=(dst_shape[1], dst_shape[2]),
                                mode="bicubic", align_corners=False)
            enc_state["pos_embed"] = src.permute(0, 2, 3, 1)

    enc_state = {
    k: v for k, v in enc_state.items()
    if "rel_pos" not in k
    }

    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    # Rel-pos parameters are content-resolution-dependent and SAM handles
    # mismatches at runtime via get_rel_pos (interpolation), so warn but continue.
    if missing:
        print(f"[MedSAM] Missing keys (first 5): {missing[:5]} (total {len(missing)})")
    if unexpected:
        print(f"[MedSAM] Unexpected keys (first 5): {unexpected[:5]} (total {len(unexpected)})")
    return encoder

class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        if p is None:
            p = k // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )


class GlobalDecoder(nn.Module):
    """Lightweight decoder: 256ch @ H/16 -> 1ch @ H. Progressive upsample."""

    def __init__(self, in_ch: int = 256, mid_ch: int = 128, out_ch: int = 64):
        super().__init__()
        self.up1 = nn.Sequential(ConvBNAct(in_ch, mid_ch), ConvBNAct(mid_ch, mid_ch))
        self.up2 = nn.Sequential(ConvBNAct(mid_ch, mid_ch), ConvBNAct(mid_ch, out_ch))
        self.up3 = nn.Sequential(ConvBNAct(out_ch, out_ch), ConvBNAct(out_ch, out_ch))
        self.up4 = nn.Sequential(ConvBNAct(out_ch, out_ch), ConvBNAct(out_ch, out_ch))
        self.feat_proj = nn.Conv2d(out_ch, out_ch, 1)  # for fusion gate input
        self.head = nn.Conv2d(out_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 256, H/16, W/16)
        x = F.interpolate(self.up1(x), scale_factor=2, mode="bilinear", align_corners=False)  # H/8
        x = F.interpolate(self.up2(x), scale_factor=2, mode="bilinear", align_corners=False)  # H/4
        x = F.interpolate(self.up3(x), scale_factor=2, mode="bilinear", align_corners=False)  # H/2
        x = F.interpolate(self.up4(x), scale_factor=2, mode="bilinear", align_corners=False)  # H
        feat = self.feat_proj(x)
        logit = self.head(x)
        return logit, feat


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            ConvBNAct(in_ch, out_ch),
            ConvBNAct(out_ch, out_ch),
        )


class CompactUNet(nn.Module):
    """~2M-param U-Net for high-frequency branch."""

    def __init__(self, in_ch: int = 3, base: int = 32, out_feat_ch: int = 64):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 8
        self.enc1 = DoubleConv(in_ch, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.bottleneck = DoubleConv(c4, c5)
        self.up4 = nn.ConvTranspose2d(c5, c4, 2, 2)
        self.dec4 = DoubleConv(c4 + c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, 2)
        self.dec3 = DoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.dec1 = DoubleConv(c1 + c1, c1)
        self.feat_proj = nn.Conv2d(c1, out_feat_ch, 1)
        self.head = nn.Conv2d(c1, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        feat = self.feat_proj(d1)
        logit = self.head(d1)
        return logit, feat


class GatedFusion(nn.Module):
    """Predict spatial gate alpha in [0,1] from concatenated branch features."""

    def __init__(self, feat_ch: int = 64, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_ch * 2, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, f_low: torch.Tensor, f_high: torch.Tensor) -> torch.Tensor:
        # bias init: gate ~0.5 (sigmoid(0)=0.5)
        return torch.sigmoid(self.net(torch.cat([f_low, f_high], dim=1)))


class DualBranchVesselNet(nn.Module):
    """Frequency-decomposed dual-branch retinal vessel segmenter.

    forward returns dict with:
      logit       : (B,1,H,W) fused logit
      logit_low   : (B,1,H,W) global-branch logit (deep supervision)
      logit_high  : (B,1,H,W) detail-branch logit (deep supervision)
      alpha       : (B,1,H,W) spatial gate in [0,1]
    """

    def __init__(self, medsam_ckpt: Path, img_size: int = 512,
                 freeze_encoder_until: int = 10, sigma: float = 2.0):
        super().__init__()
        self.freq = FrequencyDecomposition(sigma=sigma, channels=3)
        self.encoder = load_medsam_image_encoder(medsam_ckpt, img_size=img_size)
        # Freeze patch_embed, pos_embed, and first `freeze_encoder_until` blocks.
        for p in self.encoder.patch_embed.parameters():
            p.requires_grad_(False)
        if self.encoder.pos_embed is not None:
            self.encoder.pos_embed.requires_grad_(False)
        for i, blk in enumerate(self.encoder.blocks):
            if i < freeze_encoder_until:
                for p in blk.parameters():
                    p.requires_grad_(False)
        # Neck and last blocks remain trainable.
        self.global_dec = GlobalDecoder(in_ch=256, mid_ch=128, out_ch=64)
        self.local_net = CompactUNet(in_ch=3, base=32, out_feat_ch=64)
        self.fusion = GatedFusion(feat_ch=64)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_low, x_high = self.freq(x)
        z_low_feat = self.encoder(x_low)             # (B,256,H/16,W/16)
        logit_low, f_low = self.global_dec(z_low_feat)
        logit_high, f_high = self.local_net(x_high)
        alpha = self.fusion(f_low, f_high)
        logit = alpha * logit_high + (1.0 - alpha) * logit_low
        return {
            "logit": logit,
            "logit_low": logit_low,
            "logit_high": logit_high,
            "alpha": alpha,
        }


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    p = probs.flatten(1)
    t = targets.flatten(1)
    inter = (p * t).sum(1)
    denom = p.sum(1) + t.sum(1)
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def tversky_loss(logits: torch.Tensor, targets: torch.Tensor,
                 alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6) -> torch.Tensor:
    """Tversky with alpha<beta penalizes FN more than FP -> better for thin vessels."""
    probs = torch.sigmoid(logits)
    p = probs.flatten(1); t = targets.flatten(1)
    tp = (p * t).sum(1)
    fp = (p * (1 - t)).sum(1)
    fn = ((1 - p) * t).sum(1)
    return 1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def gate_collapse_penalty(alpha: torch.Tensor) -> torch.Tensor:
    """Penalize gate from collapsing to 0 or 1 globally.
    We penalize the squared deviation of the *image-level mean* of alpha from 0.5.
    Spatial values can still be 0/1 — only global imbalance is discouraged.
    """
    a_mean = alpha.flatten(1).mean(1)  # (B,)
    return (a_mean - 0.5).pow(2).mean()


@torch.no_grad()
def binary_metrics(logits: torch.Tensor, targets: torch.Tensor,
                   thresh: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs > thresh).float()
    p = pred.flatten(1); t = targets.flatten(1)
    tp = (p * t).sum(1)
    fp = (p * (1 - t)).sum(1)
    fn = ((1 - p) * t).sum(1)
    tn = ((1 - p) * (1 - t)).sum(1)
    eps = 1e-6
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    sens = (tp + eps) / (tp + fn + eps)        # recall
    spec = (tn + eps) / (tn + fp + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "sensitivity": sens.mean().item(),
        "specificity": spec.mean().item(),
        "accuracy": acc.mean().item(),
    }

@dataclass
class TrainConfig:
    data_root: str
    medsam_ckpt: str
    out_dir: str
    img_size: int = 512
    batch_size: int = 4
    grad_accum: int = 1
    epochs: int = 80
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    sigma: float = 2.0
    freeze_encoder_until: int = 10
    w_dice: float = 1.0
    w_bce: float = 1.0
    w_tversky: float = 0.5
    w_aux: float = 0.3       # weight on each auxiliary branch loss
    w_gate: float = 0.05     # gate collapse penalty
    num_workers: int = 4
    seed: int = 1337
    val_frac: float = 0.1    # fraction of train held out for validation
    save_every_steps: int = 500
    amp: bool = True
    resume: bool = True


class Checkpoint:
    """Atomic checkpointing: writes to .tmp then renames."""

    def __init__(self, path: Path):
        self.path = path
        self.tmp = path.with_suffix(path.suffix + ".tmp")

    def save(self, payload: Dict[str, Any]) -> None:
        torch.save(payload, self.tmp)
        os.replace(self.tmp, self.path)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self, map_location="cpu") -> Dict[str, Any]:
        return torch.load(self.path, map_location=map_location)


def cosine_lr(step: int, total_steps: int, warmup_steps: int,
              base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    logger = Logger(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    # ---- Data ----
    full_train = FIVESDataset(Path(cfg.data_root), "train",
                              img_size=cfg.img_size, augment=True)
    n_val = max(1, int(len(full_train) * cfg.val_frac))
    n_train = len(full_train) - n_val
    g = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = torch.utils.data.random_split(full_train, [n_train, n_val], generator=g)
    # Validation should not be augmented; build a non-augmented mirror and reuse indices
    val_base = FIVESDataset(Path(cfg.data_root), "train",
                            img_size=cfg.img_size, augment=False)
    val_ds = torch.utils.data.Subset(val_base, val_ds.indices)
    test_ds = FIVESDataset(Path(cfg.data_root), "test",
                           img_size=cfg.img_size, augment=False)

    print(f"[data] train={n_train}  val={n_val}  test={len(test_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    model = DualBranchVesselNet(
        medsam_ckpt=Path(cfg.medsam_ckpt),
        img_size=cfg.img_size,
        freeze_encoder_until=cfg.freeze_encoder_until,
        sigma=cfg.sigma,
    ).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] total params: {n_total/1e6:.2f}M  trainable: {n_train_params/1e6:.2f}M",
          flush=True)

    # Parameter groups (no weight decay on norms / biases)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias") or "norm" in n.lower() or "bn" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optim = torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(0.9, 0.999))

    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # ---- Resume ----
    ckpt_last = Checkpoint(out_dir / "ckpt_last.pt")
    ckpt_best = Checkpoint(out_dir / "ckpt_best.pt")
    start_epoch = 0
    global_step = 0
    best_val_dice = -1.0

    if cfg.resume and ckpt_last.exists():
        print(f"[resume] loading {ckpt_last.path}", flush=True)
        c = ckpt_last.load(map_location="cpu")
        model.load_state_dict(c["model"])
        optim.load_state_dict(c["optim"])
        scaler.load_state_dict(c["scaler"])
        start_epoch = c["epoch"]
        global_step = c["global_step"]
        best_val_dice = c.get("best_val_dice", -1.0)
        try:
            set_rng_state(c["rng"])
        except Exception as e:
            print(f"[resume] could not restore RNG state: {e}", flush=True)
        print(f"[resume] resuming from epoch={start_epoch} step={global_step}", flush=True)

    # ---- SIGTERM handler (Colab preemption) ----
    interrupted = {"flag": False}

    def _handler(signum, frame):
        interrupted["flag"] = True
        print(f"[signal] received {signum}, will checkpoint and exit", flush=True)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass

    bce = nn.BCEWithLogitsLoss()

    def save_last(epoch: int) -> None:
        ckpt_last.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_dice": best_val_dice,
            "rng": get_rng_state(),
            "config": asdict(cfg),
        })

    # ---- Train ----
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "dice": 0.0, "n": 0}
        optim.zero_grad(set_to_none=True)

        for it, (img, mask, _) in enumerate(train_loader):
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # LR schedule per optimizer step
            lr_now = cosine_lr(global_step, total_steps, warmup_steps, cfg.lr)
            for g_ in optim.param_groups:
                g_["lr"] = lr_now

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                out = model(img)
                logit = out["logit"]
                # Main loss
                l_dice = dice_loss(logit, mask).mean()
                l_bce = bce(logit, mask)
                l_tv = tversky_loss(logit, mask).mean()
                l_main = cfg.w_dice * l_dice + cfg.w_bce * l_bce + cfg.w_tversky * l_tv
                # Auxiliary deep-supervision losses
                l_low = (cfg.w_dice * dice_loss(out["logit_low"], mask).mean()
                         + cfg.w_bce * bce(out["logit_low"], mask))
                l_high = (cfg.w_dice * dice_loss(out["logit_high"], mask).mean()
                          + cfg.w_bce * bce(out["logit_high"], mask))
                l_gate = gate_collapse_penalty(out["alpha"])
                loss = l_main + cfg.w_aux * (l_low + l_high) + cfg.w_gate * l_gate
                loss = loss / cfg.grad_accum

            scaler.scale(loss).backward()

            if (it + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                global_step += 1

                # Track running metrics
                with torch.no_grad():
                    m = binary_metrics(logit.detach(), mask)
                running["loss"] += loss.item() * cfg.grad_accum * img.size(0)
                running["dice"] += m["dice"] * img.size(0)
                running["n"] += img.size(0)

                if global_step % 50 == 0:
                    logger.log({
                        "epoch": epoch,
                        "step": global_step,
                        "lr": lr_now,
                        "train_loss": running["loss"] / max(1, running["n"]),
                        "train_dice": running["dice"] / max(1, running["n"]),
                        "alpha_mean": float(out["alpha"].mean().item()),
                    }, step=global_step, tb_prefix="train/")

                if global_step % cfg.save_every_steps == 0:
                    save_last(epoch)

                if interrupted["flag"]:
                    save_last(epoch)
                    logger.close()
                    print("[exit] checkpointed and exiting due to signal", flush=True)
                    return

        # ---- Validation ----
        model.eval()
        val_running = {"dice": 0.0, "iou": 0.0, "sens": 0.0, "spec": 0.0, "n": 0}
        with torch.no_grad():
            for img, mask, _ in val_loader:
                img = img.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=cfg.amp):
                    out = model(img)
                m = binary_metrics(out["logit"], mask)
                bs = img.size(0)
                val_running["dice"] += m["dice"] * bs
                val_running["iou"] += m["iou"] * bs
                val_running["sens"] += m["sensitivity"] * bs
                val_running["spec"] += m["specificity"] * bs
                val_running["n"] += bs

        n = max(1, val_running["n"])
        val_dice = val_running["dice"] / n
        val_iou = val_running["iou"] / n
        val_sens = val_running["sens"] / n
        val_spec = val_running["spec"] / n
        epoch_time = time.time() - t0

        logger.log({
            "epoch": epoch,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "val_sensitivity": val_sens,
            "val_specificity": val_spec,
            "epoch_time_s": epoch_time,
        }, step=global_step, tb_prefix="val/")

        # Save last every epoch
        save_last(epoch + 1)

        # Save best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            ckpt_best.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_dice": val_dice,
                "config": asdict(cfg),
            })
            print(f"[best] new best val_dice={val_dice:.4f} (epoch {epoch})", flush=True)

    logger.close()
    print(f"[done] training complete. best val_dice={best_val_dice:.4f}", flush=True)


@torch.no_grad()
def evaluate_and_export(cfg: TrainConfig) -> None:
    """Load best checkpoint, predict on test set, save:
        - per-image PNG predictions
        - test_metrics.json
        - training_curves.png
        - model_final.pt (clean weight-only file)
    """
    out_dir = Path(cfg.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DualBranchVesselNet(
        medsam_ckpt=Path(cfg.medsam_ckpt),
        img_size=cfg.img_size,
        freeze_encoder_until=cfg.freeze_encoder_until,
        sigma=cfg.sigma,
    ).to(device).eval()
    best_path = out_dir / "ckpt_best.pt"
    if not best_path.exists():
        best_path = out_dir / "ckpt_last.pt"
    if not best_path.exists():
        print("[export] no checkpoint found, skipping", flush=True)
        return
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model"])
    print(f"[export] loaded {best_path} (epoch {state.get('epoch', '?')})", flush=True)

    # Save clean weights file
    torch.save({"model": model.state_dict(), "config": asdict(cfg)},
               out_dir / "model_final.pt")

    # Test predictions
    test_ds = FIVESDataset(Path(cfg.data_root), "test",
                           img_size=cfg.img_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    pred_dir = out_dir / "test_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    agg = {"dice": 0.0, "iou": 0.0, "sens": 0.0, "spec": 0.0, "acc": 0.0, "n": 0}
    for img, mask, names in test_loader:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            out = model(img)
        m = binary_metrics(out["logit"], mask)
        bs = img.size(0)
        agg["dice"] += m["dice"] * bs
        agg["iou"] += m["iou"] * bs
        agg["sens"] += m["sensitivity"] * bs
        agg["spec"] += m["specificity"] * bs
        agg["acc"] += m["accuracy"] * bs
        agg["n"] += bs
        # Save predictions
        prob = torch.sigmoid(out["logit"]).cpu().numpy()
        for k in range(prob.shape[0]):
            arr = (prob[k, 0] > 0.5).astype(np.uint8) * 255
            Image.fromarray(arr).save(pred_dir / f"{names[k]}.png")

    n = max(1, agg["n"])
    metrics = {
        "dice": agg["dice"] / n,
        "iou": agg["iou"] / n,
        "sensitivity": agg["sens"] / n,
        "specificity": agg["spec"] / n,
        "accuracy": agg["acc"] / n,
        "n_test": n,
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[export] test metrics: {json.dumps(metrics, indent=2)}", flush=True)

    # Plot training curves from JSONL
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train_steps, train_loss, train_dice = [], [], []
        val_epochs, val_dice = [], []
        for line in open(out_dir / "log.jsonl"):
            r = json.loads(line)
            if "train_loss" in r and "step" in r:
                train_steps.append(r["step"]); train_loss.append(r["train_loss"])
                train_dice.append(r.get("train_dice", float("nan")))
            if "val_dice" in r and "epoch" in r:
                val_epochs.append(r["epoch"]); val_dice.append(r["val_dice"])

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(train_steps, train_loss); axes[0].set_title("train loss"); axes[0].set_xlabel("step")
        axes[1].plot(train_steps, train_dice); axes[1].set_title("train dice"); axes[1].set_xlabel("step")
        axes[2].plot(val_epochs, val_dice, marker="o"); axes[2].set_title("val dice"); axes[2].set_xlabel("epoch")
        for ax in axes: ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "training_curves.png", dpi=120)
        plt.close()
        print(f"[export] saved training_curves.png", flush=True)
    except Exception as e:
        print(f"[export] could not plot curves: {e}", flush=True)


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--medsam_ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--freeze_encoder_until", type=int, default=10)
    p.add_argument("--w_dice", type=float, default=1.0)
    p.add_argument("--w_bce", type=float, default=1.0)
    p.add_argument("--w_tversky", type=float, default=0.5)
    p.add_argument("--w_aux", type=float, default=0.3)
    p.add_argument("--w_gate", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--save_every_steps", type=int, default=500)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training, only run final evaluation/export.")
    a = p.parse_args()
    cfg = TrainConfig(
        data_root=a.data_root, medsam_ckpt=a.medsam_ckpt, out_dir=a.out_dir,
        img_size=a.img_size, batch_size=a.batch_size, grad_accum=a.grad_accum,
        epochs=a.epochs, lr=a.lr, weight_decay=a.weight_decay,
        warmup_epochs=a.warmup_epochs, sigma=a.sigma,
        freeze_encoder_until=a.freeze_encoder_until,
        w_dice=a.w_dice, w_bce=a.w_bce, w_tversky=a.w_tversky,
        w_aux=a.w_aux, w_gate=a.w_gate,
        num_workers=a.num_workers, seed=a.seed, val_frac=a.val_frac,
        save_every_steps=a.save_every_steps,
        amp=not a.no_amp, resume=not a.no_resume,
    )
    return cfg, a.eval_only


def main() -> None:
    cfg, eval_only = parse_args()
    if not eval_only:
        train(cfg)
    evaluate_and_export(cfg)


if __name__ == "__main__":
    main()
