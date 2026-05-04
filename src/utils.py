"""
utils.py
--------
Small helpers used across the LPG-SAM project:
- device selection (CUDA / MPS / CPU)
- reproducibility (seeding)
- parameter counting
- pretty printing of tensor shapes during debugging

Keep this file dependency-light. It should be importable from anywhere
without pulling in heavy modules like SAM or scikit-image.
"""

from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(prefer: str | None = None) -> torch.device:
    """
    Pick the best available device.

    Priority order (when `prefer` is None):
        1. CUDA  (Kaggle T4/P100, Colab, lab GPUs)
        2. MPS   (Apple Silicon Macs - dev only, do not train here)
        3. CPU   (fallback)

    Pass `prefer="cpu"` to force CPU (useful for debugging or unit tests).

    Returns
    -------
    torch.device
        The selected device.
    """
    if prefer is not None:
        return torch.device(prefer)

    if torch.cuda.is_available():
        return torch.device("cuda")

    # MPS is Apple's GPU backend. Available on M-series Macs.
    # Note: training SAM on MPS is impractical (slow, memory-tight).
    # We allow it here for local development and smoke tests only.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def device_info(device: torch.device) -> str:
    """Human-readable string describing the active device."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        total_mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        return f"CUDA: {name} ({total_mem_gb:.1f} GB)"
    if device.type == "mps":
        return "MPS: Apple Silicon GPU (dev only - do not train here)"
    return "CPU"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch for reproducible runs.

    Parameters
    ----------
    seed : int
        Seed value used everywhere.
    deterministic : bool
        If True, force cuDNN into deterministic mode. This can slow training
        down by 10-30 percent and is only worth enabling when you need
        bit-exact reproducibility (e.g. when chasing a bug).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Faster but non-deterministic across runs. Fine for normal training.
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(module: torch.nn.Module, only_trainable: bool = True) -> int:
    """
    Count parameters in a module.

    Parameters
    ----------
    module : nn.Module
    only_trainable : bool
        If True, count only parameters with `requires_grad=True`.
        If False, count every parameter.

    Returns
    -------
    int
        Number of parameters.
    """
    if only_trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def format_param_count(n: int) -> str:
    """Format a parameter count as a human-readable string (e.g. '1.2M')."""
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.2f}K"
    return str(n)


def summarize_trainable(module: torch.nn.Module, name: str = "module") -> None:
    """
    Print a one-line summary of trainable / total parameter counts.

    Useful for confirming that SAM is fully frozen and only the adapter
    is being trained.
    """
    trainable = count_parameters(module, only_trainable=True)
    total = count_parameters(module, only_trainable=False)
    pct = 100.0 * trainable / max(total, 1)
    print(
        f"[{name}] trainable: {format_param_count(trainable)} / "
        f"total: {format_param_count(total)} ({pct:.2f}%)"
    )


# ---------------------------------------------------------------------------
# Freezing helpers
# ---------------------------------------------------------------------------

def freeze_module(module: torch.nn.Module) -> None:
    """Set requires_grad=False on every parameter and put the module in eval mode."""
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def freeze_named_modules(root: torch.nn.Module, names: Iterable[str]) -> None:
    """
    Freeze sub-modules of `root` by attribute name.

    Example
    -------
    >>> freeze_named_modules(sam, ["image_encoder", "mask_decoder"])
    """
    for name in names:
        if not hasattr(root, name):
            raise AttributeError(f"Module has no submodule named '{name}'")
        freeze_module(getattr(root, name))


# ---------------------------------------------------------------------------
# Tensor debugging
# ---------------------------------------------------------------------------

def shape_str(t: torch.Tensor) -> str:
    """Compact shape string for debug prints, e.g. '(2, 256, 64, 64) f32 cuda'."""
    return f"{tuple(t.shape)} {str(t.dtype).replace('torch.', '')} {t.device}"
