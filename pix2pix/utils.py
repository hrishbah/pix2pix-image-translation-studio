"""Shared utilities for pix2pix training and inference."""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

DEFAULT_INPUT_SIZE = 128
DEFAULT_NORMALIZE = (-1.0, 1.0)
SEED = 42
DEFAULT_LOG_INTERVAL = 100


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility (torch, numpy, python)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import random
        random.seed(seed)
    except Exception:
        pass


def get_device() -> torch.device:
    """Pick CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchTextLogger(nn.Module):
    """Lightweight text logger that writes per-step metrics to a plain-text log."""

    def __init__(self, path: str = "outputs/train.log", freq: int = 1):
        super().__init__()
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        self.path = path
        self.freq = max(1, freq)
        self._count = 0

    def __call__(self, scalars: dict[str, float], *args, **kwargs) -> None:
        if self._count % self.freq == 0:
            with open(self.path, "a", encoding="utf-8") as fh:
                ts = ""
                try:
                    import time
                    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                except Exception:
                    pass
                lines = [f"{ts} step={self._count}"]
                for k, v in scalars.items():
                    lines.append(f"  {k}: {v:.6g}")
                fh.write("\n".join(lines) + "\n")
                fh.flush()
        self._count += 1


def save_checkpoint(
    model: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    scheduler_d,
    epoch: int,
    path: str,
    extra: Optional[dict] = None,
) -> None:
    """Save a full checkpoint capturing model state dicts, optimizers, schedulers, epoch."""
    torch.save(
        {
            "model": model.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "epoch": epoch,
            "scheduler_d": scheduler_d.state_dict() if scheduler_d is not None else None,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    opt_g: Optional[torch.optim.Optimizer] = None,
    opt_d: Optional[torch.optim.Optimizer] = None,
    scheduler_d: Optional[object] = None,
    map_location: Optional[str] = None,
) -> Tuple[int, bool]:
    """Load checkpoint restoring model and optimizer state. Returns (epoch, loaded)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=True)
    try:
        model.load_state_dict(ckpt["model"])
    except Exception:
        # Fallback to full weights when old format saved metadata under 'model'.
        model.load_state_dict(ckpt)
    opt_g and opt_g.load_state_dict(ckpt.get("opt_g", {}))
    opt_d and opt_d.load_state_dict(ckpt.get("opt_d", {}))
    if scheduler_d is not None:
        scheduler_d.load_state_dict(ckpt.get("scheduler_d", {}))
    return ckpt.get("epoch", 0), True
