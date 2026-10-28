"""Run inference with a trained pix2pix generator."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pix2pix import Generator, get_device, load_checkpoint  # noqa: E402
from pix2pix.utils import DEFAULT_INPUT_SIZE  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pix2pix inference.")
    p.add_argument("--input", required=True, help="Input image path")
    p.add_argument("--checkpoint", default="checkpoints/latest.pth")
    p.add_argument("--output", default="outputs/translated.png")
    p.add_argument("--size", type=int, default=DEFAULT_INPUT_SIZE)
    return p.parse_args()


def infer(args: argparse.Namespace) -> int:
    device = get_device()
    g = Generator().to(device)
    try:
        load_checkpoint(args.checkpoint, g, map_location=str(device))
    except FileNotFoundError:
        print(f"[infer] Checkpoint not found: {args.checkpoint}")
        return 1
    g.eval()
    img = Image.open(args.input).convert("RGB")
    # Use standard transforms from training (resize, normalize to [-1,1])
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    inp = t(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = g(inp)
    # Convert from [-1,1] back to [0,1], then to PIL.
    out = (out.squeeze(0).cpu().clamp(-1, 1) + 1.0) / 2.0
    out_np = (out.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    Image.fromarray(out_np).save(args.output)
    print(f"[infer] Saved translated image to {args.output}")
    return 0


if __name__ == "__main__":
    import argparse as argparse_local
    infer(parse_args())
