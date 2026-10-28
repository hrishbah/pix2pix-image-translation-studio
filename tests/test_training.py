"""Smoke test for minimal training step, checkpoint save/load, gradient flow."""
from __future__ import annotations
import sys, tempfile
sys.path.insert(0, ".")
import torch
from pix2pix import Generator, PatchDiscriminator, save_checkpoint, load_checkpoint
from pix2pix.utils import get_device

def test_checkpoint_save_load():
    g = Generator()
    d = PatchDiscriminator()
    opt_g = torch.optim.Adam(g.parameters(), lr=0.001)
    opt_d = torch.optim.Adam(d.parameters(), lr=0.001)
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        path = f.name
    save_checkpoint(g, opt_g, opt_d, None, 5, path)
    g2 = Generator()
    load_checkpoint(path, g2, opt_g, opt_d, map_location="cpu")
    # Just verify it loads; no exceptions.

def test_gradient_flow():
    g = Generator()
    d = PatchDiscriminator()
    x = torch.randn(1, 3, 128, 128)
    y = g(x)
    out = d(x, y)
    loss = out.mean()
    loss.backward()
    assert any(p.grad is not None for p in g.parameters())
