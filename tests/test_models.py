"""Tests for generator/discriminator structures and output ranges."""
from __future__ import annotations
import sys
sys.path.insert(0, ".")
import torch
from pix2pix import Generator, PatchDiscriminator

def test_generator_output_range():
    g = Generator()
    x = torch.randn(2, 3, 128, 128)
    y = g(x)
    assert y.shape == (2, 3, 128, 128)
    assert y.min() >= -1.0 and y.max() <= 1.0

def test_discriminator_logits():
    d = PatchDiscriminator()
    a = torch.randn(1, 3, 128, 128)
    b = torch.randn(1, 3, 128, 128)
    out = d(a, b)
    assert out.dim() == 4
    assert out.shape[1] == 1
