"""Smoke tests for inference pipeline."""
from __future__ import annotations
import sys
sys.path.insert(0, ".")
import torch
from pix2pix import Generator

def test_inference_shape():
    g = Generator()
    x = torch.randn(1, 3, 128, 128)
    out = g(x)
    assert out.shape == (1, 3, 128, 128)
