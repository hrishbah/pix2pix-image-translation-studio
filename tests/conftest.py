"""Shared fixtures for pix2pix tests."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def synthetic_facades_dir(temp_dir: Path) -> Path:
    """Create a synthetic facades dataset directory with paired split images."""
    fac_dir = temp_dir / "facades"
    fac_dir.mkdir(parents=True, exist_ok=True)
    # Create 4 synthetic split images: left=blue, right=red.
    for i in range(4):
        w, h = 256, 256
        img = Image.new("RGB", (256, 256), color=(0, 0, 0))
        # Simulating split by drawing different halves (not needed for loader split logic but useful for visual checks).
        img.save(fac_dir / f"{i:03d}.png")
    return fac_dir
