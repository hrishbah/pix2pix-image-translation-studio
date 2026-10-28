#!/usr/bin/env python3
"""Download and extract official Berkeley pix2pix Facades dataset safely."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, ".")
from pix2pix import download_facades  # noqa: E402

if __name__ == "__main__":
    result = download_facades(Path("data/raw/facades"))
    print("Dataset extracted successfully at:", result)
    print("To use for training, copy/link contents to data/processed/facades/")
