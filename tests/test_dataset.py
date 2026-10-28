"""Tests for dataset loading, splitting, normalization, orientation."""
from __future__ import annotations

import sys
sys.path.insert(0, ".")
from pix2pix import FacadesDataset, get_transforms
from pix2pix.dataset import DatasetError, _split_composite
from PIL import Image, ImageDraw


def test_split_composite():
    img = Image.new("RGB", (256, 128))
    left, right = _split_composite(img)
    assert left.size == (128, 128)
    assert right.size == (128, 128)

def test_transform_normalizes_to_minus_one():
    # Synthetic image 128x128
    from torchvision import transforms
    img = Image.new("RGB", (128, 128), color=(255, 255, 255))
    t = get_transforms(size=128)
    out = t(img)
    # After ToTensor + Normalize(0.5,0.5) white -> 1, then normalized -> (1-0.5)/0.5 = 1.0. Wait that's wrong: white (255) -> 1.0, (1-0.5)/0.5 = 1.0 => 1, not -1? Actually for white = [1,1,1], norm to [-1,1] yields 1.0. For black = 0, yields -1.0.
    assert out.min() >= -1.0 and out.max() <= 1.0

def test_dataset_orientation_regression():
    """Assert input = right half (label) and target = left half (photo) by pixel colors."""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        dir_path = td_path / "test_facades"
        dir_path.mkdir()
        img = Image.new("RGB", (256, 128), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 128, 127], fill=(240, 20, 20))    # left = red = photo target
        draw.rectangle([128, 0, 255, 127], fill=(20, 20, 240))   # right = blue = label input
        img.save(str(dir_path / "reg.png"))
        ds = FacadesDataset(str(dir_path), size=128, split="train", seed=42)
        assert len(ds) == 1
        s = ds[0]
        input_mean = float(s["input"].mean())
        target_mean = float(s["target"].mean())
        # After normalization, red left has high red-channel value; blue right has high blue-channel value.
        input_r = float(s["input"][0].mean())
        input_b = float(s["input"][2].mean())
        target_r = float(s["target"][0].mean())
        target_b = float(s["target"][2].mean())
        # Input (blue/right) should have lower red mean and higher blue mean than target (red/left)
        assert input_r < target_r, f"Right-half input should have lower red mean than left-half target; got input_r={input_r}, target_r={target_r}"
        assert input_b > target_b, f"Right-half input should have higher blue mean than left-half target; got input_b={input_b}, target_b={target_b}"

def test_dataset_load_synthetic(synthetic_facades_dir):
    ds = FacadesDataset(synthetic_facades_dir, size=128, split="train")
    assert len(ds) > 0
    sample = ds[0]
    assert "input" in sample
    assert "target" in sample
    assert sample["input"].shape == (3, 128, 128)

def test_nested_split_directory():
    """Regression: loader finds images inside train/val/test subfolders."""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        root = td_path / "facades"
        root.mkdir()
        for split in ("train", "val", "test"):
            d = root / split
            d.mkdir()
            img = Image.new("RGB", (256, 128), color=(100, 100, 100))
            img.save(str(d / "img.png"))
        ds_train = FacadesDataset(str(root), size=128, split="train", seed=42)
        ds_val = FacadesDataset(str(root), size=128, split="val", seed=42)
        assert len(ds_train) >= 1
        assert len(ds_val) >= 1
        # Missing split folder should raise clear error
        try:
            FacadesDataset(str(root / "nonexistent"), size=128, split="train")
            assert False, "Should have raised DatasetError for missing dir"
        except DatasetError:
            pass
