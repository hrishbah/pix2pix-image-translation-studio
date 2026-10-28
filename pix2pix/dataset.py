"""Pix2pix Facades dataset loading, splitting, and preprocessing."""
from __future__ import annotations

import os
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Literal, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .utils import DEFAULT_INPUT_SIZE, SEED, get_device

# Official Pix2pix Facades release URL (Berkeley efrosgans).
# Layout convention in the dataset: each image is a horizontal composite of two halves.
# Per pix2pix cGAN canonical convention:
#   - LEFT half  = real building photograph  -> generator TARGET (output)
#   - RIGHT half = architectural label/segmentation -> generator INPUT (source)
# We therefore extract left=target, right=input; this matches tensorflow/pix2pix tutorials.
FACARDES_URL = (
    "https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"
)
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")
SplitName = Literal["train", "val", "test"]

# Normalization for pix2pix: scale from [0,1] (ToTensor) to [-1,1].
# Therefore mean/std are 0.5/0.5.
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


class DatasetError(Exception):
    """Raised when the dataset fails structural validation."""


def _ensure_dir(p: Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def download_facades(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    url: str = FACARDES_URL,
    overwrite: bool = False,
) -> Path:
    """Download and safely extract the facades archive.

    Raises ``DatasetError`` when download is incomplete, archive is empty,
    or extracted image files are missing. Never fabricates missing data.
    Returns the directory containing the extracted images.
    """
    raw = Path(raw_dir)
    extracted_name = "facades"
    extracted_dir = raw / extracted_name
    archive_name = "facades.tar.gz"
    archive_path = raw / archive_name
    partial_path = raw / (archive_name + ".part")

    if extracted_dir.exists() and not overwrite:
        # If the outer directory contains the actual split tree inside a nested folder,
        # prefer the inner dataset root so paths stay consistent.
        inner = extracted_dir / extracted_name
        if inner.is_dir() and any((inner / s).is_dir() for s in ("train", "val", "test")):
            images = [p for p in inner.rglob("*") if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")]
            if len(images) > 0:
                return inner
        images = [p for p in extracted_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        if len(images) == 0:
            raise DatasetError(f"Existing dataset at {extracted_dir} contains no image files.")
        return extracted_dir

    _ensure_dir(raw)
    try:
        urllib.request.urlretrieve(url, partial_path)
    except Exception as exc:
        raise DatasetError(f"Could not download dataset from {url}: {exc}") from exc

    size_bytes = partial_path.stat().st_size
    if size_bytes == 0:
        raise DatasetError("Downloaded facades archive is empty (0 bytes).")

    # Check that tarball is non-corrupt (at minimum has members).
    try:
        with tarfile.open(str(partial_path), "r:gz") as tf:
            members = tf.getmembers()
        if not members:
            raise DatasetError("Downloaded archive contains no members.")
    except (tarfile.TarError, Exception) as exc:
        raise DatasetError(f"Downloaded facades tar archive is corrupt: {exc}") from exc

    # Extract safely (filter="data" protects against malicious paths).
    with tarfile.open(str(partial_path), "r:gz") as tf:
        tf.extractall(path=str(raw), filter="data")

    # Move partial to final archive path once verified.
    partial_path.rename(archive_path)

    # After extraction, resolve the image directory.
    # Prefer the inner directory that contains split subfolders (train/val/test).
    candidates = [
        raw / extracted_name / extracted_name,
        raw / extracted_name,
        raw / (extracted_name + "_dataset"),
        raw / "facades_dataset",
    ]
    image_dir: Path | None = None
    for c in candidates:
        if c.is_dir() and any((c / s).is_dir() for s in ("train", "val", "test")):
            files = [
                f for f in c.rglob("*")
                if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg")
            ]
            if files:
                image_dir = c
                break
    if image_dir is None:
        for c in candidates:
            if c.is_dir():
                files = [
                    f for f in c.rglob("*")
                    if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg")
                ]
                if files:
                    image_dir = c
                    break
    if image_dir is None:
        raise DatasetError(
            "Facades extraction produced no recognized image files. Checked: "
            + ", ".join(str(x) for x in candidates)
        )
    # Validate using first file found recursively (images may be nested inside train/val/test).
    sample_candidates = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if not sample_candidates:
        raise DatasetError(f"No usable images found recursively under {image_dir}")
    sample_file = sample_candidates[0]
    try:
        with Image.open(str(sample_file)) as im:
            if im.width < 2 or im.height < 1:
                raise DatasetError(f"Image {sample_file} has suspicious dimensions {im.size}.")
    except Exception as exc:
        raise DatasetError(f"Failed to validate extracted dataset image {sample_file}: {exc}") from exc
    return image_dir


def get_transforms(
    size: int = DEFAULT_INPUT_SIZE,
    normalize_mean: Tuple[float, float, float] = NORMALIZE_MEAN,
    normalize_std: Tuple[float, float, float] = NORMALIZE_STD,
) -> transforms.Compose:
    """Consistent geometric transforms applied to both halves independently."""
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),  # Converts [0,255] -> [0,1]
            transforms.Normalize(mean=normalize_mean, std=normalize_std),  # -> [-1,1]
        ]
    )


def split_indices(total: int, split: SplitName, seed: int = SEED) -> List[int]:
    """Deterministic train/val/test split (default ratios: 80/10/10)."""
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    train_n = int(round(total * ratios["train"]))
    val_n = int(round(total * ratios["val"]))
    # Handle small datasets gracefully.
    train_n = max(1, min(train_n, total - 2))
    val_n = max(1, min(val_n, total - train_n - 1)) if split != "test" else max(0, total - train_n)
    if split == "train":
        return indices[:train_n]
    if split == "val":
        start = train_n
        return indices[start : start + val_n]
    # test
    return indices[train_n + val_n:]


def _split_composite(img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """Split a horizontally-composite image into (left=target, right=input).

    Left half = real building photograph = generator TARGET.
    Right half = architectural label/segmentation = generator INPUT.
    """
    w = img.width
    # For standard facades: left=output(building), right=input(label map).
    left = img.crop((0, 0, w // 2, img.height))
    right = img.crop((w // 2, 0, w, img.height))
    return left, right


class FacadesDataset(Dataset):
    """Paired Facades dataset read from a flat image directory or archive tree.

    The loader reads each image file as a horizontal split composite:
        - LEFT half  -> generator TARGET (real building photograph)
        - RIGHT half -> generator INPUT (label/segmentation map)
    Both halves are resized independently and normalized identically to [-1,1].
    """

    def __init__(
        self,
        data_dir: str | Path,
        size: int = DEFAULT_INPUT_SIZE,
        split: SplitName = "train",
        seed: int = SEED,
        max_samples: int | None = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.size = size
        self.split = split
        self.seed = seed
        if not self.data_dir.is_dir():
            raise DatasetError(f"Data directory does not exist: {self.data_dir}")
        # If dataset has nested split directories (train/val/test), use them directly.
        split_subdir = self.data_dir / split
        if split_subdir.is_dir() and any(p.is_file() for p in split_subdir.rglob("*")):
            self.image_paths = sorted([
                p for p in split_subdir.rglob("*")
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
            ])
        else:
            # Flat directory: collect all images and split deterministically.
            self.image_paths = sorted([
                p for p in self.data_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
            ])
        if len(self.image_paths) == 0:
            raise DatasetError(f"No image files found under {self.data_dir} for split '{split}'.")
        # When using nested splits, don't re-split; use all files in this split folder.
        if split_subdir.is_dir() and any(p.is_file() for p in split_subdir.rglob("*")):
            self.id_indices = list(range(len(self.image_paths)))
        else:
            self.id_indices = split_indices(len(self.image_paths), self.split, seed=self.seed)
        self.id_paths = [self.image_paths[i] for i in self.id_indices]
        if max_samples is not None and max_samples > 0:
            self.id_paths = self.id_paths[:max_samples]
            self.id_indices = self.id_indices[:max_samples]
        self.transform = get_transforms(size=size)

    def __len__(self) -> int:
        return len(self.id_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.id_paths[idx]
        try:
            img = Image.open(str(path)).convert("RGB")
            if img.width < 2:
                raise DatasetError(f"Image {path} width < 2 pixels; cannot split halves.")
            left_half, right_half = _split_composite(img)
            # Correct pix2pix orientation: right half = label map input, left half = photo target
            sample = {"input": self.transform(right_half), "target": self.transform(left_half)}
            return sample
        except FileNotFoundError:
            raise DatasetError(f"Expected image file missing: {path}")
        except Exception as exc:
            raise DatasetError(f"Failed to load dataset item from {path}: {exc}") from exc


def make_loader(
    dataset: FacadesDataset,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
