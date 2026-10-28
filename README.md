# Pix2pix cGAN — Image-to-Image Translation (Task 04)

Built by Prodigy InfoTech Generative AI Internship.

## Overview

This is a genuine paired-image conditional GAN implementing the Berkeley pix2pix architecture (Isola et al. 2017). It translates between image halves: architectural label-map input (right half) → real building photograph output (left half). The loader treats LEFT=output and RIGHT=input.

## Architecture

- **Generator**: U-Net with skip connections (base ~32 channels) and Tanh output.
- **Discriminator**: PatchGAN (conditioned on concatenated input + target), outputs logits (no sigmoid).
- **Losses**: Adversarial (`BCEWithLogitsLoss`) + L1 reconstruction (`λ = 100` default).
- **Optimization**: Alternating G/D updates with Adam (`β = (0.5, 0.999)`).

## Setup

Requires Python 3.12, `torch>=2.2`, `torchvision`, `numpy<2` (to avoid binary compatibility issues with torch 2.2.2), `Pillow`, `matplotlib`, `streamlit`.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

The official Facades dataset URL is:
`https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz`

Download and extract:
```bash
python download_dataset.py
```

The loader expects composite split images (each file is two halves side by side):
- LEFT half = real building photograph (generator TARGET / output)
- RIGHT half = label/segmentation map (generator INPUT / source)
This aligns with the official pix2pix Facades convention (see tensorflow.org/tutorials/generative/pix2pix).

Files live in `data/raw/facades/facades/` by default with nested `train/val/test/` subdirectories; the loader supports this directly.

## Training

```bash
python train.py --epochs 10 --batch-size 1 --lambda-l1 100
```

Defaults: image size 128×128, base model width ~32, batch size 1, num workers 0. MPS/CPU fallback automatic (`get_device`).

Checkpoints saved to `checkpoints/` (latest.pth, final.pth). Logs in `outputs/train.log`.

## Inference

```bash
python infer.py --input <image_path> --checkpoint checkpoints/latest.pth --output outputs/translated.png
```

## Streamlit App

```bash
streamlit run app.py
```

The app shows architecture, dataset status, paired image previews, checkpoint selection, real inference from a loaded checkpoint (not fabricated), output download, and clear guidance when data/checkpoints are missing.

## Tests

```bash
source venv/bin/activate
pytest tests/ -q
```

Covers split orientation, normalization, synthetic pairs, model dimensions/outputs, gradient flow, minimal training step, checkpoint save/load, inference validation.

## References

- [Pix2pix paper (Isola et al., CVPR 2017)](https://arxiv.org/abs/1611.07070)
- [Official Berkeley pix2pix dataset](https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz)
- [Official PyTorch installation guide](https://pytorch.org/get-started/locally/)

## Hardware & Limitations

- MacBook Air M2 (8 GB unified memory) preferred MPS; CPU fallback available.
- Default compact settings (128×128, ~32 base channels, batch size 1, workers 0) safe for 8 GB.
- Full-scale training is expensive; a minimal `--epochs 10` run is sufficient for smoke verification.
- NumPy 2.x is avoided to prevent `torch 2.2.2` binary conflicts.

## Modified / Added Files

- `requirements.txt` — pinned versions
- `.gitignore`
- `README.md`
- `download_dataset.py`
- `pix2pix/__init__.py`
- `pix2pix/dataset.py`
- `pix2pix/models.py`
- `pix2pix/utils.py`
- `train.py`
- `infer.py`
- `app.py`
- `tests/conftest.py`
- `tests/test_dataset.py`
- `tests/test_models.py`
- `tests/test_training.py`
- `tests/test_inference.py`
- `.gitkeep` files in `outputs/`, `checkpoints/`, `data/raw/`, `data/processed/`
