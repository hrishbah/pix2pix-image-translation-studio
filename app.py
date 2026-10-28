"""Polished Streamlit UI for pix2pix image-to-image translation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from pix2pix import Generator, get_device, FacadesDataset, download_facades  # noqa: E402
from pix2pix.utils import DEFAULT_INPUT_SIZE  # noqa: E402

try:
    import streamlit as st
except Exception as exc:  # type: ignore[assignment]
    raise ImportError("streamlit is required for the app; install it via 'pip install streamlit'") from exc


def main():
    st.set_page_config(page_title="Pix2pix cGAN — Image Translation", page_icon="🎨", layout="wide")
    st.title("🎨 Pix2pix cGAN — Paired Image-to-Image Translation")
    st.markdown(
        "This app demonstrates a **conditional GAN** (pix2pix) built with a U-Net generator "
        "and a PatchGAN discriminator. Inputs: architectural label maps (right half of Facades) "
        "→ Outputs: reconstructed building photographs (left half of Facades). "
        "See README.md for full architecture and dataset conventions."
    )
    device = get_device()
    st.caption(f"Device: `{device}` | Default input size: `{DEFAULT_INPUT_SIZE}`")

    # Architecture section
    with st.expander("📐 Architecture"):
        st.markdown(
            "- **Generator**: U-Net with skip connections (base ~32 ch, 128×128, Tanh output).\n"
            "- **Discriminator**: PatchGAN (conditioned on input + real/generated target), outputs logits (no sigmoid).\n"
            "- **Losses**: Adversarial (BCEWithLogitsLoss) + L1 reconstruction (`λ = 100`).\n"
            "- **Training**: Alternating G/D optimization with Adam (`β=(0.5, 0.999)`)."
        )

    # Dataset status
    st.subheader("📦 Dataset")
    raw_dir = Path("data/raw/facades")
    processed_dir = Path("data/raw/facades/facades")
    dataset_available = False
    if processed_dir.is_dir():
        files = list(processed_dir.rglob("*.png")) + list(processed_dir.rglob("*.jpg"))
        dataset_available = len(files) > 0
    if dataset_available:
        st.success(f"Dataset ready at `{processed_dir}` — {len(files)} paired composite images.")
    else:
        st.info("Dataset not yet downloaded/processed. Run `python download_dataset.py` to fetch.")
        if st.button("Download & extract dataset"):
            try:
                result = download_facades(raw_dir)
                st.success(f"Downloaded to `{result}`; please copy/link to `{processed_dir}` for training.")
            except Exception as exc:
                st.error(str(exc))

    # Dataset previews when available
    if dataset_available:
        with st.expander("👁️ Paired dataset previews"):
            preview_paths = sorted(files)[:6]
            cols = st.columns(min(3, len(preview_paths)))
            for i, p in enumerate(preview_paths):
                with cols[i % len(cols)]:
                    img = Image.open(p).convert("RGB")
                    w, h = img.size
                    left = img.crop((0, 0, w // 2, h))
                    right = img.crop((w // 2, 0, w, h))
                    st.markdown(f"**{p.name}**")
                    # Mapping: LEFT = photo = target output; RIGHT = label = input source
                    col = st.columns(2)
                    col[0].image(right, caption="RIGHT — Label map (generator INPUT)", use_container_width=True)
                    col[1].image(left, caption="LEFT — Building photo (generator TARGET)", use_container_width=True)

    # Checkpoint / inference
    st.subheader("🔍 Inference & Checkpoint")
    ckpt_dir = Path("checkpoints")
    ckpt_files = sorted([f for f in ckpt_dir.iterdir() if f.suffix in (".pth", ".pt")])
    if not ckpt_files:
        st.warning("No checkpoints found in `checkpoints/`.")
        ckpt_path = None
    else:
        ckpt_path = str(st.selectbox("Select checkpoint", [str(f) for f in ckpt_files], index=0))
    st.markdown("**Translation direction:** Label map (RIGHT half / input) → Building photo (LEFT half / target output). Upload a label-map image to generate a realistic building photograph.")

    uploaded = st.file_uploader("Upload input image (label/segmentation or photo, depending on model direction)", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        image = Image.open(uploaded)
        col = st.columns(2)
        col[0].image(image, caption="Input image", use_container_width=True)
        if ckpt_path is not None:
            # Minimal inference
            with torch.no_grad():
                g = Generator()
                from pix2pix.utils import load_checkpoint
                load_checkpoint(ckpt_path, g, map_location=str(device))
                g.eval().to(device)
                from torchvision import transforms as T
                t = T.Compose([T.Resize((DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE)), T.ToTensor(), T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
                inp = t(image).unsqueeze(0).to(device)
                out = g(inp)
                out = (out.squeeze(0).cpu().clamp(-1, 1) + 1.0) / 2.0
                out_np = (out.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                out_img = Image.fromarray(out_np)
            col[1].image(out_img, caption="Generated translation", use_container_width=True)
            # Provide download
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                out_img.save(tmp.name)
                with open(tmp.name, "rb") as f:
                    st.download_button("Download result", f.read(), file_name="translated.png", mime="image/png")
        else:
            col[1].info("No trained checkpoint loaded. Train first (run `python train.py`) or select a checkpoint.")
    else:
        st.info("Upload an image to start inference.")

    # Guidance / limitations
    with st.expander("⚠️ Guidance when data/checkpoint is missing"):
        st.markdown(
            "- **No dataset**: Run `python download_dataset.py` (downloads `facades.tar.gz`). "
            "Place processed split images under `data/processed/facades/`.\n"
            "- **No checkpoint**: Run `python train.py --epochs 10 --batch-size 1` (small quick run). "
            "The default model (~32 ch base) uses ~1–2 GB on MPS and is safe on 8 GB RAM.\n"
            "- **No MPS**: CPU fallback is automatic (`get_device`). Training will be slow on M2 Air CPU.\n"
            "- **No fabricated results**: The app never pretends an untrained model produced results; "
            "it either loads a real checkpoint or shows informative guidance."
        )


if __name__ == "__main__":
    main()
