"""Train a pix2pix conditional GAN on Facades images."""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch import optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pix2pix import (  # noqa: E402
    Generator,
    PatchDiscriminator,
    FacadesDataset,
    make_loader,
    set_seed,
    save_checkpoint,
    get_device,
    DEFAULT_INPUT_SIZE,
)
from pix2pix.utils import TorchTextLogger  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a pix2pix cGAN on Facades.")
    p.add_argument("--data-dir", default="data/raw/facades/facades")
    p.add_argument("--raw-dir", default="data/raw/facades")
    p.add_argument("--size", type=int, default=DEFAULT_INPUT_SIZE)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--g-lr", type=float, default=2e-4)
    p.add_argument("--d-lr", type=float, default=2e-4)
    p.add_argument("--lambda-l1", type=float, default=100.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-dir", default="checkpoints")
    p.add_argument("--start-epoch", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=100, help="Log interval in steps")
    p.add_argument("--max-train-samples", type=int, default=None, help="Limit train images for smoke tests")
    p.add_argument("--max-val-samples", type=int, default=None, help="Limit val images for smoke tests")
    return p.parse_args()


def train_epoch(
    *, g_net: nn.Module, d_net: nn.Module, loader: torch.utils.data.DataLoader, device, opt_g, opt_d, l1_lambda,
) -> dict[str, float]:
    """One full epoch of alternating generator/discriminator updates."""
    n = len(loader) or 1
    g_losses = d_losses = l1_losses = {"G": 0.0, "D": 0.0, "L1": 0.0}
    for batch in loader:
        input_img = batch["input"].to(device)
        target_img = batch["target"].to(device)
        with torch.no_grad():
            fake_tgt = g_net(input_img)
        d_logits = d_net(input_img, fake_tgt)  # condition on fake
        real_logits = d_net(input_img, target_img)  # condition on real target
        # Standard pix2pix discriminator loss: mean of real (label=1) and fake (label=0) BCE losses.
        d_loss_real = nn.BCEWithLogitsLoss()(real_logits, real_logits.new_ones(real_logits.shape, dtype=torch.float32))
        d_loss_fake = nn.BCEWithLogitsLoss()(d_logits, d_logits.new_zeros(d_logits.shape, dtype=torch.float32))
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        # Generator losses
        # Generator adversarial: wants to fool discriminator -> targets = 1 (real)
        g_adv = nn.BCEWithLogitsLoss()(d_logits, d_logits.new_ones(d_logits.shape, dtype=torch.float32))
        recon = nn.L1Loss()(fake_tgt, target_img)
        g_total = g_adv * 2.0 + l1_lambda * recon
        g_losses["G"] += float(g_total.item())
        d_losses["D"] += float(d_loss.item())
        l1_losses["L1"] += float(recon.item())
        # Alternating update
        g_total.backward()
        opt_g.step()
        opt_d.step()
    return {k: v / n for k, v in g_losses.items()} | {"D": d_losses["D"] / n, "L1": l1_losses["L1"] / n}


def main(args: argparse.Namespace) -> int:
    SEED = 42
    set_seed(SEED)
    device = get_device()
    print(f"[pix2pix] Device={device} Size={args.size} Batch={args.batch_size}")

    dataset = FacadesDataset(
        data_dir=args.data_dir, size=args.size, split="train", seed=SEED,
        max_samples=args.max_train_samples,
    )
    print(f"[pix2pix] Loaded {len(dataset)} train images.")
    loader = make_loader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    g_net = Generator().to(device)
    d_net = PatchDiscriminator().to(device)
    opt_g = optim.Adam(g_net.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(d_net.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

    log_path = os.path.join(args.save_dir or "outputs", "train.log")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = TorchTextLogger(log_path, freq=max(1, args.log_interval // max(1, args.epochs)))

    best_g = float("inf")
    start_epoch = args.start_epoch
    for epoch in range(start_epoch, args.epochs):
        metrics = train_epoch(
            g_net=g_net, d_net=d_net, loader=loader, device=device,
            opt_g=opt_g, opt_d=opt_d, l1_lambda=args.lambda_l1,
        )
        if metrics["G"] < best_g:
            best_g = metrics["G"]
            ckpt_path = os.path.join(args.save_dir, "latest.pth")
            save_checkpoint(g_net, opt_g, opt_d, None, epoch, ckpt_path)
        print(f"Epoch {epoch+1}/{args.epochs}  G~{metrics['G']:.4f} D~{metrics['D']:.4f} L1~{metrics['L1']:.4f}")

    final_path = os.path.join(args.save_dir, "final.pth")
    save_checkpoint(g_net, opt_g, opt_d, None, args.epochs, final_path)
    print(f"[pix2pix] Training complete. final checkpoint at {final_path}")
    return 0


if __name__ == "__main__":
    main(parse_args())
