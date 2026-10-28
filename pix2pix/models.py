"""U-Net Generator and PatchGAN Discriminator for pix2pix cGAN."""
from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Compact U-Net generator (~32 base channels, 128x128, Tanh output)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 32):
        super().__init__()
        self.enc1 = self._conv(in_ch, base)
        self.enc2 = self._conv(base, base * 2)
        self.enc3 = self._conv(base * 2, base * 4)
        self.enc4 = self._conv(base * 4, base * 8)
        self.dec4 = self._deconv(base * 8, base * 4)
        self.dec3 = self._deconv(base * 4 + base * 4, base * 2)
        self.dec2 = self._deconv(base * 2 + base * 2, base)
        self.dec1 = self._deconv(base + base, out_ch)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.enc_act = nn.LeakyReLU(0.2, inplace=True)

    def _conv(self, in_c: int, out_c: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _deconv(self, in_c: int, out_c: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)          # 32  @ 64x64  (for 128 in)
        e2 = self.enc2(e1)         # 64  @ 32x32
        e3 = self.enc3(e2)         # 128 @ 16x16
        e4 = self.enc4(e3)         # 256 @ 8x8
        # Decoder with skip connections: concat encoder output at same scale
        d4 = self.dec4(torch.cat([e3, e4], dim=1))  # wait this is wrong dimensionally.
        # Correct pairing: e3 at 16x16 matches d4 after deconv to 16x16.
        # Actually d4 takes e4 (256) then produces 128 at 16x16; skip from e3 (128) -> concat 256.
        # Let's design correctly.
        d4_in = torch.cat([self.dec4(e4), e3], dim=1)  # dec4 produces 128 @ 16x16; concat e3 128 -> 256
        d4 = self.dec3(d4_in)  # dec3 takes 256 -> 64 @ 32x32 (with exception: wait)
        # Hmm let's fix architecture exactly.
        return self.tanh(x)


# Re-implement with correct pairing and dimensions
class Generator(nn.Module):
    """U-Net generator for 128x128 with skip connections. Base ~32 channels. Tanh output."""
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(in_ch, base, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(base)
        self.enc2 = nn.Conv2d(base, base*2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(base*2)
        self.enc3 = nn.Conv2d(base*2, base*4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(base*4)
        self.enc4 = nn.Conv2d(base*4, base*8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(base*8)
        # Decoder
        self.dec4 = nn.ConvTranspose2d(base*8, base*4, 4, 2, 1, bias=False)
        self.bn_d4 = nn.BatchNorm2d(base*4)
        self.dec3 = nn.ConvTranspose2d(base*4 + base*4, base*2, 4, 2, 1, bias=False)
        self.bn_d3 = nn.BatchNorm2d(base*2)
        self.dec2 = nn.ConvTranspose2d(base*2 + base*2, base, 4, 2, 1, bias=False)
        self.bn_d2 = nn.BatchNorm2d(base)
        self.dec1 = nn.ConvTranspose2d(base + base, out_ch, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.leaky(self.bn1(self.enc1(x)))          # 64x64  ch=32
        e2 = self.leaky(self.bn2(self.enc2(e1)))         # 32x32  ch=64
        e3 = self.leaky(self.bn3(self.enc3(e2)))         # 16x16  ch=128
        e4 = self.leaky(self.bn4(self.enc4(e3)))         # 8x8    ch=256
        # Decoder with skip connections (channel concat)
        d4 = self.relu(self.bn_d4(self.dec4(e4)))        # 16x16 ch=128
        d4 = torch.cat([d4, e3], dim=1)                  # 16x16 ch=256
        d3 = self.relu(self.bn_d3(self.dec3(d4)))        # 32x32 ch=64
        d3 = torch.cat([d3, e2], dim=1)                  # 32x32 ch=128
        d2 = self.relu(self.bn_d2(self.dec2(d3)))        # 64x64 ch=32
        d2 = torch.cat([d2, e1], dim=1)                  # 64x64 ch=64
        d1 = self.tanh(self.dec1(d2))                    # 128x128 ch=3
        return d1


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator. Input = concatenated (input, target) -> logits."""
    def __init__(self, in_ch: int = 6, base: int = 32):
        super().__init__()
        def block(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        self.block1 = block(in_ch, base, False)          # 64x64 ch=32
        self.block2 = block(base, base*2)                # 32x32 ch=64
        self.block3 = block(base*2, base*4)              # 16x16 ch=128
        self.block4 = block(base*4, base*8)              # 8x8   ch=256
        self.block5 = block(base*8, 1, False)             # 8x8? Actually stride 1 -> 7x7 for 128; let's keep stride=1 for final
        # Rebuild final layer for 128 input -> ~8x8 patch output
        self.final = nn.Conv2d(base*8, 1, 4, 1, 1, bias=True)

    def forward(self, input_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        x = torch.cat([input_img, target_img], dim=1)     # condition
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.final(x)                               # logits, no sigmoid
