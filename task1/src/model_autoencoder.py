"""
Autoencoder pretraining model and the segmentation model built from it.

The encoder is structurally identical to the U-Net encoder so that
pretrained weights can be loaded directly into the segmentation model.

Autoencoder:
  Encoder: same 4-level conv blocks + pooling as UNet (base_ch=64)
  Decoder: symmetric upsampling path for pixel reconstruction
  Output: (B, 3, H, W) reconstructed image

SegWithPretrainedEncoder reuses the trained encoder and attaches a fresh
segmentation decoder.
"""

import torch
import torch.nn as nn

from model_unet import conv_block, count_parameters


class Autoencoder(nn.Module):
    """Plain convolutional autoencoder with the same encoder as the U-Net."""

    def __init__(self, in_channels=3, base_ch=64):
        super().__init__()

        # Match the U-Net encoder so weights can be transferred directly later.
        self.enc1 = conv_block(in_channels, base_ch)
        self.enc2 = conv_block(base_ch,     base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.enc4 = conv_block(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base_ch * 8, base_ch * 16)

        # The reconstruction decoder is symmetric but has no skip connections.
        self.up4  = nn.ConvTranspose2d(base_ch * 16, base_ch * 8,  kernel_size=2, stride=2)
        self.rdec4 = conv_block(base_ch * 8,  base_ch * 8)

        self.up3  = nn.ConvTranspose2d(base_ch * 8,  base_ch * 4,  kernel_size=2, stride=2)
        self.rdec3 = conv_block(base_ch * 4,  base_ch * 4)

        self.up2  = nn.ConvTranspose2d(base_ch * 4,  base_ch * 2,  kernel_size=2, stride=2)
        self.rdec2 = conv_block(base_ch * 2,  base_ch * 2)

        self.up1  = nn.ConvTranspose2d(base_ch * 2,  base_ch,      kernel_size=2, stride=2)
        self.rdec1 = conv_block(base_ch,      base_ch)

        self.recon_head = nn.Conv2d(base_ch, in_channels, kernel_size=1)

    def encode(self, x):
        """Run encoder and return all intermediate features + bottleneck."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b

    def decode_reconstruction(self, b):
        """Decoder path for image reconstruction (no skip connections)."""
        x = self.rdec4(self.up4(b))
        x = self.rdec3(self.up3(x))
        x = self.rdec2(self.up2(x))
        x = self.rdec1(self.up1(x))
        return self.recon_head(x)

    def forward(self, x):
        _, _, _, _, b = self.encode(x)
        return self.decode_reconstruction(b)


class SegWithPretrainedEncoder(nn.Module):
    """U-Net segmentation model with encoder initialised from an Autoencoder."""

    def __init__(self, pretrained_autoencoder, num_classes=3, base_ch=64):
        super().__init__()

        ae = pretrained_autoencoder

        # Reuse the trained encoder blocks exactly as they were learned.
        self.enc1 = ae.enc1
        self.enc2 = ae.enc2
        self.enc3 = ae.enc3
        self.enc4 = ae.enc4
        self.pool = ae.pool
        self.bottleneck = ae.bottleneck

        # The decoder is newly initialised for the supervised segmentation task.
        self.up4  = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = conv_block(base_ch * 16, base_ch * 8)

        self.up3  = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)

        self.up2  = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)

        self.up1  = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        self.head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)

    def encoder_features(self, x):
        """Return encoder activations and bottleneck from the pretrained encoder."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b


if __name__ == "__main__":
    # Small shape check for both model variants.
    ae = Autoencoder()
    x = torch.randn(2, 3, 512, 512)
    recon = ae(x)
    print(f"Autoencoder  input: {x.shape} → recon: {recon.shape}")
    print(f"Autoencoder params: {count_parameters(ae):,}")

    seg = SegWithPretrainedEncoder(ae, num_classes=3)
    out = seg(x)
    print(f"Seg (pretrained) output: {out.shape}")
    print(f"Seg params: {count_parameters(seg):,}")
