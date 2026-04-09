"""
U-Net variants used for the tissue segmentation experiments.

Architecture:
  Encoder: 4 levels of (conv→BN→ReLU) × 2 + MaxPool
  Bottleneck: (conv→BN→ReLU) × 2
  Decoder: 4 levels of Upsample + concat skip + (conv→BN→ReLU) × 2
  Head: 1×1 conv → num_classes logits

Channel sizes: 64 → 128 → 256 → 512 → bottleneck 1024

Input : (B, 3, H, W)   float tensor
Output: (B, num_classes, H, W) logit tensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gabor_compblock import FirstOrderCompetitionBlock, SecondOrderCompetitionBlock


def conv_block(in_ch, out_ch):
    """Two conv layers, each followed by BatchNorm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Baseline four-level U-Net used for the main segmentation runs."""

    def __init__(self, in_channels=3, num_classes=3, base_ch=64):
        super().__init__()

        # Standard encoder path.
        self.enc1 = conv_block(in_channels, base_ch)
        self.enc2 = conv_block(base_ch,     base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.enc4 = conv_block(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck between encoder and decoder.
        self.bottleneck = conv_block(base_ch * 8, base_ch * 16)

        # Decoder mirrors the encoder and concatenates skip features.
        self.up4  = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = conv_block(base_ch * 16, base_ch * 8)

        self.up3  = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)

        self.up2  = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)

        self.up1  = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        # Final 1x1 projection to class logits.
        self.head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        # Store encoder activations for the decoder skip connections.
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Lowest-resolution representation.
        b = self.bottleneck(self.pool(e4))

        # Rebuild spatial detail with the matching encoder features.
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)

    def encoder_features(self, x):
        """Return bottleneck features (used by autoencoder transfer)."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b


class GaborCompetitionUNet(nn.Module):
    """Baseline U-Net with first-layer Gabor features concatenated before fusion."""

    def __init__(self, in_channels=3, num_classes=3, base_ch=64, gabor_channels=16):
        super().__init__()

        self.enc1 = conv_block(in_channels, base_ch)
        self.enc2 = conv_block(base_ch,     base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.enc4 = conv_block(base_ch * 4, base_ch * 8)

        self.comp_block = FirstOrderCompetitionBlock(
            channel_in=in_channels,
            n_competitor=gabor_channels,
            ksize=15,
            stride=1,
            padding=7,
            weight_chan=0.5,
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(base_ch + gabor_channels, base_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base_ch * 8, base_ch * 16)

        self.up4  = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = conv_block(base_ch * 16, base_ch * 8)

        self.up3  = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)

        self.up2  = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)

        self.up1  = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        self.head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def _encode_first_level(self, x):
        e1_base = self.enc1(x)
        gabor_feat = self.comp_block(x)
        # Fuse learned Gabor responses with the standard first encoder block.
        return self.fuse1(torch.cat([e1_base, gabor_feat], dim=1))

    def forward(self, x):
        e1 = self._encode_first_level(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)

    def encoder_features(self, x):
        e1 = self._encode_first_level(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b


class GatedGaborCompetitionUNet(nn.Module):
    """U-Net variant that adds projected Gabor features through a learned gate."""

    def __init__(self, in_channels=3, num_classes=3, base_ch=64, gabor_channels=32):
        super().__init__()

        self.enc1 = conv_block(in_channels, base_ch)
        self.enc2 = conv_block(base_ch,     base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.enc4 = conv_block(base_ch * 4, base_ch * 8)

        self.comp_block = FirstOrderCompetitionBlock(
            channel_in=in_channels,
            n_competitor=gabor_channels,
            ksize=15,
            stride=1,
            padding=7,
            weight_chan=0.5,
        )
        self.gabor_proj = nn.Sequential(
            nn.Conv2d(gabor_channels, base_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.gabor_gate = nn.Parameter(torch.tensor([0.0]))

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base_ch * 8, base_ch * 16)

        self.up4  = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = conv_block(base_ch * 16, base_ch * 8)

        self.up3  = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)

        self.up2  = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)

        self.up1  = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        self.head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def _encode_first_level(self, x):
        e1_base = self.enc1(x)
        gabor_feat = self.gabor_proj(self.comp_block(x))
        # The scalar gate lets training decide how much Gabor signal to trust.
        gate = torch.sigmoid(self.gabor_gate).view(1, 1, 1, 1)
        return e1_base + gate * gabor_feat

    def forward(self, x):
        e1 = self._encode_first_level(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)

    def encoder_features(self, x):
        e1 = self._encode_first_level(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b


class BoundaryAwareGaborUNet(nn.Module):
    """Gated Gabor U-Net with an auxiliary boundary prediction head."""

    def __init__(self, in_channels=3, num_classes=3, base_ch=64, gabor_channels=32):
        super().__init__()

        self.enc1 = conv_block(in_channels, base_ch)
        self.enc2 = conv_block(base_ch,     base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.enc4 = conv_block(base_ch * 4, base_ch * 8)

        self.comp_block = FirstOrderCompetitionBlock(
            channel_in=in_channels,
            n_competitor=gabor_channels,
            ksize=15,
            stride=1,
            padding=7,
            weight_chan=0.5,
        )
        self.gabor_proj = nn.Sequential(
            nn.Conv2d(gabor_channels, base_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.gabor_gate = nn.Parameter(torch.tensor([0.0]))
        self.boundary_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, kernel_size=1),
        )

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base_ch * 8, base_ch * 16)

        self.up4  = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = conv_block(base_ch * 16, base_ch * 8)

        self.up3  = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)

        self.up2  = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)

        self.up1  = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        self.head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def _encode_first_level(self, x):
        e1_base = self.enc1(x)
        gabor_feat = self.gabor_proj(self.comp_block(x))
        gate = torch.sigmoid(self.gabor_gate).view(1, 1, 1, 1)
        e1 = e1_base + gate * gabor_feat
        # Predict boundaries from the highest-resolution fused features.
        return e1, self.boundary_head(e1)

    def forward(self, x, return_boundary=False):
        e1, boundary_logits = self._encode_first_level(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        seg_logits = self.head(d1)
        if return_boundary:
            return seg_logits, boundary_logits
        return seg_logits

    def encoder_features(self, x):
        e1, _ = self._encode_first_level(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b


class BoundaryAwareSecondOrderGaborUNet(nn.Module):
    """Boundary-aware U-Net that uses both first- and second-order Gabor responses."""

    def __init__(self, in_channels=3, num_classes=3, base_ch=64, gabor_channels=32):
        super().__init__()

        self.enc1 = conv_block(in_channels, base_ch)
        self.enc2 = conv_block(base_ch,     base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.enc4 = conv_block(base_ch * 4, base_ch * 8)

        self.comp_block = SecondOrderCompetitionBlock(
            channel_in=in_channels,
            n_competitor=gabor_channels,
            ksize=15,
            stride=1,
            padding=7,
            weight_chan=0.5,
        )
        self.gabor_proj = nn.Sequential(
            nn.Conv2d(gabor_channels * 2, base_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.gabor_gate = nn.Parameter(torch.tensor([0.0]))
        self.boundary_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, kernel_size=1),
        )

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base_ch * 8, base_ch * 16)

        self.up4  = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = conv_block(base_ch * 16, base_ch * 8)

        self.up3  = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)

        self.up2  = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)

        self.up1  = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        self.head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def _encode_first_level(self, x):
        e1_base = self.enc1(x)
        gabor_feat = self.gabor_proj(self.comp_block(x))
        gate = torch.sigmoid(self.gabor_gate).view(1, 1, 1, 1)
        e1 = e1_base + gate * gabor_feat
        return e1, self.boundary_head(e1)

    def forward(self, x, return_boundary=False):
        e1, boundary_logits = self._encode_first_level(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        seg_logits = self.head(d1)
        if return_boundary:
            return seg_logits, boundary_logits
        return seg_logits

    def encoder_features(self, x):
        e1, _ = self._encode_first_level(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b


def count_parameters(model):
    """Count trainable parameters for quick model-size comparisons."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Simple sanity check for tensor shapes and parameter count.
    model = UNet(in_channels=3, num_classes=3)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {count_parameters(model):,}")
