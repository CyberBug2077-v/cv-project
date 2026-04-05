from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Baseline(nn.Module):
    """
    Baseline classifier for Task 2 nuclei classification.

    Notes:
    - Default input: RGB image tensor of shape [B, 3, H, W]
    - Default output: logits of shape [B, num_classes]
    - The final fully-connected layer is replaced for 3-class classification
    - Can optionally freeze backbone
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return pooled feature embedding before the final classifier.
        Shape: [B, 512]
        """
        features = self.backbone(x)
        features = self.dropout(features)
        return features

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_parameter_stats(self) -> Dict[str, int]:
        return {
            "total_params": self.total_parameters(),
            "trainable_params": self.trainable_parameters(),
        }


def build_baseline_model(
    num_classes: int = 3,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
    device: Optional[torch.device] = None,
) -> ResNet18Baseline:
    model = ResNet18Baseline(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )

    if device is not None:
        model = model.to(device)

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_baseline_model(
        num_classes=3,
        pretrained=True,
        freeze_backbone=False,
        dropout=0.2,
        device=device,
    )

    x = torch.randn(4, 3, 100, 100).to(device)
    logits = model(x)

    print("Output shape:", logits.shape)
    print("Parameter stats:", model.get_parameter_stats())