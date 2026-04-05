from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ProjectionHead(nn.Module):
    """
    MLP projection head used for contrastive learning.

    Default structure:
        512 -> hidden_dim -> projection_dim
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 512,
        out_dim: int = 128,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        layers = [nn.Linear(in_dim, hidden_dim)]

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        layers.extend([
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet18Contrastive(nn.Module):
    """
    Contrastive model for Task 2 nuclei representation learning.

    Structure:
        input image -> ResNet-18 encoder -> pooled feature -> projection head

    Notes:
    - input:  RGB image tensor of shape [B, 3, H, W]
    - feature output: encoder representation of shape [B, 512]
    - projection output: contrastive embedding of shape [B, projection_dim]
    - projection is L2-normalized by default for NT-Xent / cosine-style losses
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        feature_dropout: float = 0.0,
        projection_hidden_dim: int = 512,
        projection_dim: int = 128,
        use_projection_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.encoder = backbone
        self.feature_dropout = (
            nn.Dropout(p=feature_dropout) if feature_dropout > 0 else nn.Identity()
        )
        self.projection_head = ProjectionHead(
            in_dim=in_features,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_dim,
            use_batchnorm=use_projection_batchnorm,
        )

        self.feature_dim = in_features
        self.projection_dim = projection_dim

        if freeze_encoder:
            self.freeze_encoder()

    def forward(
        self,
        x: torch.Tensor,
        normalize_projection: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            features:    [B, 512]
            projections: [B, projection_dim]
        """
        features = self.extract_features(x)
        projections = self.project(features, normalize=normalize_projection)
        return features, projections

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return encoder features before projection head.
        Shape: [B, 512]
        """
        features = self.encoder(x)
        features = self.feature_dropout(features)
        return features

    def encode(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """
        Alias for extracting encoder features.

        Args:
            x: input images
            normalize: whether to L2-normalize encoder features

        Returns:
            features of shape [B, 512]
        """
        features = self.extract_features(x)
        if normalize:
            features = F.normalize(features, dim=1)
        return features

    def project(
        self,
        features: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Apply projection head to encoder features.

        Args:
            features: encoder output of shape [B, 512]
            normalize: whether to L2-normalize projection vectors

        Returns:
            projections of shape [B, projection_dim]
        """
        projections = self.projection_head(features)
        if normalize:
            projections = F.normalize(projections, dim=1)
        return projections

    def encode_and_project(
        self,
        x: torch.Tensor,
        normalize_features: bool = False,
        normalize_projection: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience wrapper returning both encoder features and projections.
        """
        features = self.encode(x, normalize=normalize_features)
        projections = self.project(features, normalize=normalize_projection)
        return features, projections

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_projection_head(self) -> None:
        for param in self.projection_head.parameters():
            param.requires_grad = False

    def unfreeze_projection_head(self) -> None:
        for param in self.projection_head.parameters():
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
            "feature_dim": self.feature_dim,
            "projection_dim": self.projection_dim,
        }


def build_contrastive_model(
    pretrained: bool = True,
    freeze_encoder: bool = False,
    feature_dropout: float = 0.0,
    projection_hidden_dim: int = 512,
    projection_dim: int = 128,
    use_projection_batchnorm: bool = True,
    device: Optional[torch.device] = None,
) -> ResNet18Contrastive:
    model = ResNet18Contrastive(
        pretrained=pretrained,
        freeze_encoder=freeze_encoder,
        feature_dropout=feature_dropout,
        projection_hidden_dim=projection_hidden_dim,
        projection_dim=projection_dim,
        use_projection_batchnorm=use_projection_batchnorm,
    )

    if device is not None:
        model = model.to(device)

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_contrastive_model(
        pretrained=True,
        freeze_encoder=False,
        feature_dropout=0.1,
        projection_hidden_dim=512,
        projection_dim=128,
        use_projection_batchnorm=True,
        device=device,
    )

    x = torch.randn(4, 3, 100, 100).to(device)
    features, projections = model(x)

    print("Feature shape:", features.shape)
    print("Projection shape:", projections.shape)
    print("Parameter stats:", model.get_parameter_stats())