"""Loss functions used by the segmentation experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weighting.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=0 reduces to standard weighted CE.
    Higher gamma down-weights easy (well-classified) examples,
    forcing the model to focus on hard/minority pixels.
    """

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # Same class-weight tensor style as CrossEntropyLoss.

    def forward(self, logits, targets):
        # Keep the per-pixel CE values so focal weighting can be applied afterwards.
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        # p_t is the predicted probability of the true class.
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        return loss.mean()


class DiceLoss(nn.Module):
    """Soft multi-class Dice loss.

    Computes 1 - mean(per-class Dice) using softmax probabilities.
    smooth avoids division by zero for empty classes.
    """

    def __init__(self, num_classes, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)                   # (B, C, H, W)
        targets_one_hot = (
            F.one_hot(targets, self.num_classes)            # (B, H, W, C)
            .permute(0, 3, 1, 2)                            # (B, C, H, W)
            .float()
        )
        dims = (0, 2, 3)                                    # Sum over batch and pixels.
        intersection  = (probs * targets_one_hot).sum(dims) # Per-class soft overlap.
        cardinality   = probs.sum(dims) + targets_one_hot.sum(dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()
