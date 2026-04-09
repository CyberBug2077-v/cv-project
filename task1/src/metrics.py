"""
Compute the segmentation metrics used throughout the project.

Computes per-class and mean:
  - Dice coefficient
  - Intersection over Union (IoU)
  - Pixel accuracy

Metrics are accumulated through a confusion matrix so they can be computed
consistently over a whole split.
"""

import numpy as np


class ConfusionMatrix:
    """Accumulates a confusion matrix over batches for later metric computation."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        """Add one prediction/target pair, flattening any spatial dimensions."""
        pred = pred.flatten().astype(np.int64)
        target = target.flatten().astype(np.int64)
        mask = (target >= 0) & (target < self.num_classes)
        # Encode each (target, pred) pair as a single bincount index.
        idx = self.num_classes * target[mask] + pred[mask]
        self.matrix += np.bincount(idx, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )

    def reset(self):
        self.matrix[:] = 0

    def compute(self):
        """Return dict with per-class and mean Dice, IoU, and pixel accuracy."""
        cm = self.matrix.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        # Small epsilons keep empty-class cases numerically stable.
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        iou  = tp / (tp + fp + fn + 1e-8)
        pixel_acc = tp.sum() / (cm.sum() + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)

        return {
            "dice_per_class":      dice,
            "iou_per_class":       iou,
            "precision_per_class": precision,
            "recall_per_class":    recall,
            "mean_dice":           dice.mean(),
            "mean_iou":            iou.mean(),
            "pixel_accuracy":      pixel_acc,
            "mean_precision":      precision.mean(),
            "mean_recall":         recall.mean(),
        }


def format_metrics(metrics, class_names):
    """Format a compact console summary for training and evaluation logs."""
    lines = []
    lines.append(f"  Mean Dice       : {metrics['mean_dice']:.4f}")
    lines.append(f"  Mean IoU        : {metrics['mean_iou']:.4f}")
    lines.append(f"  Pixel Accuracy  : {metrics['pixel_accuracy']:.4f}")
    lines.append("  Per-class Dice  :")
    for name, d in zip(class_names, metrics["dice_per_class"]):
        lines.append(f"    {name:<12}: {d:.4f}")
    lines.append("  Per-class IoU   :")
    for name, d in zip(class_names, metrics["iou_per_class"]):
        lines.append(f"    {name:<12}: {d:.4f}")
    return "\n".join(lines)
