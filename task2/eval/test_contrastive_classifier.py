from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from task2.config import (
    TASK2_BATCH_SIZE,
    TASK2_DEVICE,
    TASK2_NUM_CLASSES,
    TASK2_NUM_WORKERS,
    TASK2_OUTPUT_DIR,
    TASK2_TEST_DIR,
)

from task2.data.dataset import Task2TestDataset
from task2.models.contrastive_model import build_contrastive_model


CLASS_NAMES = ["Tumor", "Lymphocyte", "Histiocyte"]


def get_config_value(name: str, default):
    try:
        from task2 import config as task2_config
        return getattr(task2_config, name, default)
    except Exception:
        return default


TASK2_CONTRASTIVE_CLASSIFIER_CHECKPOINT = get_config_value(
    "TASK2_CONTRASTIVE_CLASSIFIER_CHECKPOINT",
    str(Path(TASK2_OUTPUT_DIR) / "contrastive_classifier" / "checkpoints" / "best.pt"),
)

TASK2_CONTRASTIVE_PROJECTION_HIDDEN_DIM = get_config_value(
    "TASK2_CONTRASTIVE_PROJECTION_HIDDEN_DIM",
    512,
)

TASK2_CONTRASTIVE_PROJECTION_DIM = get_config_value(
    "TASK2_CONTRASTIVE_PROJECTION_DIM",
    128,
)

TASK2_CONTRASTIVE_FEATURE_DROPOUT = get_config_value(
    "TASK2_CONTRASTIVE_FEATURE_DROPOUT",
    0.1,
)


def get_device() -> torch.device:
    if TASK2_DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_test_loader() -> DataLoader:
    dataset = Task2TestDataset(
        root_dir=TASK2_TEST_DIR,
        transform=build_eval_transform(),
        return_metadata=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=TASK2_BATCH_SIZE,
        shuffle=False,
        num_workers=TASK2_NUM_WORKERS,
        pin_memory=True,
    )
    return loader


class FrozenEncoderLinearClassifier(nn.Module):
    """
    Same model structure as used in train_contrastive_classifier.py:
    pretrained contrastive encoder + frozen encoder + linear classifier head
    """

    def __init__(
        self,
        encoder_model: nn.Module,
        feature_dim: int = 512,
        num_classes: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder_model = encoder_model
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(feature_dim, num_classes)

        if hasattr(self.encoder_model, "freeze_encoder"):
            self.encoder_model.freeze_encoder()
        else:
            for p in self.encoder_model.parameters():
                p.requires_grad = False

        self.encoder_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder_model.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if hasattr(self.encoder_model, "encode"):
                features = self.encoder_model.encode(x, normalize=False)
            else:
                features = self.encoder_model.extract_features(x)

        features = self.dropout(features)
        logits = self.classifier(features)
        return logits

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_parameter_stats(self) -> Dict[str, int]:
        encoder_total = sum(p.numel() for p in self.encoder_model.parameters())
        head_total = sum(p.numel() for p in self.classifier.parameters())
        return {
            "total_params": self.total_parameters(),
            "trainable_params": self.trainable_parameters(),
            "encoder_total_params": encoder_total,
            "classifier_head_params": head_total,
        }


def load_model(device: torch.device):
    checkpoint_path = Path(TASK2_CONTRASTIVE_CLASSIFIER_CHECKPOINT)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    encoder_model = build_contrastive_model(
        pretrained=False,
        freeze_encoder=False,
        feature_dropout=TASK2_CONTRASTIVE_FEATURE_DROPOUT,
        projection_hidden_dim=TASK2_CONTRASTIVE_PROJECTION_HIDDEN_DIM,
        projection_dim=TASK2_CONTRASTIVE_PROJECTION_DIM,
        use_projection_batchnorm=True,
        device=device,
    )

    model = FrozenEncoderLinearClassifier(
        encoder_model=encoder_model,
        feature_dim=512,
        num_classes=TASK2_NUM_CLASSES,
        dropout=0.2,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    return model, checkpoint_path


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )

    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
    }

    for i, class_name in enumerate(CLASS_NAMES):
        metrics[f"{class_name}_precision"] = float(precision_per_class[i])
        metrics[f"{class_name}_recall"] = float(recall_per_class[i])
        metrics[f"{class_name}_f1"] = float(f1_per_class[i])
        metrics[f"{class_name}_support"] = int(support_per_class[i])

    return metrics


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[List[float]] = []
    all_metadata: List[Dict[str, str]] = []

    for images, labels, metadata in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

        batch_size = len(labels)
        for i in range(batch_size):
            item = {}
            for key, value in metadata.items():
                item[key] = value[i]
            all_metadata.append(item)

    return all_labels, all_preds, all_probs, all_metadata


def compute_group_metrics(
    y_true: List[int],
    y_pred: List[int],
    metadata: List[Dict[str, str]],
    group_key: str = "sample_type",
):
    grouped = {}
    for yt, yp, meta in zip(y_true, y_pred, metadata):
        group = meta.get(group_key, "unknown")
        grouped.setdefault(group, {"y_true": [], "y_pred": []})
        grouped[group]["y_true"].append(yt)
        grouped[group]["y_pred"].append(yp)

    group_metrics = {}
    for group_name, group_data in grouped.items():
        group_metrics[group_name] = compute_metrics(
            group_data["y_true"],
            group_data["y_pred"],
        )
    return group_metrics


def save_predictions_csv(
    metadata: List[Dict[str, str]],
    y_true: List[int],
    y_pred: List[int],
    probs: List[List[float]],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "path",
            "filename",
            "sample_type",
            "true_label",
            "true_class",
            "pred_label",
            "pred_class",
            "prob_tumor",
            "prob_lymphocyte",
            "prob_histiocyte",
            "correct",
        ])

        for meta, true_label, pred_label, prob in zip(metadata, y_true, y_pred, probs):
            writer.writerow([
                meta["path"],
                meta["filename"],
                meta["sample_type"],
                true_label,
                CLASS_NAMES[true_label],
                pred_label,
                CLASS_NAMES[pred_label],
                float(prob[0]),
                float(prob[1]),
                float(prob[2]),
                int(true_label == pred_label),
            ])


def save_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, cm)


def main():
    device = get_device()

    eval_dir = Path(TASK2_OUTPUT_DIR) / "contrastive_classifier" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    test_loader = build_test_loader()
    model, checkpoint_path = load_model(device)

    y_true, y_pred, probs, metadata = evaluate(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    sample_type_metrics = compute_group_metrics(
        y_true=y_true,
        y_pred=y_pred,
        metadata=metadata,
        group_key="sample_type",
    )

    param_stats = model.get_parameter_stats()
    metrics["total_params"] = param_stats["total_params"]
    metrics["trainable_params"] = param_stats["trainable_params"]
    metrics["encoder_total_params"] = param_stats["encoder_total_params"]
    metrics["classifier_head_params"] = param_stats["classifier_head_params"]
    metrics["checkpoint_path"] = str(checkpoint_path)

    with (eval_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (eval_dir / "test_metrics_by_sample_type.json").open("w", encoding="utf-8") as f:
        json.dump(sample_type_metrics, f, indent=2)

    save_confusion_matrix(cm, eval_dir / "test_confusion_matrix.npy")
    save_predictions_csv(metadata, y_true, y_pred, probs, eval_dir / "test_predictions.csv")

    print("Contrastive classifier test results")
    print(f"Loaded checkpoint : {checkpoint_path}")
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Macro Precision   : {metrics['precision_macro']:.4f}")
    print(f"Macro Recall      : {metrics['recall_macro']:.4f}")
    print(f"Macro F1          : {metrics['f1_macro']:.4f}")
    print(f"Trainable Params  : {metrics['trainable_params']:,}")

    for class_name in CLASS_NAMES:
        print(
            f"{class_name:<12} "
            f"P={metrics[f'{class_name}_precision']:.4f} "
            f"R={metrics[f'{class_name}_recall']:.4f} "
            f"F1={metrics[f'{class_name}_f1']:.4f} "
            f"N={metrics[f'{class_name}_support']}"
        )

    if sample_type_metrics:
        print("\nBy sample_type:")
        for group_name, group_metrics in sample_type_metrics.items():
            print(
                f"{group_name:<12} "
                f"Acc={group_metrics['accuracy']:.4f} "
                f"MacroF1={group_metrics['f1_macro']:.4f}"
            )

    print(f"Saved metrics to              : {eval_dir / 'test_metrics.json'}")
    print(f"Saved sample-type metrics to  : {eval_dir / 'test_metrics_by_sample_type.json'}")
    print(f"Saved confusion matrix to     : {eval_dir / 'test_confusion_matrix.npy'}")
    print(f"Saved predictions to          : {eval_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()