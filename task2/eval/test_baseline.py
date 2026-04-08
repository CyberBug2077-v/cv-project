from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import torch
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
from task2.models.baseline import build_baseline_model


CLASS_NAMES = ["Tumor", "Lymphocyte", "Histiocyte"]


def get_config_value(name: str, default):
    try:
        from task2 import config as task2_config
        return getattr(task2_config, name, default)
    except Exception:
        return default


TASK2_USE_TEST_TIME_AUGMENTATION = get_config_value(
    "TASK2_USE_TEST_TIME_AUGMENTATION",
    True,
)

TASK2_TEST_TIME_AUGMENTATION_VIEWS = tuple(
    get_config_value(
        "TASK2_TEST_TIME_AUGMENTATION_VIEWS",
        ("identity",),
    )
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


def apply_tta_view(images: torch.Tensor, view_name: str) -> torch.Tensor:
    if view_name == "identity":
        return images
    if view_name == "hflip":
        return torch.flip(images, dims=[3])
    if view_name == "vflip":
        return torch.flip(images, dims=[2])
    if view_name == "hvflip":
        return torch.flip(images, dims=[2, 3])
    if view_name == "rot90":
        return torch.rot90(images, k=1, dims=[2, 3])
    if view_name == "rot180":
        return torch.rot90(images, k=2, dims=[2, 3])
    if view_name == "rot270":
        return torch.rot90(images, k=3, dims=[2, 3])
    raise ValueError(
        f"Unsupported TTA view {view_name!r}. "
        "Expected one of identity, hflip, vflip, hvflip, rot90, rot180, rot270."
    )


def get_tta_views() -> List[str]:
    if not TASK2_USE_TEST_TIME_AUGMENTATION:
        return ["identity"]

    if not TASK2_TEST_TIME_AUGMENTATION_VIEWS:
        return ["identity"]

    return [str(view) for view in TASK2_TEST_TIME_AUGMENTATION_VIEWS]


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
    tta_views = get_tta_views()

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[List[float]] = []
    all_metadata = []

    for images, labels, metadata in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        probs_sum = None
        for view_name in tta_views:
            view_images = apply_tta_view(images, view_name)
            logits = model(view_images)
            view_probs = torch.softmax(logits, dim=1)
            if probs_sum is None:
                probs_sum = view_probs
            else:
                probs_sum = probs_sum + view_probs

        probs = probs_sum / len(tta_views)
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

    return all_labels, all_preds, all_probs, all_metadata, tta_views


def save_predictions_csv(
    paths: List[str],
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
            "true_label",
            "true_class",
            "pred_label",
            "pred_class",
            "prob_tumor",
            "prob_lymphocyte",
            "prob_histiocyte",
            "correct",
        ])

        for path, true_label, pred_label, prob in zip(paths, y_true, y_pred, probs):
            writer.writerow([
                path,
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

    eval_dir = Path(TASK2_OUTPUT_DIR) / "baseline" / "eval"
    checkpoint_path = Path(TASK2_OUTPUT_DIR) / "baseline" / "checkpoints" / "best.pt"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    test_loader = build_test_loader()

    model = build_baseline_model(
        num_classes=TASK2_NUM_CLASSES,
        pretrained=False,
        freeze_backbone=False,
        dropout=0.2,
        device=device,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred, probs, paths, tta_views = evaluate(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # parameter stats
    param_stats = model.get_parameter_stats()
    metrics["total_params"] = param_stats["total_params"]
    metrics["trainable_params"] = param_stats["trainable_params"]
    metrics["tta_enabled"] = bool(TASK2_USE_TEST_TIME_AUGMENTATION)
    metrics["tta_views"] = list(tta_views)

    with (eval_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(cm, eval_dir / "test_confusion_matrix.npy")
    save_predictions_csv(paths, y_true, y_pred, probs, eval_dir / "test_predictions.csv")

    print("Test results")
    print(f"TTA enabled     : {metrics['tta_enabled']}")
    print(f"TTA views       : {', '.join(metrics['tta_views'])}")
    print(f"Accuracy        : {metrics['accuracy']:.4f}")
    print(f"Macro Precision : {metrics['precision_macro']:.4f}")
    print(f"Macro Recall    : {metrics['recall_macro']:.4f}")
    print(f"Macro F1        : {metrics['f1_macro']:.4f}")
    print(f"Trainable Params: {metrics['trainable_params']:,}")

    for class_name in CLASS_NAMES:
        print(
            f"{class_name:<12} "
            f"P={metrics[f'{class_name}_precision']:.4f} "
            f"R={metrics[f'{class_name}_recall']:.4f} "
            f"F1={metrics[f'{class_name}_f1']:.4f} "
            f"N={metrics[f'{class_name}_support']}"
        )

    print(f"Saved metrics to      : {eval_dir / 'test_metrics.json'}")
    print(f"Saved confusion matrix: {eval_dir / 'test_confusion_matrix.npy'}")
    print(f"Saved predictions to  : {eval_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
