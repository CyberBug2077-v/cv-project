from __future__ import annotations

import csv
import json
import random
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from task2.config import (
    TASK2_RANDOM_SEED,
    TASK2_NUM_CLASSES,
    TASK2_BATCH_SIZE,
    TASK2_NUM_WORKERS,
    TASK2_DEVICE,
    TASK2_TRAIN_CSV,
    TASK2_VAL_CSV,
    TASK2_OUTPUT_DIR,
)

from task2.data.dataset import build_classification_datasets
from task2.models.contrastive_model import build_contrastive_model


CLASS_NAMES = ["Tumor", "Lymphocyte", "Histiocyte"]


# -------------------------
# Optional config fallbacks
# -------------------------
def get_config_value(name: str, default):
    try:
        from task2 import config as task2_config
        return getattr(task2_config, name, default)
    except Exception:
        return default


TASK2_CONTRASTIVE_ENCODER_CHECKPOINT = get_config_value(
    "TASK2_CONTRASTIVE_ENCODER_CHECKPOINT",
    str(Path(TASK2_OUTPUT_DIR) / "contrastive" / "checkpoints" / "best.pt"),
)

TASK2_FROZEN_HEAD_NUM_EPOCHS = get_config_value(
    "TASK2_FROZEN_HEAD_NUM_EPOCHS",
    20,
)

TASK2_FROZEN_HEAD_LEARNING_RATE = get_config_value(
    "TASK2_FROZEN_HEAD_LEARNING_RATE",
    1e-3,
)

TASK2_FROZEN_HEAD_WEIGHT_DECAY = get_config_value(
    "TASK2_FROZEN_HEAD_WEIGHT_DECAY",
    1e-4,
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if TASK2_DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform


def build_dataloaders():
    train_transform, eval_transform = build_transforms()

    train_dataset, val_dataset = build_classification_datasets(
        train_csv=TASK2_TRAIN_CSV,
        val_csv=TASK2_VAL_CSV,
        train_transform=train_transform,
        eval_transform=eval_transform,
        return_metadata=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TASK2_BATCH_SIZE,
        shuffle=True,
        num_workers=TASK2_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if TASK2_NUM_WORKERS > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TASK2_BATCH_SIZE,
        shuffle=False,
        num_workers=TASK2_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if TASK2_NUM_WORKERS > 0 else False,
    )

    return train_loader, val_loader


class FrozenEncoderLinearClassifier(nn.Module):
    """
    Load a pretrained contrastive encoder and train only a linear classifier on top.
    The encoder is fixed and kept in eval mode.
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

        # Freeze encoder
        if hasattr(self.encoder_model, "freeze_encoder"):
            self.encoder_model.freeze_encoder()
        else:
            for p in self.encoder_model.parameters():
                p.requires_grad = False

        self.encoder_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep encoder fixed in eval mode
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


def load_pretrained_contrastive_encoder(device: torch.device):
    checkpoint_path = Path(TASK2_CONTRASTIVE_ENCODER_CHECKPOINT)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Contrastive checkpoint not found: {checkpoint_path}"
        )

    encoder_model = build_contrastive_model(
        pretrained=False,
        freeze_encoder=False,
        feature_dropout=TASK2_CONTRASTIVE_FEATURE_DROPOUT,
        projection_hidden_dim=TASK2_CONTRASTIVE_PROJECTION_HIDDEN_DIM,
        projection_dim=TASK2_CONTRASTIVE_PROJECTION_DIM,
        use_projection_batchnorm=True,
        device=device,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder_model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    return encoder_model, checkpoint_path


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    accuracy = float((y_true_np == y_pred_np).mean())

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        average="macro",
        zero_division=0,
    )

    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
    }

    for i, class_name in enumerate(CLASS_NAMES):
        metrics[f"{class_name}_precision"] = float(precision_per_class[i])
        metrics[f"{class_name}_recall"] = float(recall_per_class[i])
        metrics[f"{class_name}_f1"] = float(f1_per_class[i])

    return metrics


def build_progress_bar(loader, epoch: int, total_epochs: int, training: bool):
    if tqdm is None:
        return loader

    phase = "train" if training else "val"
    return tqdm(
        loader,
        total=len(loader),
        desc=f"Epoch {epoch:02d}/{total_epochs:02d} [{phase}]",
        dynamic_ncols=True,
        leave=False,
    )


def run_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    training: bool,
    epoch: int,
    total_epochs: int,
):
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    running_samples = 0
    all_labels: List[int] = []
    all_preds: List[int] = []

    progress_bar = build_progress_bar(
        loader,
        epoch=epoch,
        total_epochs=total_epochs,
        training=training,
    )

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == labels).sum().item()
        running_samples += labels.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

        if tqdm is not None:
            progress_bar.set_postfix(
                loss=f"{running_loss / max(running_samples, 1):.4f}",
                acc=f"{running_correct / max(running_samples, 1):.4f}",
            )

    if tqdm is not None:
        progress_bar.close()

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = float(epoch_loss)

    return metrics, all_labels, all_preds


def save_history_csv(history: List[Dict[str, float]], output_path: Path) -> None:
    if not history:
        return

    fieldnames = list(history[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main():
    set_seed(TASK2_RANDOM_SEED)
    device = get_device()

    output_dir = Path(TASK2_OUTPUT_DIR) / "contrastive_classifier"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders()

    encoder_model, encoder_checkpoint_path = load_pretrained_contrastive_encoder(device)

    model = FrozenEncoderLinearClassifier(
        encoder_model=encoder_model,
        feature_dim=512,
        num_classes=TASK2_NUM_CLASSES,
        dropout=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.classifier.parameters(),
        lr=TASK2_FROZEN_HEAD_LEARNING_RATE,
        weight_decay=TASK2_FROZEN_HEAD_WEIGHT_DECAY,
    )

    param_stats = model.get_parameter_stats()
    print(f"Loaded encoder from: {encoder_checkpoint_path}")
    print(f"Total params: {param_stats['total_params']:,}")
    print(f"Trainable params: {param_stats['trainable_params']:,}")
    print(f"Encoder total params: {param_stats['encoder_total_params']:,}")
    print(f"Classifier head params: {param_stats['classifier_head_params']:,}")

    history: List[Dict[str, float]] = []
    best_score = -1.0
    best_epoch = -1
    best_val_labels = None
    best_val_preds = None

    for epoch in range(1, TASK2_FROZEN_HEAD_NUM_EPOCHS + 1):
        train_metrics, _, _ = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True,
            epoch=epoch,
            total_epochs=TASK2_FROZEN_HEAD_NUM_EPOCHS,
        )

        val_metrics, val_labels, val_preds = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=False,
            epoch=epoch,
            total_epochs=TASK2_FROZEN_HEAD_NUM_EPOCHS,
        )

        current_score = val_metrics["f1_macro"]

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision_macro": train_metrics["precision_macro"],
            "train_recall_macro": train_metrics["recall_macro"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
        }
        history.append(row)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1_macro": current_score,
            "param_stats": param_stats,
            "class_names": CLASS_NAMES,
            "encoder_checkpoint_path": str(encoder_checkpoint_path),
        }

        torch.save(checkpoint, checkpoint_dir / "last.pt")

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_val_labels = val_labels
            best_val_preds = val_preds
            torch.save(checkpoint, checkpoint_dir / "best.pt")

    save_history_csv(history, log_dir / "history.csv")

    summary = {
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_score,
        "param_stats": param_stats,
        "encoder_checkpoint_path": str(encoder_checkpoint_path),
    }
    with (log_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if best_val_labels is not None and best_val_preds is not None:
        cm = confusion_matrix(best_val_labels, best_val_preds, labels=[0, 1, 2])
        np.save(log_dir / "best_val_confusion_matrix.npy", cm)

        per_class_metrics = compute_metrics(best_val_labels, best_val_preds)
        with (log_dir / "best_val_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(per_class_metrics, f, indent=2)

    print(f"Best epoch: {best_epoch}")
    print(f"Best val macro F1: {best_score:.4f}")
    print(f"Saved checkpoint to: {checkpoint_dir / 'best.pt'}")
    print(f"Saved logs to: {log_dir}")


if __name__ == "__main__":
    main()
