from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
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
    TASK2_NUM_EPOCHS,
    TASK2_LEARNING_RATE,
    TASK2_WEIGHT_DECAY,
    TASK2_DEVICE,
    TASK2_TRAIN_CSV,
    TASK2_VAL_CSV,
    TASK2_OUTPUT_DIR,
)
from task2.data.dataset import build_classification_datasets
from task2.models.baseline import build_baseline_model


CLASS_NAMES = ["Tumor", "Lymphocyte", "Histiocyte"]


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
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.08,
            contrast=0.08,
            saturation=0.05,
            hue=0.01,
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


def format_class_distribution(dataset) -> str:
    label_counts = dataset.df["label"].value_counts().to_dict()
    return ", ".join(
        f"{class_name}={int(label_counts.get(label_idx, 0))}"
        for label_idx, class_name in enumerate(CLASS_NAMES)
    )


def build_dataloaders():
    train_transform, eval_transform = build_transforms()

    train_dataset, val_dataset = build_classification_datasets(
        train_csv=TASK2_TRAIN_CSV,
        val_csv=TASK2_VAL_CSV,
        train_transform=train_transform,
        eval_transform=eval_transform,
        return_metadata=False,
    )

    print(f"Loaded train dataset class counts: {format_class_distribution(train_dataset)}")
    print(f"Loaded val dataset class counts: {format_class_distribution(val_dataset)}")

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


def run_one_epoch(model, loader, criterion, optimizer, device, training: bool, epoch: int, total_epochs: int):
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    running_samples = 0
    all_labels: List[int] = []
    all_preds: List[int] = []

    progress_bar = build_progress_bar(loader, epoch=epoch, total_epochs=total_epochs, training=training)

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

    output_dir = Path(TASK2_OUTPUT_DIR) / "baseline"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders()

    model = build_baseline_model(
        num_classes=TASK2_NUM_CLASSES,
        pretrained=True,
        freeze_backbone=False,
        dropout=0.2,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=TASK2_LEARNING_RATE,
        weight_decay=TASK2_WEIGHT_DECAY,
    )

    param_stats = model.get_parameter_stats()
    print(f"Total params: {param_stats['total_params']:,}")
    print(f"Trainable params: {param_stats['trainable_params']:,}")

    history: List[Dict[str, float]] = []
    best_score = -1.0
    best_epoch = -1
    best_val_labels = None
    best_val_preds = None

    for epoch in range(1, TASK2_NUM_EPOCHS + 1):
        train_metrics, _, _ = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True,
            epoch=epoch,
            total_epochs=TASK2_NUM_EPOCHS,
        )

        val_metrics, val_labels, val_preds = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=False,
            epoch=epoch,
            total_epochs=TASK2_NUM_EPOCHS,
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

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_val_labels = val_labels
            best_val_preds = val_preds

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1_macro": best_score,
                "param_stats": param_stats,
                "class_names": CLASS_NAMES,
            }
            torch.save(checkpoint, checkpoint_dir / "best.pt")

    save_history_csv(history, log_dir / "history.csv")

    summary = {
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_score,
        "param_stats": param_stats,
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
