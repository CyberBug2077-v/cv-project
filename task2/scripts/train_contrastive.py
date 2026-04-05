from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from task2.config import (
    TASK2_RANDOM_SEED,
    TASK2_DEVICE,
    TASK2_NUM_WORKERS,
    TASK2_OUTPUT_DIR,
    TASK2_PATCH_SIZE,
    TASK2_CONTRASTIVE_CSV,
)
from task2.data.dataset import build_contrastive_dataset
from task2.models.contrastive_model import build_contrastive_model


# -------------------------
# Fallback defaults
# -------------------------
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_EPOCHS = 30
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_TEMPERATURE = 0.2
DEFAULT_PROJECTION_DIM = 128
DEFAULT_PROJECTION_HIDDEN_DIM = 512
DEFAULT_FEATURE_DROPOUT = 0.1


def get_config_value(name: str, default):
    try:
        from task2 import config as task2_config
        return getattr(task2_config, name, default)
    except Exception:
        return default


TASK2_CONTRASTIVE_BATCH_SIZE = get_config_value(
    "TASK2_CONTRASTIVE_BATCH_SIZE", DEFAULT_BATCH_SIZE
)
TASK2_CONTRASTIVE_NUM_EPOCHS = get_config_value(
    "TASK2_CONTRASTIVE_NUM_EPOCHS", DEFAULT_NUM_EPOCHS
)
TASK2_CONTRASTIVE_LEARNING_RATE = get_config_value(
    "TASK2_CONTRASTIVE_LEARNING_RATE", DEFAULT_LEARNING_RATE
)
TASK2_CONTRASTIVE_WEIGHT_DECAY = get_config_value(
    "TASK2_CONTRASTIVE_WEIGHT_DECAY", DEFAULT_WEIGHT_DECAY
)
TASK2_CONTRASTIVE_TEMPERATURE = get_config_value(
    "TASK2_CONTRASTIVE_TEMPERATURE", DEFAULT_TEMPERATURE
)
TASK2_CONTRASTIVE_PROJECTION_DIM = get_config_value(
    "TASK2_CONTRASTIVE_PROJECTION_DIM", DEFAULT_PROJECTION_DIM
)
TASK2_CONTRASTIVE_PROJECTION_HIDDEN_DIM = get_config_value(
    "TASK2_CONTRASTIVE_PROJECTION_HIDDEN_DIM", DEFAULT_PROJECTION_HIDDEN_DIM
)
TASK2_CONTRASTIVE_FEATURE_DROPOUT = get_config_value(
    "TASK2_CONTRASTIVE_FEATURE_DROPOUT", DEFAULT_FEATURE_DROPOUT
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


def build_contrastive_transform():
    """
    Conservative augmentations for 100x100 nuclei patches.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=TASK2_PATCH_SIZE,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
        ),
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


def build_contrastive_loader() -> DataLoader:
    dataset = build_contrastive_dataset(
        contrastive_csv=TASK2_CONTRASTIVE_CSV,
        view_transform=build_contrastive_transform(),
        return_metadata=False,
    )

    if len(dataset) < 2:
        raise ValueError(
            f"Contrastive dataset too small: {len(dataset)} samples found."
        )

    loader = DataLoader(
        dataset,
        batch_size=TASK2_CONTRASTIVE_BATCH_SIZE,
        shuffle=True,
        num_workers=TASK2_NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if TASK2_NUM_WORKERS > 0 else False,
    )
    return loader


class NTXentLoss(nn.Module):
    """
    Standard SimCLR-style NT-Xent loss.
    Input:
        z1, z2: normalized projections of shape [B, D]
    """

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if z1.ndim != 2 or z2.ndim != 2:
            raise ValueError("z1 and z2 must be 2D tensors of shape [B, D].")
        if z1.shape != z2.shape:
            raise ValueError(f"Shape mismatch: {z1.shape} vs {z2.shape}")

        batch_size = z1.size(0)
        if batch_size < 2:
            raise ValueError("NT-Xent loss requires batch_size >= 2.")

        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        similarity = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]

        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        similarity = similarity.masked_fill(mask, float("-inf"))

        positive_indices = torch.arange(batch_size, device=z.device)
        positive_indices = torch.cat(
            [positive_indices + batch_size, positive_indices], dim=0
        )

        loss = F.cross_entropy(similarity, positive_indices)
        return loss


@torch.no_grad()
def compute_batch_stats(z1: torch.Tensor, z2: torch.Tensor) -> Dict[str, float]:
    """
    Useful diagnostics for training logs.
    """
    positive_sim = F.cosine_similarity(z1, z2, dim=1).mean().item()

    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T)

    n = sim.size(0)
    diag_mask = torch.eye(n, device=sim.device, dtype=torch.bool)
    off_diag = sim.masked_select(~diag_mask)

    return {
        "positive_cosine": float(positive_sim),
        "mean_pairwise_cosine": float(off_diag.mean().item()),
    }


def build_progress_bar(loader, epoch: int, total_epochs: int, training: bool):
    if tqdm is None:
        return loader

    phase = "train" if training else "eval"
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
) -> Dict[str, float]:
    if training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_positive_cosine = 0.0
    running_pairwise_cosine = 0.0
    num_batches = 0

    progress_bar = build_progress_bar(
        loader,
        epoch=epoch,
        total_epochs=total_epochs,
        training=training,
    )

    for view1, view2 in progress_bar:
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            _, proj1 = model(view1, normalize_projection=True)
            _, proj2 = model(view2, normalize_projection=True)

            loss = criterion(proj1, proj2)

            if training:
                loss.backward()
                optimizer.step()

        stats = compute_batch_stats(proj1.detach(), proj2.detach())

        running_loss += loss.item()
        running_positive_cosine += stats["positive_cosine"]
        running_pairwise_cosine += stats["mean_pairwise_cosine"]
        num_batches += 1

        if tqdm is not None:
            progress_bar.set_postfix(
                loss=f"{running_loss / num_batches:.4f}",
                pos_cos=f"{running_positive_cosine / num_batches:.4f}",
                mean_pair_cos=f"{running_pairwise_cosine / num_batches:.4f}",
            )

    if tqdm is not None:
        progress_bar.close()

    if num_batches == 0:
        raise ValueError("No batches were produced. Check batch_size/drop_last settings.")

    return {
        "loss": running_loss / num_batches,
        "positive_cosine": running_positive_cosine / num_batches,
        "mean_pairwise_cosine": running_pairwise_cosine / num_batches,
    }


def save_history_csv(history: List[Dict[str, float]], output_path: Path) -> None:
    if not history:
        return

    fieldnames = list(history[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main():
    set_seed(TASK2_RANDOM_SEED)
    device = get_device()

    output_dir = Path(TASK2_OUTPUT_DIR) / "contrastive"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    loader = build_contrastive_loader()

    model = build_contrastive_model(
        pretrained=True,
        freeze_encoder=False,
        feature_dropout=TASK2_CONTRASTIVE_FEATURE_DROPOUT,
        projection_hidden_dim=TASK2_CONTRASTIVE_PROJECTION_HIDDEN_DIM,
        projection_dim=TASK2_CONTRASTIVE_PROJECTION_DIM,
        use_projection_batchnorm=True,
        device=device,
    )

    criterion = NTXentLoss(temperature=TASK2_CONTRASTIVE_TEMPERATURE)
    optimizer = AdamW(
        model.parameters(),
        lr=TASK2_CONTRASTIVE_LEARNING_RATE,
        weight_decay=TASK2_CONTRASTIVE_WEIGHT_DECAY,
    )

    param_stats = model.get_parameter_stats()
    print(f"Total params: {param_stats['total_params']:,}")
    print(f"Trainable params: {param_stats['trainable_params']:,}")
    print(f"Feature dim: {param_stats['feature_dim']}")
    print(f"Projection dim: {param_stats['projection_dim']}")

    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, TASK2_CONTRASTIVE_NUM_EPOCHS + 1):
        train_metrics = run_one_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True,
            epoch=epoch,
            total_epochs=TASK2_CONTRASTIVE_NUM_EPOCHS,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_positive_cosine": train_metrics["positive_cosine"],
            "train_mean_pairwise_cosine": train_metrics["mean_pairwise_cosine"],
        }
        history.append(row)

        print(
            f"[Epoch {epoch:02d}] "
            f"loss={train_metrics['loss']:.4f} "
            f"pos_cos={train_metrics['positive_cosine']:.4f} "
            f"mean_pair_cos={train_metrics['mean_pairwise_cosine']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_metrics["loss"],
            "param_stats": param_stats,
            "config": {
                "batch_size": TASK2_CONTRASTIVE_BATCH_SIZE,
                "num_epochs": TASK2_CONTRASTIVE_NUM_EPOCHS,
                "learning_rate": TASK2_CONTRASTIVE_LEARNING_RATE,
                "weight_decay": TASK2_CONTRASTIVE_WEIGHT_DECAY,
                "temperature": TASK2_CONTRASTIVE_TEMPERATURE,
                "projection_hidden_dim": TASK2_CONTRASTIVE_PROJECTION_HIDDEN_DIM,
                "projection_dim": TASK2_CONTRASTIVE_PROJECTION_DIM,
                "feature_dropout": TASK2_CONTRASTIVE_FEATURE_DROPOUT,
            },
        }

        torch.save(checkpoint, checkpoint_dir / "last.pt")

        if train_metrics["loss"] < best_loss:
            best_loss = train_metrics["loss"]
            best_epoch = epoch
            torch.save(checkpoint, checkpoint_dir / "best.pt")

    save_history_csv(history, log_dir / "history.csv")

    summary = {
        "best_epoch": best_epoch,
        "best_train_loss": best_loss,
        "param_stats": param_stats,
        "contrastive_csv": str(TASK2_CONTRASTIVE_CSV),
    }
    with (log_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Best epoch: {best_epoch}")
    print(f"Best train loss: {best_loss:.4f}")
    print(f"Saved best checkpoint to: {checkpoint_dir / 'best.pt'}")
    print(f"Saved logs to: {log_dir}")


if __name__ == "__main__":
    main()
