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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
    TASK2_BATCH_SIZE,
    TASK2_NUM_WORKERS,
    TASK2_OUTPUT_DIR,
    TASK2_PATCH_SIZE,
    TASK2_CONTRASTIVE_CSV,
    TASK2_TRAIN_CSV,
    TASK2_VAL_CSV,
)
from task2.data.dataset import build_classification_datasets, build_contrastive_dataset
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
DEFAULT_SELECTION_METRIC = "f1_macro"
DEFAULT_SELECTION_METRIC_TOLERANCE = 0.005
DEFAULT_PROBE_MAX_ITER = 1000


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
TASK2_CONTRASTIVE_SELECTION_METRIC = get_config_value(
    "TASK2_CONTRASTIVE_SELECTION_METRIC", DEFAULT_SELECTION_METRIC
)
TASK2_CONTRASTIVE_SELECTION_METRIC_TOLERANCE = get_config_value(
    "TASK2_CONTRASTIVE_SELECTION_METRIC_TOLERANCE",
    DEFAULT_SELECTION_METRIC_TOLERANCE,
)
TASK2_CONTRASTIVE_PROBE_MAX_ITER = get_config_value(
    "TASK2_CONTRASTIVE_PROBE_MAX_ITER", DEFAULT_PROBE_MAX_ITER
)

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


def get_selection_metric_name() -> str:
    valid_metrics = {"f1_macro", "accuracy"}
    if TASK2_CONTRASTIVE_SELECTION_METRIC not in valid_metrics:
        raise ValueError(
            f"Unsupported TASK2_CONTRASTIVE_SELECTION_METRIC="
            f"{TASK2_CONTRASTIVE_SELECTION_METRIC!r}. "
            f"Expected one of {sorted(valid_metrics)}."
        )
    return TASK2_CONTRASTIVE_SELECTION_METRIC


def should_select_checkpoint(
    current_score: float,
    current_tiebreak: float,
    current_loss: float,
    best_metric_seen_before_epoch: float,
    selected_score: float,
    selected_tiebreak: float,
    selected_loss: float,
    score_tolerance: float,
) -> Tuple[bool, str]:
    if selected_score == float("-inf"):
        return True, "first_checkpoint"

    if current_score > best_metric_seen_before_epoch:
        return True, "new_best_downstream_metric"

    eligibility_floor = best_metric_seen_before_epoch - score_tolerance
    if current_score < eligibility_floor:
        return False, "outside_metric_tolerance"

    if current_loss < selected_loss:
        return True, "within_metric_tolerance_lower_loss"

    if math.isclose(current_loss, selected_loss) and current_tiebreak > selected_tiebreak:
        return True, "within_metric_tolerance_better_tiebreak"

    return False, "within_metric_tolerance_not_better_than_selected"


def build_contrastive_transform():
    """
    Conservative augmentations for 100x100 nuclei patches.
    """
    return transforms.Compose([
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


def build_classification_eval_transform():
    return transforms.Compose([
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
        return_label=True,
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


def build_linear_probe_loaders() -> Tuple[DataLoader, DataLoader]:
    eval_transform = build_classification_eval_transform()
    train_dataset, val_dataset = build_classification_datasets(
        train_csv=TASK2_TRAIN_CSV,
        val_csv=TASK2_VAL_CSV,
        train_transform=eval_transform,
        eval_transform=eval_transform,
        return_metadata=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TASK2_BATCH_SIZE,
        shuffle=False,
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


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss from Khosla et al.
    Input:
        features: normalized projections of shape [B, n_views, D]
        labels: class labels of shape [B]
    """

    def __init__(
        self,
        temperature: float = 0.2,
        base_temperature: float | None = None,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.base_temperature = (
            temperature if base_temperature is None else base_temperature
        )

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                "features must be a 3D tensor of shape [batch_size, n_views, dim]."
            )

        batch_size, n_views, _ = features.shape
        if labels.ndim != 1 or labels.size(0) != batch_size:
            raise ValueError(
                f"labels must have shape [{batch_size}], got {tuple(labels.shape)}"
            )

        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_features = contrast_features
        anchor_count = n_views
        contrast_count = n_views

        logits = torch.matmul(anchor_features, contrast_features.T) / self.temperature
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positive_counts = mask.sum(dim=1)
        if torch.any(positive_counts <= 0):
            raise ValueError(
                "Each anchor must have at least one positive pair. "
                "Check batch composition and labels."
            )

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positive_counts
        loss = -(
            self.temperature / self.base_temperature
        ) * mean_log_prob_pos
        return loss.mean()


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


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    accuracy = float((y_true == y_pred).mean())
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
    }

    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    for i, class_name in enumerate(CLASS_NAMES):
        metrics[f"{class_name}_precision"] = float(precision_per_class[i])
        metrics[f"{class_name}_recall"] = float(recall_per_class[i])
        metrics[f"{class_name}_f1"] = float(f1_per_class[i])

    return metrics


@torch.no_grad()
def extract_encoder_features(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    all_features: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if hasattr(model, "encode"):
            features = model.encode(images, normalize=False)
        else:
            features = model.extract_features(images)

        all_features.append(features.cpu().numpy().astype(np.float32))
        all_labels.append(labels.cpu().numpy().astype(np.int64))

    if not all_features:
        raise ValueError("No features were extracted for downstream probe evaluation.")

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def evaluate_downstream_linear_probe(model, train_loader, val_loader, device) -> Dict[str, Dict[str, float]]:
    train_features, train_labels = extract_encoder_features(model, train_loader, device)
    val_features, val_labels = extract_encoder_features(model, val_loader, device)

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=TASK2_CONTRASTIVE_PROBE_MAX_ITER,
            random_state=TASK2_RANDOM_SEED,
        ),
    )
    probe.fit(train_features, train_labels)

    train_preds = probe.predict(train_features)
    val_preds = probe.predict(val_features)

    return {
        "train": compute_classification_metrics(train_labels, train_preds),
        "val": compute_classification_metrics(val_labels, val_preds),
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

    for view1, view2, labels in progress_bar:
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            _, proj1 = model(view1, normalize_projection=True)
            _, proj2 = model(view2, normalize_projection=True)
            projections = torch.stack([proj1, proj2], dim=1)

            loss = criterion(projections, labels)

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


def save_history_csv(history: List[Dict[str, object]], output_path: Path) -> None:
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
    selection_metric_name = get_selection_metric_name()
    pretrain_objective = "supervised_contrastive"

    output_dir = Path(TASK2_OUTPUT_DIR) / "contrastive"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    loader = build_contrastive_loader()
    probe_train_loader, probe_val_loader = build_linear_probe_loaders()

    model = build_contrastive_model(
        pretrained=True,
        freeze_encoder=False,
        feature_dropout=TASK2_CONTRASTIVE_FEATURE_DROPOUT,
        projection_hidden_dim=TASK2_CONTRASTIVE_PROJECTION_HIDDEN_DIM,
        projection_dim=TASK2_CONTRASTIVE_PROJECTION_DIM,
        use_projection_batchnorm=True,
        device=device,
    )

    criterion = SupervisedContrastiveLoss(
        temperature=TASK2_CONTRASTIVE_TEMPERATURE
    )
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
    print(f"Pretrain objective: {pretrain_objective}")
    print(f"Selection metric: downstream_val_{selection_metric_name}")
    print(
        "Selection strategy: downstream val metric with loss-aware selection "
        f"inside tolerance {TASK2_CONTRASTIVE_SELECTION_METRIC_TOLERANCE:.4f}"
    )

    history: List[Dict[str, object]] = []
    best_metric_seen = float("-inf")
    best_selection_score = float("-inf")
    best_selection_tiebreak = float("-inf")
    best_selected_loss = float("inf")
    best_epoch = -1
    best_probe_metrics = None
    best_selection_reason = None
    lowest_train_loss = float("inf")
    lowest_train_loss_epoch = -1

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

        probe_metrics = evaluate_downstream_linear_probe(
            model=model,
            train_loader=probe_train_loader,
            val_loader=probe_val_loader,
            device=device,
        )
        current_selection_score = probe_metrics["val"][selection_metric_name]
        current_tiebreak = (
            probe_metrics["val"]["accuracy"]
            if selection_metric_name != "accuracy"
            else probe_metrics["val"]["f1_macro"]
        )
        metric_seen_before_epoch = best_metric_seen
        should_update_best, selection_reason = should_select_checkpoint(
            current_score=current_selection_score,
            current_tiebreak=current_tiebreak,
            current_loss=train_metrics["loss"],
            best_metric_seen_before_epoch=metric_seen_before_epoch,
            selected_score=best_selection_score,
            selected_tiebreak=best_selection_tiebreak,
            selected_loss=best_selected_loss,
            score_tolerance=TASK2_CONTRASTIVE_SELECTION_METRIC_TOLERANCE,
        )
        best_metric_seen = max(best_metric_seen, current_selection_score)
        selected_as_best = 0

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_positive_cosine": train_metrics["positive_cosine"],
            "train_mean_pairwise_cosine": train_metrics["mean_pairwise_cosine"],
            "probe_train_accuracy": probe_metrics["train"]["accuracy"],
            "probe_train_f1_macro": probe_metrics["train"]["f1_macro"],
            "probe_val_accuracy": probe_metrics["val"]["accuracy"],
            "probe_val_f1_macro": probe_metrics["val"]["f1_macro"],
            "selection_metric": current_selection_score,
            "selection_tiebreak": current_tiebreak,
            "best_metric_seen_so_far": best_metric_seen,
            "selection_metric_tolerance": TASK2_CONTRASTIVE_SELECTION_METRIC_TOLERANCE,
            "selected_as_best": selected_as_best,
            "selection_reason": selection_reason,
        }

        print(
            f"[Epoch {epoch:02d}] "
            f"loss={train_metrics['loss']:.4f} "
            f"pos_cos={train_metrics['positive_cosine']:.4f} "
            f"mean_pair_cos={train_metrics['mean_pairwise_cosine']:.4f} "
            f"probe_val_acc={probe_metrics['val']['accuracy']:.4f} "
            f"probe_val_f1={probe_metrics['val']['f1_macro']:.4f} "
            f"selection_reason={selection_reason}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_metrics["loss"],
            "probe_metrics": probe_metrics,
            "selection_metric": selection_metric_name,
            "param_stats": param_stats,
            "config": {
                "objective": pretrain_objective,
                "batch_size": TASK2_CONTRASTIVE_BATCH_SIZE,
                "num_epochs": TASK2_CONTRASTIVE_NUM_EPOCHS,
                "learning_rate": TASK2_CONTRASTIVE_LEARNING_RATE,
                "weight_decay": TASK2_CONTRASTIVE_WEIGHT_DECAY,
                "temperature": TASK2_CONTRASTIVE_TEMPERATURE,
                "projection_hidden_dim": TASK2_CONTRASTIVE_PROJECTION_HIDDEN_DIM,
                "projection_dim": TASK2_CONTRASTIVE_PROJECTION_DIM,
                "feature_dropout": TASK2_CONTRASTIVE_FEATURE_DROPOUT,
                "selection_metric": selection_metric_name,
                "selection_metric_tolerance": TASK2_CONTRASTIVE_SELECTION_METRIC_TOLERANCE,
                "probe_max_iter": TASK2_CONTRASTIVE_PROBE_MAX_ITER,
            },
        }

        torch.save(checkpoint, checkpoint_dir / "last.pt")

        if train_metrics["loss"] < lowest_train_loss:
            lowest_train_loss = train_metrics["loss"]
            lowest_train_loss_epoch = epoch

        if should_update_best:
            best_selection_score = current_selection_score
            best_selection_tiebreak = current_tiebreak
            best_selected_loss = train_metrics["loss"]
            best_epoch = epoch
            best_probe_metrics = probe_metrics
            best_selection_reason = selection_reason
            row["selected_as_best"] = 1
            torch.save(checkpoint, checkpoint_dir / "best.pt")

        history.append(row)

    save_history_csv(history, log_dir / "history.csv")

    summary = {
        "best_epoch": best_epoch,
        "selection_metric": selection_metric_name,
        "selection_metric_tolerance": TASK2_CONTRASTIVE_SELECTION_METRIC_TOLERANCE,
        "selection_strategy": (
            "downstream_val_metric_with_lower_loss_preference_within_tolerance"
        ),
        "best_downstream_val_score": best_selection_score,
        "best_downstream_val_accuracy": (
            best_probe_metrics["val"]["accuracy"] if best_probe_metrics is not None else None
        ),
        "best_downstream_val_f1_macro": (
            best_probe_metrics["val"]["f1_macro"] if best_probe_metrics is not None else None
        ),
        "best_selected_train_loss": best_selected_loss if best_epoch != -1 else None,
        "best_selection_reason": best_selection_reason,
        "best_metric_seen": best_metric_seen if best_epoch != -1 else None,
        "lowest_train_loss": lowest_train_loss,
        "lowest_train_loss_epoch": lowest_train_loss_epoch,
        "pretrain_objective": pretrain_objective,
        "param_stats": param_stats,
        "contrastive_csv": str(TASK2_CONTRASTIVE_CSV),
        "train_csv": str(TASK2_TRAIN_CSV),
        "val_csv": str(TASK2_VAL_CSV),
    }
    with (log_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if best_probe_metrics is not None:
        with (log_dir / "best_downstream_probe_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(best_probe_metrics, f, indent=2)

    print(f"Best epoch: {best_epoch}")
    print(f"Best downstream val score ({selection_metric_name}): {best_selection_score:.4f}")
    if best_probe_metrics is not None:
        print(f"Best downstream val accuracy: {best_probe_metrics['val']['accuracy']:.4f}")
        print(f"Best downstream val macro F1: {best_probe_metrics['val']['f1_macro']:.4f}")
    if best_epoch != -1:
        print(f"Best selected train loss: {best_selected_loss:.4f}")
        print(f"Best selection reason: {best_selection_reason}")
    print(f"Lowest pretrain loss: {lowest_train_loss:.4f} (epoch {lowest_train_loss_epoch})")
    print(f"Saved best checkpoint to: {checkpoint_dir / 'best.pt'}")
    print(f"Saved logs to: {log_dir}")


if __name__ == "__main__":
    main()
