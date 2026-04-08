from __future__ import annotations

from collections import Counter
import copy
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
from torch.utils.data import DataLoader, WeightedRandomSampler
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

TASK2_USE_WEIGHTED_RANDOM_SAMPLER = get_config_value(
    "TASK2_USE_WEIGHTED_RANDOM_SAMPLER",
    True,
)

TASK2_WEIGHTED_SAMPLER_MODE = get_config_value(
    "TASK2_WEIGHTED_SAMPLER_MODE",
    "label_and_sample_type",
)

TASK2_FINETUNE_NUM_EPOCHS = get_config_value(
    "TASK2_FINETUNE_NUM_EPOCHS",
    10,
)

TASK2_FINETUNE_ENCODER_LEARNING_RATE = get_config_value(
    "TASK2_FINETUNE_ENCODER_LEARNING_RATE",
    1e-5,
)

TASK2_FINETUNE_CLASSIFIER_LEARNING_RATE = get_config_value(
    "TASK2_FINETUNE_CLASSIFIER_LEARNING_RATE",
    1e-4,
)

TASK2_FINETUNE_WEIGHT_DECAY = get_config_value(
    "TASK2_FINETUNE_WEIGHT_DECAY",
    1e-4,
)

TASK2_FULL_FINETUNE_NUM_EPOCHS = get_config_value(
    "TASK2_FULL_FINETUNE_NUM_EPOCHS",
    10,
)

TASK2_FULL_FINETUNE_ENCODER_LEARNING_RATE = get_config_value(
    "TASK2_FULL_FINETUNE_ENCODER_LEARNING_RATE",
    5e-6,
)

TASK2_FULL_FINETUNE_CLASSIFIER_LEARNING_RATE = get_config_value(
    "TASK2_FULL_FINETUNE_CLASSIFIER_LEARNING_RATE",
    5e-5,
)

TASK2_FULL_FINETUNE_WEIGHT_DECAY = get_config_value(
    "TASK2_FULL_FINETUNE_WEIGHT_DECAY",
    1e-4,
)

TASK2_EARLY_STOPPING_PATIENCE = get_config_value(
    "TASK2_EARLY_STOPPING_PATIENCE",
    5,
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


def build_dataloaders():
    train_transform, eval_transform = build_transforms()

    train_dataset, val_dataset = build_classification_datasets(
        train_csv=TASK2_TRAIN_CSV,
        val_csv=TASK2_VAL_CSV,
        train_transform=train_transform,
        eval_transform=eval_transform,
        return_metadata=False,
    )

    train_sampler, sampler_summary = build_weighted_train_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TASK2_BATCH_SIZE,
        shuffle=train_sampler is None,
        sampler=train_sampler,
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

    return train_loader, val_loader, sampler_summary


def build_sampler_group_keys(train_dataset) -> List[str]:
    train_df = train_dataset.df.copy()
    train_df["label"] = train_df["label"].astype(int)

    if TASK2_WEIGHTED_SAMPLER_MODE == "label":
        return [str(label) for label in train_df["label"].tolist()]

    if TASK2_WEIGHTED_SAMPLER_MODE == "label_and_sample_type":
        if "sample_type" not in train_df.columns:
            raise ValueError(
                "TASK2_WEIGHTED_SAMPLER_MODE='label_and_sample_type' requires "
                "'sample_type' column in the training CSV."
            )
        sample_types = train_df["sample_type"].fillna("unknown").astype(str).tolist()
        labels = train_df["label"].tolist()
        return [f"{label}|{sample_type}" for label, sample_type in zip(labels, sample_types)]

    raise ValueError(
        f"Unsupported TASK2_WEIGHTED_SAMPLER_MODE={TASK2_WEIGHTED_SAMPLER_MODE!r}. "
        "Expected 'label' or 'label_and_sample_type'."
    )


def build_weighted_train_sampler(train_dataset):
    summary = {
        "enabled": bool(TASK2_USE_WEIGHTED_RANDOM_SAMPLER),
        "mode": str(TASK2_WEIGHTED_SAMPLER_MODE),
        "group_counts": {},
        "group_weights": {},
    }

    if not TASK2_USE_WEIGHTED_RANDOM_SAMPLER:
        return None, summary

    group_keys = build_sampler_group_keys(train_dataset)
    group_counts = Counter(group_keys)
    sample_weights = [1.0 / group_counts[group_key] for group_key in group_keys]
    sample_weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)

    generator = torch.Generator()
    generator.manual_seed(TASK2_RANDOM_SEED)

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )

    summary["group_counts"] = {
        group_key: int(count)
        for group_key, count in sorted(group_counts.items())
    }
    summary["group_weights"] = {
        group_key: float(1.0 / count)
        for group_key, count in sorted(group_counts.items())
    }

    return sampler, summary


def print_sampler_summary(sampler_summary: Dict[str, object]) -> None:
    print(f"Weighted sampler enabled: {sampler_summary['enabled']}")
    print(f"Weighted sampler mode: {sampler_summary['mode']}")

    group_counts = sampler_summary.get("group_counts", {})
    if not group_counts:
        return

    print("Weighted sampler group counts:")
    for group_key, count in group_counts.items():
        print(f"  {group_key}: {count}")


class ContrastiveEncoderLinearClassifier(nn.Module):
    """
    Contrastive encoder + linear classifier wrapper supporting:
    1. frozen linear probing
    2. last-block fine-tuning
    3. full-encoder fine-tuning

    The projection head is always frozen because downstream classification
    uses encoder features only.
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
        self.encoder_frozen = True
        self.encoder_train_mode = "frozen"
        self.freeze_projection_head()
        self.freeze_encoder()

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self.encoder_model, "encoder"):
            self._set_encoder_module_mode(mode)

        if hasattr(self.encoder_model, "projection_head"):
            self.encoder_model.projection_head.eval()

        return self

    def _set_encoder_module_mode(self, mode: bool) -> None:
        encoder = self.encoder_model.encoder

        if self.encoder_train_mode == "frozen":
            encoder.eval()
            return

        if self.encoder_train_mode == "last_block":
            encoder.eval()
            if hasattr(encoder, "layer4"):
                encoder.layer4.train(mode)
            return

        if self.encoder_train_mode == "full":
            encoder.train(mode)
            return

        raise ValueError(f"Unsupported encoder_train_mode: {self.encoder_train_mode}")

    def freeze_projection_head(self) -> None:
        if hasattr(self.encoder_model, "freeze_projection_head"):
            self.encoder_model.freeze_projection_head()
        elif hasattr(self.encoder_model, "projection_head"):
            for p in self.encoder_model.projection_head.parameters():
                p.requires_grad = False

    def freeze_encoder(self) -> None:
        if hasattr(self.encoder_model, "freeze_encoder"):
            self.encoder_model.freeze_encoder()
        elif hasattr(self.encoder_model, "encoder"):
            for p in self.encoder_model.encoder.parameters():
                p.requires_grad = False
        else:
            for p in self.encoder_model.parameters():
                p.requires_grad = False
        self.encoder_frozen = True
        self.encoder_train_mode = "frozen"

    def unfreeze_last_block(self) -> None:
        if not hasattr(self.encoder_model, "encoder"):
            raise ValueError("Partial fine-tuning requires encoder_model.encoder.")

        self.freeze_encoder()

        if not hasattr(self.encoder_model.encoder, "layer4"):
            raise ValueError("Expected a ResNet-style encoder with layer4 for partial fine-tuning.")

        for p in self.encoder_model.encoder.layer4.parameters():
            p.requires_grad = True

        self.freeze_projection_head()
        self.encoder_frozen = False
        self.encoder_train_mode = "last_block"

    def unfreeze_encoder(self) -> None:
        if hasattr(self.encoder_model, "unfreeze_encoder"):
            self.encoder_model.unfreeze_encoder()
        elif hasattr(self.encoder_model, "encoder"):
            for p in self.encoder_model.encoder.parameters():
                p.requires_grad = True
        else:
            for p in self.encoder_model.parameters():
                p.requires_grad = True
        self.freeze_projection_head()
        self.encoder_frozen = False
        self.encoder_train_mode = "full"

    def get_last_block_parameters(self):
        if not hasattr(self.encoder_model, "encoder") or not hasattr(self.encoder_model.encoder, "layer4"):
            raise ValueError("Expected encoder.layer4 for partial fine-tuning.")
        return self.encoder_model.encoder.layer4.parameters()

    def get_full_encoder_parameters(self):
        if not hasattr(self.encoder_model, "encoder"):
            raise ValueError("Expected encoder_model.encoder for full fine-tuning.")
        return self.encoder_model.encoder.parameters()

    def _extract_encoder_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder_model, "encoder"):
            return self.encoder_model.encoder(x)
        if hasattr(self.encoder_model, "encode"):
            return self.encoder_model.encode(x, normalize=False)
        return self.encoder_model.extract_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_frozen:
            with torch.no_grad():
                features = self._extract_encoder_features(x)
        else:
            features = self._extract_encoder_features(x)

        features = self.dropout(features)
        logits = self.classifier(features)
        return logits

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_parameter_stats(self) -> Dict[str, object]:
        if hasattr(self.encoder_model, "encoder"):
            encoder_total = sum(p.numel() for p in self.encoder_model.encoder.parameters())
        else:
            encoder_total = sum(p.numel() for p in self.encoder_model.parameters())

        projection_total = sum(
            p.numel() for p in getattr(self.encoder_model, "projection_head", nn.Identity()).parameters()
        )
        head_total = sum(p.numel() for p in self.classifier.parameters())

        return {
            "total_params": self.total_parameters(),
            "trainable_params": self.trainable_parameters(),
            "encoder_backbone_params": encoder_total,
            "projection_head_params": projection_total,
            "classifier_head_params": head_total,
            "encoder_train_mode": self.encoder_train_mode,
        }


def build_stage1_optimizer(model: ContrastiveEncoderLinearClassifier) -> AdamW:
    return AdamW(
        model.classifier.parameters(),
        lr=TASK2_FROZEN_HEAD_LEARNING_RATE,
        weight_decay=TASK2_FROZEN_HEAD_WEIGHT_DECAY,
    )


def get_trainable_params(parameters) -> List[torch.nn.Parameter]:
    return [param for param in parameters if param.requires_grad]


def build_stage2_optimizer(model: ContrastiveEncoderLinearClassifier) -> AdamW:
    last_block_params = get_trainable_params(model.get_last_block_parameters())
    classifier_params = get_trainable_params(model.classifier.parameters())

    return AdamW(
        [
            {
                "params": last_block_params,
                "lr": TASK2_FINETUNE_ENCODER_LEARNING_RATE,
            },
            {
                "params": classifier_params,
                "lr": TASK2_FINETUNE_CLASSIFIER_LEARNING_RATE,
            },
        ],
        weight_decay=TASK2_FINETUNE_WEIGHT_DECAY,
    )


def build_stage3_optimizer(model: ContrastiveEncoderLinearClassifier) -> AdamW:
    encoder_params = get_trainable_params(model.get_full_encoder_parameters())
    classifier_params = get_trainable_params(model.classifier.parameters())

    return AdamW(
        [
            {
                "params": encoder_params,
                "lr": TASK2_FULL_FINETUNE_ENCODER_LEARNING_RATE,
            },
            {
                "params": classifier_params,
                "lr": TASK2_FULL_FINETUNE_CLASSIFIER_LEARNING_RATE,
            },
        ],
        weight_decay=TASK2_FULL_FINETUNE_WEIGHT_DECAY,
    )


def print_stage_header(stage_name: str, num_epochs: int, model: ContrastiveEncoderLinearClassifier, stage: Dict[str, object]) -> None:
    param_stats = model.get_parameter_stats()
    print(f"\nStarting stage: {stage_name}")
    print(f"Stage epochs: {num_epochs}")
    print(f"Encoder mode: {param_stats['encoder_train_mode']}")
    print(f"Trainable params: {param_stats['trainable_params']:,}")
    print(f"Encoder params: {param_stats['encoder_backbone_params']:,}")
    print(f"Projection head params (frozen): {param_stats['projection_head_params']:,}")
    print(f"Classifier head params: {param_stats['classifier_head_params']:,}")
    print(f"Encoder LR: {float(stage['encoder_lr']):.2e}")
    print(f"Classifier LR: {float(stage['classifier_lr']):.2e}")
    print(f"Weight decay: {float(stage['weight_decay']):.2e}")
    print(f"Early stopping patience: {int(stage['early_stopping_patience'])}")


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


def save_history_csv(history: List[Dict[str, object]], output_path: Path) -> None:
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

    train_loader, val_loader, sampler_summary = build_dataloaders()

    encoder_model, encoder_checkpoint_path = load_pretrained_contrastive_encoder(device)

    model = ContrastiveEncoderLinearClassifier(
        encoder_model=encoder_model,
        feature_dim=512,
        num_classes=TASK2_NUM_CLASSES,
        dropout=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    param_stats = model.get_parameter_stats()
    print(f"Loaded encoder from: {encoder_checkpoint_path}")
    print(f"Total params: {param_stats['total_params']:,}")
    print(f"Trainable params: {param_stats['trainable_params']:,}")
    print(f"Encoder backbone params: {param_stats['encoder_backbone_params']:,}")
    print(f"Projection head params: {param_stats['projection_head_params']:,}")
    print(f"Classifier head params: {param_stats['classifier_head_params']:,}")
    print_sampler_summary(sampler_summary)

    training_config = {
        "stage1_num_epochs": TASK2_FROZEN_HEAD_NUM_EPOCHS,
        "stage1_classifier_lr": TASK2_FROZEN_HEAD_LEARNING_RATE,
        "stage1_weight_decay": TASK2_FROZEN_HEAD_WEIGHT_DECAY,
        "use_weighted_random_sampler": bool(TASK2_USE_WEIGHTED_RANDOM_SAMPLER),
        "weighted_sampler_mode": str(TASK2_WEIGHTED_SAMPLER_MODE),
        "stage2_num_epochs": TASK2_FINETUNE_NUM_EPOCHS,
        "stage2_encoder_lr": TASK2_FINETUNE_ENCODER_LEARNING_RATE,
        "stage2_classifier_lr": TASK2_FINETUNE_CLASSIFIER_LEARNING_RATE,
        "stage2_weight_decay": TASK2_FINETUNE_WEIGHT_DECAY,
        "stage3_num_epochs": TASK2_FULL_FINETUNE_NUM_EPOCHS,
        "stage3_encoder_lr": TASK2_FULL_FINETUNE_ENCODER_LEARNING_RATE,
        "stage3_classifier_lr": TASK2_FULL_FINETUNE_CLASSIFIER_LEARNING_RATE,
        "stage3_weight_decay": TASK2_FULL_FINETUNE_WEIGHT_DECAY,
        "early_stopping_patience": TASK2_EARLY_STOPPING_PATIENCE,
    }

    stages = [
        {
            "name": "linear_probe",
            "num_epochs": TASK2_FROZEN_HEAD_NUM_EPOCHS,
            "setup": model.freeze_encoder,
            "optimizer_builder": build_stage1_optimizer,
            "encoder_lr": 0.0,
            "classifier_lr": TASK2_FROZEN_HEAD_LEARNING_RATE,
            "weight_decay": TASK2_FROZEN_HEAD_WEIGHT_DECAY,
            "early_stopping_patience": TASK2_EARLY_STOPPING_PATIENCE,
        },
        {
            "name": "last_block_finetune",
            "num_epochs": TASK2_FINETUNE_NUM_EPOCHS,
            "setup": model.unfreeze_last_block,
            "optimizer_builder": build_stage2_optimizer,
            "encoder_lr": TASK2_FINETUNE_ENCODER_LEARNING_RATE,
            "classifier_lr": TASK2_FINETUNE_CLASSIFIER_LEARNING_RATE,
            "weight_decay": TASK2_FINETUNE_WEIGHT_DECAY,
            "early_stopping_patience": TASK2_EARLY_STOPPING_PATIENCE,
        },
        {
            "name": "full_finetune",
            "num_epochs": TASK2_FULL_FINETUNE_NUM_EPOCHS,
            "setup": model.unfreeze_encoder,
            "optimizer_builder": build_stage3_optimizer,
            "encoder_lr": TASK2_FULL_FINETUNE_ENCODER_LEARNING_RATE,
            "classifier_lr": TASK2_FULL_FINETUNE_CLASSIFIER_LEARNING_RATE,
            "weight_decay": TASK2_FULL_FINETUNE_WEIGHT_DECAY,
            "early_stopping_patience": TASK2_EARLY_STOPPING_PATIENCE,
        },
    ]
    enabled_stages = [stage for stage in stages if stage["num_epochs"] > 0]
    if not enabled_stages:
        raise ValueError("No training stages enabled. Check stage epoch configuration.")

    history: List[Dict[str, object]] = []
    best_score = -1.0
    best_epoch = -1
    best_stage_name: Optional[str] = None
    best_stage_epoch = -1
    best_val_labels = None
    best_val_preds = None
    global_epoch = 0
    stage_summaries: List[Dict[str, object]] = []

    for stage in enabled_stages:
        stage_name = str(stage["name"])
        num_epochs = int(stage["num_epochs"])
        patience = int(stage["early_stopping_patience"])
        stage["setup"]()
        optimizer = stage["optimizer_builder"](model)
        print_stage_header(stage_name, num_epochs, model, stage)

        stage_best_score = float("-inf")
        stage_best_epoch = -1
        stage_best_global_epoch = -1
        stage_best_weights = copy.deepcopy(model.state_dict())
        stage_epochs_without_improvement = 0
        stage_stopped_early = False

        for stage_epoch in range(1, num_epochs + 1):
            global_epoch += 1

            train_metrics, _, _ = run_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                training=True,
                epoch=stage_epoch,
                total_epochs=num_epochs,
            )

            val_metrics, val_labels, val_preds = run_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                training=False,
                epoch=stage_epoch,
                total_epochs=num_epochs,
            )

            current_score = val_metrics["f1_macro"]
            current_param_stats = model.get_parameter_stats()
            improved_within_stage = current_score > stage_best_score

            if improved_within_stage:
                stage_best_score = current_score
                stage_best_epoch = stage_epoch
                stage_best_global_epoch = global_epoch
                stage_best_weights = copy.deepcopy(model.state_dict())
                stage_epochs_without_improvement = 0
            else:
                stage_epochs_without_improvement += 1

            row = {
                "epoch": global_epoch,
                "stage": stage_name,
                "stage_epoch": stage_epoch,
                "encoder_frozen": int(model.encoder_frozen),
                "encoder_mode": str(model.encoder_train_mode),
                "encoder_lr": float(stage["encoder_lr"]),
                "classifier_lr": float(stage["classifier_lr"]),
                "weight_decay": float(stage["weight_decay"]),
                "early_stopping_patience": patience,
                "epochs_without_improvement": stage_epochs_without_improvement,
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
                f"[{stage_name}][Epoch {stage_epoch:02d}/{num_epochs:02d}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['f1_macro']:.4f} "
                f"patience={stage_epochs_without_improvement}/{patience}"
            )

            checkpoint = {
                "epoch": global_epoch,
                "stage_name": stage_name,
                "stage_epoch": stage_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1_macro": current_score,
                "param_stats": current_param_stats,
                "class_names": CLASS_NAMES,
                "encoder_checkpoint_path": str(encoder_checkpoint_path),
                "training_config": training_config,
            }

            torch.save(checkpoint, checkpoint_dir / "last.pt")

            if current_score > best_score:
                best_score = current_score
                best_epoch = global_epoch
                best_stage_name = stage_name
                best_stage_epoch = stage_epoch
                best_val_labels = val_labels
                best_val_preds = val_preds
                torch.save(checkpoint, checkpoint_dir / "best.pt")

            if patience > 0 and stage_epochs_without_improvement >= patience:
                stage_stopped_early = True
                print(
                    f"[{stage_name}] early stopping triggered at stage epoch {stage_epoch} "
                    f"after {patience} non-improving epochs."
                )
                break

        model.load_state_dict(stage_best_weights)
        print(
            f"[{stage_name}] restored best stage weights from stage epoch {stage_best_epoch} "
            f"(global epoch {stage_best_global_epoch}) with val_f1={stage_best_score:.4f}"
        )
        stage_summaries.append(
            {
                "stage": stage_name,
                "planned_epochs": num_epochs,
                "completed_stage_epochs": stage_epoch,
                "best_stage_epoch": stage_best_epoch,
                "best_global_epoch": stage_best_global_epoch,
                "best_val_f1_macro": stage_best_score,
                "stopped_early": stage_stopped_early,
                "early_stopping_patience": patience,
            }
        )

    save_history_csv(history, log_dir / "history.csv")

    summary = {
        "best_epoch": best_epoch,
        "best_stage": best_stage_name,
        "best_stage_epoch": best_stage_epoch,
        "best_val_f1_macro": best_score,
        "final_param_stats": model.get_parameter_stats(),
        "encoder_checkpoint_path": str(encoder_checkpoint_path),
        "training_config": training_config,
        "weighted_sampler": sampler_summary,
        "stage_summaries": stage_summaries,
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
    print(f"Best stage: {best_stage_name} (stage epoch {best_stage_epoch})")
    print(f"Best val macro F1: {best_score:.4f}")
    print(f"Saved checkpoint to: {checkpoint_dir / 'best.pt'}")
    print(f"Saved logs to: {log_dir}")


if __name__ == "__main__":
    main()
