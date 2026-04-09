from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
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
    TASK2_RANDOM_SEED,
    TASK2_TEST_DIR,
    TASK2_TRAIN_CSV,
    TASK2_VAL_CSV,
)
from task2.data.dataset import Task2TestDataset, build_classification_datasets
from task2.models.contrastive_model import build_contrastive_model


CLASS_NAMES = ["Tumor", "Lymphocyte", "Histiocyte"]
CLASS_COLORS = {
    0: "#D55E00",
    1: "#009E73",
    2: "#0072B2",
}
SAMPLE_TYPE_MARKERS = {
    "primary": "o",
    "metastatic": "^",
    "unknown": "X",
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize contrastive latent representations in 2D.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "train", "val"],
        default="test",
        help="Which dataset split to visualize.",
    )
    parser.add_argument(
        "--checkpoint-type",
        choices=["contrastive_classifier", "contrastive"],
        default="contrastive_classifier",
        help="Which checkpoint to load.",
    )
    parser.add_argument(
        "--representation",
        choices=["features", "projections"],
        default="features",
        help="Use encoder features or projection-head embeddings.",
    )
    parser.add_argument(
        "--method",
        choices=["tsne", "pca"],
        default="tsne",
        help="2D reduction method.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional class-balanced subsample size before 2D reduction.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=None,
        help="Optional t-SNE perplexity. Must be smaller than the sample count.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional output file stem. Defaults to split_representation_method.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def build_loader(split: str) -> DataLoader:
    eval_transform = build_eval_transform()

    if split == "test":
        dataset = Task2TestDataset(
            root_dir=TASK2_TEST_DIR,
            transform=eval_transform,
            return_metadata=True,
        )
    else:
        train_dataset, val_dataset = build_classification_datasets(
            train_csv=TASK2_TRAIN_CSV,
            val_csv=TASK2_VAL_CSV,
            train_transform=eval_transform,
            eval_transform=eval_transform,
            return_metadata=True,
        )
        dataset = train_dataset if split == "train" else val_dataset

    return DataLoader(
        dataset,
        batch_size=TASK2_BATCH_SIZE,
        shuffle=False,
        num_workers=TASK2_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if TASK2_NUM_WORKERS > 0 else False,
    )


class ContrastiveEncoderLinearClassifier(nn.Module):
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
        self.freeze_projection_head()
        self.freeze_encoder()

    def freeze_projection_head(self) -> None:
        if hasattr(self.encoder_model, "freeze_projection_head"):
            self.encoder_model.freeze_projection_head()
        elif hasattr(self.encoder_model, "projection_head"):
            for param in self.encoder_model.projection_head.parameters():
                param.requires_grad = False

    def freeze_encoder(self) -> None:
        if hasattr(self.encoder_model, "freeze_encoder"):
            self.encoder_model.freeze_encoder()
        elif hasattr(self.encoder_model, "encoder"):
            for param in self.encoder_model.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.encoder_model.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self.encoder_model, "encoder"):
            self.encoder_model.encoder.eval()
        if hasattr(self.encoder_model, "projection_head"):
            self.encoder_model.projection_head.eval()
        return self

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if hasattr(self.encoder_model, "encode"):
                return self.encoder_model.encode(x, normalize=False)
            if hasattr(self.encoder_model, "extract_features"):
                return self.encoder_model.extract_features(x)
            return self.encoder_model.encoder(x)

    def extract_projections(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.extract_features(x)
            if hasattr(self.encoder_model, "project"):
                return self.encoder_model.project(features, normalize=True)
            raise ValueError("Loaded encoder does not expose a projection head.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.extract_features(x)
        return self.classifier(self.dropout(features))


def build_contrastive_backbone(device: torch.device):
    return build_contrastive_model(
        pretrained=False,
        freeze_encoder=False,
        feature_dropout=TASK2_CONTRASTIVE_FEATURE_DROPOUT,
        projection_hidden_dim=TASK2_CONTRASTIVE_PROJECTION_HIDDEN_DIM,
        projection_dim=TASK2_CONTRASTIVE_PROJECTION_DIM,
        use_projection_batchnorm=True,
        device=device,
    )


def load_model_bundle(checkpoint_type: str, device: torch.device):
    if checkpoint_type == "contrastive":
        checkpoint_path = Path(TASK2_CONTRASTIVE_ENCODER_CHECKPOINT)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = build_contrastive_backbone(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        return {
            "model": model,
            "checkpoint_path": checkpoint_path,
            "output_dir": Path(TASK2_OUTPUT_DIR) / "contrastive" / "eval" / "latent_space",
            "predict_fn": None,
            "feature_fn": lambda images: model.encode(images, normalize=False),
            "projection_fn": lambda images: model.encode_and_project(
                images,
                normalize_features=False,
                normalize_projection=True,
            )[1],
        }

    checkpoint_path = Path(TASK2_CONTRASTIVE_CLASSIFIER_CHECKPOINT)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    encoder_model = build_contrastive_backbone(device)
    model = ContrastiveEncoderLinearClassifier(
        encoder_model=encoder_model,
        feature_dim=512,
        num_classes=TASK2_NUM_CLASSES,
        dropout=0.2,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    return {
        "model": model,
        "checkpoint_path": checkpoint_path,
        "output_dir": Path(TASK2_OUTPUT_DIR) / "contrastive_classifier" / "eval" / "latent_space",
        "predict_fn": model,
        "feature_fn": model.extract_features,
        "projection_fn": model.extract_projections,
    }


def normalize_metadata_value(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def metadata_batch_to_rows(
    metadata: Dict[str, Sequence[object]],
    batch_size: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for sample_idx in range(batch_size):
        row = {}
        for key, values in metadata.items():
            row[key] = normalize_metadata_value(values[sample_idx])
        rows.append(row)
    return rows


@torch.no_grad()
def extract_embeddings(
    loader: DataLoader,
    feature_fn,
    projection_fn,
    predict_fn,
    representation: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, object]]]:
    all_embeddings: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_pred_labels: List[np.ndarray] = []
    all_metadata: List[Dict[str, object]] = []

    for images, labels, metadata in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if representation == "features":
            embeddings = feature_fn(images)
        else:
            embeddings = projection_fn(images)

        if predict_fn is None:
            pred_labels = torch.full_like(labels, fill_value=-1)
        else:
            logits = predict_fn(images)
            pred_labels = torch.argmax(logits, dim=1)

        all_embeddings.append(embeddings.detach().cpu().numpy().astype(np.float32))
        all_labels.append(labels.detach().cpu().numpy().astype(np.int64))
        all_pred_labels.append(pred_labels.detach().cpu().numpy().astype(np.int64))
        all_metadata.extend(metadata_batch_to_rows(metadata, batch_size=len(labels)))

    if not all_embeddings:
        raise ValueError("No embeddings were extracted from the requested split.")

    return (
        np.concatenate(all_embeddings, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_pred_labels, axis=0),
        all_metadata,
    )


def build_balanced_subsample_indices(
    labels: np.ndarray,
    max_samples: Optional[int],
    seed: int,
) -> np.ndarray:
    total_samples = len(labels)
    if max_samples is None or max_samples >= total_samples:
        return np.arange(total_samples)

    unique_labels = sorted(np.unique(labels).tolist())
    rng = np.random.default_rng(seed)
    label_to_indices = {
        label: rng.permutation(np.where(labels == label)[0])
        for label in unique_labels
    }

    selected: List[int] = []
    target_per_class = max_samples // max(len(unique_labels), 1)
    remaining_budget = max_samples

    for label in unique_labels:
        class_indices = label_to_indices[label]
        take = min(len(class_indices), target_per_class)
        selected.extend(class_indices[:take].tolist())
        label_to_indices[label] = class_indices[take:]
        remaining_budget -= take

    while remaining_budget > 0:
        progressed = False
        for label in unique_labels:
            class_indices = label_to_indices[label]
            if len(class_indices) == 0:
                continue
            selected.append(int(class_indices[0]))
            label_to_indices[label] = class_indices[1:]
            remaining_budget -= 1
            progressed = True
            if remaining_budget == 0:
                break
        if not progressed:
            break

    return np.array(sorted(selected), dtype=np.int64)


def resolve_tsne_perplexity(num_samples: int, requested_perplexity: Optional[float]) -> float:
    if num_samples < 3:
        raise ValueError("Need at least 3 samples to build a 2D latent-space visualization.")

    if requested_perplexity is not None:
        if requested_perplexity <= 0 or requested_perplexity >= num_samples:
            raise ValueError(
                f"Invalid perplexity {requested_perplexity}. It must satisfy 0 < perplexity < {num_samples}."
            )
        return float(requested_perplexity)

    auto_perplexity = min(30.0, max(5.0, (num_samples - 1) / 3.0))
    if auto_perplexity >= num_samples:
        auto_perplexity = max(1.0, num_samples - 1.0)
    return float(auto_perplexity)


def reduce_to_2d(
    embeddings: np.ndarray,
    method: str,
    seed: int,
    perplexity: Optional[float],
) -> Tuple[np.ndarray, Dict[str, object]]:
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    if method == "pca":
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(scaled_embeddings)
        reducer_summary = {
            "method": "pca",
            "explained_variance_ratio": [float(x) for x in reducer.explained_variance_ratio_],
        }
        return reduced.astype(np.float32), reducer_summary

    resolved_perplexity = resolve_tsne_perplexity(
        num_samples=len(embeddings),
        requested_perplexity=perplexity,
    )
    reducer = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=resolved_perplexity,
        random_state=seed,
    )
    reduced = reducer.fit_transform(scaled_embeddings)
    reducer_summary = {
        "method": "tsne",
        "perplexity": float(resolved_perplexity),
    }
    return reduced.astype(np.float32), reducer_summary


def get_sample_path(metadata: Dict[str, object]) -> str:
    if "path" in metadata:
        return str(metadata["path"])
    if "patch_path" in metadata:
        return str(metadata["patch_path"])
    return ""


def get_filename(metadata: Dict[str, object]) -> str:
    if "filename" in metadata:
        return str(metadata["filename"])

    sample_path = get_sample_path(metadata)
    if sample_path:
        return Path(sample_path).name

    return ""


def get_sample_type(metadata: Dict[str, object]) -> str:
    sample_type = str(metadata.get("sample_type", "unknown"))
    if sample_type not in SAMPLE_TYPE_MARKERS:
        return "unknown"
    return sample_type


def save_coordinates_csv(
    output_path: Path,
    reduced_embeddings: np.ndarray,
    labels: np.ndarray,
    pred_labels: np.ndarray,
    metadata_rows: List[Dict[str, object]],
    split: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "split",
        "x",
        "y",
        "label",
        "class_name",
        "pred_label",
        "pred_class_name",
        "sample_type",
        "path",
        "filename",
    ]

    extra_keys = sorted(
        {
            key
            for metadata in metadata_rows
            for key in metadata.keys()
            if key not in {"sample_type", "path", "filename", "patch_path"}
        }
    )
    fieldnames.extend(extra_keys)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for point, label, pred_label, metadata in zip(
            reduced_embeddings,
            labels,
            pred_labels,
            metadata_rows,
        ):
            row = {
                "split": split,
                "x": float(point[0]),
                "y": float(point[1]),
                "label": int(label),
                "class_name": CLASS_NAMES[int(label)],
                "pred_label": int(pred_label),
                "pred_class_name": CLASS_NAMES[int(pred_label)] if pred_label >= 0 else "",
                "sample_type": get_sample_type(metadata),
                "path": get_sample_path(metadata),
                "filename": get_filename(metadata),
            }
            for key in extra_keys:
                row[key] = normalize_metadata_value(metadata.get(key, ""))
            writer.writerow(row)


def plot_latent_space(
    output_path: Path,
    reduced_embeddings: np.ndarray,
    labels: np.ndarray,
    metadata_rows: List[Dict[str, object]],
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)

    sample_types = sorted({get_sample_type(metadata) for metadata in metadata_rows})
    unique_labels = sorted(np.unique(labels).tolist())

    for label in unique_labels:
        for sample_type in sample_types:
            mask = np.array([
                sample_label == label and get_sample_type(metadata) == sample_type
                for sample_label, metadata in zip(labels, metadata_rows)
            ])
            if not mask.any():
                continue
            ax.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                s=20,
                alpha=0.78,
                c=CLASS_COLORS.get(label, "#444444"),
                marker=SAMPLE_TYPE_MARKERS.get(sample_type, "o"),
                linewidths=0.25,
                edgecolors="white",
            )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.6)

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label=CLASS_NAMES[label],
            markerfacecolor=CLASS_COLORS.get(label, "#444444"),
            markeredgecolor="white",
            markersize=8,
            linestyle="None",
        )
        for label in unique_labels
    ]
    sample_type_handles = [
        Line2D(
            [0],
            [0],
            marker=SAMPLE_TYPE_MARKERS.get(sample_type, "o"),
            color="#555555",
            label=sample_type,
            markerfacecolor="#B0B0B0",
            markeredgecolor="#555555",
            markersize=8,
            linestyle="None",
        )
        for sample_type in sample_types
    ]

    class_legend = ax.legend(
        handles=class_handles,
        title="Class",
        loc="upper right",
        frameon=True,
    )
    ax.add_artist(class_legend)
    ax.legend(
        handles=sample_type_handles,
        title="Sample Type",
        loc="lower right",
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_summary_json(
    output_path: Path,
    split: str,
    checkpoint_type: str,
    checkpoint_path: Path,
    representation: str,
    reducer_summary: Dict[str, object],
    figure_path: Path,
    csv_path: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    metadata_rows: List[Dict[str, object]],
    max_samples: Optional[int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_counts = {
        CLASS_NAMES[int(label)]: int(count)
        for label, count in sorted(Counter(labels.tolist()).items())
    }
    sample_type_counts = Counter(get_sample_type(metadata) for metadata in metadata_rows)

    summary = {
        "split": split,
        "checkpoint_type": checkpoint_type,
        "checkpoint_path": str(checkpoint_path),
        "representation": representation,
        "embedding_dim": int(embeddings.shape[1]),
        "num_samples": int(len(labels)),
        "max_samples": max_samples,
        "class_counts": class_counts,
        "sample_type_counts": {key: int(value) for key, value in sorted(sample_type_counts.items())},
        "figure_path": str(figure_path),
        "csv_path": str(csv_path),
        "reducer": reducer_summary,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(TASK2_RANDOM_SEED)
    device = get_device()

    loader = build_loader(args.split)
    model_bundle = load_model_bundle(args.checkpoint_type, device)

    embeddings, labels, pred_labels, metadata_rows = extract_embeddings(
        loader=loader,
        feature_fn=model_bundle["feature_fn"],
        projection_fn=model_bundle["projection_fn"],
        predict_fn=model_bundle["predict_fn"],
        representation=args.representation,
        device=device,
    )

    selected_indices = build_balanced_subsample_indices(
        labels=labels,
        max_samples=args.max_samples,
        seed=TASK2_RANDOM_SEED,
    )
    embeddings = embeddings[selected_indices]
    labels = labels[selected_indices]
    pred_labels = pred_labels[selected_indices]
    metadata_rows = [metadata_rows[index] for index in selected_indices.tolist()]

    reduced_embeddings, reducer_summary = reduce_to_2d(
        embeddings=embeddings,
        method=args.method,
        seed=TASK2_RANDOM_SEED,
        perplexity=args.perplexity,
    )

    output_stem = args.output_name or f"{args.split}_{args.representation}_{args.method}"
    output_dir = model_bundle["output_dir"]
    figure_path = output_dir / f"{output_stem}.png"
    csv_path = output_dir / f"{output_stem}.csv"
    summary_path = output_dir / f"{output_stem}.json"

    title = (
        f"{args.checkpoint_type} latent space"
        f" | split={args.split}"
        f" | repr={args.representation}"
        f" | method={args.method.upper()}"
    )

    save_coordinates_csv(
        output_path=csv_path,
        reduced_embeddings=reduced_embeddings,
        labels=labels,
        pred_labels=pred_labels,
        metadata_rows=metadata_rows,
        split=args.split,
    )
    plot_latent_space(
        output_path=figure_path,
        reduced_embeddings=reduced_embeddings,
        labels=labels,
        metadata_rows=metadata_rows,
        title=title,
    )
    save_summary_json(
        output_path=summary_path,
        split=args.split,
        checkpoint_type=args.checkpoint_type,
        checkpoint_path=model_bundle["checkpoint_path"],
        representation=args.representation,
        reducer_summary=reducer_summary,
        figure_path=figure_path,
        csv_path=csv_path,
        embeddings=embeddings,
        labels=labels,
        metadata_rows=metadata_rows,
        max_samples=args.max_samples,
    )

    print("Saved 2D latent-space visualization")
    print(f"Checkpoint type : {args.checkpoint_type}")
    print(f"Checkpoint path : {model_bundle['checkpoint_path']}")
    print(f"Split           : {args.split}")
    print(f"Representation  : {args.representation}")
    print(f"Method          : {args.method}")
    print(f"Num samples     : {len(labels)}")
    if "perplexity" in reducer_summary:
        print(f"t-SNE perplexity: {reducer_summary['perplexity']:.2f}")
    print(f"Saved figure to : {figure_path}")
    print(f"Saved coords to : {csv_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
