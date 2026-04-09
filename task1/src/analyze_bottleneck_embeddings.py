"""
Analyse how cleanly different encoders separate tissue mixtures in bottleneck space.

For each selected test image, this script:
1. Resizes the image and mask to 256x256 using the same pipeline as training.
2. Runs a frozen encoder and extracts its bottleneck tensor of shape
   (1024, 16, 16).
3. Treats each bottleneck spatial location as one datapoint with a 1024-d
   feature vector.
4. Computes the class mixture of the corresponding 16x16 mask patch as a soft
   label vector [p_other, p_tumor, p_stroma].
5. Compares a random frozen encoder against a pretrained frozen encoder using
   PCA / UMAP plots and a small set of composition-alignment metrics.

Usage:
    conda run -n ML python3 src/analyze_bottleneck_embeddings.py

Saves to:
    outputs_analysis/bottleneck_patch_purity/
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from data import IMAGE_SIZE, geojson_to_mask
from model_autoencoder import Autoencoder
from model_unet import UNet, BoundaryAwareGaborUNet


SHORTLIST_PATH = pathlib.Path("outputs_analysis/top5_testset_class_area.json")
OUTPUT_DIR = pathlib.Path("outputs_analysis/bottleneck_patch_purity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


ENCODER_SPECS = [
    {
        "key": "random_frozen_encoder",
        "label": "Random frozen encoder",
        "kind": "autoencoder",
        "checkpoint": None,
        "seed": 42,
    },
    {
        "key": "pretrained_frozen_encoder",
        "label": "Pretrained frozen encoder (Masked MSE)",
        "kind": "autoencoder",
        "checkpoint": pathlib.Path("outputs_masked_mse/autoencoder/best_autoencoder.pth"),
        "seed": None,
    },
    {
        "key": "baseline_unet_ce",
        "label": "Baseline U-Net encoder (CE)",
        "kind": "unet",
        "checkpoint": pathlib.Path("outputs_trial1/unet/best_model.pth"),
        "seed": None,
    },
    {
        "key": "best_unet_gabor_boundary",
        "label": "Best U-Net encoder (Gabor boundary)",
        "kind": "gabor_boundary_unet",
        "checkpoint": pathlib.Path("outputs_gabor_boundary_dice/unet/best_model.pth"),
        "seed": None,
    },
]


BASE_COLORS = np.array(
    [
        [44, 109, 178],   # Other
        [192, 57, 43],    # Tumor
        [46, 139, 87],    # Stroma
    ],
    dtype=np.float32,
) / 255.0


CLASS_NAMES = ["Other", "Tumor", "Stroma"]


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_shortlist_images(shortlist_path: pathlib.Path):
    with shortlist_path.open() as f:
        raw = json.load(f)["top5_by_class"]

    samples = {}
    for group_name, entries in raw.items():
        for entry in entries:
            stem = entry["stem"]
            item = samples.setdefault(
                stem,
                {
                    "stem": stem,
                    "image_path": entry["image_path"],
                    "source_groups": [],
                },
            )
            item["source_groups"].append(
                {
                    "group": group_name,
                    "percentage": float(entry["percentage"]),
                }
            )

    return [samples[stem] for stem in sorted(samples)]


def build_image_transform():
    return transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_image_tensor(image_path: pathlib.Path, image_transform):
    image = Image.open(image_path).convert("RGB")
    return image_transform(image).unsqueeze(0)


def load_resized_mask(image_path: pathlib.Path):
    tissue_dir = image_path.parent.parent / "tissue"
    geojson_path = tissue_dir / f"{image_path.stem}_tissue.geojson"

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    mask_np = geojson_to_mask(geojson_path, height=orig_h, width=orig_w)
    mask_pil = Image.fromarray(mask_np.astype(np.uint8), mode="L")
    mask_pil = mask_pil.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.NEAREST)
    return np.array(mask_pil, dtype=np.int64)


def compute_patch_mixtures(mask_np):
    patch_size = IMAGE_SIZE // 16
    mixtures = np.zeros((16, 16, 3), dtype=np.float32)

    for row in range(16):
        for col in range(16):
            patch = mask_np[
                row * patch_size:(row + 1) * patch_size,
                col * patch_size:(col + 1) * patch_size,
            ]
            total = float(patch.size)
            for cls_idx in range(3):
                mixtures[row, col, cls_idx] = float(np.sum(patch == cls_idx)) / total

    return mixtures


def load_encoder(spec, device):
    kind = spec["kind"]

    if kind == "autoencoder":
        if spec["seed"] is not None:
            torch.manual_seed(spec["seed"])
        encoder = Autoencoder(in_channels=3)
        if spec["checkpoint"] is not None:
            encoder.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu"))
    elif kind == "unet":
        encoder = UNet(in_channels=3, num_classes=3)
        encoder.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu"))
    elif kind == "gabor_boundary_unet":
        encoder = BoundaryAwareGaborUNet(in_channels=3, num_classes=3)
        encoder.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu"))
    else:
        raise ValueError(f"Unknown encoder kind: {kind}")

    encoder = encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def extract_spatial_features(encoder, image_tensor, device):
    with torch.no_grad():
        x = image_tensor.to(device)
        if hasattr(encoder, "encode"):
            _, _, _, _, bottleneck = encoder.encode(x)
        elif hasattr(encoder, "encoder_features"):
            _, _, _, _, bottleneck = encoder.encoder_features(x)
        else:
            raise AttributeError("Encoder does not expose encode() or encoder_features()")
        bottleneck = bottleneck.detach().cpu().numpy()[0]
    return np.transpose(bottleneck, (1, 2, 0))


def standardize_features(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (x - mean) / std


def compute_pca(x):
    x_centered = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x_centered, full_matrices=False)
    coords = u[:, :2] * s[:2]
    explained = (s ** 2) / max(np.sum(s ** 2), 1e-12)
    return coords, explained[:2]


def compute_umap(x):
    with contextlib.redirect_stderr(io.StringIO()):
        from umap import UMAP

    n_neighbors = min(20, max(5, x.shape[0] // 40))
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    return reducer.fit_transform(x)


def build_rgba_from_mixtures(mixtures):
    rgb = mixtures @ BASE_COLORS
    purity = mixtures.max(axis=1)
    alpha = 0.2 + 0.75 * purity
    return np.concatenate([rgb, alpha[:, None]], axis=1), purity


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def sample_distance_correlation(features, mixtures, n_pairs=50000, seed=42):
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    keep = idx_a != idx_b
    idx_a = idx_a[keep]
    idx_b = idx_b[keep]

    feat_dist = np.linalg.norm(features[idx_a] - features[idx_b], axis=1)
    mix_dist = np.linalg.norm(mixtures[idx_a] - mixtures[idx_b], axis=1)
    return pearson_corr(feat_dist, mix_dist)


def pairwise_squared_distances(x):
    x = x.astype(np.float32, copy=False)
    squared_norm = np.sum(x * x, axis=1, keepdims=True)
    d2 = squared_norm + squared_norm.T - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def compute_knn_metrics(features, mixtures, k=10):
    dominant = mixtures.argmax(axis=1)
    purity = mixtures.max(axis=1)

    d2 = pairwise_squared_distances(features)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argpartition(d2, kth=k, axis=1)[:, :k]

    pred_mixture = mixtures[nn_idx].mean(axis=1)
    composition_mse = float(np.mean((pred_mixture - mixtures) ** 2))
    dominant_agreement = float(np.mean(dominant[nn_idx] == dominant[:, None]))

    pure_tumor = (dominant == 1) & (purity >= 0.75)
    pure_stroma = (dominant == 2) & (purity >= 0.75)
    mixed = purity < 0.60

    def rate(mask):
        if int(mask.sum()) == 0:
            return None
        return float(np.mean(dominant[nn_idx[mask]] == dominant[mask, None]))

    return {
        "knn_k": k,
        "composition_knn_mse": composition_mse,
        "dominant_class_neighbor_agreement": dominant_agreement,
        "pure_tumor_neighbor_agreement": rate(pure_tumor),
        "pure_stroma_neighbor_agreement": rate(pure_stroma),
        "num_pure_tumor_locations": int(pure_tumor.sum()),
        "num_pure_stroma_locations": int(pure_stroma.sum()),
        "num_mixed_locations": int(mixed.sum()),
    }


def normalize_simplex(pred):
    pred = np.clip(pred, 0.0, None)
    row_sum = pred.sum(axis=1, keepdims=True)
    empty = row_sum[:, 0] < 1e-12
    if np.any(empty):
        pred[empty] = 1.0 / pred.shape[1]
        row_sum = pred.sum(axis=1, keepdims=True)
    return pred / row_sum


def ridge_regression_predict(train_x, train_y, test_x, l2=1.0):
    train_mean = train_x.mean(axis=0, keepdims=True)
    train_std = train_x.std(axis=0, keepdims=True)
    train_std[train_std < 1e-8] = 1.0

    train_x = (train_x - train_mean) / train_std
    test_x = (test_x - train_mean) / train_std

    train_x = np.concatenate(
        [train_x, np.ones((train_x.shape[0], 1), dtype=train_x.dtype)],
        axis=1,
    )
    test_x = np.concatenate(
        [test_x, np.ones((test_x.shape[0], 1), dtype=test_x.dtype)],
        axis=1,
    )

    reg = np.eye(train_x.shape[1], dtype=train_x.dtype) * l2
    reg[-1, -1] = 0.0

    weights = np.linalg.solve(train_x.T @ train_x + reg, train_x.T @ train_y)
    return test_x @ weights


def evaluate_local_proportion_prediction(features, mixtures, stems, l2=1.0):
    unique_stems = sorted(set(stems))
    pred_rows = []
    fold_rows = []

    for stem in unique_stems:
        test_mask = np.array([item == stem for item in stems], dtype=bool)
        train_mask = ~test_mask

        pred = ridge_regression_predict(
            features[train_mask],
            mixtures[train_mask],
            features[test_mask],
            l2=l2,
        )
        pred = normalize_simplex(pred)
        true = mixtures[test_mask]

        pred_rows.append(pred)
        dominant_true = true.argmax(axis=1)
        dominant_pred = pred.argmax(axis=1)

        fold_rows.append(
            {
                "held_out_stem": stem,
                "num_locations": int(test_mask.sum()),
                "dominant_accuracy": float(np.mean(dominant_pred == dominant_true)),
                "proportion_mse": float(np.mean((pred - true) ** 2)),
            }
        )

    pred = np.concatenate(pred_rows, axis=0)
    true = mixtures
    dominant_true = true.argmax(axis=1)
    dominant_pred = pred.argmax(axis=1)

    class_corr = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_corr[class_name.lower()] = pearson_corr(pred[:, class_idx], true[:, class_idx])

    return {
        "regressor": "ridge",
        "ridge_l2": l2,
        "dominant_class_accuracy": float(np.mean(dominant_pred == dominant_true)),
        "proportion_mse": float(np.mean((pred - true) ** 2)),
        "class_proportion_correlation": class_corr,
        "per_image_folds": fold_rows,
        "predicted_proportions": pred,
    }


def plot_projection_grid(results, output_path):
    order = []
    for encoder_key in results:
        order.append((encoder_key, "pca"))
        order.append((encoder_key, "umap"))

    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    axes = np.atleast_2d(axes)

    for ax, (encoder_key, method) in zip(axes.ravel(), order):
        item = results[encoder_key]
        coords = item[method]
        rgba = item["rgba"]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=8,
            c=rgba,
            linewidths=0,
        )
        ax.set_title(f"{item['label']} — {method.upper()}")
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        ax.grid(alpha=0.18, linewidth=0.5)

    handles = []
    for class_name, color in zip(CLASS_NAMES, BASE_COLORS):
        handle = plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8,
            linewidth=0,
            label=class_name,
        )
        handles.append(handle)

    fig.legend(
        handles=handles,
        labels=CLASS_NAMES,
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Bottleneck patch-mixture projections\n"
        "Point colour is the soft class composition of the aligned 16x16 mask patch",
        y=0.98,
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_comparison(results, output_path):
    labels = [results[key]["label"] for key in results]
    accuracy = [
        results[key]["prediction_metrics"]["dominant_class_accuracy"]
        for key in results
    ]
    mse = [
        results[key]["prediction_metrics"]["proportion_mse"]
        for key in results
    ]
    colors = ["#6c8ebf", "#7cb37c", "#d5a253", "#b07cc6"][: len(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(labels, accuracy, color=colors)
    axes[0].set_title("Dominant class accuracy\n(leave-one-image-out)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].tick_params(axis="x", rotation=12)

    axes[1].bar(labels, mse, color=colors)
    axes[1].set_title("Class proportion MSE\n(leave-one-image-out)")
    axes[1].set_ylabel("MSE")
    axes[1].tick_params(axis="x", rotation=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_metadata(rows, output_path):
    with output_path.open("w") as f:
        json.dump(rows, f, indent=2)


def save_patch_table(rows, output_path):
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stem",
                "row",
                "col",
                "p_other",
                "p_tumor",
                "p_stroma",
                "dominant_class",
                "purity",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["stem"],
                    row["row"],
                    row["col"],
                    row["mixture"][0],
                    row["mixture"][1],
                    row["mixture"][2],
                    row["dominant_class"],
                    row["purity"],
                ]
            )


def save_projection_coordinates(results, patch_rows, output_path):
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "encoder_key",
                "encoder_label",
                "method",
                "stem",
                "row",
                "col",
                "p_other",
                "p_tumor",
                "p_stroma",
                "x",
                "y",
            ]
        )
        for encoder_key, item in results.items():
            for method in ["pca", "umap"]:
                coords = item[method]
                for idx, patch in enumerate(patch_rows):
                    writer.writerow(
                        [
                            encoder_key,
                            item["label"],
                            method,
                            patch["stem"],
                            patch["row"],
                            patch["col"],
                            patch["mixture"][0],
                            patch["mixture"][1],
                            patch["mixture"][2],
                            float(coords[idx, 0]),
                            float(coords[idx, 1]),
                        ]
                    )


def save_prediction_rows(results, patch_rows, output_path):
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "encoder_key",
                "encoder_label",
                "stem",
                "row",
                "col",
                "true_other",
                "true_tumor",
                "true_stroma",
                "pred_other",
                "pred_tumor",
                "pred_stroma",
                "true_dominant",
                "pred_dominant",
            ]
        )
        for encoder_key, item in results.items():
            pred = item["prediction_metrics"]["predicted_proportions"]
            for idx, patch in enumerate(patch_rows):
                true = np.asarray(patch["mixture"], dtype=np.float32)
                pred_vec = pred[idx]
                writer.writerow(
                    [
                        encoder_key,
                        item["label"],
                        patch["stem"],
                        patch["row"],
                        patch["col"],
                        true[0],
                        true[1],
                        true[2],
                        float(pred_vec[0]),
                        float(pred_vec[1]),
                        float(pred_vec[2]),
                        patch["dominant_class"],
                        CLASS_NAMES[int(np.argmax(pred_vec))],
                    ]
                )


def main():
    device = select_device()
    print(f"Device: {device}")

    shortlist = load_shortlist_images(SHORTLIST_PATH)
    image_transform = build_image_transform()

    print(f"Selected unique shortlist images: {len(shortlist)}")
    for sample in shortlist:
        source = ", ".join(
            f"{item['group']}={item['percentage']:.4f}%"
            for item in sorted(sample["source_groups"], key=lambda x: x["group"])
        )
        print(f"  {sample['stem']} ({source})")

    patch_rows = []
    feature_bank = {}
    patch_stems = []

    for spec in ENCODER_SPECS:
        print(f"\nLoading {spec['label']}")
        encoder = load_encoder(spec, device)

        encoder_features = []
        encoder_mixtures = []
        bottleneck_shape = None

        for sample in shortlist:
            image_path = pathlib.Path(sample["image_path"])
            image_tensor = load_image_tensor(image_path, image_transform)
            mask_np = load_resized_mask(image_path)
            mixture_map = compute_patch_mixtures(mask_np)
            feature_map = extract_spatial_features(encoder, image_tensor, device)

            bottleneck_shape = [feature_map.shape[2], feature_map.shape[0], feature_map.shape[1]]

            for row in range(16):
                for col in range(16):
                    mixture = mixture_map[row, col]
                    encoder_features.append(feature_map[row, col])
                    encoder_mixtures.append(mixture)

                    if spec["key"] == ENCODER_SPECS[0]["key"]:
                        patch_rows.append(
                            {
                                "stem": sample["stem"],
                                "row": row,
                                "col": col,
                                "mixture": mixture.tolist(),
                                "dominant_class": CLASS_NAMES[int(np.argmax(mixture))],
                                "purity": float(np.max(mixture)),
                            }
                        )
                        patch_stems.append(sample["stem"])

        encoder_features = np.asarray(encoder_features, dtype=np.float32)
        encoder_mixtures = np.asarray(encoder_mixtures, dtype=np.float32)
        scaled_features = standardize_features(encoder_features)

        pca_coords, pca_explained = compute_pca(scaled_features)
        umap_coords = compute_umap(scaled_features)
        rgba, _ = build_rgba_from_mixtures(encoder_mixtures)

        metrics = {
            "bottleneck_shape": list(bottleneck_shape),
            "num_locations": int(encoder_features.shape[0]),
            "pca_explained_variance": [float(x) for x in pca_explained],
            "feature_composition_distance_corr": sample_distance_correlation(
                scaled_features,
                encoder_mixtures,
            ),
        }
        metrics.update(compute_knn_metrics(scaled_features, encoder_mixtures, k=10))
        prediction_metrics = evaluate_local_proportion_prediction(
            encoder_features.astype(np.float64),
            encoder_mixtures.astype(np.float64),
            patch_stems,
            l2=1.0,
        )

        feature_bank[spec["key"]] = encoder_features
        feature_bank[f"{spec['key']}_mixtures"] = encoder_mixtures
        feature_bank[f"{spec['key']}_pca"] = pca_coords.astype(np.float32)
        feature_bank[f"{spec['key']}_umap"] = umap_coords.astype(np.float32)
        feature_bank[f"{spec['key']}_predicted_proportions"] = prediction_metrics[
            "predicted_proportions"
        ].astype(np.float32)

        feature_bank[f"{spec['key']}_rgba"] = rgba.astype(np.float32)

        feature_bank[f"{spec['key']}_purity"] = encoder_mixtures.max(axis=1).astype(np.float32)

        print(
            f"  bottleneck feature map={bottleneck_shape}  "
            f"locations={encoder_features.shape[0]}  "
            f"corr={metrics['feature_composition_distance_corr']:.4f}  "
            f"knn_mse={metrics['composition_knn_mse']:.4f}  "
            f"dominant_acc={prediction_metrics['dominant_class_accuracy']:.4f}  "
            f"prop_mse={prediction_metrics['proportion_mse']:.4f}"
        )

        spec["metrics"] = metrics
        spec["prediction_metrics"] = prediction_metrics

    results = {}
    for spec in ENCODER_SPECS:
        key = spec["key"]
        results[key] = {
            "label": spec["label"],
            "pca": feature_bank[f"{key}_pca"],
            "umap": feature_bank[f"{key}_umap"],
            "rgba": feature_bank[f"{key}_rgba"],
            "metrics": spec["metrics"],
            "prediction_metrics": spec["prediction_metrics"],
        }

    np.savez(OUTPUT_DIR / "patch_features_and_projections.npz", **feature_bank)
    save_metadata(shortlist, OUTPUT_DIR / "selected_images.json")
    save_metadata(
        {
            spec["key"]: {
                "label": spec["label"],
                **spec["metrics"],
                "prediction_metrics": {
                    k: v
                    for k, v in spec["prediction_metrics"].items()
                    if k != "predicted_proportions"
                },
            }
            for spec in ENCODER_SPECS
        },
        OUTPUT_DIR / "analysis_summary.json",
    )
    save_patch_table(patch_rows, OUTPUT_DIR / "patch_mixtures.csv")
    save_projection_coordinates(results, patch_rows, OUTPUT_DIR / "projection_coordinates.csv")
    save_prediction_rows(results, patch_rows, OUTPUT_DIR / "prediction_rows.csv")
    plot_projection_grid(results, OUTPUT_DIR / "projection_grid.png")
    plot_prediction_comparison(results, OUTPUT_DIR / "prediction_comparison.png")

    print(f"\nSaved analysis outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
