"""
Recompute the full metric table for the main saved checkpoints.

Outputs a single JSON file: outputs_analysis/all_metrics.json

Usage:
    conda run -n ML python3 src/compute_all_metrics.py
"""

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import NUM_CLASSES, CLASS_NAMES, DATA_ROOT, TissueDataset
from model_unet import UNet, BoundaryAwareGaborUNet, count_parameters
from model_autoencoder import Autoencoder, SegWithPretrainedEncoder
from metrics import ConfusionMatrix

BASE = pathlib.Path(__file__).parent.parent

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")


def evaluate_model(model, split):
    """Run one model over a split and accumulate confusion-matrix metrics."""
    ds = TissueDataset(
        image_dir=DATA_ROOT / split / "image",
        tissue_dir=DATA_ROOT / split / "tissue",
        augment=False,
    )
    cm = ConfusionMatrix(NUM_CLASSES)
    for idx in range(len(ds)):
        img_tensor, mask_tensor = ds[idx]
        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        cm.update(pred, mask_tensor.numpy())
    return cm.compute()


def metrics_to_dict(m, params):
    """Round metrics into a compact JSON-friendly structure."""
    def fl(arr):
        return [round(float(x), 4) for x in arr]
    return {
        "mean_dice":           round(float(m["mean_dice"]), 4),
        "mean_iou":            round(float(m["mean_iou"]), 4),
        "pixel_accuracy":      round(float(m["pixel_accuracy"]), 4),
        "mean_precision":      round(float(m["mean_precision"]), 4),
        "mean_recall":         round(float(m["mean_recall"]), 4),
        "dice_per_class":      fl(m["dice_per_class"]),
        "iou_per_class":       fl(m["iou_per_class"]),
        "precision_per_class": fl(m["precision_per_class"]),
        "recall_per_class":    fl(m["recall_per_class"]),
        "num_params":          params,
    }


def load_unet(path):
    model = UNet(in_channels=3, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to(device).eval()
    return model


def load_pretrained_seg(path):
    ae = Autoencoder(in_channels=3)
    model = SegWithPretrainedEncoder(ae, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to(device).eval()
    return model


def load_gabor_boundary_unet(path):
    model = BoundaryAwareGaborUNet(in_channels=3, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to(device).eval()
    return model


MODELS = [
    # Each entry names the run, how to build the model, and which checkpoint to load.
    ("trial1_unet_ce",
     load_unet,
     BASE / "outputs_trial1/unet/best_model.pth"),

    ("trial1_pretrained_ce",
     load_pretrained_seg,
     BASE / "outputs_trial1/pretrained_seg/best_model.pth"),

    ("trial3_unet_focal",
     load_unet,
     BASE / "outputs_focal/unet/best_model.pth"),

    ("trial3_pretrained_focal",
     load_pretrained_seg,
     BASE / "outputs_focal/pretrained_seg/best_model.pth"),

    ("trial4_unet_dice",
     load_unet,
     BASE / "outputs_dice/unet/best_model.pth"),

    ("trial4_pretrained_dice",
     load_pretrained_seg,
     BASE / "outputs_dice/pretrained_seg/best_model.pth"),

    ("trial6_unet_dice_focal",
     load_unet,
     BASE / "outputs_dice_focal/unet/best_model.pth"),

    ("trial6_pretrained_dice_focal",
     load_pretrained_seg,
     BASE / "outputs_dice_focal/pretrained_seg/best_model.pth"),

    ("trial7_unet_dice_ce",
     load_unet,
     BASE / "outputs_dice_ce/unet/best_model.pth"),

    ("trial7_pretrained_dice_ce",
     load_pretrained_seg,
     BASE / "outputs_dice_ce/pretrained_seg/best_model.pth"),

    ("g3_gabor_boundary",
     load_gabor_boundary_unet,
     BASE / "outputs_gabor_boundary_dice/unet/best_model.pth"),

    ("trialA_masked_mse_pretrained",
     load_pretrained_seg,
     BASE / "outputs_masked_mse/pretrained_seg/best_model.pth"),

    ("trialB_masked_mse_l1_pretrained",
     load_pretrained_seg,
     BASE / "outputs_masked_mse_l1/pretrained_seg/best_model.pth"),
]

all_results = {}

for label, loader_fn, ckpt_path in MODELS:
    if not ckpt_path.exists():
        print(f"SKIP {label}: checkpoint not found at {ckpt_path}")
        continue

    print(f"\n--- {label} ---")
    model = loader_fn(ckpt_path)
    params = count_parameters(model)
    print(f"  Parameters: {params:,}")

    entry = {}
    for split in ["validation", "test"]:
        print(f"  Evaluating {split}...", end=" ", flush=True)
        m = evaluate_model(model, split)
        entry[split] = metrics_to_dict(m, params)
        print(
            f"Dice={entry[split]['mean_dice']:.4f}  "
            f"Prec={entry[split]['mean_precision']:.4f}  "
            f"Rec={entry[split]['mean_recall']:.4f}"
        )

    all_results[label] = entry

out_dir = BASE / "outputs_analysis"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "all_metrics.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved to {out_path}")
