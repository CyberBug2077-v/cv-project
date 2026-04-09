"""
Evaluate the U-Net and pretrained-segmentation models on validation and test data.

Computes per-class Dice, IoU, pixel accuracy for each split.
Saves comparison panels for all samples.
Identifies and saves hard samples (lowest per-sample mean Dice on test set).

Usage:
    conda run -n ML python3 src/evaluate.py <output_dir>

Example:
    conda run -n ML python3 src/evaluate.py outputs_trial1

Requires:
    <output_dir>/unet/best_model.pth
    <output_dir>/pretrained_seg/best_model.pth

Saves to:
    <output_dir>/visualisations/
    <output_dir>/evaluation_results.json
"""

import json
import sys
import pathlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import get_dataloaders, NUM_CLASSES, CLASS_NAMES, DATA_ROOT, TissueDataset
from model_unet import UNet, count_parameters
from model_autoencoder import Autoencoder, SegWithPretrainedEncoder
from metrics import ConfusionMatrix, format_metrics
from visualise import save_comparison, save_legend

# Read the output directory from the command line.

if len(sys.argv) < 2:
    print("Usage: python3 src/evaluate.py <output_dir>")
    sys.exit(1)

OUTPUT_DIR = pathlib.Path(sys.argv[1])
VIS_ROOT   = OUTPUT_DIR / "visualisations"
VIS_ROOT.mkdir(parents=True, exist_ok=True)

# Pick the available accelerator, with CPU as a fallback.

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")
print(f"Output dir: {OUTPUT_DIR}")


# Checkpoint loading helpers.

def load_unet(ckpt_path):
    model = UNet(in_channels=3, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device).eval()
    return model


def load_pretrained_seg(ckpt_path):
    ae = Autoencoder(in_channels=3)
    model = SegWithPretrainedEncoder(ae, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device).eval()
    return model


# Per-sample Dice used to rank hard examples.

def per_sample_dice(pred_np, target_np):
    """Mean Dice across classes for one sample."""
    dice_scores = []
    for c in range(NUM_CLASSES):
        tp = ((pred_np == c) & (target_np == c)).sum()
        fp = ((pred_np == c) & (target_np != c)).sum()
        fn = ((pred_np != c) & (target_np == c)).sum()
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        dice_scores.append(float(dice))
    return float(np.mean(dice_scores)), dice_scores


# Evaluate one model on one dataset split.

def evaluate_model(model, model_name, split):
    """Evaluate model on 'validation' or 'test' split. Returns metrics + per-sample results."""
    ds = TissueDataset(
        image_dir=DATA_ROOT / split / "image",
        tissue_dir=DATA_ROOT / split / "tissue",
        augment=False,
    )

    vis_dir = VIS_ROOT / split / model_name
    vis_dir.mkdir(parents=True, exist_ok=True)

    cm = ConfusionMatrix(NUM_CLASSES)
    sample_results = []

    for idx in range(len(ds)):
        img_tensor, mask_tensor = ds[idx]
        sample_name = ds.get_sample_name(idx)

        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0).to(device))
        pred   = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        target = mask_tensor.numpy()

        cm.update(pred, target)

        mean_d, per_class_d = per_sample_dice(pred, target)
        sample_results.append({
            "name":        sample_name,
            "mean_dice":   mean_d,
            "dice_other":  per_class_d[0],
            "dice_tumor":  per_class_d[1],
            "dice_stroma": per_class_d[2],
        })

        save_comparison(
            img_tensor=img_tensor,
            gt_mask=target,
            pred_mask=pred,
            save_path=vis_dir / f"{sample_name}.png",
            title=f"{model_name} | {split} | dice={mean_d:.3f}",
        )

    metrics = cm.compute()
    print(f"\n=== {model_name} — {split} ===")
    print(format_metrics(metrics, CLASS_NAMES))

    return metrics, sample_results


# Save the hardest test samples so they can be inspected later.

def save_hard_samples(unet_results, pretrained_results, unet_model, pretrained_model):
    hard_dir = VIS_ROOT / "test" / "hard_samples"
    hard_dir.mkdir(parents=True, exist_ok=True)

    # The hardest cases are defined by the U-Net's lowest mean Dice scores.
    sorted_results = sorted(unet_results, key=lambda r: r["mean_dice"])
    hard_n = min(10, len(sorted_results))

    test_ds = TissueDataset(
        image_dir=DATA_ROOT / "test" / "image",
        tissue_dir=DATA_ROOT / "test" / "tissue",
        augment=False,
    )
    name_to_idx      = {test_ds.get_sample_name(i): i for i in range(len(test_ds))}
    pretrained_by_name = {r["name"]: r for r in pretrained_results}

    hard_records = []
    for result in sorted_results[:hard_n]:
        name = result["name"]
        idx  = name_to_idx.get(name)
        if idx is None:
            continue

        img_tensor, mask_tensor = test_ds[idx]
        target = mask_tensor.numpy()

        with torch.no_grad():
            pred_unet = unet_model(img_tensor.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu().numpy()
            pred_pre  = pretrained_model(img_tensor.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu().numpy()

        save_comparison(img_tensor, target, pred_unet,
                        hard_dir / f"{name}_unet.png",
                        title=f"U-Net dice={result['mean_dice']:.3f}")
        save_comparison(img_tensor, target, pred_pre,
                        hard_dir / f"{name}_pretrained.png",
                        title=f"Pretrained dice={pretrained_by_name[name]['mean_dice']:.3f}")

        hard_records.append({
            "name":             name,
            "unet_dice":        result["mean_dice"],
            "pretrained_dice":  pretrained_by_name[name]["mean_dice"],
            "unet_dice_other":  result["dice_other"],
            "unet_dice_tumor":  result["dice_tumor"],
            "unet_dice_stroma": result["dice_stroma"],
        })
        print(f"  Hard: {name}  unet={result['mean_dice']:.3f}  "
              f"pretrained={pretrained_by_name[name]['mean_dice']:.3f}")

    return hard_records


def metrics_dict(m, model, sample_results):
    return {
        "mean_dice":      float(m["mean_dice"]),
        "mean_iou":       float(m["mean_iou"]),
        "pixel_accuracy": float(m["pixel_accuracy"]),
        "dice_per_class": m["dice_per_class"].tolist(),
        "iou_per_class":  m["iou_per_class"].tolist(),
        "num_params":     count_parameters(model),
        "per_sample":     sample_results,
    }


# Main entry point.

if __name__ == "__main__":
    save_legend(VIS_ROOT / "legend.png")

    unet_ckpt = OUTPUT_DIR / "unet" / "best_model.pth"
    pre_ckpt  = OUTPUT_DIR / "pretrained_seg" / "best_model.pth"

    if not unet_ckpt.exists():
        raise FileNotFoundError(f"Missing {unet_ckpt}")
    if not pre_ckpt.exists():
        raise FileNotFoundError(f"Missing {pre_ckpt}")

    unet_model       = load_unet(unet_ckpt)
    pretrained_model = load_pretrained_seg(pre_ckpt)

    print(f"U-Net params      : {count_parameters(unet_model):,}")
    print(f"Pretrained params : {count_parameters(pretrained_model):,}")

    # Run the full validation split for both models.
    unet_val_m,  unet_val_r  = evaluate_model(unet_model,       "unet",          "validation")
    pre_val_m,   pre_val_r   = evaluate_model(pretrained_model,  "pretrained_seg", "validation")

    # Then repeat on the held-out test split.
    unet_test_m, unet_test_r = evaluate_model(unet_model,       "unet",          "test")
    pre_test_m,  pre_test_r  = evaluate_model(pretrained_model,  "pretrained_seg", "test")

    # Save side-by-side outputs for the hardest test cases.
    print("\n=== Hard Sample Analysis (10 lowest U-Net Dice on test set) ===")
    hard_records = save_hard_samples(
        unet_test_r, pre_test_r, unet_model, pretrained_model
    )

    # Write the metrics and per-sample results to disk.
    results = {
        "unet": {
            "validation": metrics_dict(unet_val_m,  unet_model, unet_val_r),
            "test":       metrics_dict(unet_test_m, unet_model, unet_test_r),
        },
        "pretrained_seg": {
            "validation": metrics_dict(pre_val_m,  pretrained_model, pre_val_r),
            "test":       metrics_dict(pre_test_m, pretrained_model, pre_test_r),
        },
        "hard_samples": hard_records,
    }

    out_path = OUTPUT_DIR / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")

    print("\n=== Summary ===")
    for label, um, pm in [("Validation", unet_val_m, pre_val_m),
                           ("Test",       unet_test_m, pre_test_m)]:
        print(f"\n  {label}:")
        print(f"    U-Net      Dice={um['mean_dice']:.4f}  IoU={um['mean_iou']:.4f}  PixAcc={um['pixel_accuracy']:.4f}")
        print(f"    Pretrained Dice={pm['mean_dice']:.4f}  IoU={pm['mean_iou']:.4f}  PixAcc={pm['pixel_accuracy']:.4f}")
