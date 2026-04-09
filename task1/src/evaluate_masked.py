"""
Evaluate masked-pretrained segmentation models against the standard pretrained baseline.

Compares the new masked pretrained seg model against the Trial 4 standard
pretrained seg (outputs_dice/pretrained_seg/best_model.pth) on both validation
and test sets.  Hard sample analysis highlights cases where masked pretraining
helps or hurts relative to the standard MSE baseline.

Usage:
    python3 src/evaluate_masked.py <output_dir>

Example:
    python3 src/evaluate_masked.py outputs_masked_mse
    python3 src/evaluate_masked.py outputs_masked_mse_l1

Requires:
    <output_dir>/pretrained_seg/best_model.pth    (masked-pretrained model)
    outputs_dice/pretrained_seg/best_model.pth    (Trial 4 standard baseline)

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
from model_autoencoder import Autoencoder, SegWithPretrainedEncoder
from metrics import ConfusionMatrix, format_metrics
from visualise import save_comparison, save_legend

# Read the trial output directory from the command line.

if len(sys.argv) < 2:
    print("Usage: python3 src/evaluate_masked.py <output_dir>")
    sys.exit(1)

OUTPUT_DIR = pathlib.Path(sys.argv[1])
VIS_ROOT   = OUTPUT_DIR / "visualisations"
VIS_ROOT.mkdir(parents=True, exist_ok=True)

BASELINE_CKPT = pathlib.Path("outputs_dice/pretrained_seg/best_model.pth")

# Pick the available accelerator, with CPU as a fallback.

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")
print(f"Output dir: {OUTPUT_DIR}")

# Checkpoint loading helper.

def load_pretrained_seg(ckpt_path):
    ae = Autoencoder(in_channels=3)
    model = SegWithPretrainedEncoder(ae, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device).eval()
    return model

# Per-sample Dice used to rank hard examples.

def per_sample_dice(pred_np, target_np):
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

# Save the hardest masked-pretrained test cases for qualitative comparison.

def save_hard_samples(masked_results, baseline_results, masked_model, baseline_model):
    """Compare masked vs standard pretrained on the 10 hardest test cases."""
    hard_dir = VIS_ROOT / "test" / "hard_samples"
    hard_dir.mkdir(parents=True, exist_ok=True)

    # Rank difficulty by the masked-pretrained model's own test Dice.
    sorted_results = sorted(masked_results, key=lambda r: r["mean_dice"])
    hard_n = min(10, len(sorted_results))

    test_ds = TissueDataset(
        image_dir=DATA_ROOT / "test" / "image",
        tissue_dir=DATA_ROOT / "test" / "tissue",
        augment=False,
    )
    name_to_idx     = {test_ds.get_sample_name(i): i for i in range(len(test_ds))}
    baseline_by_name = {r["name"]: r for r in baseline_results}

    hard_records = []
    for result in sorted_results[:hard_n]:
        name = result["name"]
        idx  = name_to_idx.get(name)
        if idx is None:
            continue

        img_tensor, mask_tensor = test_ds[idx]
        target = mask_tensor.numpy()

        with torch.no_grad():
            pred_masked   = masked_model(img_tensor.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu().numpy()
            pred_baseline = baseline_model(img_tensor.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu().numpy()

        save_comparison(img_tensor, target, pred_masked,
                        hard_dir / f"{name}_masked.png",
                        title=f"Masked dice={result['mean_dice']:.3f}")
        save_comparison(img_tensor, target, pred_baseline,
                        hard_dir / f"{name}_baseline.png",
                        title=f"Baseline (T4) dice={baseline_by_name[name]['mean_dice']:.3f}")

        hard_records.append({
            "name":           name,
            "masked_dice":    result["mean_dice"],
            "baseline_dice":  baseline_by_name[name]["mean_dice"],
        })
        print(f"  Hard: {name}  masked={result['mean_dice']:.3f}  "
              f"baseline={baseline_by_name[name]['mean_dice']:.3f}")

    return hard_records

# Main entry point.

if __name__ == "__main__":
    save_legend(VIS_ROOT / "legend.png")

    masked_ckpt = OUTPUT_DIR / "pretrained_seg" / "best_model.pth"
    if not masked_ckpt.exists():
        raise FileNotFoundError(f"Missing {masked_ckpt}")
    if not BASELINE_CKPT.exists():
        raise FileNotFoundError(f"Missing baseline {BASELINE_CKPT}")

    masked_model   = load_pretrained_seg(masked_ckpt)
    baseline_model = load_pretrained_seg(BASELINE_CKPT)

    # Run the full validation split for both pretrained variants.
    masked_val_m,  masked_val_r  = evaluate_model(masked_model,   "masked_pretrained", "validation")
    baseline_val_m, baseline_val_r = evaluate_model(baseline_model, "baseline_pretrained", "validation")

    # Then evaluate the held-out test split.
    masked_test_m,  masked_test_r  = evaluate_model(masked_model,   "masked_pretrained", "test")
    baseline_test_m, baseline_test_r = evaluate_model(baseline_model, "baseline_pretrained", "test")

    # Save side-by-side outputs for the hardest masked-pretrained cases.
    print("\n=== Hard Sample Analysis (10 lowest masked-pretrained Dice on test) ===")
    hard_records = save_hard_samples(
        masked_test_r, baseline_test_r, masked_model, baseline_model
    )

    # Write the metrics and per-sample results to disk.
    def metrics_dict(m, sample_results):
        return {
            "mean_dice":      float(m["mean_dice"]),
            "mean_iou":       float(m["mean_iou"]),
            "pixel_accuracy": float(m["pixel_accuracy"]),
            "dice_per_class": m["dice_per_class"].tolist(),
            "iou_per_class":  m["iou_per_class"].tolist(),
            "per_sample":     sample_results,
        }

    results = {
        "masked_pretrained": {
            "validation": metrics_dict(masked_val_m,  masked_val_r),
            "test":       metrics_dict(masked_test_m, masked_test_r),
        },
        "baseline_pretrained_trial4": {
            "validation": metrics_dict(baseline_val_m,  baseline_val_r),
            "test":       metrics_dict(baseline_test_m, baseline_test_r),
        },
        "hard_samples": hard_records,
    }

    out_path = OUTPUT_DIR / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")

    print("\n=== Summary ===")
    for label, mm, bm in [("Validation", masked_val_m, baseline_val_m),
                           ("Test",       masked_test_m, baseline_test_m)]:
        print(f"\n  {label}:")
        print(f"    Masked pretrained  Dice={mm['mean_dice']:.4f}  IoU={mm['mean_iou']:.4f}  PixAcc={mm['pixel_accuracy']:.4f}")
        print(f"    Baseline (Trial 4) Dice={bm['mean_dice']:.4f}  IoU={bm['mean_iou']:.4f}  PixAcc={bm['pixel_accuracy']:.4f}")
        delta = mm["mean_dice"] - bm["mean_dice"]
        print(f"    Delta (masked - baseline): {delta:+.4f}")
