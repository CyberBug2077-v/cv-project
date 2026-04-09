"""
Compare the Dice baseline U-Net against the second-order boundary-aware Gabor U-Net.

Usage:
    conda run -n ML python3 src/evaluate_gabor_boundary_second_order_unet.py <output_dir>
"""

import json
import sys
import pathlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import NUM_CLASSES, CLASS_NAMES, DATA_ROOT, TissueDataset
from model_unet import UNet, BoundaryAwareSecondOrderGaborUNet, count_parameters
from metrics import ConfusionMatrix, format_metrics
from visualise import save_comparison, save_legend

BASELINE_CKPT = pathlib.Path("outputs_dice/unet/best_model.pth")

if len(sys.argv) < 2:
    print("Usage: python3 src/evaluate_gabor_boundary_second_order_unet.py <output_dir>")
    sys.exit(1)

OUTPUT_DIR = pathlib.Path(sys.argv[1])
VIS_ROOT = OUTPUT_DIR / "visualisations"
VIS_ROOT.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")
print(f"Output dir: {OUTPUT_DIR}")


def load_baseline_unet(ckpt_path):
    model = UNet(in_channels=3, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device).eval()
    return model


def load_gabor_unet(ckpt_path):
    model = BoundaryAwareSecondOrderGaborUNet(in_channels=3, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device).eval()
    return model


def per_sample_dice(pred_np, target_np):
    dice_scores = []
    for c in range(NUM_CLASSES):
        tp = ((pred_np == c) & (target_np == c)).sum()
        fp = ((pred_np == c) & (target_np != c)).sum()
        fn = ((pred_np != c) & (target_np == c)).sum()
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        dice_scores.append(float(dice))
    return float(np.mean(dice_scores)), dice_scores


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
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        target = mask_tensor.numpy()

        cm.update(pred, target)

        mean_d, per_class_d = per_sample_dice(pred, target)
        sample_results.append({
            "name": sample_name,
            "mean_dice": mean_d,
            "dice_other": per_class_d[0],
            "dice_tumor": per_class_d[1],
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


def save_hard_samples(gabor_results, baseline_results, gabor_model, baseline_model):
    hard_dir = VIS_ROOT / "test" / "hard_samples"
    hard_dir.mkdir(parents=True, exist_ok=True)

    sorted_results = sorted(gabor_results, key=lambda r: r["mean_dice"])
    hard_n = min(10, len(sorted_results))

    test_ds = TissueDataset(
        image_dir=DATA_ROOT / "test" / "image",
        tissue_dir=DATA_ROOT / "test" / "tissue",
        augment=False,
    )
    name_to_idx = {test_ds.get_sample_name(i): i for i in range(len(test_ds))}
    baseline_by_name = {r["name"]: r for r in baseline_results}

    hard_records = []
    for result in sorted_results[:hard_n]:
        name = result["name"]
        idx = name_to_idx.get(name)
        if idx is None:
            continue

        img_tensor, mask_tensor = test_ds[idx]
        target = mask_tensor.numpy()

        with torch.no_grad():
            pred_gabor = gabor_model(img_tensor.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu().numpy()
            pred_base = baseline_model(img_tensor.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu().numpy()

        save_comparison(
            img_tensor, target, pred_gabor,
            hard_dir / f"{name}_gabor_boundary_second_order_unet.png",
            title=f"Gabor boundary second-order U-Net dice={result['mean_dice']:.3f}",
        )
        save_comparison(
            img_tensor, target, pred_base,
            hard_dir / f"{name}_baseline_unet.png",
            title=f"Baseline U-Net dice={baseline_by_name[name]['mean_dice']:.3f}",
        )

        hard_records.append({
            "name": name,
            "gabor_boundary_second_order_dice": result["mean_dice"],
            "baseline_unet_dice": baseline_by_name[name]["mean_dice"],
            "gabor_dice_other": result["dice_other"],
            "gabor_dice_tumor": result["dice_tumor"],
            "gabor_dice_stroma": result["dice_stroma"],
        })
        print(
            f"  Hard: {name}  gabor={result['mean_dice']:.3f}  "
            f"baseline={baseline_by_name[name]['mean_dice']:.3f}"
        )

    return hard_records


def metrics_dict(m, model, sample_results):
    return {
        "mean_dice": float(m["mean_dice"]),
        "mean_iou": float(m["mean_iou"]),
        "pixel_accuracy": float(m["pixel_accuracy"]),
        "dice_per_class": m["dice_per_class"].tolist(),
        "iou_per_class": m["iou_per_class"].tolist(),
        "num_params": count_parameters(model),
        "per_sample": sample_results,
    }


if __name__ == "__main__":
    save_legend(VIS_ROOT / "legend.png")

    gabor_ckpt = OUTPUT_DIR / "unet" / "best_model.pth"

    if not BASELINE_CKPT.exists():
        raise FileNotFoundError(f"Missing {BASELINE_CKPT}")
    if not gabor_ckpt.exists():
        raise FileNotFoundError(f"Missing {gabor_ckpt}")

    baseline_model = load_baseline_unet(BASELINE_CKPT)
    gabor_model = load_gabor_unet(gabor_ckpt)

    print(f"Baseline U-Net params                    : {count_parameters(baseline_model):,}")
    print(f"Gabor boundary second-order U-Net params : {count_parameters(gabor_model):,}")

    base_val_m, base_val_r = evaluate_model(baseline_model, "baseline_unet", "validation")
    gabor_val_m, gabor_val_r = evaluate_model(gabor_model, "gabor_boundary_second_order_unet", "validation")

    base_test_m, base_test_r = evaluate_model(baseline_model, "baseline_unet", "test")
    gabor_test_m, gabor_test_r = evaluate_model(gabor_model, "gabor_boundary_second_order_unet", "test")

    print("\n=== Hard Sample Analysis (10 lowest Gabor boundary second-order U-Net Dice on test set) ===")
    hard_records = save_hard_samples(
        gabor_test_r, base_test_r, gabor_model, baseline_model
    )

    results = {
        "baseline_unet": {
            "validation": metrics_dict(base_val_m, baseline_model, base_val_r),
            "test": metrics_dict(base_test_m, baseline_model, base_test_r),
        },
        "gabor_boundary_second_order_unet": {
            "validation": metrics_dict(gabor_val_m, gabor_model, gabor_val_r),
            "test": metrics_dict(gabor_test_m, gabor_model, gabor_test_r),
        },
        "hard_samples": hard_records,
    }

    out_path = OUTPUT_DIR / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")

    print("\n=== Summary ===")
    for label, bm, gm in [("Validation", base_val_m, gabor_val_m),
                          ("Test", base_test_m, gabor_test_m)]:
        print(f"\n  {label}:")
        print(f"    Baseline U-Net                     Dice={bm['mean_dice']:.4f}  IoU={bm['mean_iou']:.4f}  PixAcc={bm['pixel_accuracy']:.4f}")
        print(f"    Gabor boundary second-order U-Net Dice={gm['mean_dice']:.4f}  IoU={gm['mean_iou']:.4f}  PixAcc={gm['pixel_accuracy']:.4f}")
