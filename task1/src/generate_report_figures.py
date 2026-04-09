"""
Generate the figures used in the Task 1 report section.

Outputs to cv_project_overleaf/figures/task1/

Figures generated:
  arch_diagram.pdf          — two-branch architecture schematic
  unet_val_dice_curves.pdf  — val Dice over epochs for key U-Net variants
  unet_per_class_dice.pdf   — per-class test Dice bar chart for U-Net variants
  pretrained_val_dice_curves.pdf — val Dice over epochs for pretrained variants
  qual_unet_ce_vs_dice.png  — qualitative: CE vs Dice U-Net (2 cases)
  qual_pretrained_comparison.png — qualitative: baseline vs masked pretrained (2 cases)
  training_stability.pdf    — val Dice curves pretrained branch comparison

Usage:
    conda run -n ML python3 src/generate_report_figures.py
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as mpatch
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import NUM_CLASSES, CLASS_NAMES, DATA_ROOT, TissueDataset
from model_unet import UNet, BoundaryAwareGaborUNet, count_parameters
from model_autoencoder import Autoencoder, SegWithPretrainedEncoder
from visualise import mask_to_rgb, denormalise

BASE = pathlib.Path(__file__).parent.parent
FIG_DIR = BASE / "cv_project_overleaf" / "figures" / "task1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Shared colours and style settings for all report figures.
COLOURS = {
    "CE":            "#e06c75",
    "Focal":         "#c678dd",
    "Dice":          "#61afef",
    "Dice+CE":       "#e5c07b",
    "Dice+Focal":    "#56b6c2",
    "Gabor+Bound":   "#98c379",
    "Gabor (no bnd)": "#d4a373",
    "Pretrained CE": "#e06c75",
    "Pretrained Dice": "#61afef",
    "Masked MSE":    "#98c379",
    "Masked MSE+L1": "#e5c07b",
}

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# Load the saved training logs that feed the plots.

def load_log(path):
    with open(path) as f:
        return json.load(f)


def get_unet_logs():
    return {
        "CE":          load_log(BASE / "outputs_trial1/unet/training_log.json"),
        "Focal":       load_log(BASE / "outputs_focal/unet/training_log.json"),
        "Dice":        load_log(BASE / "outputs_dice/unet/training_log.json"),
        "Dice+CE":     load_log(BASE / "outputs_dice_ce/unet/training_log.json"),
        "Dice+Focal":  load_log(BASE / "outputs_dice_focal/unet/training_log.json"),
        "Gabor+Bound": load_log(BASE / "outputs_gabor_boundary_dice/unet/training_log.json"),
    }


def get_pretrained_logs():
    return {
        "Pretrained CE":    load_log(BASE / "outputs_trial1/pretrained_seg/training_log.json"),
        "Pretrained Dice":  load_log(BASE / "outputs_dice/pretrained_seg/training_log.json"),
        "Masked MSE":       load_log(BASE / "outputs_masked_mse/pretrained_seg/training_log.json"),
        "Masked MSE+L1":    load_log(BASE / "outputs_masked_mse_l1/pretrained_seg/training_log.json"),
    }


# Figure 1: architecture diagram.

def make_arch_diagram():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    def box(ax, x, y, w, h, label, color="#aec6e8", fontsize=8.5, alpha=0.9):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor="#555", linewidth=1.0, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="#222", wrap=True,
                multialignment="center")

    def arrow(ax, x1, y1, x2, y2):
        # arrowhead is at (x2, y2)
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))

    def dim_label(ax, x, y, text):
        ax.text(x, y, text, ha="center", va="top", fontsize=6.5,
                color="#555", style="italic")

    # ----- Branch A: Vanilla U-Net -----
    ax = axes[0]
    ax.set_title("(A)  Vanilla U-Net", fontsize=11, fontweight="bold", pad=8)

    # Encoder: descending in y (top → bottom = high y → low y)
    enc_cols  = ["#b5d3f5", "#8fbce8", "#6aa3db", "#4a8bce"]
    enc_labels = ["Conv 64", "Conv 128", "Conv 256", "Conv 512"]
    enc_dims  = ["256²", "128²", "64²", "32²"]
    x_enc  = [0.4, 1.4, 2.3, 3.1]
    y_enc  = [5.8, 4.3, 3.2, 2.2]   # y_bottom; top = y + h
    widths = [1.2, 1.0, 0.8, 0.7]
    heights = [2.0, 1.3, 1.0, 0.9]

    for i, (xi, yi, wi, hi, lbl, col, dim) in enumerate(
            zip(x_enc, y_enc, widths, heights, enc_labels, enc_cols, enc_dims)):
        box(ax, xi, yi, wi, hi, lbl, color=col)
        dim_label(ax, xi + wi / 2, yi - 0.05, dim)

    # Bottleneck
    box(ax, 4.1, 1.2, 1.6, 0.8, "Bottleneck\n1024", color="#5c7bb0")
    dim_label(ax, 4.9, 1.15, "16²")

    # Decoder (mirrored positions)
    dec_cols  = ["#a2d4a2", "#80c680", "#60b860", "#3faa3f"]
    dec_labels = ["Up 512", "Up 256", "Up 128", "Up 64"]
    dec_dims  = ["32²", "64²", "128²", "256²"]
    x_dec  = [6.0, 6.9, 7.7, 8.5]
    # mirror: Up512↔Conv512, Up256↔Conv256, Up128↔Conv128, Up64↔Conv64
    y_dec  = y_enc[::-1]
    w_dec  = widths[::-1]
    h_dec  = heights[::-1]

    for i, (xi, yi, wi, hi, lbl, col, dim) in enumerate(
            zip(x_dec, y_dec, w_dec, h_dec, dec_labels, dec_cols, dec_dims)):
        box(ax, xi, yi, wi, hi, lbl, color=col)
        dim_label(ax, xi + wi / 2, yi - 0.05, dim)

    # Output box (above Up64, same x)
    box(ax, 8.5, 8.3, 1.2, 0.6, "3-class\noutput", color="#f0c070", fontsize=7.5)
    dim_label(ax, 9.1, 8.25, "256²")

    # Skip connections (horizontal dashed lines, encoder right → decoder left)
    for i in range(4):
        xi_right = x_enc[i] + widths[i]
        xd_left  = x_dec[3 - i]
        yi_mid   = y_enc[i] + heights[i] / 2
        ax.annotate("", xy=(xd_left, yi_mid), xytext=(xi_right, yi_mid),
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=1.0,
                                    linestyle="dashed"))

    # Input label + arrow pointing DOWN into Conv64
    ax.text(1.0, 9.0, "Input\n256×256", ha="center", fontsize=8, color="#333")
    # arrowhead at top of Conv64 (y_enc[0]+heights[0]=7.8), tail above
    arrow(ax, 1.0, 8.65, 1.0, y_enc[0] + heights[0])

    # Encoder flow: bottom of box i → top of box i+1 (downward)
    for i in range(3):
        arrow(ax, x_enc[i] + widths[i] / 2, y_enc[i],
              x_enc[i + 1] + widths[i + 1] / 2, y_enc[i + 1] + heights[i + 1])

    # Conv512 → Bottleneck (downward)
    arrow(ax, x_enc[3] + widths[3] / 2, y_enc[3],
          4.1 + 0.8, 1.2 + 0.8)

    # Bottleneck → Up512 (rightward)
    arrow(ax, 4.1 + 1.6, 1.6, x_dec[0], y_dec[0] + h_dec[0] / 2)

    # Decoder flow: top of box i → bottom of box i+1 (upward)
    for i in range(3):
        arrow(ax, x_dec[i] + w_dec[i] / 2, y_dec[i] + h_dec[i],
              x_dec[i + 1] + w_dec[i + 1] / 2, y_dec[i + 1])

    # Up64 top → output box (upward)
    arrow(ax, x_dec[3] + w_dec[3] / 2, y_dec[3] + h_dec[3], 9.1, 8.3)

    # Skip connection label
    ax.text(5.25, 9.1, "skip connections", ha="center", fontsize=7,
            color="#999", style="italic")

    # ----- Branch B: Autoencoder pretraining + seg -----
    ax = axes[1]
    ax.set_title("(B)  Autoencoder Pretraining → Segmentation", fontsize=11,
                 fontweight="bold", pad=8)

    ax.text(0.3, 9.5, "Phase 1: Pretraining (no labels)",
            fontsize=8.5, color="#555", fontweight="bold")
    ax.text(0.3, 4.7, "Phase 2: Segmentation fine-tuning (with labels)",
            fontsize=8.5, color="#555", fontweight="bold")
    ax.axhline(y=5.0, xmin=0, xmax=1, color="#ccc", lw=1.0, linestyle="--")

    # Phase 1
    box(ax, 0.3, 8.5, 2.2, 0.8, "Corrupted input\n(mask + noise + jitter)",
        color="#e8cfc0", fontsize=7.5)
    box(ax, 0.3, 6.5, 2.2, 1.6, "Shared Encoder\n(4 levels, 64→512)", color="#6aa3db")
    box(ax, 3.0, 7.0, 2.2, 1.1, "Reconstruction\nDecoder\n(no skip conn.)",
        color="#c8a8e0", fontsize=7.5)
    box(ax, 5.6, 7.0, 2.2, 1.1, "MSE loss vs\nclean image", color="#f0c8a0", fontsize=7.5)

    # corrupted input → encoder (downward: from bottom of input box to top of encoder)
    arrow(ax, 1.4, 8.5, 1.4, 8.1)
    # encoder → reconstruction decoder (rightward from encoder right edge to decoder left)
    arrow(ax, 2.5, 7.55, 3.0, 7.55)
    # reconstruction decoder → MSE loss
    arrow(ax, 5.2, 7.55, 5.6, 7.55)

    # Phase 2
    box(ax, 0.3, 1.5, 2.2, 2.8,
        "Pretrained\nEncoder\n(init. Phase 1)", color="#5c7bb0")
    box(ax, 3.0, 1.5, 2.2, 2.8,
        "Seg Decoder\n(fresh weights,\nskip conn.,\nDice loss)", color="#80c680", fontsize=7.5)
    box(ax, 5.6, 2.3, 2.2, 1.2, "3-class\nseg output", color="#f0c070")
    box(ax, 0.3, 0.2, 2.2, 0.8, "Labelled input\n256×256", color="#e8cfc0", fontsize=7.5)

    arrow(ax, 1.4, 1.0, 1.4, 1.5)   # input → encoder
    arrow(ax, 2.5, 2.9, 3.0, 2.9)   # encoder → seg decoder
    arrow(ax, 5.2, 2.9, 5.6, 2.9)   # seg decoder → output

    # Phase 1 → Phase 2: shared encoder initialises pretrained encoder (downward)
    ax.annotate("", xy=(1.4, 4.3), xytext=(1.4, 6.5),
                arrowprops=dict(arrowstyle="->", color="#2266aa", lw=1.5))

    plt.tight_layout(pad=1.5)
    out = FIG_DIR / "arch_diagram.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out}")


# Figure 2: U-Net validation Dice curves.

def make_unet_training_curves():
    logs = get_unet_logs()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: val loss
    ax = axes[0]
    for name, log in logs.items():
        epochs = [e["epoch"] for e in log]
        val_loss = [e["val_loss"] for e in log]
        ax.plot(epochs, val_loss, label=name, color=COLOURS[name], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation loss — U-Net variants")
    ax.legend(framealpha=0.8, fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(1, 60)

    # Right: val mean Dice
    ax = axes[1]
    for name, log in logs.items():
        epochs = [e["epoch"] for e in log]
        val_dice = [e["mean_dice"] for e in log]
        ax.plot(epochs, val_dice, label=name, color=COLOURS[name], linewidth=1.5)
    ax.axhline(y=0.467, color="#888", lw=1.0, linestyle=":", label="Baseline (0.467)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val mean Dice")
    ax.set_title("Validation Dice — U-Net variants")
    ax.legend(framealpha=0.8, fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(1, 60)

    plt.tight_layout()
    out = FIG_DIR / "unet_val_dice_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# Figure 3: per-class test Dice for the U-Net variants.

def make_unet_per_class_dice():
    with open(BASE / "outputs_analysis/all_metrics.json") as f:
        all_metrics = json.load(f)

    # Gabor (no boundary) metrics from outputs_gabor_dice_ce/evaluation_results.json
    with open(BASE / "outputs_gabor_dice_ce/evaluation_results.json") as f:
        gabor_eval = json.load(f)
    gabor_no_bound = gabor_eval["gabor_unet"]["test"]

    models = [
        ("CE",            "trial1_unet_ce"),
        ("Focal",         "trial3_unet_focal"),
        ("Dice",          "trial4_unet_dice"),
        ("Dice+Focal",    "trial6_unet_dice_focal"),
        ("Gabor\n(no bnd)", None),
        ("Gabor\n+Bound", "g3_gabor_boundary"),
    ]

    labels = [m[0] for m in models]
    other_d, tumor_d, stroma_d, mean_d = [], [], [], []
    for lbl, key in models:
        if key is None:
            other_d.append(gabor_no_bound["dice_per_class"][0])
            tumor_d.append(gabor_no_bound["dice_per_class"][1])
            stroma_d.append(gabor_no_bound["dice_per_class"][2])
            mean_d.append(gabor_no_bound["mean_dice"])
        else:
            other_d.append(all_metrics[key]["test"]["dice_per_class"][0])
            tumor_d.append(all_metrics[key]["test"]["dice_per_class"][1])
            stroma_d.append(all_metrics[key]["test"]["dice_per_class"][2])
            mean_d.append(all_metrics[key]["test"]["mean_dice"])

    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.5))

    b1 = ax.bar(x - 1.5 * w, other_d,  w, label="Other",  color="#b0b0b0", edgecolor="white")
    b2 = ax.bar(x - 0.5 * w, tumor_d,  w, label="Tumor",  color="#e06c75", edgecolor="white")
    b3 = ax.bar(x + 0.5 * w, stroma_d, w, label="Stroma", color="#61afef", edgecolor="white")
    b4 = ax.bar(x + 1.5 * w, mean_d,   w, label="Mean",   color="#98c379", edgecolor="white", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test Dice")
    ax.set_title("Per-class and mean test Dice — U-Net variants")
    ax.legend(fontsize=8.5)
    ax.axhline(y=0.467, color="#888", lw=1.0, linestyle=":", label="Baseline")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Annotate mean Dice values on bars
    for xi, val in zip(x + 1.5 * w, mean_d):
        ax.text(xi, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out = FIG_DIR / "unet_per_class_dice.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# Figure 4: pretrained-branch validation Dice curves.

def make_pretrained_training_curves():
    logs = get_pretrained_logs()
    fig, ax = plt.subplots(figsize=(7, 4))

    for name, log in logs.items():
        epochs = [e["epoch"] for e in log]
        val_dice = [e["mean_dice"] for e in log]
        col = COLOURS[name]
        ax.plot(epochs, val_dice, label=name, color=col, linewidth=1.8)

    ax.axhline(y=0.467, color="#888", lw=1.0, linestyle=":", label="Baseline (0.467)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val mean Dice")
    ax.set_title("Validation Dice — pretrained segmentation variants")
    ax.legend(framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(1, 60)

    plt.tight_layout()
    out = FIG_DIR / "pretrained_val_dice_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# Shared helpers for the qualitative figure panels.

def load_unet_model(ckpt_path):
    model = UNet(in_channels=3, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device).eval()
    return model


def load_pretrained_seg_model(ckpt_path):
    ae = Autoencoder(in_channels=3)
    model = SegWithPretrainedEncoder(ae, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device).eval()
    return model


def load_gabor_boundary_model(ckpt_path):
    model = BoundaryAwareGaborUNet(in_channels=3, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(device).eval()
    return model


def get_predictions(models_dict, img_tensor):
    """Run a dict of {label: model} on img_tensor. Returns dict of {label: pred_np}."""
    results = {}
    with torch.no_grad():
        for label, model in models_dict.items():
            logits = model(img_tensor.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            results[label] = pred
    return results


def make_qual_panel(cases, models_dict, test_ds, name_to_idx, out_path,
                    col_labels, title="", per_class_dice_fns=None):
    """
    cases        : list of sample names
    models_dict  : ordered dict of {col_label: model}
    test_ds      : TissueDataset
    out_path     : save path
    col_labels   : labels for prediction columns (same order as models_dict)
    """
    n_rows = len(cases)
    n_pred_cols = len(models_dict)
    n_cols = 2 + n_pred_cols  # Image, GT, pred1, pred2, ...
    cell = 256
    pad = 6
    header_h = 28

    W = n_cols * cell + (n_cols - 1) * pad
    H = n_rows * (cell + header_h) + header_h

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    from PIL import Image as PILImage, ImageDraw as PILDraw
    pil = PILImage.fromarray(canvas)
    draw = PILDraw.Draw(pil)

    col_names = ["Image", "Ground Truth"] + list(col_labels)
    x_positions = [i * (cell + pad) for i in range(n_cols)]

    # Draw column headers
    for ci, (cname, xp) in enumerate(zip(col_names, x_positions)):
        draw.text((xp + cell // 2 - len(cname) * 3, 8), cname, fill=(50, 50, 50))

    for ri, sample_name in enumerate(cases):
        idx = name_to_idx.get(sample_name)
        if idx is None:
            continue
        img_tensor, mask_tensor = test_ds[idx]
        target = mask_tensor.numpy()
        preds = get_predictions(models_dict, img_tensor)

        y_top = header_h + ri * (cell + header_h)

        # Image
        img_rgb = denormalise(img_tensor)
        pil.paste(PILImage.fromarray(img_rgb), (x_positions[0], y_top))

        # GT
        gt_rgb = mask_to_rgb(target)
        pil.paste(PILImage.fromarray(gt_rgb), (x_positions[1], y_top))

        # Predictions
        for ci, (col_label, pred) in enumerate(preds.items()):
            pred_rgb = mask_to_rgb(pred)
            pil.paste(PILImage.fromarray(pred_rgb), (x_positions[2 + ci], y_top))

        # Row label
        draw.text((4, y_top + 4), sample_name.replace("test_set_", ""), fill=(80, 80, 80))

    pil.save(str(out_path))
    print(f"Saved {out_path}")


# Figure 5: qualitative CE-versus-Dice U-Net comparison.

def make_qual_unet_ce_vs_dice():
    test_ds = TissueDataset(
        image_dir=DATA_ROOT / "test" / "image",
        tissue_dir=DATA_ROOT / "test" / "tissue",
        augment=False,
    )
    name_to_idx = {test_ds.get_sample_name(i): i for i in range(len(test_ds))}

    ce_model   = load_unet_model(BASE / "outputs_trial1/unet/best_model.pth")
    dice_model = load_unet_model(BASE / "outputs_dice/unet/best_model.pth")
    gabor_model = load_gabor_boundary_model(
        BASE / "outputs_gabor_boundary_dice/unet/best_model.pth")

    models_dict = {
        "CE U-Net": ce_model,
        "Dice U-Net": dice_model,
        "Gabor+Bound": gabor_model,
    }

    # Best improvement cases: roi_058 (big Dice gain), roi_088 (Stroma gain),
    # roi_013 (roughly similar, both models work), roi_007 (CE was better — counterexample)
    cases = [
        "test_set_metastatic_roi_058",   # CE=0.031 → Dice=0.329 (huge recovery)
        "test_set_metastatic_roi_088",   # CE=0.215 → Dice=0.468 (Stroma recovery)
        "test_set_primary_roi_007",      # CE=0.565, Dice=0.488 (CE was better — honest)
        "test_set_primary_roi_061",      # Other-heavy: CE=0.448, Dice=0.319
    ]

    out_path = FIG_DIR / "qual_unet_ce_vs_dice.png"
    make_qual_panel(
        cases, models_dict, test_ds, name_to_idx, out_path,
        col_labels=list(models_dict.keys()),
        title="CE vs Dice vs Gabor+Boundary U-Net"
    )


# Figure 6: qualitative standard-versus-masked pretraining comparison.

def make_qual_pretrained_comparison():
    test_ds = TissueDataset(
        image_dir=DATA_ROOT / "test" / "image",
        tissue_dir=DATA_ROOT / "test" / "tissue",
        augment=False,
    )
    name_to_idx = {test_ds.get_sample_name(i): i for i in range(len(test_ds))}

    baseline_model = load_pretrained_seg_model(
        BASE / "outputs_dice/pretrained_seg/best_model.pth")
    masked_model   = load_pretrained_seg_model(
        BASE / "outputs_masked_mse/pretrained_seg/best_model.pth")

    models_dict = {
        "Pretrained (MSE)": baseline_model,
        "Masked MSE Pretrained": masked_model,
    }

    # Best cases for masked improvement, especially Other class
    cases = [
        "test_set_primary_roi_047",    # biggest Other gain: 0.070→0.405
        "test_set_primary_roi_007",    # big Other gain: 0.081→0.512
        "test_set_primary_roi_061",    # Other: 0.235→0.361
        "test_set_primary_roi_080",    # both do well (positive example)
    ]

    out_path = FIG_DIR / "qual_pretrained_comparison.png"
    make_qual_panel(
        cases, models_dict, test_ds, name_to_idx, out_path,
        col_labels=list(models_dict.keys()),
        title="Baseline vs Masked MSE Pretrained"
    )


# Figure 7: per-class test Dice for the pretrained branch.

def make_pretrained_per_class_dice():
    with open(BASE / "outputs_analysis/all_metrics.json") as f:
        all_metrics = json.load(f)

    models = [
        ("Pretrained\nCE",       "trial1_pretrained_ce"),
        ("Pretrained\nDice",     "trial4_pretrained_dice"),
        ("Pretrained\nDice+CE",  "trial7_pretrained_dice_ce"),
        ("Masked\nMSE",          "trialA_masked_mse_pretrained"),
        ("Masked\nMSE+L1",       "trialB_masked_mse_l1_pretrained"),
    ]

    labels = [m[0] for m in models]
    other_d  = [all_metrics[m[1]]["test"]["dice_per_class"][0] for m in models]
    tumor_d  = [all_metrics[m[1]]["test"]["dice_per_class"][1] for m in models]
    stroma_d = [all_metrics[m[1]]["test"]["dice_per_class"][2] for m in models]
    mean_d   = [all_metrics[m[1]]["test"]["mean_dice"]          for m in models]

    x = np.arange(len(labels))
    w = 0.18
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(x - 1.5 * w, other_d,  w, label="Other",  color="#b0b0b0", edgecolor="white")
    ax.bar(x - 0.5 * w, tumor_d,  w, label="Tumor",  color="#e06c75", edgecolor="white")
    ax.bar(x + 0.5 * w, stroma_d, w, label="Stroma", color="#61afef", edgecolor="white")
    ax.bar(x + 1.5 * w, mean_d,   w, label="Mean",   color="#98c379", edgecolor="white", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Test Dice")
    ax.set_title("Per-class and mean test Dice — pretrained segmentation variants")
    ax.legend(fontsize=8.5)
    ax.axhline(y=0.467, color="#888", lw=1.0, linestyle=":", label="Baseline")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    for xi, val in zip(x + 1.5 * w, mean_d):
        ax.text(xi, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out = FIG_DIR / "pretrained_per_class_dice.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# Figure 8: summary comparison bar chart.

def make_summary_comparison():
    with open(BASE / "outputs_analysis/all_metrics.json") as f:
        all_metrics = json.load(f)

    models = [
        ("Baseline\n(125M)", None, 0.467, 0.4670),
        ("Best U-Net\nGabor+Bound\n(31M)", "g3_gabor_boundary", None, None),
        ("Pretrained\nDice\n(31M)", "trial4_pretrained_dice", None, None),
        ("Masked MSE\nPretrained\n(31M)", "trialA_masked_mse_pretrained", None, None),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_cols = ["#aaa", "#61afef", "#e5c07b", "#98c379"]

    mean_dices = []
    for name, key, manual_dice, _ in models:
        if key:
            mean_dices.append(all_metrics[key]["test"]["mean_dice"])
        else:
            mean_dices.append(manual_dice)

    bars = ax.bar(range(len(models)), mean_dices, color=bar_cols, width=0.55,
                  edgecolor="white")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m[0] for m in models], fontsize=8.5)
    ax.set_ylabel("Test mean Dice")
    ax.set_title("Model comparison — test mean Dice")
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, mean_dices):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    out = FIG_DIR / "summary_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# Main entry point.

if __name__ == "__main__":
    print("Generating report figures...")

    print("\n[1/8] Architecture diagram")
    make_arch_diagram()

    print("\n[2/8] U-Net training curves")
    make_unet_training_curves()

    print("\n[3/8] U-Net per-class Dice bar chart")
    make_unet_per_class_dice()

    print("\n[4/8] Pretrained training curves")
    make_pretrained_training_curves()

    print("\n[5/8] Qualitative: CE vs Dice vs Gabor U-Net")
    make_qual_unet_ce_vs_dice()

    print("\n[6/8] Qualitative: baseline vs masked pretrained")
    make_qual_pretrained_comparison()

    print("\n[7/8] Pretrained per-class Dice bar chart")
    make_pretrained_per_class_dice()

    print("\n[8/8] Summary comparison")
    make_summary_comparison()

    print(f"\nAll figures saved to {FIG_DIR}")
