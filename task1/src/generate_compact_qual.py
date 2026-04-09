"""
Generate compact qualitative panels for the report figures.

Outputs to cv_project_overleaf/figures/task1/
"""

import pathlib
import sys

import numpy as np
import torch
from PIL import Image as PILImage, ImageDraw as PILDraw, ImageFont

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


def load_unet(path):
    m = UNet(in_channels=3, num_classes=NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.to(device).eval()
    return m


def load_pretrained_seg(path):
    ae = Autoencoder(in_channels=3)
    m = SegWithPretrainedEncoder(ae, num_classes=NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.to(device).eval()
    return m


def load_gabor(path):
    m = BoundaryAwareGaborUNet(in_channels=3, num_classes=NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.to(device).eval()
    return m


def run_model(model, img_tensor):
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(device))
    return logits.argmax(dim=1).squeeze(0).cpu().numpy()


def make_panel_row(img_tensor, target, preds_and_labels, dice_per_model=None):
    """
    Returns a numpy array: [Image | GT | pred1 | pred2 | ...]
    with a header row showing labels.
    """
    cell = 256
    pad = 4
    header_h = 22
    n_panels = 2 + len(preds_and_labels)

    W = n_panels * cell + (n_panels - 1) * pad
    H = cell + header_h

    canvas = np.full((H, W, 3), 250, dtype=np.uint8)
    pil = PILImage.fromarray(canvas)
    draw = PILDraw.Draw(pil)

    cols = ["Image", "Ground Truth"] + [lbl for lbl, _ in preds_and_labels]
    x_positions = [i * (cell + pad) for i in range(n_panels)]

    # Headers
    for ci, (col, xp) in enumerate(zip(cols, x_positions)):
        draw.text((xp + 4, 4), col, fill=(50, 50, 50))

    # Images
    pil.paste(PILImage.fromarray(denormalise(img_tensor)), (x_positions[0], header_h))
    pil.paste(PILImage.fromarray(mask_to_rgb(target)), (x_positions[1], header_h))
    for ci, (lbl, pred) in enumerate(preds_and_labels):
        pil.paste(PILImage.fromarray(mask_to_rgb(pred)), (x_positions[2 + ci], header_h))
        if dice_per_model and lbl in dice_per_model:
            draw.text((x_positions[2 + ci] + 4, header_h + cell - 16),
                      f"Dice={dice_per_model[lbl]:.3f}", fill=(255, 255, 200))

    return np.array(pil)


def stack_rows(rows, gap=8):
    W = rows[0].shape[1]
    H = sum(r.shape[0] for r in rows) + gap * (len(rows) - 1)
    canvas = np.full((H, W, 3), 240, dtype=np.uint8)
    y = 0
    for r in rows:
        canvas[y:y + r.shape[0]] = r
        y += r.shape[0] + gap
    return canvas


# Load the test set once so all figure panels use the same source images.
test_ds = TissueDataset(
    image_dir=DATA_ROOT / "test" / "image",
    tissue_dir=DATA_ROOT / "test" / "tissue",
    augment=False,
)
name_to_idx = {test_ds.get_sample_name(i): i for i in range(len(test_ds))}


# First figure: CE-trained U-Net against Dice-trained U-Net.
print("Loading U-Net models...")
ce_model    = load_unet(BASE / "outputs_trial1/unet/best_model.pth")
dice_model  = load_unet(BASE / "outputs_dice/unet/best_model.pth")
gabor_model = load_gabor(BASE / "outputs_gabor_boundary_dice/unet/best_model.pth")

# Case 1: metastatic_roi_058 — CE fails completely, Dice recovers
# CE=0.031 Dice=0.329 Gabor=~0.33
# Case 2: primary_roi_007 — CE does well, Dice does slightly worse (honest counterexample)
# CE=0.565 Dice=0.488 Gabor=~0.5

rows = []
for sample_name, case_label in [
    ("test_set_metastatic_roi_058", "Case (i): metastatic\\_roi\\_058  CE failed, Dice recovered"),
    ("test_set_primary_roi_007", "Case (ii): primary\\_roi\\_007  CE slightly better; Gabor recovers Stroma"),
]:
    idx = name_to_idx[sample_name]
    img, mask = test_ds[idx]
    target = mask.numpy()

    ce_pred    = run_model(ce_model, img)
    dice_pred  = run_model(dice_model, img)
    gabor_pred = run_model(gabor_model, img)

    # per-sample dice
    def pdice(pred, tgt, c):
        tp = ((pred == c) & (tgt == c)).sum()
        fp = ((pred == c) & (tgt != c)).sum()
        fn = ((pred != c) & (tgt == c)).sum()
        return float(2 * tp / (2 * tp + fp + fn + 1e-8))

    dice_vals = {}
    for lbl, pr in [("CE U-Net", ce_pred), ("Dice U-Net", dice_pred), ("Gabor+Boundary", gabor_pred)]:
        md = sum(pdice(pr, target, c) for c in range(NUM_CLASSES)) / NUM_CLASSES
        dice_vals[lbl] = md

    row = make_panel_row(
        img, target,
        [("CE U-Net", ce_pred), ("Dice U-Net", dice_pred), ("Gabor+Boundary", gabor_pred)],
        dice_per_model=dice_vals
    )

    # Add left-margin case label
    pil_row = PILImage.fromarray(row)
    draw = PILDraw.Draw(pil_row)
    draw.text((4, row.shape[0] - 14), case_label.replace("\\", ""),
              fill=(50, 50, 50))
    rows.append(np.array(pil_row))

out_img = PILImage.fromarray(stack_rows(rows))
out_path = FIG_DIR / "qual_unet_compact.png"
out_img.save(str(out_path), dpi=(200, 200))
print(f"Saved {out_path}")


# Second figure: standard pretraining against masked-MSE pretraining.
print("Loading pretrained models...")
baseline_model = load_pretrained_seg(BASE / "outputs_dice/pretrained_seg/best_model.pth")
masked_model   = load_pretrained_seg(BASE / "outputs_masked_mse/pretrained_seg/best_model.pth")

# Case 1: primary_roi_047 — big Other gain 0.070→0.405
# Case 2: primary_roi_007 — big Other gain 0.081→0.512
rows = []
for sample_name, case_label in [
    ("test_set_primary_roi_047", "Case (i): primary\\_roi\\_047  Other Dice: 0.070 to 0.405"),
    ("test_set_primary_roi_007", "Case (ii): primary\\_roi\\_007  Other Dice: 0.081 to 0.512"),
]:
    idx = name_to_idx[sample_name]
    img, mask = test_ds[idx]
    target = mask.numpy()

    base_pred   = run_model(baseline_model, img)
    masked_pred = run_model(masked_model, img)

    def pdice(pred, tgt, c):
        tp = ((pred == c) & (tgt == c)).sum()
        fp = ((pred == c) & (tgt != c)).sum()
        fn = ((pred != c) & (tgt == c)).sum()
        return float(2 * tp / (2 * tp + fp + fn + 1e-8))

    dice_vals = {}
    for lbl, pr in [("Pretrained (MSE)", base_pred), ("Masked MSE Pretrained", masked_pred)]:
        md = sum(pdice(pr, target, c) for c in range(NUM_CLASSES)) / NUM_CLASSES
        dice_vals[lbl] = md

    row = make_panel_row(
        img, target,
        [("Pretrained (MSE)", base_pred), ("Masked MSE Pretrained", masked_pred)],
        dice_per_model=dice_vals
    )
    pil_row = PILImage.fromarray(row)
    draw = PILDraw.Draw(pil_row)
    draw.text((4, row.shape[0] - 14), case_label.replace("\\", ""),
              fill=(50, 50, 50))
    rows.append(np.array(pil_row))

out_img = PILImage.fromarray(stack_rows(rows))
out_path = FIG_DIR / "qual_pretrained_compact.png"
out_img.save(str(out_path), dpi=(200, 200))
print(f"Saved {out_path}")
print("Done.")
