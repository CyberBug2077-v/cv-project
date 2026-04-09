"""
Train the boundary-aware Gabor U-Net with Dice and boundary supervision.

Usage:
    conda run -n ML python3 src/train_unet_gabor_boundary_dice.py

Saves to: outputs_gabor_boundary_dice/unet/
  - best_model.pth
  - last_model.pth
  - training_log.json
"""

import json
import sys
import time
import pathlib

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from boundary import boundary_from_mask
from data import get_dataloaders, NUM_CLASSES, CLASS_NAMES
from model_unet import BoundaryAwareGaborUNet, count_parameters
from metrics import ConfusionMatrix, format_metrics
from losses import DiceLoss

EPOCHS = 60
BATCH_SIZE = 8
LR = 1e-4
BOUNDARY_WIDTH = 2
BOUNDARY_WEIGHT = 0.1
BOUNDARY_POS_WEIGHT = 4.0
OUTPUT_DIR = pathlib.Path("outputs_gabor_boundary_dice/unet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

model = BoundaryAwareGaborUNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
print(f"Trainable parameters: {count_parameters(model):,}")

train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE, num_workers=0)
print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

seg_criterion = DiceLoss(num_classes=NUM_CLASSES)
boundary_criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([BOUNDARY_POS_WEIGHT], device=device)
)


def compute_losses(seg_logits, boundary_logits, masks):
    # Combine the main segmentation loss with a lighter auxiliary boundary loss.
    boundary_targets = boundary_from_mask(masks, width=BOUNDARY_WIDTH).unsqueeze(1)
    seg_loss = seg_criterion(seg_logits, masks)
    boundary_loss = boundary_criterion(boundary_logits, boundary_targets)
    total_loss = seg_loss + BOUNDARY_WEIGHT * boundary_loss
    return total_loss, seg_loss, boundary_loss


optimiser = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode="max", factor=0.5, patience=8
)

log = []
best_dice = 0.0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    model.train()
    train_loss = 0.0
    train_seg_loss = 0.0
    train_boundary_loss = 0.0

    for imgs, masks in train_loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimiser.zero_grad()
        seg_logits, boundary_logits = model(imgs, return_boundary=True)
        loss, seg_loss, boundary_loss = compute_losses(seg_logits, boundary_logits, masks)
        loss.backward()
        optimiser.step()

        train_loss += loss.item()
        train_seg_loss += seg_loss.item()
        train_boundary_loss += boundary_loss.item()

    train_loss /= len(train_loader)
    train_seg_loss /= len(train_loader)
    train_boundary_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    val_seg_loss = 0.0
    val_boundary_loss = 0.0
    cm = ConfusionMatrix(NUM_CLASSES)

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            seg_logits, boundary_logits = model(imgs, return_boundary=True)
            loss, seg_loss, boundary_loss = compute_losses(seg_logits, boundary_logits, masks)

            val_loss += loss.item()
            val_seg_loss += seg_loss.item()
            val_boundary_loss += boundary_loss.item()

            preds = seg_logits.argmax(dim=1)
            cm.update(preds.cpu().numpy(), masks.cpu().numpy())

    val_loss /= len(val_loader)
    val_seg_loss /= len(val_loader)
    val_boundary_loss /= len(val_loader)
    m = cm.compute()
    mean_dice = m["mean_dice"]

    scheduler.step(mean_dice)

    if mean_dice > best_dice:
        best_dice = mean_dice
        torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")

    elapsed = time.time() - t0
    print(
        f"Epoch {epoch:3d}/{EPOCHS}  "
        f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
        f"train_boundary={train_boundary_loss:.4f}  val_boundary={val_boundary_loss:.4f}  "
        f"mean_dice={mean_dice:.4f}  best={best_dice:.4f}  "
        f"[{elapsed:.0f}s]"
    )
    print(format_metrics(m, CLASS_NAMES))

    log.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_seg_loss": train_seg_loss,
        "train_boundary_loss": train_boundary_loss,
        "val_loss": val_loss,
        "val_seg_loss": val_seg_loss,
        "val_boundary_loss": val_boundary_loss,
        "mean_dice": float(mean_dice),
        "mean_iou": float(m["mean_iou"]),
        "pixel_acc": float(m["pixel_accuracy"]),
        "dice_other": float(m["dice_per_class"][0]),
        "dice_tumor": float(m["dice_per_class"][1]),
        "dice_stroma": float(m["dice_per_class"][2]),
    })
    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

torch.save(model.state_dict(), OUTPUT_DIR / "last_model.pth")
print(f"\nTraining complete. Best val mean Dice: {best_dice:.4f}")
print(f"Checkpoints saved to {OUTPUT_DIR}")
