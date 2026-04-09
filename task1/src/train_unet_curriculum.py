"""
Train the U-Net with Dice loss and a patch-to-full-image curriculum.

Curriculum schedule (cosine):
  - Epoch 1:  75% samples drawn from minority-class patches
  - Epoch 60:  5% samples drawn from patches (mostly original images)
  - Smooth cosine decay between the two extremes

Each epoch a WeightedRandomSampler is rebuilt with the current ratio.
Total samples per epoch stays fixed at len(full training set).

Usage:
    conda run -n ML python3 src/train_unet_curriculum.py

Saves to: outputs_curriculum/unet/
  - best_model.pth
  - last_model.pth
  - training_log.json   (includes patch_ratio column)
"""

import json
import math
import sys
import time
import pathlib

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import get_dataloaders, PatchDataset, TissueDataset, DATA_ROOT, NUM_CLASSES, CLASS_NAMES
from model_unet import UNet, count_parameters
from metrics import ConfusionMatrix, format_metrics
from losses import DiceLoss

# Training configuration.

EPOCHS               = 60
BATCH_SIZE           = 8
LR                   = 1e-4
PATCH_RATIO_START    = 0.75  # patch fraction at epoch 1
PURE_ORIGINAL_EPOCHS = 20    # final N epochs train on original images only
OUTPUT_DIR           = pathlib.Path("outputs_curriculum/unet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pick the available accelerator, with CPU as a fallback.

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

# Build the U-Net.

model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
print(f"Trainable parameters: {count_parameters(model):,}")

# Load the full-image dataset and the pre-generated patch dataset.

full_ds  = TissueDataset(
    image_dir=DATA_ROOT / "train" / "image",
    tissue_dir=DATA_ROOT / "train" / "tissue",
    augment=True,
)
patch_ds = PatchDataset(augment=True)

print(f"Full training images : {len(full_ds)}")
print(f"Minority-class patches: {len(patch_ds)}")

_, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE, num_workers=0)
print(f"Val batches: {len(val_loader)}")

# Use pure Dice loss for the supervised objective.

criterion = DiceLoss(num_classes=NUM_CLASSES)

# Adam optimiser with validation-Dice LR scheduling.

optimiser = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode="max", factor=0.5, patience=8
)

# Smoothly reduce the patch sampling ratio over training.

def patch_ratio(epoch):
    """Cosine decay from PATCH_RATIO_START → 0.0 over the first (EPOCHS - PURE_ORIGINAL_EPOCHS)
    epochs, then 0.0 for the remaining PURE_ORIGINAL_EPOCHS epochs."""
    transition_epochs = EPOCHS - PURE_ORIGINAL_EPOCHS   # 40
    if epoch > transition_epochs:
        return 0.0
    t = (epoch - 1) / (transition_epochs - 1)           # 0.0 → 1.0 over epochs 1–40
    cosine_t = 0.5 * (1.0 - math.cos(math.pi * t))      # slow start, fast middle, slow end
    return PATCH_RATIO_START * (1.0 - cosine_t)          # 0.75 → 0.0


def make_train_loader(epoch):
    """Build a DataLoader with the current epoch's patch/full mixing ratio.
    When patch_ratio reaches 0.0 (final PURE_ORIGINAL_EPOCHS), uses a plain
    shuffled DataLoader over the original training set only."""
    p = patch_ratio(epoch)
    if p == 0.0:
        return DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    n_patch = len(patch_ds)
    n_full  = len(full_ds)
    w_patch = p / n_patch
    w_full  = (1.0 - p) / n_full
    weights = [w_patch] * n_patch + [w_full] * n_full
    sampler = WeightedRandomSampler(weights, num_samples=n_full, replacement=True)
    return DataLoader(ConcatDataset([patch_ds, full_ds]), batch_size=BATCH_SIZE,
                      sampler=sampler, num_workers=0)


# Main training loop.

log      = []
best_dice = 0.0

for epoch in range(1, EPOCHS + 1):
    t0    = time.time()
    p     = patch_ratio(epoch)
    train_loader = make_train_loader(epoch)

    # ---- Train ----
    model.train()
    train_loss = 0.0
    for imgs, masks in train_loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimiser.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, masks)
        loss.backward()
        optimiser.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- Validate ----
    model.eval()
    val_loss = 0.0
    cm = ConfusionMatrix(NUM_CLASSES)

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            val_loss += criterion(logits, masks).item()

            preds = logits.argmax(dim=1)
            cm.update(preds.cpu().numpy(), masks.cpu().numpy())

    val_loss /= len(val_loader)
    m = cm.compute()
    mean_dice = m["mean_dice"]

    scheduler.step(mean_dice)

    if mean_dice > best_dice:
        best_dice = mean_dice
        torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")

    elapsed = time.time() - t0
    print(
        f"Epoch {epoch:3d}/{EPOCHS}  patch_ratio={p:.3f}  "
        f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
        f"mean_dice={mean_dice:.4f}  best={best_dice:.4f}  [{elapsed:.0f}s]"
    )
    print(format_metrics(m, CLASS_NAMES))

    log.append({
        "epoch":       epoch,
        "patch_ratio": round(p, 4),
        "train_loss":  train_loss,
        "val_loss":    val_loss,
        "mean_dice":   float(mean_dice),
        "mean_iou":    float(m["mean_iou"]),
        "pixel_acc":   float(m["pixel_accuracy"]),
        "dice_other":  float(m["dice_per_class"][0]),
        "dice_tumor":  float(m["dice_per_class"][1]),
        "dice_stroma": float(m["dice_per_class"][2]),
    })
    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

torch.save(model.state_dict(), OUTPUT_DIR / "last_model.pth")
print(f"\nTraining complete. Best val mean Dice: {best_dice:.4f}")
print(f"Checkpoints saved to {OUTPUT_DIR}")
