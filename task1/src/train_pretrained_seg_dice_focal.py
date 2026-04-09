"""
Fine-tune the pretrained segmentation model with Dice plus focal loss.

Loads the trial1 autoencoder checkpoint (reconstruction-only pretraining),
attaches a fresh segmentation decoder, and trains with:
  combined loss = DiceLoss + FocalLoss(gamma=1.0, weight=class_weights)

Usage:
    conda run -n ML python3 src/train_pretrained_seg_dice_focal.py

Saves to: outputs_dice_focal/pretrained_seg/
  - best_model.pth
  - last_model.pth
  - training_log.json
"""

import json
import sys
import time
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import get_dataloaders, NUM_CLASSES, CLASS_NAMES, compute_class_weights
from model_autoencoder import Autoencoder, SegWithPretrainedEncoder
from model_unet import count_parameters
from metrics import ConfusionMatrix, format_metrics
from losses import DiceLoss, FocalLoss

# Training configuration.

EPOCHS     = 60
BATCH_SIZE = 8
LR         = 1e-4
GAMMA      = 1.0
AE_CKPT    = pathlib.Path("outputs_trial1/autoencoder/best_autoencoder.pth")
OUTPUT_DIR = pathlib.Path("outputs_dice_focal/pretrained_seg")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pick the available accelerator, with CPU as a fallback.

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

# Load the pretrained encoder and attach the segmentation decoder.

ae = Autoencoder(in_channels=3)
ae.load_state_dict(torch.load(AE_CKPT, map_location="cpu"))
print(f"Loaded autoencoder from {AE_CKPT}")

model = SegWithPretrainedEncoder(ae, num_classes=NUM_CLASSES).to(device)
print(f"Trainable parameters: {count_parameters(model):,}")

# Build the training and validation dataloaders.

train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE, num_workers=0)
print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

# Dice and focal loss are summed with equal weight.

class_weights = compute_class_weights().to(device)
print(f"Class weights: {class_weights.tolist()}")

dice_criterion  = DiceLoss(num_classes=NUM_CLASSES)
focal_criterion = FocalLoss(gamma=GAMMA, weight=class_weights)

def criterion(logits, masks):
    return dice_criterion(logits, masks) + focal_criterion(logits, masks)

# Adam optimiser with validation-Dice LR scheduling.

optimiser = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode="max", factor=0.5, patience=8
)

# Main training loop.

log = []
best_dice = 0.0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

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
        f"Epoch {epoch:3d}/{EPOCHS}  "
        f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
        f"mean_dice={mean_dice:.4f}  best={best_dice:.4f}  "
        f"[{elapsed:.0f}s]"
    )
    print(format_metrics(m, CLASS_NAMES))

    log.append({
        "epoch":       epoch,
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
