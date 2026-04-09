"""
Train the plain reconstruction autoencoder on the raw tissue images.

Trains the encoder to reconstruct input images (no labels used).
The encoder weights are saved for transfer to the segmentation model.

Usage:
    conda run -n ML python3 src/train_autoencoder.py

Saves to: outputs/autoencoder/
  - best_autoencoder.pth    (lowest validation reconstruction loss)
  - last_autoencoder.pth    (final epoch)
  - training_log.json
"""

import json
import sys
import time
import pathlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import DATA_ROOT, TissueDataset
from model_autoencoder import Autoencoder, count_parameters

# Training configuration.

EPOCHS     = 40
BATCH_SIZE = 4
LR         = 1e-4
OUTPUT_DIR = pathlib.Path("outputs_trial2/autoencoder")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pick the available accelerator, with CPU as a fallback.

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

# Build image-only training and validation datasets.
# Reuse TissueDataset but only use the image tensor.

train_ds = TissueDataset(
    image_dir=DATA_ROOT / "train" / "image",
    tissue_dir=DATA_ROOT / "train" / "tissue",
    augment=True,
)
val_ds = TissueDataset(
    image_dir=DATA_ROOT / "validation" / "image",
    tissue_dir=DATA_ROOT / "validation" / "tissue",
    augment=False,
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

# Build the autoencoder and reconstruction loss.

model = Autoencoder(in_channels=3).to(device)
print(f"Autoencoder parameters: {count_parameters(model):,}")

criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode="min", factor=0.5, patience=6
)

# Main training loop.

log = []
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # ---- Train ----
    model.train()
    train_loss = 0.0
    for imgs, _ in train_loader:      # ignore masks
        imgs = imgs.to(device)
        optimiser.zero_grad()
        recon = model(imgs)
        loss  = criterion(recon, imgs)
        loss.backward()
        optimiser.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # ---- Validate ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            recon = model(imgs)
            val_loss += criterion(recon, imgs).item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), OUTPUT_DIR / "best_autoencoder.pth")

    elapsed = time.time() - t0
    print(
        f"Epoch {epoch:3d}/{EPOCHS}  "
        f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
        f"best={best_val_loss:.5f}  [{elapsed:.0f}s]"
    )

    log.append({
        "epoch":      epoch,
        "train_loss": train_loss,
        "val_loss":   val_loss,
    })
    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

torch.save(model.state_dict(), OUTPUT_DIR / "last_autoencoder.pth")
print(f"\nAutoencoder training complete. Best val loss: {best_val_loss:.5f}")
print(f"Checkpoints saved to {OUTPUT_DIR}")
