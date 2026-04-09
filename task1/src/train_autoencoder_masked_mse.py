"""
Train the masked-reconstruction autoencoder with plain MSE loss.

Instead of reconstructing the clean input from itself, the model receives a
corrupted version and must reconstruct the original clean image.  This is a
stronger pretext task than plain reconstruction: the encoder must infer missing
content and colour distributions rather than just compress and copy pixels.

Corruption applied per image (independently in each batch):
  1. Random block masking (always): 4–8 blocks of 24–48 px, filled with 0.0
     (≈ normalised ImageNet mean).  Forces the encoder to infer missing tissue
     context — analogous to masked autoencoders in histopathology SSL.
  2. Gaussian noise (p=0.7): σ=0.15 in normalised space.
  3. Stain jitter (p=0.7): independent per-channel scale in [0.8, 1.2],
     simulating H&E staining variation across slides.

Reconstruction target: always the clean, unmodified image.
Loss: MSE(reconstruction, clean_image).

Usage:
    conda run -n ML python3 src/train_autoencoder_masked_mse.py

Saves to: outputs_masked_mse/autoencoder/
  - best_autoencoder.pth
  - last_autoencoder.pth
  - training_log.json
"""

import json
import random
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

EPOCHS     = 60
BATCH_SIZE = 8
LR         = 1e-4
OUTPUT_DIR = pathlib.Path("outputs_masked_mse/autoencoder")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pick the available accelerator, with CPU as a fallback.

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

# Corruption pipeline used for the masked-reconstruction task.

def corrupt_batch(imgs):
    """Apply random corruption to a batch of normalised image tensors.

    Returns the corrupted batch.  The clean imgs tensor is used as the
    reconstruction target and must not be modified.
    """
    B, C, H, W = imgs.shape
    corrupted = imgs.clone()

    for i in range(B):
        # 1. Random block masking (always applied)
        n_blocks = random.randint(4, 8)
        for _ in range(n_blocks):
            bh = random.randint(24, 48)
            bw = random.randint(24, 48)
            y  = random.randint(0, H - bh)
            x  = random.randint(0, W - bw)
            corrupted[i, :, y:y + bh, x:x + bw] = 0.0  # normalised mean ≈ 0

        # 2. Gaussian noise
        if random.random() < 0.7:
            corrupted[i] = corrupted[i] + torch.randn_like(corrupted[i]) * 0.15

        # 3. Stain jitter — per-channel scaling mimics slide staining variation
        if random.random() < 0.7:
            for c in range(C):
                corrupted[i, c] = corrupted[i, c] * random.uniform(0.8, 1.2)

    return corrupted

# Build the image-only training and validation datasets.

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
    optimiser, mode="min", factor=0.5, patience=8
)

# Main training loop.

log = []
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # ---- Train ----
    model.train()
    train_loss = 0.0
    for imgs, _ in train_loader:      # masks not needed
        imgs      = imgs.to(device)
        corrupted = corrupt_batch(imgs)

        optimiser.zero_grad()
        recon = model(corrupted)
        loss  = criterion(recon, imgs)   # reconstruct clean from corrupted
        loss.backward()
        optimiser.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # ---- Validate ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs      = imgs.to(device)
            corrupted = corrupt_batch(imgs)
            recon     = model(corrupted)
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

    log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

torch.save(model.state_dict(), OUTPUT_DIR / "last_autoencoder.pth")
print(f"\nMasked MSE pretraining complete. Best val loss: {best_val_loss:.5f}")
print(f"Checkpoint saved to {OUTPUT_DIR}")
