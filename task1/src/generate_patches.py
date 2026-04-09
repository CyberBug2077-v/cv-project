"""
Pre-generate minority-class-focused training crops from the full images.

For each training image that contains Stroma or Other pixels:
  - Samples K random anchor pixels from those minority classes
  - Crops a 512x512 window centred on each anchor (clamped to image bounds)
  - Saves the cropped image and mask at crop resolution
  - PatchDataset will resize to IMAGE_SIZE at training time

Also saves debug comparison panels to data/Task1/patches/debug/
Each panel shows: original (with crop rectangle) | original mask | patch | patch mask

Usage:
    conda run -n ML python3 src/generate_patches.py
"""

import pathlib
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import DATA_ROOT, geojson_to_mask, NUM_CLASSES
from visualise import CLASS_COLOURS, mask_to_rgb

# Patch-generation settings.

PATCHES_PER_IMAGE = 4       # crops to attempt per eligible image
CROP_SIZE         = 512     # crop window size in original 1024x1024 space
MINORITY_CLASSES  = [0, 2]  # Other=0, Stroma=2
DEBUG_LIMIT       = 15      # how many debug panels to save
SEED              = 42

PATCH_DIR = DATA_ROOT / "patches"
DEBUG_DIR = PATCH_DIR / "debug"
PATCH_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

# Helper functions for crop selection and visual debugging.

def crop_at_anchor(img_np, mask_np, anchor_y, anchor_x, crop_size):
    """Crop a crop_size x crop_size window centred on (anchor_y, anchor_x).
    Clamps to image bounds. Returns (cropped_img, cropped_mask, y0, x0)."""
    H, W = mask_np.shape
    half = crop_size // 2
    y0 = int(np.clip(anchor_y - half, 0, H - crop_size))
    x0 = int(np.clip(anchor_x - half, 0, W - crop_size))
    return (
        img_np[y0:y0 + crop_size, x0:x0 + crop_size],
        mask_np[y0:y0 + crop_size, x0:x0 + crop_size],
        y0, x0,
    )


def save_debug_panel(orig_img_np, orig_mask_np, patch_img_np, patch_mask_np,
                     y0, x0, stem, k, display_size=256):
    """Save a 4-panel debug image.

    Top row: original image (with crop rectangle) | original mask
    Bottom row: patch image | patch mask
    """
    scale = display_size / orig_img_np.shape[0]  # 256/1024 = 0.25

    # Scale original down to display size
    orig_pil  = Image.fromarray(orig_img_np).resize((display_size, display_size), Image.BILINEAR)
    omask_pil = Image.fromarray(mask_to_rgb(orig_mask_np)).resize((display_size, display_size), Image.NEAREST)

    # Draw crop rectangle on original image
    rx0, ry0 = int(x0 * scale), int(y0 * scale)
    rx1 = int(rx0 + CROP_SIZE * scale)
    ry1 = int(ry0 + CROP_SIZE * scale)
    orig_ann = orig_pil.copy()
    draw = ImageDraw.Draw(orig_ann)
    draw.rectangle([rx0, ry0, rx1, ry1], outline=(255, 200, 0), width=2)

    # Scale patch to display size
    patch_pil  = Image.fromarray(patch_img_np).resize((display_size, display_size), Image.BILINEAR)
    pmask_pil  = Image.fromarray(mask_to_rgb(patch_mask_np)).resize((display_size, display_size), Image.NEAREST)

    # Header height
    hdr = 28
    gap = 6
    panels = [orig_ann, omask_pil, patch_pil, pmask_pil]
    labels = ["Original (crop box)", "Original mask", "Patch", "Patch mask"]

    total_w = len(panels) * display_size + (len(panels) - 1) * gap
    canvas = Image.new("RGB", (total_w, display_size + hdr), (255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    for i, (panel, label) in enumerate(zip(panels, labels)):
        x = i * (display_size + gap)
        canvas.paste(panel, (x, hdr))
        draw.text((x + 4, 6), label, fill=(40, 40, 40))

    # Classes present in patch
    classes_in_patch = sorted({int(v) for v in np.unique(patch_mask_np)})
    class_names = ["Other", "Tumor", "Stroma"]
    cls_str = " ".join(class_names[c] for c in classes_in_patch)
    draw.text((4, hdr + display_size - 18), f"classes: {cls_str}", fill=(255, 255, 255))

    out = DEBUG_DIR / f"{stem}_patch_{k:02d}.png"
    canvas.save(str(out))


# Main entry point.

if __name__ == "__main__":
    image_dir  = DATA_ROOT / "train" / "image"
    tissue_dir = DATA_ROOT / "train" / "tissue"

    img_paths = sorted(image_dir.glob("*.tif"))
    total_patches = 0
    debug_count   = 0

    for img_path in img_paths:
        stem        = img_path.stem
        geojson_path = tissue_dir / f"{stem}_tissue.geojson"
        if not geojson_path.exists():
            continue

        # Load original image as RGB at full resolution
        orig_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = orig_pil.size    # (1024, 1024) — PIL gives (W, H)
        orig_np  = np.array(orig_pil)     # (H, W, 3)

        # Generate mask at original resolution
        mask_np = geojson_to_mask(geojson_path, height=orig_h, width=orig_w)  # (H, W) uint8

        # Find minority class pixels
        minority_ys, minority_xs = np.where(
            np.isin(mask_np, MINORITY_CLASSES) & (mask_np != 255)
        )
        if len(minority_ys) == 0:
            continue  # pure-tumour tile — skip

        # Sample K anchors from minority pixels (with replacement if needed)
        n_anchors = min(PATCHES_PER_IMAGE, len(minority_ys))
        idxs = np.random.choice(len(minority_ys), size=PATCHES_PER_IMAGE, replace=True)
        anchors = list(zip(minority_ys[idxs].tolist(), minority_xs[idxs].tolist()))

        for k, (ay, ax) in enumerate(anchors):
            patch_img, patch_mask, y0, x0 = crop_at_anchor(orig_np, mask_np, ay, ax, CROP_SIZE)

            # Save patch image as PNG
            img_save = PATCH_DIR / f"{stem}_patch_{k:02d}_img.png"
            Image.fromarray(patch_img).save(str(img_save))

            # Save patch mask as single-channel PNG (values 0, 1, 2)
            mask_save = PATCH_DIR / f"{stem}_patch_{k:02d}_mask.png"
            Image.fromarray(patch_mask, mode="L").save(str(mask_save))

            total_patches += 1

            # Debug panel for the first DEBUG_LIMIT patches
            if debug_count < DEBUG_LIMIT:
                save_debug_panel(orig_np, mask_np, patch_img, patch_mask,
                                 y0, x0, stem, k)
                debug_count += 1

        print(f"  {stem}: {PATCHES_PER_IMAGE} patches  "
              f"(minority px: {len(minority_ys):,})")

    print(f"\nDone. {total_patches} patches saved to {PATCH_DIR}")
    print(f"{debug_count} debug panels saved to {DEBUG_DIR}")
