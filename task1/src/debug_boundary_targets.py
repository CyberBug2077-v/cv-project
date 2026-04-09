"""
Save quick visual checks for the boundary targets used in training.

Usage:
    conda run -n ML python3 src/debug_boundary_targets.py
    conda run -n ML python3 src/debug_boundary_targets.py validation 6
"""

import sys
import pathlib

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from boundary import boundary_from_mask_np
from data import DATA_ROOT, TissueDataset
from visualise import denormalise, mask_to_rgb


SPLIT = sys.argv[1] if len(sys.argv) > 1 else "validation"
COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 6
OUT_DIR = pathlib.Path("outputs_boundary_debug") / SPLIT
OUT_DIR.mkdir(parents=True, exist_ok=True)


def boundary_to_rgb(boundary_np):
    rgb = np.zeros((boundary_np.shape[0], boundary_np.shape[1], 3), dtype=np.uint8)
    rgb[boundary_np > 0] = np.array([40, 220, 80], dtype=np.uint8)
    return rgb


def overlay_boundary(image_rgb, boundary_np):
    overlay = image_rgb.copy()
    overlay[boundary_np > 0] = (
        0.5 * overlay[boundary_np > 0] + 0.5 * np.array([40, 220, 80])
    ).astype(np.uint8)
    return overlay


def save_panel(img_tensor, mask_np, boundary_np, save_path, title):
    # Show the raw image, class mask, binary boundary map, and overlay together.
    image_rgb = denormalise(img_tensor)
    mask_rgb = mask_to_rgb(mask_np)
    boundary_rgb = boundary_to_rgb(boundary_np)
    overlay_rgb = overlay_boundary(image_rgb, boundary_np)

    h, w = image_rgb.shape[:2]
    gap = 8
    canvas = np.full((h + 30, 4 * w + 3 * gap, 3), 255, dtype=np.uint8)
    canvas[30:, :w] = image_rgb
    canvas[30:, w + gap:2 * w + gap] = mask_rgb
    canvas[30:, 2 * w + 2 * gap:3 * w + 2 * gap] = boundary_rgb
    canvas[30:, 3 * w + 3 * gap:] = overlay_rgb

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    labels = ["Input", "Mask", "Boundary", "Overlay"]
    x_positions = [
        w // 2,
        w + gap + w // 2,
        2 * w + 2 * gap + w // 2,
        3 * w + 3 * gap + w // 2,
    ]
    for label, x in zip(labels, x_positions):
        draw.text((x - 30, 5), label, fill=(0, 0, 0))
    draw.text((4, 5), title, fill=(80, 80, 80))
    pil.save(str(save_path))


if __name__ == "__main__":
    ds = TissueDataset(
        image_dir=DATA_ROOT / SPLIT / "image",
        tissue_dir=DATA_ROOT / SPLIT / "tissue",
        augment=False,
    )

    count = min(COUNT, len(ds))
    for idx in range(count):
        img_tensor, mask_tensor = ds[idx]
        sample_name = ds.get_sample_name(idx)
        mask_np = mask_tensor.numpy()
        boundary_np = boundary_from_mask_np(mask_np, width=2)
        boundary_ratio = float(boundary_np.mean())
        save_panel(
            img_tensor,
            mask_np,
            boundary_np,
            OUT_DIR / f"{sample_name}.png",
            title=f"{sample_name} | boundary_ratio={boundary_ratio:.4f}",
        )

    print(f"Saved {count} boundary debug panels to {OUT_DIR}")
