"""
Save small visual summaries of segmentation predictions.

Panels are laid out as input image, ground-truth mask, and predicted mask.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Fixed colours make it easier to compare figures across scripts and runs.
CLASS_COLOURS = np.array([
    [180, 180, 180],   # 0 Other     — grey
    [220,  40,  40],   # 1 Tumor     — red
    [ 40, 100, 200],   # 2 Stroma    — blue
], dtype=np.uint8)

CLASS_NAMES = ["Other", "Tumor", "Stroma"]


def mask_to_rgb(mask_np):
    """Convert integer class mask (H, W) to RGB image (H, W, 3)."""
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c, colour in enumerate(CLASS_COLOURS):
        rgb[mask_np == c] = colour
    return rgb


def denormalise(tensor):
    """Undo ImageNet normalisation and return a (H, W, 3) uint8 numpy array."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def save_comparison(img_tensor, gt_mask, pred_mask, save_path, title=""):
    """Save a three-panel comparison image: Input | GT | Prediction.

    img_tensor : (3, H, W) float tensor (normalised)
    gt_mask    : (H, W) int numpy array
    pred_mask  : (H, W) int numpy array
    save_path  : path to save the PNG
    title      : optional text label in the top-left
    """
    img_rgb   = denormalise(img_tensor)
    gt_rgb    = mask_to_rgb(gt_mask)
    pred_rgb  = mask_to_rgb(pred_mask)

    h, w = img_rgb.shape[:2]
    gap = 8  # Small white gap keeps the three panels visually separate.
    canvas = np.full((h + 30, 3 * w + 2 * gap, 3), 255, dtype=np.uint8)

    canvas[30:, :w] = img_rgb
    canvas[30:, w + gap: 2 * w + gap] = gt_rgb
    canvas[30:, 2 * w + 2 * gap:] = pred_rgb

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    labels = ["Input", "Ground Truth", "Prediction"]
    x_positions = [w // 2, w + gap + w // 2, 2 * w + 2 * gap + w // 2]
    for label, x in zip(labels, x_positions):
        draw.text((x - 40, 5), label, fill=(0, 0, 0))
    if title:
        draw.text((4, 5), title, fill=(80, 80, 80))

    pil.save(str(save_path))


def save_legend(save_path):
    """Save a small legend that matches the project-wide mask colours."""
    h, w = 30, 120
    canvas = np.full((len(CLASS_NAMES) * h, w, 3), 255, dtype=np.uint8)
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    for i, (name, colour) in enumerate(zip(CLASS_NAMES, CLASS_COLOURS)):
        y = i * h
        draw.rectangle([4, y + 4, 26, y + h - 4], fill=tuple(colour))
        draw.text((32, y + 8), name, fill=(0, 0, 0))
    pil.save(str(save_path))
