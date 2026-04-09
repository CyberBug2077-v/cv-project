"""Helpers for turning segmentation masks into binary boundary targets."""

import numpy as np
import torch
import torch.nn.functional as F


def boundary_from_mask(mask, width=1):
    """Mark pixels that touch a different class in an 8-neighbourhood."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
        squeeze = True
    elif mask.ndim == 3:
        squeeze = False
    else:
        raise ValueError(f"Expected mask with 2 or 3 dims, got shape {tuple(mask.shape)}")

    boundary = torch.zeros_like(mask, dtype=torch.bool)

    # Mark any horizontal, vertical, or diagonal class change as boundary.
    diff = mask[:, 1:, :] != mask[:, :-1, :]
    boundary[:, 1:, :] |= diff
    boundary[:, :-1, :] |= diff

    diff = mask[:, :, 1:] != mask[:, :, :-1]
    boundary[:, :, 1:] |= diff
    boundary[:, :, :-1] |= diff

    diff = mask[:, 1:, 1:] != mask[:, :-1, :-1]
    boundary[:, 1:, 1:] |= diff
    boundary[:, :-1, :-1] |= diff

    diff = mask[:, 1:, :-1] != mask[:, :-1, 1:]
    boundary[:, 1:, :-1] |= diff
    boundary[:, :-1, 1:] |= diff

    boundary = boundary.float().unsqueeze(1)
    if width > 1:
        # A small max-pool widens the one-pixel boundary for supervision.
        kernel = 2 * width - 1
        boundary = F.max_pool2d(boundary, kernel_size=kernel, stride=1, padding=width - 1)

    boundary = boundary.squeeze(1)
    if squeeze:
        boundary = boundary.squeeze(0)
    return boundary


def boundary_from_mask_np(mask_np, width=1):
    """Numpy wrapper used by debugging and visualisation scripts."""
    boundary = boundary_from_mask(torch.from_numpy(mask_np).long(), width=width)
    return boundary.numpy().astype(np.uint8)
