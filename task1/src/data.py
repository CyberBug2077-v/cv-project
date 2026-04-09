"""
Load images, rasterise GeoJSON tissue labels, and build dataloaders.

Three classes:
    0 = Other  (blood_vessel, epidermis, white_background, necrosis, unlabeled)
    1 = Tumor
    2 = Stroma

Images are loaded as RGB and resized to 256x256 before training.
"""

import json
import pathlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Dataset label definitions.

CLASS_NAMES = ["Other", "Tumor", "Stroma"]
NUM_CLASSES = 3

# Any source label not listed here is folded into class 0 ("Other").
GEOJSON_TO_CLASS = {
    "tissue_tumor": 1,
    "tissue_stroma": 2,
}

# GeoJSON to mask conversion.

def geojson_to_mask(geojson_path, height, width):
    """Convert a tissue GeoJSON annotation file to an integer class mask.

    Pixels not covered by an annotation stay as class 0 ("Other").
    Later polygons overwrite earlier ones if they overlap.

    Returns a numpy array of shape (height, width) with dtype uint8.
    """
    from PIL import ImageDraw

    mask = Image.new("L", (width, height), 0)  # Start with "Other" everywhere.
    draw = ImageDraw.Draw(mask)

    with open(geojson_path) as f:
        data = json.load(f)

    for feature in data.get("features", []):
        cls_name = (
            feature.get("properties", {})
            .get("classification", {})
            .get("name", "")
        )
        class_id = GEOJSON_TO_CLASS.get(cls_name, 0)

        geometry = feature.get("geometry", {})
        geom_type = geometry.get("type", "")
        coords = geometry.get("coordinates", [])

        if geom_type == "Polygon":
            _draw_polygon(draw, coords, class_id)
        elif geom_type == "MultiPolygon":
            for poly_coords in coords:
                _draw_polygon(draw, poly_coords, class_id)

    return np.array(mask, dtype=np.uint8)


def _draw_polygon(draw, coords, class_id):
    """Draw one polygon (exterior ring only) onto a PIL ImageDraw canvas."""
    exterior = coords[0]
    # PIL expects a flat list of (x, y) vertices for the exterior ring.
    flat = [(float(x), float(y)) for x, y in exterior]
    if len(flat) >= 3:
        draw.polygon(flat, fill=int(class_id))


# Main full-image dataset.

IMAGE_SIZE = 256  # All training and evaluation runs use the same resized input.


class TissueDataset(Dataset):
    """Loads tissue images and their segmentation masks.

    image_dir : directory containing .tif images
    tissue_dir: directory containing _tissue.geojson files
    augment   : whether to apply random augmentation (training only)
    """

    def __init__(self, image_dir, tissue_dir, augment=False):
        self.image_dir = pathlib.Path(image_dir)
        self.tissue_dir = pathlib.Path(tissue_dir)
        self.augment = augment

        # Keep only images that have a matching tissue annotation file.
        self.samples = []
        for img_path in sorted(self.image_dir.glob("*.tif")):
            stem = img_path.stem  # e.g. training_set_metastatic_roi_001
            geojson_path = self.tissue_dir / f"{stem}_tissue.geojson"
            if geojson_path.exists():
                self.samples.append((img_path, geojson_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No paired samples found in {image_dir}")

        # Reuse the same normalisation everywhere so training and evaluation match.
        self.img_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, geojson_path = self.samples[idx]

        # Load image as RGB
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size  # PIL gives (width, height)

        # Rasterise at the original image size before resizing to the model size.
        mask_np = geojson_to_mask(geojson_path, height=orig_h, width=orig_w)
        mask_pil = Image.fromarray(mask_np, mode="L")
        mask_pil = mask_pil.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.NEAREST)
        mask_np = np.array(mask_pil, dtype=np.int64)

        # Apply the same spatial transform to image and mask.
        if self.augment:
            if torch.rand(1).item() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask_np = np.fliplr(mask_np).copy()
            if torch.rand(1).item() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask_np = np.flipud(mask_np).copy()
            # Random 90-degree rotation
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                img = img.rotate(90 * k)
                mask_np = np.rot90(mask_np, k=k).copy()

        image_tensor = self.img_transform(img)
        mask_tensor = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor

    def get_sample_name(self, idx):
        return self.samples[idx][0].stem


# Dataset for the pre-generated minority-class crops.

DATA_ROOT = pathlib.Path("/Users/boud/CV_project/data/Task1")


class PatchDataset(Dataset):
    """Loads pre-generated minority-class patches from data/Task1/patches/.

    Patches are saved as:
      {stem}_patch_{k:02d}_img.png   — cropped RGB image (crop resolution)
      {stem}_patch_{k:02d}_mask.png  — cropped mask, single-channel (values 0/1/2)

    Applies the same resize + normalise pipeline as TissueDataset.
    Augmentation (random flips + rotations) applied when augment=True.
    """

    PATCH_DIR = DATA_ROOT / "patches"

    def __init__(self, augment=True):
        self.augment   = augment
        self.img_paths = sorted(self.PATCH_DIR.glob("*_img.png"))
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No patches found in {self.PATCH_DIR}. "
                               "Run generate_patches.py first.")

        self.img_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        mask_path = img_path.parent / img_path.name.replace("_img.png", "_mask.png")

        img  = Image.open(img_path).convert("RGB")
        mask_np = np.array(Image.open(mask_path), dtype=np.int64)

        # Use nearest-neighbour resizing so class ids are not interpolated.
        mask_pil = Image.fromarray(mask_np.astype(np.uint8), mode="L")
        mask_pil = mask_pil.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.NEAREST)
        mask_np  = np.array(mask_pil, dtype=np.int64)

        if self.augment:
            if torch.rand(1).item() > 0.5:
                img     = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask_np = np.fliplr(mask_np).copy()
            if torch.rand(1).item() > 0.5:
                img     = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask_np = np.flipud(mask_np).copy()
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                img     = img.rotate(90 * k)
                mask_np = np.rot90(mask_np, k=k).copy()

        return self.img_transform(img), torch.from_numpy(mask_np).long()


# Dataloader helpers.


def get_dataloaders(batch_size=4, num_workers=2):
    """Build the standard train, validation, and test dataloaders."""
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
    test_ds = TissueDataset(
        image_dir=DATA_ROOT / "test" / "image",
        tissue_dir=DATA_ROOT / "test" / "tissue",
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False,
                              num_workers=0)

    return train_loader, val_loader, test_loader


def compute_class_weights(num_samples=50):
    """Estimate per-class pixel frequencies from a subset of training images
    and return inverse-frequency weights for use in weighted cross-entropy."""
    ds = TissueDataset(
        image_dir=DATA_ROOT / "train" / "image",
        tissue_dir=DATA_ROOT / "train" / "tissue",
        augment=False,
    )
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    n = min(num_samples, len(ds))
    for i in range(n):
        _, mask = ds[i]
        for c in range(NUM_CLASSES):
            counts[c] += (mask == c).sum().item()

    # Normalise the inverse-frequency weights so their scale stays easy to read.
    freq = counts / counts.sum()
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    # Simple sanity check for dataset loading and class mapping.
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=2)
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")
    print(f"Test batches  : {len(test_loader)}")

    imgs, masks = next(iter(train_loader))
    print(f"Image batch   : {imgs.shape}  dtype={imgs.dtype}")
    print(f"Mask batch    : {masks.shape}  dtype={masks.dtype}")
    print(f"Mask classes  : {masks.unique().tolist()}")

    w = compute_class_weights()
    print(f"Class weights : {w.tolist()}")
