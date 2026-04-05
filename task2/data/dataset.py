from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import tifffile


def _read_image(image_path: str | Path) -> np.ndarray:
    """
    Read an image patch from disk and return HWC uint8 RGB numpy array.
    Supports .npy, .png, .jpg, .jpeg, .tif, .tiff
    """
    image_path = Path(image_path)
    suffix = image_path.suffix.lower()

    if suffix == ".npy":
        image = np.load(image_path)
    elif suffix in {".tif", ".tiff"}:
        image = tifffile.imread(image_path)
    else:
        image = np.array(Image.open(image_path))

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape {image.shape} for {image_path}")

    if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.transpose(image, (1, 2, 0))

    # RGBA -> RGB
    if image.shape[-1] == 4:
        image = image[..., :3]

    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels, got shape {image.shape} for {image_path}")

    # normalize dtype to uint8 if needed
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 255.0)
            if image.max() <= 1.0:
                image = image * 255.0
            image = image.astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def _default_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Convert HWC uint8 numpy image to CHW float tensor in [0, 1].
    """
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return tensor


def _is_macos_artifact(path: Path) -> bool:
    return "__MACOSX" in path.parts or path.name.startswith("._")


class NucleiClassificationDataset(Dataset):
    """
    Supervised dataset for Task 2 nuclei classification.

    Expected CSV columns:
    - patch_path
    - label
    Optional:
    - class_name
    - raw_class_name
    - feature_id
    - source_image
    - source_geojson
    - center_x
    - center_y
    - split
    - sample_type
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)

        required_cols = {"patch_path", "label"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {self.csv_path}: {missing}")

        self.transform = transform
        self.return_metadata = return_metadata

        # ensure label is int
        self.df["label"] = self.df["label"].astype(int)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image = _read_image(row["patch_path"])
        label = int(row["label"])

        if self.transform is not None:
            transformed = self.transform(image=image) if self._is_albumentations() else self.transform(Image.fromarray(image))
            if self._is_albumentations():
                image_tensor = transformed["image"]
            else:
                image_tensor = transformed
        else:
            image_tensor = _default_to_tensor(image)

        if not self.return_metadata:
            return image_tensor, label

        metadata = row.to_dict()
        return image_tensor, label, metadata

    def _is_albumentations(self) -> bool:
        return hasattr(self.transform, "__class__") and self.transform.__class__.__module__.startswith("albumentations")


class ContrastiveNucleiDataset(Dataset):
    """
    Dataset for contrastive pretraining.
    Returns two augmented views of the same patch.

    Expected CSV columns:
    - patch_path
    Optional metadata columns are preserved if return_metadata=True.
    """

    def __init__(
        self,
        csv_path: str | Path,
        view_transform: Callable,
        return_metadata: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)

        if "patch_path" not in self.df.columns:
            raise ValueError(f"'patch_path' column not found in {self.csv_path}")

        self.view_transform = view_transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = _read_image(row["patch_path"])

        if self._is_albumentations():
            view1 = self.view_transform(image=image)["image"]
            view2 = self.view_transform(image=image)["image"]
        else:
            pil_image = Image.fromarray(image)
            view1 = self.view_transform(pil_image)
            view2 = self.view_transform(pil_image)

        if not self.return_metadata:
            return view1, view2

        metadata = row.to_dict()
        return view1, view2, metadata

    def _is_albumentations(self) -> bool:
        return hasattr(self.view_transform, "__class__") and self.view_transform.__class__.__module__.startswith("albumentations")


class Task2TestDataset(Dataset):
    """
    Dataset for Task 2 test patches.

    Expected filename examples:
    - test_set_metastatic_roi_013_nuclei_lymphocyte_xxx.npy
    - test_set_metastatic_roi_013_nuclei_tumor_xxx.npy
    - test_set_primary_roi_095_nuclei_histiocyte_xxx.npy
    """

    FILENAME_CLASS_MAP = {
        "nuclei_tumor": 0,
        "nuclei_lymphocyte": 1,
        "nuclei_histiocyte": 2,
    }

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        return_paths: bool = False,
        return_metadata: bool = False,
    ) -> None:
        
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.return_paths = return_paths
        self.return_metadata = return_metadata

        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}
        self.samples = []
        self.skipped_artifact_files = 0

        for path in sorted(self.root_dir.rglob("*")):
            if not path.is_file():
                continue
            if _is_macos_artifact(path):
                self.skipped_artifact_files += 1
                continue
            if path.suffix.lower() not in valid_exts:
                continue

            filename = path.name.lower()
            label = self._infer_label_from_filename(filename)
            sample_type = self._infer_sample_type_from_filename(filename)

            self.samples.append({
                "path": str(path),
                "label": label,
                "sample_type": sample_type,
                "filename": path.name,
            })

        if len(self.samples) == 0:
            raise ValueError(f"No valid test samples found in {self.root_dir}")

    def _infer_label_from_filename(self, filename: str) -> int:
        matched = [
            label
            for pattern, label in self.FILENAME_CLASS_MAP.items()
            if pattern in filename
        ]

        if len(matched) != 1:
            raise ValueError(
                f"Could not uniquely infer class from filename: {filename}"
            )

        return matched[0]

    def _infer_sample_type_from_filename(self, filename: str) -> str:
        if "primary" in filename:
            return "primary"
        if "metastatic" in filename:
            return "metastatic"
        return "unknown"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        image = _read_image(sample["path"])
        label = sample["label"]

        if self.transform is not None:
            transformed = (
                self.transform(image=image)
                if self._is_albumentations()
                else self.transform(Image.fromarray(image))
            )
            if self._is_albumentations():
                image_tensor = transformed["image"]
            else:
                image_tensor = transformed
        else:
            image_tensor = _default_to_tensor(image)

        if self.return_metadata:
            metadata = {
                "path": sample["path"],
                "filename": sample["filename"],
                "sample_type": sample["sample_type"],
            }
            return image_tensor, label, metadata

        if self.return_paths:
            return image_tensor, label, sample["path"]

        return image_tensor, label

    def _is_albumentations(self) -> bool:
        return (
            hasattr(self.transform, "__class__")
            and self.transform.__class__.__module__.startswith("albumentations")
        )


def build_classification_datasets(
    train_csv: str | Path,
    val_csv: str | Path,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    return_metadata: bool = False,
) -> Tuple[NucleiClassificationDataset, NucleiClassificationDataset]:
    train_dataset = NucleiClassificationDataset(
        csv_path=train_csv,
        transform=train_transform,
        return_metadata=return_metadata,
    )
    val_dataset = NucleiClassificationDataset(
        csv_path=val_csv,
        transform=eval_transform,
        return_metadata=return_metadata,
    )
    return train_dataset, val_dataset


def build_contrastive_dataset(
    contrastive_csv: str | Path,
    view_transform: Callable,
    return_metadata: bool = False,
) -> ContrastiveNucleiDataset:
    return ContrastiveNucleiDataset(
        csv_path=contrastive_csv,
        view_transform=view_transform,
        return_metadata=return_metadata,
    )
