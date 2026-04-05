from collections import Counter
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from task2.data.extract import (  # noqa: E402
    load_tif_image,
    load_geojson,
    parse_nuclei_annotations,
    extract_patch,
)
from task2.config import TASK2_CLASS_MAP, TASK2_LABEL_TO_NAME  # noqa: E402


def summarize_annotations(records):
    raw_counter = Counter()
    mapped_counter = Counter()

    for r in records:
        raw_counter[r["raw_class_name"]] += 1
        if "label" in r:
            mapped_counter[TASK2_LABEL_TO_NAME[r["label"]]] += 1

    print(f"Total parsed annotations: {len(records)}")
    print("\nRaw class distribution:")
    for cls_name, count in raw_counter.most_common():
        print(f"  {cls_name}: {count}")

    print("\nMapped Task 2 class distribution:")
    for cls_name, count in mapped_counter.most_common():
        print(f"  {cls_name}: {count}")


def visualize_random_patches(image, records, num_samples=3, patch_size=100, seed=42):
    valid_records = [r for r in records if "label" in r]
    if len(valid_records) == 0:
        print("\nNo valid Task 2 classes found in this file.")
        return

    random.seed(seed)
    samples = random.sample(valid_records, min(num_samples, len(valid_records)))

    fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4))
    if len(samples) == 1:
        axes = [axes]

    for ax, record in zip(axes, samples):
        patch = extract_patch(
            image=image,
            center_x=record["center_x"],
            center_y=record["center_y"],
            patch_size=patch_size,
        )

        ax.imshow(patch)
        ax.set_title(
            f'{TASK2_LABEL_TO_NAME[record["label"]]}\n'
            f'raw={record["raw_class_name"]}\n'
            f'center=({record["center_x"]:.1f}, {record["center_y"]:.1f})',
            fontsize=9,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    image_path = "data/Dataset_Splits/train/image/training_set_metastatic_roi_001.tif"
    geojson_path = "data/Dataset_Splits/train/nuclei/training_set_metastatic_roi_001_nuclei.geojson"

    print(f"Loading image from: {image_path}")
    print(f"Loading annotations from: {geojson_path}")

    image = load_tif_image(image_path)
    geojson_data = load_geojson(geojson_path)
    records = parse_nuclei_annotations(geojson_data, class_map=TASK2_CLASS_MAP)

    print(f"\nImage shape: {image.shape}")
    summarize_annotations(records)

    print("\nExample parsed annotations:")
    for r in records[:5]:
        print(
            {
                "feature_id": r["feature_id"],
                "raw_class_name": r["raw_class_name"],
                "center_x": round(r["center_x"], 2),
                "center_y": round(r["center_y"], 2),
                "num_points": len(r["polygon"]),
                "label": r.get("label", None),
            }
        )

    visualize_random_patches(image, records, num_samples=3, patch_size=100, seed=42)


if __name__ == "__main__":
    main()