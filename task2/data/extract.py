import json
import csv
import random
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import tifffile
from shapely.geometry import Polygon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from task2.config import (
    TASK2_CLASS_MAP,
    TASK2_LABEL_TO_NAME,
    TASK2_PATCH_SIZE,
    TASK2_RANDOM_SEED,
    TASK2_TRAIN_SAMPLES_PER_CLASS,
    TASK2_VAL_SAMPLES_PER_CLASS,
    TASK2_CONTRASTIVE_SAMPLES_PER_CLASS,
    TASK2_TRAIN_IMAGE_DIR,
    TASK2_TRAIN_NUCLEI_DIR,
    TASK2_VAL_IMAGE_DIR,
    TASK2_VAL_NUCLEI_DIR,
    TASK2_PATCH_OUTPUT_DIR,
    TASK2_TRAIN_CSV,
    TASK2_VAL_CSV,
    TASK2_CONTRASTIVE_CSV,
)

def list_geojson_files(geojson_dir):
    geojson_dir = Path(geojson_dir)

    if not geojson_dir.exists():
        raise FileNotFoundError(f"GeoJSON directory does not exist: {geojson_dir}")

    files = sorted(
        [
            p for p in geojson_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".geojson"
        ]
    )
    return files

def load_tif_image(image_path):
    image = tifffile.imread(image_path)

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.ndim == 3 and image.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return image


def load_geojson(geojson_path):
    with open(geojson_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_centroid_from_polygon(points):
    try:
        poly = Polygon(points)
        if poly.is_valid and not poly.is_empty:
            return float(poly.centroid.x), float(poly.centroid.y)
    except Exception:
        pass

    points = np.asarray(points, dtype=np.float32)
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def parse_nuclei_annotations(geojson_data, class_map=None):
    records = []
    features = geojson_data.get("features", [])

    for feature in features:
        geometry = feature.get("geometry", {})
        properties = feature.get("properties", {})
        classification = properties.get("classification", {})

        if geometry.get("type") != "Polygon":
            continue

        coords = geometry.get("coordinates", [])
        if not coords or not coords[0]:
            continue

        polygon_points = coords[0]
        raw_class_name = classification.get("name")

        if class_map is not None and raw_class_name not in class_map:
            continue

        center_x, center_y = compute_centroid_from_polygon(polygon_points)
        label = class_map[raw_class_name]
        class_name = TASK2_LABEL_TO_NAME[label]

        records.append(
            {
                "feature_id": feature.get("id"),
                "raw_class_name": raw_class_name,
                "label": label,
                "class_name": class_name,
                "polygon": polygon_points,
                "center_x": center_x,
                "center_y": center_y,
            }
        )

    return records


def extract_patch(image, center_x, center_y, patch_size=100):
    h, w = image.shape[:2]
    half = patch_size // 2

    cx = int(round(center_x))
    cy = int(round(center_y))

    x1, x2 = cx - half, cx + half
    y1, y2 = cy - half, cy + half

    patch = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    patch[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    return patch


def find_matching_image_path(geojson_path, image_dir):
    image_dir = Path(image_dir)
    stem = Path(geojson_path).stem

    candidate_stems = [stem]
    if stem.endswith("_nuclei"):
        candidate_stems.append(stem[: -len("_nuclei")])

    for candidate_stem in candidate_stems:
        candidate = image_dir / f"{candidate_stem}.tif"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No matching image found for {geojson_path}")

def infer_sample_type_from_source_name(source_image_name):
    name = str(source_image_name).lower()
    if "primary" in name:
        return "primary"
    if "metastatic" in name:
        return "metastatic"
    return "unknown"

def collect_candidate_records(split_name, image_dir, nuclei_dir, class_map):
    image_dir = Path(image_dir)
    nuclei_dir = Path(nuclei_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not nuclei_dir.exists():
        raise FileNotFoundError(f"Nuclei directory does not exist: {nuclei_dir}")

    geojson_paths = list_geojson_files(nuclei_dir)

    print(f"[{split_name}] image_dir  = {image_dir}")
    print(f"[{split_name}] nuclei_dir = {nuclei_dir}")
    print(f"[{split_name}] found {len(geojson_paths)} geojson files")

    all_records = []

    for i, geojson_path in enumerate(geojson_paths):
        image_path = find_matching_image_path(geojson_path, image_dir)
        geojson_data = load_geojson(geojson_path)
        records = parse_nuclei_annotations(geojson_data, class_map=class_map)

        if i < 3:
            print(
                f"[{split_name}] sample file {geojson_path.name}: "
                f"{len(records)} target records"
            )

        for r in records:
            r["source_image_name"] = image_path.name
            r["source_image_path"] = str(image_path)
            r["source_geojson_path"] = str(geojson_path)
            r["split"] = split_name
            r["sample_type"] = infer_sample_type_from_source_name(image_path.name)

        all_records.extend(records)

    print(f"[{split_name}] total collected target records: {len(all_records)}")
    return all_records

def make_record_uid(record):
    return (
        str(record["source_image_name"]),
        str(record["source_geojson_path"]),
        str(record["feature_id"]),
    )


def filter_records_by_excluded_uids(records, excluded_uids):
    if not excluded_uids:
        return list(records)

    filtered = [
        r for r in records
        if make_record_uid(r) not in excluded_uids
    ]
    return filtered

def group_records_by_label(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[r["label"]].append(r)
    return grouped


def sample_balanced_records(grouped_records, samples_per_class, seed=42):
    rng = random.Random(seed)
    sampled = []

    for label, class_name in TASK2_LABEL_TO_NAME.items():
        candidates = grouped_records[label]
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"Not enough samples for class {class_name}: "
                f"required {samples_per_class}, found {len(candidates)}"
            )

        sampled.extend(rng.sample(candidates, samples_per_class))

    return sampled

def sample_balanced_records_from_records(records, samples_per_class, seed=42):
    grouped_records = group_records_by_label(records)
    return sample_balanced_records(
        grouped_records=grouped_records,
        samples_per_class=samples_per_class,
        seed=seed,
    )

def save_patch(patch, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, patch)


def export_records_as_patches_and_csv(records, output_patch_dir, output_csv_path, patch_size=100):
    output_patch_dir = Path(output_patch_dir)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "patch_path",
        "label",
        "class_name",
        "raw_class_name",
        "feature_id",
        "source_image_name",
        "source_image_path",
        "source_geojson_path",
        "center_x",
        "center_y",
        "split",
    ]

    image_cache = {}
    rows = []

    for idx, record in enumerate(records):
        source_image_path = record["source_image_path"]

        if source_image_path not in image_cache:
            image_cache[source_image_path] = load_tif_image(source_image_path)

        image = image_cache[source_image_path]
        patch = extract_patch(
            image=image,
            center_x=record["center_x"],
            center_y=record["center_y"],
            patch_size=patch_size,
        )

        patch_filename = (
            f'{record["class_name"]}_'
            f'{Path(record["source_image_name"]).stem}_'
            f'{record["feature_id"]}.npy'
        )
        patch_path = output_patch_dir / record["class_name"] / patch_filename
        save_patch(patch, patch_path)

        row = {
            "patch_path": str(patch_path),
            "label": record["label"],
            "class_name": record["class_name"],
            "raw_class_name": record["raw_class_name"],
            "feature_id": record["feature_id"],
            "source_image_name": record["source_image_name"],
            "source_image_path": record["source_image_path"],
            "source_geojson_path": record["source_geojson_path"],
            "center_x": record["center_x"],
            "center_y": record["center_y"],
            "split": record["split"],
        }
        rows.append(row)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} samples to {output_csv_path}")


def build_supervised_split(
    split_name,
    image_dir,
    nuclei_dir,
    output_patch_dir,
    output_csv_path,
    samples_per_class,
    class_map,
    patch_size=100,
    seed=42,
):
    print(f"\nBuilding supervised split: {split_name}")
    all_records = collect_candidate_records(split_name, image_dir, nuclei_dir, class_map)
    grouped_records = group_records_by_label(all_records)

    for label, class_name in TASK2_LABEL_TO_NAME.items():
        print(f"  {class_name}: {len(grouped_records[label])} candidates")

    sampled_records = sample_balanced_records(
        grouped_records=grouped_records,
        samples_per_class=samples_per_class,
        seed=seed,
    )

    export_records_as_patches_and_csv(
        records=sampled_records,
        output_patch_dir=output_patch_dir,
        output_csv_path=output_csv_path,
        patch_size=patch_size,
    )

    return sampled_records

def build_contrastive_split(
    split_name,
    image_dir,
    nuclei_dir,
    output_patch_dir,
    output_csv_path,
    samples_per_class,
    class_map,
    exclude_records=None,
    patch_size=100,
    seed=42,
):
    print(f"\nBuilding contrastive split: {split_name}")

    all_records = collect_candidate_records(split_name, image_dir, nuclei_dir, class_map)

    excluded_uids = set()
    if exclude_records is not None:
        excluded_uids = {make_record_uid(r) for r in exclude_records}

    candidate_records = filter_records_by_excluded_uids(all_records, excluded_uids)
    grouped_records = group_records_by_label(candidate_records)

    print(f"[{split_name}] excluded {len(excluded_uids)} supervised-train nuclei")
    for label, class_name in TASK2_LABEL_TO_NAME.items():
        print(f"  {class_name}: {len(grouped_records[label])} candidates after exclusion")

    sampled_records = sample_balanced_records(
        grouped_records=grouped_records,
        samples_per_class=samples_per_class,
        seed=seed,
    )

    export_records_as_patches_and_csv(
        records=sampled_records,
        output_patch_dir=output_patch_dir,
        output_csv_path=output_csv_path,
        patch_size=patch_size,
    )

    return sampled_records


def main():
    train_records = build_supervised_split(
        split_name="train",
        image_dir=TASK2_TRAIN_IMAGE_DIR,
        nuclei_dir=TASK2_TRAIN_NUCLEI_DIR,
        output_patch_dir=TASK2_PATCH_OUTPUT_DIR / "train",
        output_csv_path=TASK2_TRAIN_CSV,
        samples_per_class=TASK2_TRAIN_SAMPLES_PER_CLASS,
        class_map=TASK2_CLASS_MAP,
        patch_size=TASK2_PATCH_SIZE,
        seed=TASK2_RANDOM_SEED,
    )

    build_supervised_split(
        split_name="val",
        image_dir=TASK2_VAL_IMAGE_DIR,
        nuclei_dir=TASK2_VAL_NUCLEI_DIR,
        output_patch_dir=TASK2_PATCH_OUTPUT_DIR / "val",
        output_csv_path=TASK2_VAL_CSV,
        samples_per_class=TASK2_VAL_SAMPLES_PER_CLASS,
        class_map=TASK2_CLASS_MAP,
        patch_size=TASK2_PATCH_SIZE,
        seed=TASK2_RANDOM_SEED,
    )

    build_contrastive_split(
        split_name="contrastive",
        image_dir=TASK2_TRAIN_IMAGE_DIR,
        nuclei_dir=TASK2_TRAIN_NUCLEI_DIR,
        output_patch_dir=TASK2_PATCH_OUTPUT_DIR / "contrastive",
        output_csv_path=TASK2_CONTRASTIVE_CSV,
        samples_per_class=TASK2_CONTRASTIVE_SAMPLES_PER_CLASS,
        class_map=TASK2_CLASS_MAP,
        exclude_records=train_records,
        patch_size=TASK2_PATCH_SIZE,
        seed=TASK2_RANDOM_SEED + 1,
    )

if __name__ == "__main__":
    main()
