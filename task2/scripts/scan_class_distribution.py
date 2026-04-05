from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from task2.data.extract import load_geojson, parse_nuclei_annotations  # noqa: E402
from task2.config import TASK2_CLASS_MAP, TASK2_LABEL_TO_NAME  # noqa: E402


def scan_split(split_name: str, geojson_dir):
    geojson_dir = Path(geojson_dir)
    geojson_paths = sorted(geojson_dir.glob("*.geojson"))

    raw_counter = Counter()
    mapped_counter = Counter()
    files_with_target_class = 0
    raw_classes_per_file = defaultdict(set)

    print(f"\nScanning split: {split_name}")
    print(f"GeoJSON directory: {geojson_dir}")
    print(f"Found {len(geojson_paths)} geojson files")

    for geojson_path in geojson_paths:
        geojson_data = load_geojson(geojson_path)
        records = parse_nuclei_annotations(geojson_data, class_map=TASK2_CLASS_MAP)

        has_target_class = False
        for r in records:
            raw_class = r["raw_class_name"]
            raw_counter[raw_class] += 1
            raw_classes_per_file[geojson_path.name].add(raw_class)

            if "label" in r:
                mapped_name = TASK2_LABEL_TO_NAME[r["label"]]
                mapped_counter[mapped_name] += 1
                has_target_class = True

        if has_target_class:
            files_with_target_class += 1

    return {
        "num_files": len(geojson_paths),
        "files_with_target_class": files_with_target_class,
        "raw_counter": raw_counter,
        "mapped_counter": mapped_counter,
        "raw_classes_per_file": raw_classes_per_file,
    }


def print_summary(split_name: str, stats: dict):
    print(f"\n=== Summary for {split_name} ===")
    print(f"Total geojson files: {stats['num_files']}")
    print(f"Files containing Task 2 target classes: {stats['files_with_target_class']}")

    print("\nRaw class distribution:")
    if len(stats["raw_counter"]) == 0:
        print("  (none)")
    else:
        for cls_name, count in stats["raw_counter"].most_common():
            print(f"  {cls_name}: {count}")

    print("\nMapped Task 2 class distribution:")
    if len(stats["mapped_counter"]) == 0:
        print("  (none)")
    else:
        for cls_name, count in stats["mapped_counter"].most_common():
            print(f"  {cls_name}: {count}")


def print_possible_histiocyte_like_classes(train_stats: dict, val_stats: dict):
    all_raw_classes = set(train_stats["raw_counter"].keys()) | set(val_stats["raw_counter"].keys())

    candidates = sorted(
        cls for cls in all_raw_classes
        if cls is not None and ("histi" in cls.lower() or "macro" in cls.lower())
    )

    print("\nPossible histiocyte-like raw class names:")
    if len(candidates) == 0:
        print("  No obvious class names containing 'histi' or 'macro' were found.")
    else:
        for cls in candidates:
            train_count = train_stats["raw_counter"].get(cls, 0)
            val_count = val_stats["raw_counter"].get(cls, 0)
            print(f"  {cls}: train={train_count}, validation={val_count}")


def main():
    train_geojson_dir = PROJECT_ROOT / "data" / "Dataset_Splits" / "train" / "nuclei"
    val_geojson_dir = PROJECT_ROOT / "data" / "Dataset_Splits" / "validation" / "nuclei"

    train_stats = scan_split("train", train_geojson_dir)
    val_stats = scan_split("validation", val_geojson_dir)

    print_summary("train", train_stats)
    print_summary("validation", val_stats)
    print_possible_histiocyte_like_classes(train_stats, val_stats)


if __name__ == "__main__":
    main()