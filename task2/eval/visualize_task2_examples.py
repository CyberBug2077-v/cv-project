from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from task2.config import TASK2_OUTPUT_DIR, TASK2_RANDOM_SEED


CLASS_NAMES = ["Tumor", "Lymphocyte", "Histiocyte"]
CLASS_PRIORITY_FOR_CORRECT = ["Histiocyte", "Lymphocyte", "Tumor"]
PANEL_STYLES = {
    "input": {
        "title": "Input patch",
        "border_color": "#6B7280",
    },
    "correct": {
        "title": "Correct example",
        "border_color": "#2E8B57",
    },
    "failure": {
        "title": "Failure example",
        "border_color": "#C0392B",
    },
}


def get_default_predictions_csv() -> Path:
    return Path(TASK2_OUTPUT_DIR) / "contrastive_classifier" / "eval" / "test_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create representative Task 2 qualitative-example figures.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default=str(get_default_predictions_csv()),
        help="Path to a test_predictions.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for saved figures and metadata. Defaults to <eval>/qualitative_examples.",
    )
    parser.add_argument(
        "--model-label",
        type=str,
        default="Contrastive classifier",
        help="Short label used in the combined figure title.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Optional explicit patch path for the input example.",
    )
    parser.add_argument(
        "--correct-path",
        type=str,
        default=None,
        help="Optional explicit patch path for the correct example.",
    )
    parser.add_argument(
        "--failure-path",
        type=str,
        default=None,
        help="Optional explicit patch path for the failure example.",
    )
    parser.add_argument(
        "--prefer-correct-class",
        choices=CLASS_NAMES,
        default=None,
        help="Optional class to prioritise for the correct example.",
    )
    parser.add_argument(
        "--prefer-failure-class",
        choices=CLASS_NAMES,
        default=None,
        help="Optional true class to prioritise for the failure example.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI.",
    )
    return parser.parse_args()


def infer_sample_type(text: str) -> str:
    lowered = text.lower()
    if "primary" in lowered:
        return "primary"
    if "metastatic" in lowered:
        return "metastatic"
    return "unknown"


def parse_path_field(raw_value: str) -> Dict[str, str]:
    patch_path = str(raw_value)
    filename = ""
    sample_type = ""

    if patch_path.startswith("{") and patch_path.endswith("}"):
        try:
            parsed = ast.literal_eval(patch_path)
            patch_path = str(parsed.get("path", ""))
            filename = str(parsed.get("filename", ""))
            sample_type = str(parsed.get("sample_type", ""))
        except Exception:
            pass

    if not filename and patch_path:
        filename = Path(patch_path).name
    if not sample_type:
        sample_type = infer_sample_type(patch_path or filename)

    return {
        "patch_path": patch_path,
        "filename": filename,
        "sample_type": sample_type,
    }


def normalize_prediction_row(row: Dict[str, str]) -> Dict[str, object]:
    parsed_path = parse_path_field(row.get("path", ""))
    prob_tumor = float(row.get("prob_tumor", 0.0))
    prob_lymphocyte = float(row.get("prob_lymphocyte", 0.0))
    prob_histiocyte = float(row.get("prob_histiocyte", 0.0))
    probabilities = [prob_tumor, prob_lymphocyte, prob_histiocyte]

    true_label = int(row["true_label"])
    pred_label = int(row["pred_label"])
    confidence = probabilities[pred_label]
    true_probability = probabilities[true_label]
    sorted_probs = sorted(probabilities, reverse=True)
    margin = (
        sorted_probs[0] - sorted_probs[1]
        if len(sorted_probs) >= 2
        else sorted_probs[0]
    )

    return {
        "patch_path": parsed_path["patch_path"],
        "filename": parsed_path["filename"],
        "sample_type": parsed_path["sample_type"],
        "true_label": true_label,
        "true_class": row["true_class"],
        "pred_label": pred_label,
        "pred_class": row["pred_class"],
        "probabilities": probabilities,
        "confidence": float(confidence),
        "true_probability": float(true_probability),
        "margin": float(margin),
        "correct": bool(int(row["correct"])),
    }


def load_predictions(csv_path: Path) -> List[Dict[str, object]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [normalize_prediction_row(row) for row in reader]

    if not rows:
        raise ValueError(f"No prediction rows found in {csv_path}")

    return rows


def read_patch_image(patch_path: str) -> np.ndarray:
    path = Path(patch_path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        image = np.load(path)
    else:
        raise ValueError(
            f"Unsupported patch format {suffix!r}. "
            "This script currently expects .npy patch files."
        )

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    if image.ndim != 3:
        raise ValueError(f"Unsupported patch shape {image.shape} for {path}")

    if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.transpose(image, (1, 2, 0))

    if image.shape[-1] == 4:
        image = image[..., :3]

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 255.0)
            if image.max() <= 1.0:
                image = image * 255.0
            image = image.astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def find_record_by_path(records: List[Dict[str, object]], patch_path: str) -> Dict[str, object]:
    requested = str(Path(patch_path).resolve())
    for record in records:
        current = str(Path(str(record["patch_path"])).resolve())
        if current == requested:
            return record
    raise ValueError(f"Could not find patch in predictions CSV: {patch_path}")


def sort_correct_candidates(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        records,
        key=lambda item: (
            item["confidence"],
            item["true_probability"],
            item["margin"],
        ),
        reverse=True,
    )


def sort_failure_candidates(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        records,
        key=lambda item: (
            item["confidence"],
            item["confidence"] - item["true_probability"],
            item["margin"],
        ),
        reverse=True,
    )


def select_correct_example(
    records: List[Dict[str, object]],
    preferred_class: Optional[str],
) -> Dict[str, object]:
    candidates = [record for record in records if bool(record["correct"])]
    if not candidates:
        raise ValueError("No correct predictions found in the predictions CSV.")

    if preferred_class is not None:
        preferred = [record for record in candidates if record["true_class"] == preferred_class]
        if preferred:
            return sort_correct_candidates(preferred)[0]

    for class_name in CLASS_PRIORITY_FOR_CORRECT:
        preferred = [record for record in candidates if record["true_class"] == class_name]
        if preferred:
            return sort_correct_candidates(preferred)[0]

    return sort_correct_candidates(candidates)[0]


def select_failure_example(
    records: List[Dict[str, object]],
    preferred_class: Optional[str],
) -> Dict[str, object]:
    candidates = [record for record in records if not bool(record["correct"])]
    if not candidates:
        raise ValueError("No failure cases found in the predictions CSV.")

    if preferred_class is not None:
        preferred = [record for record in candidates if record["true_class"] == preferred_class]
        if preferred:
            return sort_failure_candidates(preferred)[0]

    return sort_failure_candidates(candidates)[0]


def select_input_example(
    records: List[Dict[str, object]],
    used_patch_paths: List[str],
    avoid_class: Optional[str],
) -> Dict[str, object]:
    candidates = [
        record for record in records
        if str(record["patch_path"]) not in used_patch_paths
    ]
    if not candidates:
        candidates = records

    preferred = [record for record in candidates if bool(record["correct"])]
    if preferred:
        candidates = preferred

    if avoid_class is not None:
        different_class = [
            record for record in candidates
            if record["true_class"] != avoid_class
        ]
        if different_class:
            candidates = different_class

    return sort_correct_candidates(candidates)[0]


def build_panel_footer(panel_kind: str, record: Dict[str, object]) -> str:
    if panel_kind == "input":
        return (
            f"GT: {record['true_class']} | sample: {record['sample_type']}"
        )

    return (
        f"GT: {record['true_class']} | Pred: {record['pred_class']} | "
        f"p={float(record['confidence']):.3f}"
    )


def draw_panel(
    ax,
    image: np.ndarray,
    panel_kind: str,
    record: Dict[str, object],
) -> None:
    panel_style = PANEL_STYLES[panel_kind]

    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(panel_style["title"], fontsize=11, pad=8)
    footer = build_panel_footer(panel_kind, record)
    ax.text(
        0.5,
        -0.10,
        footer,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "edgecolor": panel_style["border_color"],
            "boxstyle": "round,pad=0.28",
            "alpha": 0.92,
        },
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.8)
        spine.set_edgecolor(panel_style["border_color"])


def save_single_panel(
    output_path: Path,
    image: np.ndarray,
    panel_kind: str,
    record: Dict[str, object],
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.8), dpi=dpi)
    draw_panel(ax, image, panel_kind, record)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_combined_figure(
    output_path: Path,
    images: Dict[str, np.ndarray],
    records: Dict[str, Dict[str, object]],
    model_label: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8), dpi=dpi)
    panel_order = ["input", "correct", "failure"]

    for ax, panel_kind in zip(axes, panel_order):
        draw_panel(ax, images[panel_kind], panel_kind, records[panel_kind])

    fig.suptitle(
        f"Task 2 qualitative examples | {model_label}",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def serialize_record(record: Dict[str, object]) -> Dict[str, object]:
    serializable = dict(record)
    serializable["probabilities"] = [float(value) for value in record["probabilities"]]
    serializable["confidence"] = float(record["confidence"])
    serializable["true_probability"] = float(record["true_probability"])
    serializable["margin"] = float(record["margin"])
    serializable["correct"] = bool(record["correct"])
    return serializable


def save_selection_summary(
    output_path: Path,
    predictions_csv: Path,
    model_label: str,
    records: Dict[str, Dict[str, object]],
    figure_path: Path,
    panel_paths: Dict[str, Path],
) -> None:
    summary = {
        "predictions_csv": str(predictions_csv),
        "model_label": model_label,
        "combined_figure": str(figure_path),
        "panel_files": {key: str(value) for key, value in panel_paths.items()},
        "selected_examples": {
            key: serialize_record(record)
            for key, record in records.items()
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()

    predictions_csv = Path(args.predictions_csv)
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else predictions_csv.parent / "qualitative_examples"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_predictions(predictions_csv)

    if args.correct_path is not None:
        correct_record = find_record_by_path(records, args.correct_path)
    else:
        correct_record = select_correct_example(
            records=records,
            preferred_class=args.prefer_correct_class,
        )

    if args.failure_path is not None:
        failure_record = find_record_by_path(records, args.failure_path)
    else:
        failure_record = select_failure_example(
            records=records,
            preferred_class=args.prefer_failure_class,
        )

    if args.input_path is not None:
        input_record = find_record_by_path(records, args.input_path)
    else:
        input_record = select_input_example(
            records=records,
            used_patch_paths=[
                str(correct_record["patch_path"]),
                str(failure_record["patch_path"]),
            ],
            avoid_class=str(correct_record["true_class"]),
        )

    selected_records = {
        "input": input_record,
        "correct": correct_record,
        "failure": failure_record,
    }
    selected_images = {
        key: read_patch_image(str(record["patch_path"]))
        for key, record in selected_records.items()
    }

    panel_paths = {
        "input": output_dir / "input_patch.png",
        "correct": output_dir / "correct_example.png",
        "failure": output_dir / "failure_example.png",
    }
    combined_figure_path = output_dir / "task2_examples.png"
    summary_path = output_dir / "task2_examples.json"

    for panel_kind, output_path in panel_paths.items():
        save_single_panel(
            output_path=output_path,
            image=selected_images[panel_kind],
            panel_kind=panel_kind,
            record=selected_records[panel_kind],
            dpi=args.dpi,
        )

    save_combined_figure(
        output_path=combined_figure_path,
        images=selected_images,
        records=selected_records,
        model_label=args.model_label,
        dpi=args.dpi,
    )
    save_selection_summary(
        output_path=summary_path,
        predictions_csv=predictions_csv,
        model_label=args.model_label,
        records=selected_records,
        figure_path=combined_figure_path,
        panel_paths=panel_paths,
    )

    print("Saved Task 2 qualitative-example figures")
    print(f"Predictions CSV : {predictions_csv}")
    print(f"Output dir      : {output_dir}")
    print(f"Combined figure : {combined_figure_path}")
    print(f"Input patch     : {panel_paths['input']}")
    print(f"Correct example : {panel_paths['correct']}")
    print(f"Failure example : {panel_paths['failure']}")
    print(f"Summary JSON    : {summary_path}")
    for panel_kind, record in selected_records.items():
        print(
            f"{panel_kind:>7} | "
            f"path={record['patch_path']} | "
            f"GT={record['true_class']} | "
            f"Pred={record['pred_class']} | "
            f"conf={float(record['confidence']):.3f}"
        )


if __name__ == "__main__":
    main()
