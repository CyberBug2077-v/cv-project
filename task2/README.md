# Task 2

Task 2 focuses on 3-class nuclei classification:

- `Tumor`
- `Lymphocyte`
- `Histiocyte`

The current implementation includes two pipelines:

- A supervised `ResNet-18` baseline classifier
- A contrastive pretraining pipeline followed by a frozen-encoder linear classifier

All commands below assume you run them from the repository root:

## Environment Setup

Install the Python dependencies for `task2`:

```powershell
pip install -r task2/requirements.txt
```

Notes:

- `torch` and `torchvision` are listed in the requirements file, but on Windows you may still want to install matching CPU/CUDA wheels from the official PyTorch installer page.
- If you do not have a CUDA-capable environment, set `TASK2_DEVICE = "cpu"` in [config.py](/i:/cv/project/task2/config.py).

## Data Layout

The code expects the following structure under the repository-level `data/` directory:

```text
data/
  Dataset_Splits/
    train/
      image/
        *.tif
      nuclei/
        *.geojson
    validation/
      image/
        *.tif
      nuclei/
        *.geojson
  Task2_Test_Set/
    ...
```

Path configuration lives in [config.py](/i:/cv/project/task2/config.py).

Important assumptions:

- Training and validation images are `.tif` files.
- Nuclei annotations are stored in `.geojson`.
- Task 2 class names are mapped from GeoJSON labels:
  - `nuclei_tumor`
  - `nuclei_lymphocyte`
  - `nuclei_histiocyte`
- The test loader infers labels from filenames, so test filenames must contain one of the class substrings above.
- The test loader also infers sample type from filenames using `primary` or `metastatic`.

## Config

Main settings are defined in [config.py](/i:/cv/project/task2/config.py), including:

- class mapping and patch size
- train/validation sampling counts
- batch size, epochs, learning rate, and weight decay
- device selection
- output directories
- contrastive pretraining and frozen-head classifier settings

If you want to change dataset size, training length, or device, start there.

## Recommended Workflow

### 1. Optional: inspect class distribution

This scans the train/validation GeoJSON annotations and prints raw and mapped class counts.

```powershell
python task2/scripts/scan_class_distribution.py
```

### 2. Optional: visualize extracted patches

This is useful for checking whether the patch extraction logic is centered correctly.

```powershell
python task2/scripts/debug_extract.py
```

### 3. Generate patches and CSV metadata

This script:

- reads the train and validation `.geojson` annotations
- extracts `100 x 100` nuclei patches from the source `.tif` images
- saves patches as `.npy`
- writes `train.csv`, `val.csv`, and `contrastive.csv`

Run:

```powershell
python task2/data/extract.py
```

Generated files are written to:

- `task2/data/generated/train.csv`
- `task2/data/generated/val.csv`
- `task2/data/generated/contrastive.csv`
- `task2/data/generated/patches/...`

### 4. Train the supervised baseline

```powershell
python task2/scripts/train_baseline.py
```

This trains a `ResNet-18` classifier directly on the extracted nuclei patches.

### 5. Evaluate the supervised baseline

```powershell
python task2/eval/test_baseline.py
```

### 6. Train the contrastive encoder

```powershell
python task2/scripts/train_contrastive.py
```

This stage learns a representation encoder using two augmented views of the same nuclei patch.

### 7. Train the frozen-head contrastive classifier

```powershell
python task2/scripts/train_contrastive_classifier.py
```

This loads the best contrastive checkpoint, freezes the encoder, and trains a linear classifier on top.

### 8. Evaluate the contrastive classifier

```powershell
python task2/eval/test_contrastive_classifier.py
```

## Outputs

Training and evaluation artifacts are written under `task2/outputs/`:

```text
task2/outputs/
  baseline/
    checkpoints/
    logs/
    eval/
  contrastive/
    checkpoints/
    logs/
  contrastive_classifier/
    checkpoints/
    logs/
    eval/
```

Typical saved artifacts include:

- `best.pt` and `last.pt` checkpoints
- `history.csv`
- `summary.json`
- `best_val_metrics.json`
- `test_metrics.json`
- `test_predictions.csv`
- `test_confusion_matrix.npy`

## Notes on Re-running

- Re-running `task2/data/extract.py` will regenerate patch metadata and may overwrite previously generated CSV files.
- Training scripts save into fixed output directories under `task2/outputs/`, so re-training will update checkpoint and log files in place.
- If you want to keep multiple experiments, copy or rename the output directory before re-running.

## Main Files

- [config.py](/i:/cv/project/task2/config.py): all task-level configuration
- [extract.py](/i:/cv/project/task2/data/extract.py): patch extraction and CSV generation
- [dataset.py](/i:/cv/project/task2/data/dataset.py): dataset definitions and image loading
- [train_baseline.py](/i:/cv/project/task2/scripts/train_baseline.py): supervised baseline training
- [train_contrastive.py](/i:/cv/project/task2/scripts/train_contrastive.py): contrastive pretraining
- [train_contrastive_classifier.py](/i:/cv/project/task2/scripts/train_contrastive_classifier.py): frozen linear head training
- [test_baseline.py](/i:/cv/project/task2/eval/test_baseline.py): baseline evaluation
- [test_contrastive_classifier.py](/i:/cv/project/task2/eval/test_contrastive_classifier.py): contrastive classifier evaluation
