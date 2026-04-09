# Task 2

Task 2 is a 3-class nuclei classification task with the following labels:

- `Tumor`
- `Lymphocyte`
- `Histiocyte`

The current codebase contains two model families:

- A supervised `ResNet-18` baseline trained directly on extracted nucleus patches
- A contrastive pipeline with supervised contrastive pretraining, followed by a staged downstream classifier (`linear probe -> last-block finetune -> full finetune`)

All commands below assume you run them from the repository root:

```powershell
cd i:\cv\project
```

This matters because several paths in `task2/config.py` are resolved relative to the repository root, and the contrastive CSV / checkpoint defaults are stored as repo-relative strings.

## End-to-End Workflow

The complete Task 2 workflow is:

1. Install dependencies and review `task2/config.py`
2. Optionally inspect the raw class distribution
3. Optionally sanity-check patch extraction on a known training example
4. Generate nucleus patches plus CSV manifests
5. Train and evaluate the supervised baseline
6. Pretrain the contrastive encoder
7. Train the downstream contrastive classifier
8. Evaluate the contrastive classifier on the held-out test set
9. Optionally visualize the latent space
10. Optionally generate qualitative example figures

If you only want to reproduce evaluation or figures from existing checkpoints, you can skip the earlier training steps as long as the required checkpoints and prediction CSV files already exist.

## Environment Setup

Install the Python dependencies for `task2`:

```powershell
pip install -r task2/requirements.txt
```

Notes:

- `torch` and `torchvision` are listed in `requirements.txt`, but on Windows you may still want to install matching CPU/CUDA wheels from the official PyTorch installer page.
- If you do not have a CUDA-capable environment, change `TASK2_DEVICE = "cpu"` in `task2/config.py`.
- Both the baseline and contrastive models use `pretrained=True` during training, so the first run may download ImageNet pretrained `ResNet-18` weights if they are not already cached.
- `shapely` is used when available for polygon centroid computation during extraction. The extraction script has a fallback centroid estimate, but the recommended setup is still to install `shapely`.

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

Path configuration lives in `task2/config.py`.

Important assumptions:

- Training and validation source images are `.tif` files.
- Training and validation nuclei annotations are `.geojson` polygon files.
- Task 2 classes are mapped from the following raw GeoJSON labels:
  - `nuclei_tumor`
  - `nuclei_lymphocyte`
  - `nuclei_histiocyte`
- The test set is scanned recursively under `data/Task2_Test_Set/`.
- The test loader accepts `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, and `.npy`.
- Test labels are inferred from the filename, so each test filename must contain exactly one of:
  - `nuclei_tumor`
  - `nuclei_lymphocyte`
  - `nuclei_histiocyte`
- Test `sample_type` is also inferred from the filename using `primary` or `metastatic`. If neither substring is present, the sample type becomes `unknown`.

## Dataset Construction

Task 2 is a nuclei-level classification problem rather than a whole-image classification problem.

Each supervised training example corresponds to one annotated nucleus:

- the source image is a histopathology patch stored as `.tif`
- the annotation is a polygon stored in `.geojson`
- `task2/data/extract.py` computes the polygon centroid and crops a `100 x 100` RGB patch around it
- if the crop crosses an image boundary, the missing area is zero-padded

Each sample carries two different attributes:

- `label`: one of `Tumor`, `Lymphocyte`, `Histiocyte`
- `sample_type`: inferred from the source filename as `primary`, `metastatic`, or `unknown`

### Raw Source Distribution

Before any balancing or subsampling, the annotated nuclei distribution is:

| Split | GeoJSON files | Total nuclei | Tumor | Lymphocyte | Histiocyte | Primary | Metastatic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 163 | 67,161 | 43,710 | 17,630 | 5,821 | 32,638 | 34,523 |
| validation | 20 | 9,023 | 5,787 | 2,349 | 887 | 4,736 | 4,287 |

This raw dataset is strongly imbalanced, especially at the class level:

- `Tumor` is the dominant class
- `Histiocyte` is the rarest class by a large margin
- the `primary` / `metastatic` mix is also uneven inside each class

### Generated Splits

The code does not train directly on all raw nuclei. It first materializes patch files plus CSV metadata under `task2/data/generated/`.

With the current settings in `task2/config.py`, the generated splits are:

| Artifact | Source pool | Sampling rule | Total | Tumor | Lymphocyte | Histiocyte | Primary | Metastatic |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `train.csv` | train | class-balanced, `2500` per class | 7,500 | 2,500 | 2,500 | 2,500 | 4,665 | 2,835 |
| `val.csv` | validation | class-balanced, `700` per class | 2,100 | 700 | 700 | 700 | 1,351 | 749 |
| `contrastive.csv` | train | all remaining train nuclei after excluding `train.csv` samples | 59,661 | 41,210 | 15,130 | 3,321 | 27,973 | 31,688 |

Important details:

- The supervised train and validation splits are balanced by class, but not by `sample_type`.
- The contrastive split is much larger and remains close to the original train distribution.
- Because `TASK2_CONTRASTIVE_EXCLUDE_SUPERVISED_TRAIN = True`, the contrastive pretraining split does not reuse nuclei already selected into `train.csv`.
- With the current setup, the three CSV manifests reference `69,261` patches in total.

### Test Set

The held-out test set under `data/Task2_Test_Set/` is loaded directly from files rather than from a CSV manifest.

Current test counts inferred from filenames are:

| Split | Total | Tumor | Lymphocyte | Histiocyte | Primary | Metastatic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 1,858 | 700 | 700 | 458 | 1,138 | 720 |

### Generated CSV Schema

Each row in the generated CSVs stores both the training target and traceability metadata:

- `patch_path`
- `label`
- `class_name`
- `sample_type`
- `raw_class_name`
- `feature_id`
- `source_image_name`
- `source_image_path`
- `source_geojson_path`
- `center_x`
- `center_y`
- `split`

This metadata is used by more than just training:

- weighted sampling can rebalance by both `label` and `sample_type`
- evaluation can report metrics by `sample_type`
- latent-space plots can encode class and sample type separately
- predictions can be traced back to the original image and annotation

## Main Configuration Knobs

Most task-level settings live in `task2/config.py`.

The most useful ones to review before running are:

- Data and patch extraction:
  - `TASK2_PATCH_SIZE`
  - `TASK2_TRAIN_IMAGE_DIR`
  - `TASK2_TRAIN_NUCLEI_DIR`
  - `TASK2_VAL_IMAGE_DIR`
  - `TASK2_VAL_NUCLEI_DIR`
  - `TASK2_TEST_DIR`
- Generated split sizes:
  - `TASK2_TRAIN_SAMPLES_PER_CLASS`
  - `TASK2_VAL_SAMPLES_PER_CLASS`
  - `TASK2_CONTRASTIVE_SAMPLING_MODE`
  - `TASK2_CONTRASTIVE_SAMPLES_PER_CLASS`
  - `TASK2_CONTRASTIVE_EXCLUDE_SUPERVISED_TRAIN`
- Baseline training:
  - `TASK2_BATCH_SIZE`
  - `TASK2_NUM_EPOCHS`
  - `TASK2_LEARNING_RATE`
  - `TASK2_WEIGHT_DECAY`
- Contrastive pretraining:
  - `TASK2_CONTRASTIVE_BATCH_SIZE`
  - `TASK2_CONTRASTIVE_NUM_EPOCHS`
  - `TASK2_CONTRASTIVE_LEARNING_RATE`
  - `TASK2_CONTRASTIVE_WEIGHT_DECAY`
  - `TASK2_CONTRASTIVE_TEMPERATURE`
  - `TASK2_CONTRASTIVE_SELECTION_METRIC`
  - `TASK2_CONTRASTIVE_SELECTION_METRIC_TOLERANCE`
- Downstream contrastive classifier:
  - `TASK2_FROZEN_HEAD_NUM_EPOCHS`
  - `TASK2_FINETUNE_NUM_EPOCHS`
  - `TASK2_FULL_FINETUNE_NUM_EPOCHS`
  - `TASK2_USE_WEIGHTED_RANDOM_SAMPLER`
  - `TASK2_WEIGHTED_SAMPLER_MODE`
  - `TASK2_EARLY_STOPPING_PATIENCE`
- Device, outputs, and evaluation:
  - `TASK2_DEVICE`
  - `TASK2_OUTPUT_DIR`
  - `TASK2_USE_TEST_TIME_AUGMENTATION`
  - `TASK2_TEST_TIME_AUGMENTATION_VIEWS`
  - `TASK2_CONTRASTIVE_ENCODER_CHECKPOINT`
  - `TASK2_CONTRASTIVE_CLASSIFIER_CHECKPOINT`

## Recommended Workflow

### 0. Review config first

Before running anything, check:

```powershell
code task2/config.py
```

At minimum, confirm:

- data paths point to the right directories
- `TASK2_DEVICE` matches your hardware
- split sizes and epoch counts are appropriate for your experiment

### 1. Optional: inspect raw class distribution

This scans the train and validation GeoJSON files and prints raw as well as mapped Task 2 class counts.

```powershell
python task2/scripts/scan_class_distribution.py
```

Use this when you want to verify the raw annotation pool before generating patches.

### 2. Optional: sanity-check patch extraction

This visualizes several extracted nuclei patches from one fixed training example:

```powershell
python task2/scripts/debug_extract.py
```

Notes:

- This is a local interactive sanity-check script intended for manual inspection.
- It currently uses a hard-coded example image / GeoJSON pair from the training split.
- It calls `plt.show()`, so if you are running in a headless environment you may want to skip it or adapt it to save figures instead.

### 3. Generate patches and CSV manifests

This step is required before any training script can run.

```powershell
python task2/data/extract.py
```

What it does:

- reads training and validation `.geojson` annotations
- extracts `100 x 100` nucleus-centered patches from the source `.tif` images
- saves patches as `.npy`
- writes `train.csv`, `val.csv`, and `contrastive.csv`

Generated outputs:

- `task2/data/generated/train.csv`
- `task2/data/generated/val.csv`
- `task2/data/generated/contrastive.csv`
- `task2/data/generated/patches/train/...`
- `task2/data/generated/patches/val/...`
- `task2/data/generated/patches/contrastive/...`

Important re-run note:

- `extract.py` overwrites the CSV manifests, but it does not delete old patch files already present in `task2/data/generated/patches/`.
- If you change sampling settings and want a truly clean regeneration, manually remove the old generated patch directory and CSV files before re-running extraction.
- In other words, the number of `.npy` files on disk may be larger than the number of rows currently referenced by the CSV manifests.

### 4. Train the supervised baseline

```powershell
python task2/scripts/train_baseline.py
```

What this trains:

- `ResNet-18` backbone with ImageNet pretrained initialization
- full supervised training on `train.csv`
- model selection by validation macro F1

Main artifacts:

- `task2/outputs/baseline/checkpoints/best.pt`
- `task2/outputs/baseline/logs/history.csv`
- `task2/outputs/baseline/logs/summary.json`
- `task2/outputs/baseline/logs/best_val_metrics.json`

### 5. Evaluate the supervised baseline on the test set

```powershell
python task2/eval/test_baseline.py
```

This script:

- loads `task2/outputs/baseline/checkpoints/best.pt`
- evaluates on `data/Task2_Test_Set/`
- applies test-time augmentation if enabled in `task2/config.py`

Main artifacts:

- `task2/outputs/baseline/eval/test_metrics.json`
- `task2/outputs/baseline/eval/test_confusion_matrix.npy`
- `task2/outputs/baseline/eval/test_predictions.csv`

### 6. Train the contrastive encoder

```powershell
python task2/scripts/train_contrastive.py
```

What this stage actually does:

- builds a `ResNet-18` encoder plus MLP projection head
- trains on `contrastive.csv`
- uses a supervised contrastive objective, not an unsupervised SimCLR-style loss
- evaluates the encoder after each epoch using a downstream logistic-regression probe on `train.csv` and `val.csv`
- selects the best checkpoint using downstream validation performance, with a small tolerance that prefers lower contrastive training loss among near-tied candidates

Main artifacts:

- `task2/outputs/contrastive/checkpoints/best.pt`
- `task2/outputs/contrastive/checkpoints/last.pt`
- `task2/outputs/contrastive/logs/history.csv`
- `task2/outputs/contrastive/logs/summary.json`
- `task2/outputs/contrastive/logs/best_downstream_probe_metrics.json`

### 7. Train the downstream contrastive classifier

```powershell
python task2/scripts/train_contrastive_classifier.py
```

Prerequisite:

- `task2/outputs/contrastive/checkpoints/best.pt` must already exist, unless you changed `TASK2_CONTRASTIVE_ENCODER_CHECKPOINT` to point elsewhere.

This is no longer just a single frozen-head training stage. The script runs three possible stages:

1. `linear_probe`
2. `last_block_finetune`
3. `full_finetune`

Each stage can be enabled or disabled by setting its epoch count in `task2/config.py`.

Other important behavior:

- the projection head stays frozen in all downstream stages
- model selection still uses validation macro F1
- early stopping is applied per stage
- by default the train loader uses a `WeightedRandomSampler` with `TASK2_WEIGHTED_SAMPLER_MODE = "label_and_sample_type"` to compensate for imbalance inside the class-balanced training CSV

Main artifacts:

- `task2/outputs/contrastive_classifier/checkpoints/best.pt`
- `task2/outputs/contrastive_classifier/checkpoints/last.pt`
- `task2/outputs/contrastive_classifier/logs/history.csv`
- `task2/outputs/contrastive_classifier/logs/summary.json`
- `task2/outputs/contrastive_classifier/logs/best_val_metrics.json`

### 8. Evaluate the contrastive classifier on the test set

```powershell
python task2/eval/test_contrastive_classifier.py
```

This script:

- loads `task2/outputs/contrastive_classifier/checkpoints/best.pt`
- evaluates on the test set
- applies the configured TTA views
- reports both overall metrics and metrics grouped by `sample_type`

Main artifacts:

- `task2/outputs/contrastive_classifier/eval/test_metrics.json`
- `task2/outputs/contrastive_classifier/eval/test_metrics_by_sample_type.json`
- `task2/outputs/contrastive_classifier/eval/test_confusion_matrix.npy`
- `task2/outputs/contrastive_classifier/eval/test_predictions.csv`

### 9. Visualize the latent space in 2D

This script can visualize either:

- the raw contrastive encoder checkpoint
- the downstream contrastive classifier checkpoint

It can also visualize either:

- encoder `features`
- projection-head `projections`

Example: visualize test-set encoder features from the downstream classifier with t-SNE:

```powershell
python task2/eval/visualize_contrastive_latent_space.py --split test --checkpoint-type contrastive_classifier --representation features --method tsne
```

Example: visualize train-set projection embeddings from the pure contrastive checkpoint with PCA:

```powershell
python task2/eval/visualize_contrastive_latent_space.py --split train --checkpoint-type contrastive --representation projections --method pca --max-samples 3000
```

Useful options:

- `--split {train,val,test}`
- `--checkpoint-type {contrastive_classifier,contrastive}`
- `--representation {features,projections}`
- `--method {tsne,pca}`
- `--max-samples N`
- `--perplexity X`
- `--output-name custom_name`

Outputs are saved under:

- `task2/outputs/contrastive/eval/latent_space/`
- `task2/outputs/contrastive_classifier/eval/latent_space/`

Each run writes:

- a `.png` scatter plot
- a `.csv` of 2D coordinates plus metadata
- a `.json` summary

### 10. Generate qualitative example figures

This script creates one combined figure and three single-panel figures:

- an input patch
- a correct prediction example
- a failure example

By default it reads the contrastive classifier predictions CSV produced in Step 8:

```powershell
python task2/eval/visualize_task2_examples.py
```

You can also point it to another predictions CSV:

```powershell
python task2/eval/visualize_task2_examples.py --predictions-csv task2/outputs/contrastive_classifier/eval/test_predictions.csv
```

Notes:

- It expects the patch paths in the predictions CSV to point to `.npy` patch files.
- It automatically selects representative examples unless you provide explicit `--input-path`, `--correct-path`, or `--failure-path`.
- You can bias the chosen examples with `--prefer-correct-class` and `--prefer-failure-class`.

Outputs are saved under:

- `task2/outputs/contrastive_classifier/eval/qualitative_examples/`

## Output Layout

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
    eval/
      latent_space/
  contrastive_classifier/
    checkpoints/
    logs/
    eval/
      latent_space/
      qualitative_examples/
```

Typical saved artifacts include:

- `best.pt` and `last.pt`
- `history.csv`
- `summary.json`
- `best_val_metrics.json`
- `best_downstream_probe_metrics.json`
- `test_metrics.json`
- `test_metrics_by_sample_type.json`
- `test_predictions.csv`
- `test_confusion_matrix.npy`
- `eval/latent_space/*.png`
- `eval/latent_space/*.csv`
- `eval/latent_space/*.json`
- `eval/qualitative_examples/*.png`
- `eval/qualitative_examples/*.json`

## Current Checked-In Results

The repository already contains example outputs under `task2/outputs/`. With the current checked-in artifacts:

| Pipeline | Best validation metric | Held-out test macro F1 | Notes |
| --- | ---: | ---: | --- |
| Baseline | `0.7477` val macro F1 | `0.7010` | `task2/outputs/baseline/...` |
| Contrastive encoder | `0.7644` downstream val macro F1 | N/A | selection metric from the probe in `task2/outputs/contrastive/...` |
| Contrastive classifier | `0.7737` val macro F1 | `0.7137` | `task2/outputs/contrastive_classifier/...` |

These numbers are useful as a sanity check, but they may change if you regenerate splits or retrain with different settings.

## Re-Running Notes and Common Gotchas

- Run commands from the repository root. Running from `task2/` directly may break repo-relative paths.
- You must run `task2/data/extract.py` before any training script if the generated CSV files do not exist yet.
- `task2/eval/test_baseline.py` requires `task2/outputs/baseline/checkpoints/best.pt`.
- `task2/scripts/train_contrastive_classifier.py` requires a contrastive encoder checkpoint first.
- `task2/eval/test_contrastive_classifier.py` requires `task2/outputs/contrastive_classifier/checkpoints/best.pt`.
- `task2/eval/visualize_task2_examples.py` expects a predictions CSV and patch paths that still exist on disk.
- Test labels and sample types come from filenames, not from a manifest file. If filenames do not follow the expected pattern, evaluation will fail.
- Re-running extraction updates the CSV manifests but does not automatically clean old generated patch files.
- The current default TTA configuration is:
  - `identity`
  - `hflip`
  - `vflip`

## Main Files

- `task2/config.py`: task-level configuration
- `task2/data/extract.py`: patch extraction and CSV generation
- `task2/data/dataset.py`: dataset definitions and image loading
- `task2/models/baseline.py`: supervised `ResNet-18` baseline
- `task2/models/contrastive_model.py`: contrastive encoder + projection head
- `task2/scripts/scan_class_distribution.py`: raw annotation statistics
- `task2/scripts/debug_extract.py`: visual patch extraction sanity check
- `task2/scripts/train_baseline.py`: baseline training
- `task2/scripts/train_contrastive.py`: contrastive pretraining
- `task2/scripts/train_contrastive_classifier.py`: staged downstream classifier training
- `task2/eval/test_baseline.py`: baseline evaluation
- `task2/eval/test_contrastive_classifier.py`: contrastive classifier evaluation
- `task2/eval/visualize_contrastive_latent_space.py`: 2D latent-space visualization
- `task2/eval/visualize_task2_examples.py`: qualitative example figure generation
