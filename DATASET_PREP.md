# Fire/Smoke YOLO11-Seg Dataset Prep

This repo now has a reusable prep script for the 5-group Roboflow COCO-seg
exports. It ignores any old Roboflow train/valid/test split, creates a balanced
new split, slices only the training set, and outputs YOLO segmentation format.

## Expected Input

Put the five exports under `datasets/raw`:

```text
datasets/raw/
  01_positive_standard.zip
  02_Alley_Context.zip
  03_Negative_Hard_Samples.zip
  04_SAHI_Small_Objects.zip
  05_Ambient_Context_Null.zip
```

You can also use already-extracted folders. Each group can be either flat:

```text
01_positive_standard/
  _annotations.coco.json
  image_001.jpg
```

or already split by Roboflow:

```text
01_positive_standard/
  train/_annotations.coco.json
  valid/_annotations.coco.json
  test/_annotations.coco.json
```

The script pools each group first, then creates a new deterministic split.

On Windows, prefer leaving Roboflow downloads as `.zip` files. The script reads
the zip and re-extracts images with short filenames, which avoids Explorer's
`Path too long` extraction error.

## Install Dependencies

On Kaggle, Colab, or Vertex:

```bash
pip install -q pillow huggingface_hub
```

## Build Dataset

```bash
python scripts/prepare_yolo11_seg_dataset.py \
  --raw-root datasets/raw \
  --output datasets/fire_vn_yolo11seg_v1 \
  --make-zip \
  --overwrite
```

Output:

```text
datasets/fire_vn_yolo11seg_v1/
  data.yaml
  images/train
  images/valid
  images/test
  labels/train
  labels/valid
  labels/test
  coco/train
  coco/valid
  coco/test
  metadata/source_manifest.csv
  metadata/split_summary.csv
```

Use `data.yaml` directly with Ultralytics YOLO11 segmentation training.

## Build 1-Class Detection Dataset

The current final model uses detection-only labels with one class:
`fire_smoke`. Build it from the prepared segmentation dataset without relabeling:

```bash
python scripts/prepare_yolo11_detect_dataset.py \
  --input datasets/fire_vn_yolo11seg_v1 \
  --output datasets/fire_vn_yolo11det_fire_smoke_v2 \
  --overwrite
```

This keeps the same split and image files, converts polygons into bounding
boxes, and merges old `smoke`/`fire` classes into class `0`.

## Split and Slicing Rules

- `01_positive_standard`: split near `87/7/6`, no slicing.
- `05_Ambient_Context_Null`: split near `87/7/6`, no slicing.
- `02_Alley_Context`: split near `80/10/10`, train keeps originals plus `640` SAHI crops.
- `03_Negative_Hard_Samples`: split near `100/20/20` for 140 images, train keeps originals plus capped negative crops.
- `04_SAHI_Small_Objects`: split near `70/20/20` for 110 images, train keeps originals plus `512` SAHI crops with annotations.

Valid/test stay as original images for fair mAP and false-positive evaluation.

## Train

See `TRAINING_PIPELINE.md`. The preferred flow is:

- Kaggle: run `kaggle_s` and optionally `kaggle_m` only as baselines.
- Vertex: run `vertex_final` as the accuracy-first final model.

## Upload Prepared Dataset to Hugging Face

Use a dataset repo so Kaggle and Vertex can download the same prepared version
without manually uploading again.

```bash
python scripts/upload_hf_dataset.py \
  --repo-id YOUR_USERNAME/fire-vn-yolo11seg-v1 \
  --zip-path datasets/fire_vn_yolo11seg_v1.zip
```

Only upload again when data or preprocessing changes. Use new output names such
as `fire_vn_yolo11seg_v2` for the next version.

## Checks

After each run, inspect:

```text
datasets/fire_vn_yolo11seg_v1/metadata/split_summary.csv
```

Success criteria:

- Every split has all five groups.
- Group `03` is no longer train-only.
- Valid/test contain original images only.
- No source image appears in more than one split.
- `data.yaml` can be passed directly to YOLO11-seg training.
