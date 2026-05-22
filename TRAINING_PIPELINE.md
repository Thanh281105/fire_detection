# YOLO11 Detect Final Training Pipeline

Goal: prioritize the final model reaching single-class `mAP50(B) >= 0.70`.
The final architecture is YOLO11 detection-only with one class: `fire_smoke`.

Dataset source:

- HF public repo: `thanhhoangnvbg/fire-vn-yolo11seg-v1`
- Source file: `fire_vn_yolo11seg_v1.zip`
- Final detect dataset: generated from segmentation polygons, no relabeling.

## Install

```bash
pip install -q -r requirements-training.txt
```

No HF token is required for downloading the public dataset. `HF_TOKEN` is only
needed if uploading dataset/model artifacts to Hugging Face.

## Build Detect Dataset Locally

```bash
python scripts/prepare_yolo11_detect_dataset.py \
  --input datasets/fire_vn_yolo11seg_v1 \
  --output datasets/fire_vn_yolo11det_fire_smoke_v2 \
  --overwrite
```

The converter:

- keeps the same `train/valid/test` split and image files,
- converts YOLO-seg polygons into YOLO-detect boxes,
- merges old `smoke` and `fire` labels into class `0: fire_smoke`,
- keeps empty labels for negative/background images,
- writes `metadata/conversion_summary.csv`.

## Full-Image Fine-Tune Dataset

Use this when the sliced training set plateaus and you want a conservative
fine-tune on original full images only.

Build a cleaned full-image seg dataset from raw exports:

```bash
python scripts/prepare_yolo11_seg_dataset.py \
  --raw-root datasets/raw \
  --output work/fire_vn_yolo11seg_v1_fullimg \
  --skip-slicing \
  --dedupe-near-duplicates \
  --near-duplicate-threshold 2 \
  --make-zip \
  --overwrite
```

Upload that zip to a new HF dataset branch:

```bash
export HF_TOKEN=hf_xxx
python scripts/upload_hf_dataset.py \
  --repo-id thanhhoangnvbg/fire-vn-yolo11seg-v1 \
  --zip-path work/fire_vn_yolo11seg_v1_fullimg.zip \
  --revision full-image-v1 \
  --create-branch
```

Fine-tune from the current best checkpoint on the full-image branch:

```bash
python scripts/train_yolo11_detect.py \
  --profile vertex_detect_fullimage_finetune_l4 \
  --dataset-revision full-image-v1 \
  --dataset-zip fire_vn_yolo11seg_v1_fullimg.zip \
  --seg-dataset-dir work/fire_vn_yolo11seg_v1_fullimg \
  --det-dataset-dir work/fire_vn_yolo11det_fire_smoke_fullimg \
  --init-weights runs/final/yolo11x_detect_fire_smoke_l4_finetune/weights/best.pt \
  --val-test \
  --test-imgsz 1024 \
  --exist-ok
```

If the full-image seg and detect datasets already exist locally, add
`--skip-download --skip-convert`.

## Vertex Final Model

Recommended machine:

- GPU: `1x NVIDIA L4`
- Machine: `g2-standard-32`
- Disk: `250GB`

Run the final detection pipeline:

```bash
bash scripts/run_vertex_detect_final.sh
```

Default final profile:

- `model=yolo11x.pt`
- `imgsz=1024`
- `epochs=180`
- `batch=4`
- `optimizer=AdamW`
- `cache=disk`
- `class=fire_smoke`

The launcher downloads the public seg dataset if needed, converts it to the
1-class detect dataset, trains, validates on `valid`, validates `best.pt` on
`test`, and syncs artifacts to GCS.

## GCS And HF Artifacts

Default GCS path:

```text
gs://fire_detection_final/yolo11x_detect_fire_smoke_l4_final
```

Optional HF model upload:

```bash
export HF_TOKEN=hf_xxx
export HF_MODEL_REPO_ID=thanhhoangnvbg/fire-vn-yolo11x-detect-fire-smoke-l4-final
bash scripts/run_vertex_detect_final.sh
```

Uploaded/synced artifacts include:

```text
weights/best.pt
weights/last.pt
results.csv
training_config.json
args.yaml
confusion_matrix.png
confusion_matrix_normalized.png
```

## Sliced Test Evaluation

For the final report, run normal YOLO validation first, then run sliced
inference on the original full-size test images. This does not rewrite the test
set: predictions from overlapping tiles are mapped back to the original image,
merged with NMS, and scored against the original full-image labels.

```bash
python scripts/evaluate_yolo11_detect_sliced.py \
  --model runs/final/yolo11x_detect_fire_smoke_l4_final/weights/best.pt \
  --dataset datasets/fire_vn_yolo11det_fire_smoke_v2 \
  --split test \
  --imgsz 1024 \
  --slice-size 768 \
  --overlap 0.25 \
  --include-full-image \
  --conf 0.001 \
  --merge-iou 0.55 \
  --report-conf 0.25 \
  --output reports/sliced_eval/test_l4_best
```

Useful slower/high-recall variant:

```bash
python scripts/evaluate_yolo11_detect_sliced.py \
  --model runs/final/yolo11x_detect_fire_smoke_l4_final/weights/best.pt \
  --dataset datasets/fire_vn_yolo11det_fire_smoke_v2 \
  --split test \
  --imgsz 1280 \
  --slice-size 768 \
  --overlap 0.35 \
  --tta \
  --output reports/sliced_eval/test_l4_best_1280_tta
```

Outputs:

```text
sliced_eval_metrics.json
sliced_eval_summary.csv
sliced_eval_predictions.csv
sliced_pr_curve_iou50.png
samples/*_sliced_pred.jpg
```

## OOM Or Stability Fallbacks

If L4 OOMs, retry with smaller batch first:

```bash
BATCH=2 bash scripts/run_vertex_detect_final.sh
```

If AMP/EMA produces NaN warnings, retry with AMP disabled:

```bash
AMP=0 bash scripts/run_vertex_detect_final.sh
```

If the seg dataset is already downloaded and the detect dataset is already
converted on Vertex:

```bash
SKIP_DOWNLOAD=1 SKIP_CONVERT=1 bash scripts/run_vertex_detect_final.sh
```

## Legacy Segmentation

The previous YOLO11 segmentation pipeline is kept for comparison only:

```bash
bash scripts/run_vertex_final.sh
```

It is no longer the primary final path because the target metric is bbox
`mAP50(B)` and the 2-class segmentation run plateaued around `0.445`.

## Acceptance Criteria

- Primary: single-class `fire_smoke mAP50(B) >= 0.70` on validation and then
  confirmed on test.
- Secondary: track recall, precision, false positives on groups `03` and `05`,
  and small-object recall on group `04`.
- If epoch 60 is still below `0.55`, stop and audit data/split/labels before
  trying a different architecture.
