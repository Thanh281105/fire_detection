# YOLO11 Detect Final Training Pipeline

Goal: train a single-class YOLO11 detection model for `fire_smoke`.
The final architecture is YOLO11 detection-only with one class.

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

## Stage 1: Main Vertex Model

Recommended machine:

- GPU: `1x NVIDIA L4`
- Machine: `g2-standard-32`
- Disk: `250GB`

Run the final detection pipeline:

```bash
python scripts/train_yolo11_detect.py \
  --profile vertex_detect_final \
  --val-test
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
`test`, and syncs artifacts to GCS when `--gcs-dir` is set.

## Stage 2: Low-LR Fine-Tune

Run Stage 2 from Stage 1 `best.pt`:

```bash
python scripts/train_yolo11_detect.py \
  --profile vertex_detect_finetune_l4 \
  --init-weights runs/final/yolo11x_detect_fire_smoke_l4_final/weights/best.pt \
  --skip-download \
  --skip-convert \
  --val-test \
  --test-imgsz 1024 \
  --exist-ok
```

Stage 2 starts a new run from the best Stage 1 weights instead of resuming
optimizer state. It uses a lower learning rate and disables mosaic/mixup to
refine boxes on a more realistic image distribution.

## Plot Stage 1/2 Curves

```bash
python scripts/plot_yolo11_stage_metrics.py \
  --stage1 runs/final/yolo11x_detect_fire_smoke_l4_final/results.csv \
  --stage2 runs/final/yolo11x_detect_fire_smoke_l4_finetune/results.csv \
  --output reports/figures/stage1_stage2_training
```

Outputs:

```text
stage1_training_curves.png
stage2_training_curves.png
stage1_stage2_map_comparison.png
stage1_stage2_best_metrics.png
stage1_stage2_best_metrics.csv
```

## GCS And HF Artifacts

Default GCS paths:

```text
gs://fire_detection_final/yolo11x_detect_fire_smoke_l4_final
gs://fire_detection_final/yolo11x_detect_fire_smoke_l4_finetune
```

Optional HF model upload:

```bash
export HF_TOKEN=<paste_token_here>
export HF_MODEL_REPO_ID=thanhhoangnvbg/fire-detection-yolo11-stage12
python scripts/train_yolo11_detect.py --profile vertex_detect_final --hf-model-repo-id "$HF_MODEL_REPO_ID"
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
merged with NMS, and scored against the original labels.

```bash
python scripts/evaluate_yolo11_detect_sliced.py \
  --model runs/final/yolo11x_detect_fire_smoke_l4_finetune/weights/best.pt \
  --dataset datasets/fire_vn_yolo11det_fire_smoke_v2 \
  --split test \
  --imgsz 1024 \
  --slice-size 768 \
  --overlap 0.25 \
  --include-full-image \
  --conf 0.001 \
  --merge-iou 0.55 \
  --report-conf 0.25 \
  --output reports/sliced_eval/stage2_test
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
python scripts/train_yolo11_detect.py --profile vertex_detect_final --batch 2
```

If AMP/EMA produces NaN warnings, retry with AMP disabled:

```bash
python scripts/train_yolo11_detect.py --profile vertex_detect_final --no-amp
```

If the seg dataset is already downloaded and the detect dataset is already
converted on Vertex:

```bash
python scripts/train_yolo11_detect.py \
  --profile vertex_detect_final \
  --skip-download \
  --skip-convert
```

## Legacy Segmentation

The previous YOLO11 segmentation pipeline is kept for comparison only:

```bash
bash scripts/run_vertex_final.sh
```

It is no longer the primary final path because the target metric is bbox
`mAP50(B)`.

## Acceptance Criteria

- Primary: maximize single-class `fire_smoke mAP50(B)` on validation and then
  confirm on test.
- Secondary: track recall, precision, false positives on groups `03` and `05`,
  and small-object recall on group `04`.
- If Stage 2 plateaus near Stage 1, stop training and run error analysis before
  trying more epochs.
