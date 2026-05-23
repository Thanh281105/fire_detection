# Fire/Smoke Detection VN

End-to-end Computer Vision pipeline for Vietnam-focused fire/smoke alerting.
The final deployment target is a single YOLO11 detection model with one class:
`fire_smoke`.

The project keeps segmentation masks as source annotations, then converts them
to 1-class detection boxes for the final YOLO11x detector.

## Repository Structure

```text
scripts/
  prepare_yolo11_seg_dataset.py      # build YOLO11-seg dataset from raw COCO-seg exports
  prepare_yolo11_detect_dataset.py   # convert YOLO-seg polygons/boxes to 1-class detect labels
  upload_hf_dataset.py               # upload prepared dataset zip to Hugging Face
  train_yolo11_detect.py             # main YOLO11 detect training/fine-tune entrypoint
  evaluate_yolo11_detect_sliced.py   # sliced inference evaluation on original images
  evaluate_yolo11_detect_groups.py   # per-group evaluation helper
  generate_yolo11_data_report.py     # data audit figures/tables
  plot_yolo11_stage_metrics.py       # Stage 1/2 training plots
  run_vertex_detect_final.sh         # legacy launcher for Stage 1/2 detect flow
  run_vertex_final.sh                # legacy segmentation baseline launcher

reports/
  report.pdf                         # final compiled report
  report.tex                         # final editable LaTeX source
  data_audit_deduped/                # generated data audit artifacts

DATASET_PREP.md                      # dataset build details
TRAINING_PIPELINE.md                 # training commands and profiles
requirements-dataset-prep.txt
requirements-training.txt
```

## Hugging Face Dataset

Dataset repo:

```text
https://huggingface.co/datasets/thanhhoangnvbg/fire-vn-yolo11seg-v1
```

Main prepared artifact:

| Branch | File | Purpose |
|---|---|---|
| `main` | `fire_vn_yolo11seg_v1.zip` | Prepared dataset with train-only slicing |

The prep flow rebuilds the split from raw exports, removes duplicate or
near-duplicate images, slices selected training groups only, and keeps
validation/test images in original full resolution for fair evaluation.

Local `datasets/`, `work/`, and `runs/` folders are intentionally excluded from
the cleaned submission tree because the dataset and checkpoints are stored on
Hugging Face.

## Inference UI

Install inference dependencies:

```bash
pip install -r requirements-inference.txt
```

Download and extract the Stage 1/2 checkpoint artifact:

```bash
hf download thanhhoangnvbg/fire-detection-yolo11-stage12 model_artifacts_stage12_complete.tar.gz --local-dir weights/fire-detection-yolo11-stage12
tar -xzf weights/fire-detection-yolo11-stage12/model_artifacts_stage12_complete.tar.gz -C weights/fire-detection-yolo11-stage12
```

Run the Streamlit app:

```bash
streamlit run app.py
```

The app defaults to the Stage 2 finetune checkpoint:

```text
weights/fire-detection-yolo11-stage12/runs/final/yolo11x_detect_fire_smoke_l4_finetune/weights/best.pt
```

## Training Plan

The current final pipeline has two stages:

| Stage | Init weights | Dataset | Profile | Purpose |
|---|---|---|---|---|
| Stage 1 | `yolo11x.pt` | sliced train, original valid/test | `vertex_detect_final` | main detector training |
| Stage 2 | Stage 1 `best.pt` | same dataset, no mosaic/mixup | `vertex_detect_finetune_l4` | low-LR refinement |

Observed checkpoints so far:

| Stage | Best observed validation mAP50 | Notes |
|---|---:|---|
| Stage 1 | ~0.445 | best around epoch 96 |
| Stage 2 | ~0.454 | best seen around epoch 7 |

Stage 2 plateaued, so the next useful work is error analysis: false negatives,
per-group metrics, sample prediction montages, and inference threshold/NMS
tuning.

## Run Training

Install training dependencies:

```bash
pip install -q -r requirements-training.txt
```

Run Stage 1:

```bash
python scripts/train_yolo11_detect.py \
  --profile vertex_detect_final \
  --val-test
```

Run Stage 2 from Stage 1 best:

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

## Plot Stage Curves

After Stage 1 and Stage 2 produce `results.csv`, generate report figures:

```bash
python scripts/plot_yolo11_stage_metrics.py \
  --stage1 runs/final/yolo11x_detect_fire_smoke_l4_final/results.csv \
  --stage2 runs/final/yolo11x_detect_fire_smoke_l4_finetune/results.csv \
  --output reports/figures/stage1_stage2_training
```

Outputs include per-stage loss/metric plots, a Stage 1 vs Stage 2 mAP
comparison, and a best-metric CSV summary.

## Files To Sync To Vertex/Jupyter

Sync these files together to avoid import/version mismatch:

```text
scripts/prepare_yolo11_detect_dataset.py
scripts/train_yolo11_detect.py
scripts/plot_yolo11_stage_metrics.py
```

Quick sanity checks on Vertex:

```bash
grep -n "def convert_dataset" scripts/prepare_yolo11_detect_dataset.py
python -m py_compile scripts/prepare_yolo11_detect_dataset.py scripts/train_yolo11_detect.py
```

## Evaluation

Primary metric:

```text
single-class fire_smoke mAP50(B)
```

Secondary checks:

- recall and precision at deployment confidence thresholds
- false positives on hard negatives and ambient null images
- small-object recall on group `04`
- sliced inference evaluation on unchanged original test labels

Sliced test evaluation:

```bash
python scripts/evaluate_yolo11_detect_sliced.py \
  --model runs/final/yolo11x_detect_fire_smoke_l4_finetune/weights/best.pt \
  --dataset datasets/fire_vn_yolo11det_fire_smoke_v2 \
  --split test \
  --imgsz 1024 \
  --slice-size 768 \
  --overlap 0.25 \
  --include-full-image \
  --output reports/sliced_eval/stage2_test
```

## Notes

- `prepare_yolo11_detect_dataset.py` must be the converter file and contain
  `def convert_dataset(...)`.
- `train_yolo11_detect.py` imports `convert_dataset` from the converter file.
- Legacy segmentation scripts remain for comparison only; the final model path
  is YOLO11x detect 1-class `fire_smoke`.
