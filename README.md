# Fire/Smoke Detection VN

Pipeline for preparing a Vietnam-focused fire/smoke segmentation dataset and
training YOLO11 models. The current final target is a 1-class detection model:
`fire_smoke mAP50(B) >= 70`, stable training, reproducibility, and easy
fine-tuning.

## Current Structure

```text
scripts/
  prepare_yolo11_seg_dataset.py  # build YOLO11-seg dataset from 5 Roboflow zips
  prepare_yolo11_detect_dataset.py # convert seg polygons to 1-class detect boxes
  upload_hf_dataset.py           # upload prepared dataset zip to Hugging Face
  train_yolo11_seg.py            # Kaggle/Vertex training entrypoint
  train_yolo11_detect.py         # final 1-class detection training entrypoint
  run_vertex_final.sh            # Vertex final training launcher
  run_vertex_detect_final.sh     # Vertex final detection launcher

notebooks/
  vertex_final_training.ipynb     # optional cell-based Vertex launcher

datasets/
  raw/                           # original 5 Roboflow export zips
  fire_vn_yolo11seg_v1/          # prepared ready-to-train dataset
  fire_vn_yolo11det_fire_smoke_v2/ # generated 1-class detect dataset
  fire_vn_yolo11seg_v1.zip       # uploaded HF artifact

DATASET_PREP.md                  # dataset prep details
TRAINING_PIPELINE.md             # training profiles and commands
requirements-dataset-prep.txt
requirements-training.txt
```

## Dataset

Prepared dataset repo:

```text
https://hf.co/datasets/thanhhoangnvbg/fire-vn-yolo11seg-v1
```

The dataset was built from 5 groups:

- `01_positive_standard`
- `02_Alley_Context`
- `03_Negative_Hard_Samples`
- `04_SAHI_Small_Objects`
- `05_Ambient_Context_Null`

The prep script ignores old Roboflow splits, creates a deterministic balanced
split, slices only `train`, and keeps `valid/test` as original images for fair
evaluation.

## Final Detection Training

Install:

```bash
pip install -q -r requirements-training.txt
```

No HF token is required for downloading the public dataset.

Build the 1-class detection dataset locally:

```bash
python scripts/prepare_yolo11_detect_dataset.py --overwrite
```

Vertex final detection model:

```bash
bash scripts/run_vertex_detect_final.sh
```

The final detection profile uses `yolo11x.pt`, `imgsz=1024`, `batch=4`,
`epochs=180`, and a single class: `fire_smoke`. If L4 runs out of memory, retry
with `BATCH=2`.

Legacy segmentation training remains available for comparison:

```bash
bash scripts/run_vertex_final.sh
```

See `TRAINING_PIPELINE.md` for L4 commands, OOM fallbacks, model upload, and
the legacy segmentation notes.

## Evaluation Target

Primary metric:

```text
single-class fire_smoke mAP50(B) >= 0.70
```

Secondary metrics:

- recall and precision for `fire_smoke`
- false positives on hard negatives and ambient null images
- small-object recall on group `04`

## Notes

- Legacy Kaggle/segmentation runs are baselines/benchmarks only.
- Vertex `vertex_detect_final` is the main accuracy-focused L4 run.
- Keep `best.pt`, `last.pt`, `training_config.json`, plots, and result CSVs for
  reproducibility and future fine-tuning.
