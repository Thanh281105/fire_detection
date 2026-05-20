#!/usr/bin/env bash
set -euo pipefail

# Vertex AI final training launcher.
#
# Optional environment variables:
#   PROFILE=vertex_final
#   GCS_DIR=gs://fire_detection_final/yolo11l_seg_fire_vn_l4_final
#   HF_MODEL_REPO_ID=thanhhoangnvbg/fire-vn-yolo11l-seg-l4-final
#   HF_PRIVATE_MODEL=1
#   IMG_SIZE=896
#   BATCH=4
#   EPOCHS=240
#   MODEL=yolo11l-seg.pt
#   INIT_WEIGHTS=/home/jupyter/fire_project/runs/final/yolo11l_seg_fire_vn_l4_final/weights/epoch70.pt
#   AMP=0
#   TEST_IMG_SIZE=896
#   VAL_TTA=0
#   RESTORE_FROM_GCS=1

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

PROFILE="${PROFILE:-vertex_final}"
if [[ -z "${GCS_DIR:-}" ]]; then
  case "${PROFILE}" in
    vertex_finetune_stable)
      GCS_DIR="gs://fire_detection_final/yolo11l_seg_fire_vn_l4_finetune_stable"
      ;;
    vertex_x_l4)
      GCS_DIR="gs://fire_detection_final/yolo11x_seg_fire_vn_l4_benchmark"
      ;;
    *)
      GCS_DIR="gs://fire_detection_final/yolo11l_seg_fire_vn_l4_final"
      ;;
  esac
fi

python -m pip install -q -r requirements-training.txt

echo "Python: $(python --version)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

ARGS=(
  scripts/train_yolo11_seg.py
  --profile "${PROFILE}"
  --device 0
  --val-test
  --exist-ok
)

if [[ -n "${IMG_SIZE:-}" ]]; then
  ARGS+=(--imgsz "${IMG_SIZE}")
fi

if [[ -n "${BATCH:-}" ]]; then
  ARGS+=(--batch "${BATCH}")
fi

if [[ -n "${EPOCHS:-}" ]]; then
  ARGS+=(--epochs "${EPOCHS}")
fi

if [[ -n "${MODEL:-}" ]]; then
  ARGS+=(--model "${MODEL}")
fi

if [[ -n "${INIT_WEIGHTS:-}" ]]; then
  ARGS+=(--init-weights "${INIT_WEIGHTS}")
fi

if [[ -n "${TEST_IMG_SIZE:-}" ]]; then
  ARGS+=(--test-imgsz "${TEST_IMG_SIZE}")
fi

if [[ "${VAL_TTA:-0}" == "1" ]]; then
  ARGS+=(--tta-val)
fi

if [[ "${AMP:-}" == "0" ]]; then
  ARGS+=(--no-amp)
elif [[ "${AMP:-}" == "1" ]]; then
  ARGS+=(--amp)
fi

if [[ -n "${GCS_DIR:-}" ]]; then
  ARGS+=(--gcs-dir "${GCS_DIR}")
fi

if [[ -n "${HF_MODEL_REPO_ID:-}" ]]; then
  ARGS+=(--hf-model-repo-id "${HF_MODEL_REPO_ID}")
fi

if [[ "${HF_PRIVATE_MODEL:-0}" == "1" ]]; then
  ARGS+=(--hf-private-model)
fi

if [[ "${RESTORE_FROM_GCS:-0}" == "1" ]]; then
  ARGS+=(--restore-from-gcs --resume)
fi

python "${ARGS[@]}"
