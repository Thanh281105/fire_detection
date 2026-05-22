#!/usr/bin/env bash
set -euo pipefail

# Vertex AI final detection training launcher.
#
# Optional environment variables:
#   GCS_DIR=gs://fire_detection_final/yolo11x_detect_fire_smoke_l4_final
#   HF_MODEL_REPO_ID=thanhhoangnvbg/fire-detection-yolo11-stage12
#   HF_PRIVATE_MODEL=1
#   IMG_SIZE=1024
#   BATCH=4
#   EPOCHS=180
#   MODEL=yolo11x.pt
#   AMP=0
#   RUN_FINETUNE=1
#   FINETUNE_GCS_DIR=gs://fire_detection_final/yolo11x_detect_fire_smoke_l4_finetune
#   PROFILE=vertex_detect_final
#   FINETUNE_PROFILE=vertex_detect_finetune_l4
#   FINETUNE_EPOCHS=60
#   TEST_IMG_SIZE=1024
#   VAL_TTA=0
#   SKIP_DOWNLOAD=1
#   SKIP_CONVERT=1
#   RESTORE_FROM_GCS=1
#   FINETUNE_RESTORE_FROM_GCS=1

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

PROFILE="${PROFILE:-vertex_detect_final}"
if [[ -z "${FINETUNE_PROFILE:-}" ]]; then
  FINETUNE_PROFILE="vertex_detect_finetune_l4"
fi

if [[ -z "${GCS_DIR:-}" ]]; then
  GCS_DIR="gs://fire_detection_final/yolo11x_detect_fire_smoke_l4_final"
fi

if [[ -z "${FINETUNE_GCS_DIR:-}" ]]; then
  FINETUNE_GCS_DIR="gs://fire_detection_final/yolo11x_detect_fire_smoke_l4_finetune"
fi
RUN_FINETUNE="${RUN_FINETUNE:-1}"

python -m pip install -q -r requirements-training.txt

echo "Python: $(python --version)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

ARGS=(
  scripts/train_yolo11_detect.py
  --profile "${PROFILE}"
  --device 0
  --val-test
  --exist-ok
  --gcs-dir "${GCS_DIR}"
)

if [[ -n "${TEST_IMG_SIZE:-}" ]]; then
  ARGS+=(--test-imgsz "${TEST_IMG_SIZE}")
fi

if [[ "${VAL_TTA:-0}" == "1" ]]; then
  ARGS+=(--tta-val)
fi

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

if [[ "${AMP:-}" == "0" ]]; then
  ARGS+=(--no-amp)
elif [[ "${AMP:-}" == "1" ]]; then
  ARGS+=(--amp)
fi

if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
  ARGS+=(--skip-download)
fi

if [[ "${SKIP_CONVERT:-0}" == "1" ]]; then
  ARGS+=(--skip-convert)
fi

if [[ "${RESTORE_FROM_GCS:-0}" == "1" ]]; then
  ARGS+=(--restore-from-gcs --resume)
fi

if [[ "${RUN_FINETUNE}" != "1" && -n "${HF_MODEL_REPO_ID:-}" ]]; then
  ARGS+=(--hf-model-repo-id "${HF_MODEL_REPO_ID}")
fi

if [[ "${RUN_FINETUNE}" != "1" && "${HF_PRIVATE_MODEL:-0}" == "1" ]]; then
  ARGS+=(--hf-private-model)
fi

python "${ARGS[@]}"

if [[ "${RUN_FINETUNE}" == "1" ]]; then
  ARGS=(
    scripts/train_yolo11_detect.py
    --profile "${FINETUNE_PROFILE}"
    --device 0
    --val-test
    --exist-ok
    --skip-download
    --skip-convert
    --gcs-dir "${FINETUNE_GCS_DIR}"
  )

  if [[ -n "${FINETUNE_INIT_WEIGHTS:-}" ]]; then
    ARGS+=(--init-weights "${FINETUNE_INIT_WEIGHTS}")
  elif [[ -z "${FINETUNE_MODEL:-}" ]]; then
    ARGS+=(--init-weights "runs/final/yolo11x_detect_fire_smoke_l4_final/weights/best.pt")
  fi

  if [[ -n "${TEST_IMG_SIZE:-}" ]]; then
    ARGS+=(--test-imgsz "${TEST_IMG_SIZE}")
  fi

  if [[ "${VAL_TTA:-0}" == "1" ]]; then
    ARGS+=(--tta-val)
  fi

  if [[ -n "${FINETUNE_IMG_SIZE:-}" ]]; then
    ARGS+=(--imgsz "${FINETUNE_IMG_SIZE}")
  fi

  if [[ -n "${FINETUNE_BATCH:-}" ]]; then
    ARGS+=(--batch "${FINETUNE_BATCH}")
  fi

  if [[ -n "${FINETUNE_EPOCHS:-}" ]]; then
    ARGS+=(--epochs "${FINETUNE_EPOCHS}")
  fi

  if [[ -n "${FINETUNE_MODEL:-}" ]]; then
    ARGS+=(--model "${FINETUNE_MODEL}")
  fi

  if [[ "${FINETUNE_AMP:-${AMP:-1}}" == "0" ]]; then
    ARGS+=(--no-amp)
  elif [[ "${FINETUNE_AMP:-${AMP:-1}}" == "1" ]]; then
    ARGS+=(--amp)
  fi

  if [[ "${FINETUNE_RESTORE_FROM_GCS:-0}" == "1" ]]; then
    ARGS+=(--restore-from-gcs --resume)
  fi

  if [[ -n "${HF_MODEL_REPO_ID:-}" ]]; then
    ARGS+=(--hf-model-repo-id "${HF_MODEL_REPO_ID}")
  fi

  if [[ "${HF_PRIVATE_MODEL:-0}" == "1" ]]; then
    ARGS+=(--hf-private-model)
  fi

  python "${ARGS[@]}"
fi
