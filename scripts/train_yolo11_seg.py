#!/usr/bin/env python3
"""Train YOLO11-seg fire/smoke models from the prepared HF dataset zip.

Profiles:
  - kaggle_s: quick stable baseline on Kaggle T4.
  - kaggle_m: stronger Kaggle benchmark if VRAM allows.
  - vertex_final: L4-safe accuracy-first final model profile for Vertex AI.
  - vertex_finetune_stable: stable fine-tune from the saved final checkpoint.
  - vertex_x_l4: YOLO11x benchmark profile that fits L4 by reducing image size.

This script intentionally does not run local smoke training. It is meant for
Kaggle/Vertex environments with GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any


DATASET_REPO_ID = "thanhhoangnvbg/fire-vn-yolo11seg-v1"
DATASET_ZIP_NAME = "fire_vn_yolo11seg_v1.zip"


PROFILES: dict[str, dict[str, Any]] = {
    "kaggle_s": {
        "model": "yolo11s-seg.pt",
        "imgsz": 640,
        "epochs": 80,
        "batch": 16,
        "patience": 20,
        "workers": 2,
        "cache": False,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 3,
        "close_mosaic": 10,
        "mosaic": 0.8,
        "mixup": 0.05,
        "copy_paste": 0.10,
        "degrees": 8.0,
        "translate": 0.10,
        "scale": 0.50,
        "fliplr": 0.50,
        "dropout": 0.0,
        "save_period": 10,
        "project": "runs/kaggle_baselines",
        "name": "yolo11s_seg_fire_vn",
    },
    "kaggle_m": {
        "model": "yolo11m-seg.pt",
        "imgsz": 640,
        "epochs": 120,
        "batch": 8,
        "patience": 25,
        "workers": 2,
        "cache": False,
        "optimizer": "AdamW",
        "lr0": 0.0008,
        "lrf": 0.01,
        "warmup_epochs": 4,
        "close_mosaic": 15,
        "mosaic": 0.8,
        "mixup": 0.05,
        "copy_paste": 0.12,
        "degrees": 8.0,
        "translate": 0.10,
        "scale": 0.55,
        "fliplr": 0.50,
        "dropout": 0.0,
        "save_period": 10,
        "project": "runs/kaggle_baselines",
        "name": "yolo11m_seg_fire_vn",
    },
    "vertex_final": {
        "model": "yolo11l-seg.pt",
        "imgsz": 896,
        "epochs": 240,
        "batch": 4,
        "patience": 45,
        "workers": 8,
        "cache": "disk",
        "optimizer": "AdamW",
        "lr0": 0.0006,
        "lrf": 0.005,
        "warmup_epochs": 5,
        "close_mosaic": 30,
        "mosaic": 0.40,
        "mixup": 0.03,
        "copy_paste": 0.10,
        "degrees": 4.0,
        "translate": 0.06,
        "scale": 0.35,
        "fliplr": 0.50,
        "dropout": 0.02,
        "save_period": 10,
        "project": "runs/final",
        "name": "yolo11l_seg_fire_vn_l4_final",
    },
    "vertex_finetune_stable": {
        "model": "runs/final/yolo11l_seg_fire_vn_l4_final/weights/best.pt",
        "imgsz": 896,
        "epochs": 80,
        "batch": 4,
        "patience": 25,
        "workers": 8,
        "cache": "disk",
        "optimizer": "AdamW",
        "lr0": 0.0001,
        "lrf": 0.01,
        "warmup_epochs": 2,
        "close_mosaic": 0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 3.0,
        "translate": 0.04,
        "scale": 0.20,
        "fliplr": 0.50,
        "dropout": 0.02,
        "amp": False,
        "save_period": 5,
        "project": "runs/final",
        "name": "yolo11l_seg_fire_vn_l4_finetune_stable",
    },
    "vertex_x_l4": {
        "model": "yolo11x-seg.pt",
        "imgsz": 768,
        "epochs": 240,
        "batch": 2,
        "patience": 45,
        "workers": 8,
        "cache": "disk",
        "optimizer": "AdamW",
        "lr0": 0.0006,
        "lrf": 0.005,
        "warmup_epochs": 5,
        "close_mosaic": 30,
        "mosaic": 0.40,
        "mixup": 0.03,
        "copy_paste": 0.10,
        "degrees": 4.0,
        "translate": 0.06,
        "scale": 0.35,
        "fliplr": 0.50,
        "dropout": 0.02,
        "save_period": 10,
        "project": "runs/final",
        "name": "yolo11x_seg_fire_vn_l4_benchmark",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11-seg fire/smoke models.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="vertex_final",
        help="Training profile. Use vertex_final for the final model.",
    )
    parser.add_argument("--dataset-repo-id", default=DATASET_REPO_ID)
    parser.add_argument("--dataset-zip", default=DATASET_ZIP_NAME)
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--device", default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", help="Override profile model, e.g. yolo11l-seg.pt.")
    parser.add_argument(
        "--init-weights",
        type=Path,
        help="Checkpoint to initialize a new fine-tune run without restoring optimizer/EMA state.",
    )
    parser.add_argument("--imgsz", type=int, help="Override image size.")
    parser.add_argument("--batch", type=int, help="Override batch size.")
    parser.add_argument("--epochs", type=int, help="Override epochs.")
    parser.add_argument("--workers", type=int, help="Override DataLoader workers.")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override AMP. Use --no-amp for stable fine-tuning if AMP/EMA produced NaN.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from the profile last.pt if found.")
    parser.add_argument("--resume-path", type=Path, help="Explicit checkpoint to resume from.")
    parser.add_argument("--skip-download", action="store_true", help="Use existing extracted dataset.")
    parser.add_argument("--val-test", action="store_true", help="Run test split validation after training.")
    parser.add_argument("--test-imgsz", type=int, help="Override image size for test validation.")
    parser.add_argument("--tta-val", action="store_true", help="Use test-time augmentation for test validation.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow Ultralytics to reuse run folder.")
    parser.add_argument(
        "--gcs-dir",
        help="Optional GCS folder for Vertex backup, e.g. gs://bucket/fire-vn/yolo11l-l4-final.",
    )
    parser.add_argument(
        "--restore-from-gcs",
        action="store_true",
        help="Restore the run directory from --gcs-dir before training/resume.",
    )
    parser.add_argument(
        "--hf-model-repo-id",
        help="Optional HF model repo id for uploading final artifacts, e.g. user/fire-vn-yolo11l-seg-l4.",
    )
    parser.add_argument(
        "--hf-private-model",
        action="store_true",
        help="Create the HF model repo as private if it does not exist.",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(PROFILES[args.profile])
    for key in ("model", "imgsz", "batch", "epochs", "workers", "amp"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    cfg["profile"] = args.profile
    cfg["project"] = str(Path(cfg["project"]).resolve())
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def download_and_extract_dataset(args: argparse.Namespace) -> Path:
    dataset_dir = args.dataset_dir or (args.work_dir / "fire_vn_yolo11seg_v1")
    data_yaml = dataset_dir / "data.yaml"
    if args.skip_download and data_yaml.exists():
        normalize_data_yaml(data_yaml, dataset_dir)
        return dataset_dir
    if data_yaml.exists():
        print(f"dataset already extracted: {dataset_dir}")
        normalize_data_yaml(data_yaml, dataset_dir)
        return dataset_dir

    from huggingface_hub import hf_hub_download, list_repo_files

    token = os.environ.get("HF_TOKEN")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    repo_files = list_repo_files(args.dataset_repo_id, repo_type="dataset", token=token)
    dataset_zip = args.dataset_zip
    if dataset_zip not in repo_files:
        zip_basename = Path(dataset_zip).name
        matches = [
            name
            for name in repo_files
            if name.replace("\\", "/").split("/")[-1] == zip_basename
        ]
        if not matches:
            matches = [name for name in repo_files if name.lower().endswith(".zip")]
        if not matches:
            raise SystemExit(
                f"dataset zip not found in HF repo {args.dataset_repo_id}: {dataset_zip}"
            )
        dataset_zip = matches[0]
        print(f"dataset zip resolved from {args.dataset_zip!r} to {dataset_zip!r}")
    download_kwargs = dict(
        repo_id=args.dataset_repo_id,
        filename=dataset_zip,
        repo_type="dataset",
        local_dir=args.work_dir / "hf_cache",
    )
    if token:
        download_kwargs["token"] = token
    zip_path = hf_hub_download(**download_kwargs)

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"extracting dataset: {zip_path} -> {dataset_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dataset_dir)
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found after extraction: {data_yaml}")
    normalize_data_yaml(data_yaml, dataset_dir)
    return dataset_dir


def normalize_data_yaml(data_yaml: Path, dataset_dir: Path) -> None:
    try:
        import yaml
    except ImportError:
        print("PyYAML not found; leaving data.yaml unchanged.")
        return

    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"invalid data.yaml format: {data_yaml}")
    data["path"] = str(dataset_dir.resolve())
    with data_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"normalized data.yaml path to {data['path']}")


def find_resume_checkpoint(cfg: dict[str, Any], explicit: Path | None) -> Path | None:
    if explicit:
        return explicit
    last_pt = Path(cfg["project"]) / cfg["name"] / "weights" / "last.pt"
    return last_pt if last_pt.exists() else None


def build_train_kwargs(
    cfg: dict[str, Any], args: argparse.Namespace, data_yaml: Path
) -> dict[str, Any]:
    return {
        "data": str(data_yaml),
        "epochs": cfg["epochs"],
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "device": args.device,
        "workers": cfg["workers"],
        "project": cfg["project"],
        "name": cfg["name"],
        "exist_ok": args.exist_ok,
        "patience": cfg["patience"],
        "save": True,
        "save_period": cfg["save_period"],
        "plots": True,
        "val": True,
        "seed": args.seed,
        "deterministic": True,
        "pretrained": True,
        "optimizer": cfg["optimizer"],
        "lr0": cfg["lr0"],
        "lrf": cfg["lrf"],
        "warmup_epochs": cfg["warmup_epochs"],
        "cos_lr": cfg.get("cos_lr", True),
        "weight_decay": cfg.get("weight_decay", 0.0005),
        "amp": cfg.get("amp", True),
        "cache": cfg["cache"],
        "close_mosaic": cfg["close_mosaic"],
        "mosaic": cfg["mosaic"],
        "mixup": cfg["mixup"],
        "copy_paste": cfg["copy_paste"],
        "degrees": cfg["degrees"],
        "translate": cfg["translate"],
        "scale": cfg["scale"],
        "fliplr": cfg["fliplr"],
        "dropout": cfg["dropout"],
        "hsv_h": cfg.get("hsv_h", 0.015),
        "hsv_s": cfg.get("hsv_s", 0.60),
        "hsv_v": cfg.get("hsv_v", 0.40),
        "overlap_mask": True,
        "mask_ratio": 4,
    }


def train(args: argparse.Namespace) -> None:
    cfg = resolve_config(args)
    seed_everything(args.seed)
    run_dir = Path(cfg["project"]) / cfg["name"]
    if args.restore_from_gcs:
        restore_run_from_gcs(run_dir, args.gcs_dir)

    dataset_dir = download_and_extract_dataset(args)
    data_yaml = dataset_dir / "data.yaml"

    from ultralytics import YOLO

    resume_checkpoint = find_resume_checkpoint(cfg, args.resume_path) if args.resume else args.resume_path
    try:
        if resume_checkpoint:
            print(f"resuming from {resume_checkpoint}")
            model = YOLO(str(resume_checkpoint))
            results = model.train(resume=True)
        else:
            model_path = args.init_weights or cfg["model"]
            print(f"starting new train/fine-tune run from {model_path}")
            model = YOLO(str(model_path))
            train_kwargs = build_train_kwargs(cfg, args, data_yaml)
            results = model.train(**train_kwargs)

        save_training_config(cfg, args, data_yaml)

        best_pt = run_dir / "weights" / "best.pt"
        if args.val_test and best_pt.exists():
            test_imgsz = args.test_imgsz or cfg["imgsz"]
            print(f"running test validation on {best_pt}")
            YOLO(str(best_pt)).val(
                data=str(data_yaml),
                split="test",
                imgsz=test_imgsz,
                device=args.device,
                augment=args.tta_val,
            )

        upload_best_model_to_hf(run_dir, args.hf_model_repo_id, args.hf_private_model)

        print(results)
        print(f"best checkpoint expected at: {best_pt}")
    finally:
        sync_run_to_gcs(run_dir, args.gcs_dir)


def save_training_config(cfg: dict[str, Any], args: argparse.Namespace, data_yaml: Path) -> None:
    run_dir = Path(cfg["project"]) / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile_config": cfg,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "data_yaml": str(data_yaml),
    }
    with (run_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def restore_run_from_gcs(run_dir: Path, gcs_dir: str | None) -> None:
    if not gcs_dir:
        print("--restore-from-gcs ignored because --gcs-dir was not provided.")
        return
    if shutil.which("gsutil") is None:
        print("gsutil not found; skipping GCS restore.")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"restoring run directory from {gcs_dir} -> {run_dir}")
    subprocess.run(["gsutil", "-m", "rsync", "-r", gcs_dir, str(run_dir)], check=False)


def sync_run_to_gcs(run_dir: Path, gcs_dir: str | None) -> None:
    if not gcs_dir:
        return
    if shutil.which("gsutil") is None:
        print("gsutil not found; skipping GCS sync.")
        return
    if not run_dir.exists():
        print(f"run directory not found; skipping GCS sync: {run_dir}")
        return
    print(f"syncing run directory to {gcs_dir}")
    subprocess.run(["gsutil", "-m", "rsync", "-r", str(run_dir), gcs_dir], check=False)


def upload_best_model_to_hf(
    run_dir: Path, repo_id: str | None, private: bool
) -> None:
    if not repo_id:
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing; skipping HF model upload.")
        return

    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    if not best_pt.exists():
        print(f"best.pt not found; skipping HF model upload: {best_pt}")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed; skipping HF model upload.")
        return

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    files_to_upload = [
        (best_pt, "weights/best.pt"),
        (last_pt, "weights/last.pt"),
        (run_dir / "results.csv", "results.csv"),
        (run_dir / "training_config.json", "training_config.json"),
        (run_dir / "args.yaml", "args.yaml"),
        (run_dir / "confusion_matrix.png", "confusion_matrix.png"),
        (run_dir / "confusion_matrix_normalized.png", "confusion_matrix_normalized.png"),
        (run_dir / "results.png", "results.png"),
        (run_dir / "BoxPR_curve.png", "BoxPR_curve.png"),
        (run_dir / "BoxF1_curve.png", "BoxF1_curve.png"),
        (run_dir / "MaskPR_curve.png", "MaskPR_curve.png"),
        (run_dir / "MaskF1_curve.png", "MaskF1_curve.png"),
    ]
    for local_path, path_in_repo in files_to_upload:
        if local_path.exists():
            print(f"uploading {local_path} -> hf://{repo_id}/{path_in_repo}")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
            )

    model_card = run_dir / "README.md"
    write_model_card(model_card, repo_id, run_dir)
    api.upload_file(
        path_or_fileobj=str(model_card),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"uploaded final model artifacts to https://huggingface.co/{repo_id}")


def write_model_card(path: Path, repo_id: str, run_dir: Path) -> None:
    content = f"""---
library_name: ultralytics
tags:
- yolo
- yolo11
- image-segmentation
- fire-detection
- smoke-detection
- vietnam
---

# Fire VN YOLO11-Seg Final Model

This repository contains the final YOLO11 segmentation model artifacts for
Vietnam-focused fire/smoke detection.

## Main Files

- `weights/best.pt`: best checkpoint selected by validation performance.
- `weights/last.pt`: last checkpoint for resuming or further fine-tuning.
- `results.csv`: training metrics.
- `training_config.json`: reproducibility configuration.
- `confusion_matrix.png`: validation confusion matrix, if generated.

## Dataset

Dataset: https://huggingface.co/datasets/thanhhoangnvbg/fire-vn-yolo11seg-v1

## Local Run Directory

```text
{run_dir}
```

## Intended Use

The model is intended for fire/smoke detection experiments and downstream
deployment validation. It should be validated on real camera footage before any
production use.
"""
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    train(parse_args())
