#!/usr/bin/env python3
"""Evaluate a YOLO11 detect checkpoint on per-source-group subsets."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml


GROUPS = ("01", "02", "03", "04", "05")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO validation separately for g01..g05 detect subsets."
    )
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint path, e.g. weights/best.pt.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Prepared YOLO detect dataset. Defaults to work/..., then datasets/....",
    )
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--groups", nargs="*", default=list(GROUPS), choices=GROUPS)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=0)
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation.")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of hard-linking.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing subset work directory.")
    parser.add_argument("--work-dir", type=Path, default=Path("work/detect_group_eval"))
    return parser.parse_args()


def default_dataset() -> Path:
    candidates = (
        Path("work/fire_vn_yolo11det_fire_smoke_v2"),
        Path("datasets/fire_vn_yolo11det_fire_smoke_v2"),
    )
    for candidate in candidates:
        if (candidate / "data.yaml").exists():
            return candidate
    return candidates[0]


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_subset(
    dataset: Path,
    split: str,
    group: str,
    output: Path,
    copy_images: bool,
) -> int:
    src_images = dataset / "images" / split
    src_labels = dataset / "labels" / split
    if not src_images.exists():
        raise SystemExit(f"Missing images directory: {src_images}")
    if not src_labels.exists():
        raise SystemExit(f"Missing labels directory: {src_labels}")

    dst_images = output / "images" / "val"
    dst_labels = output / "labels" / "val"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    count = 0
    prefix = f"g{group}_"
    for src_image in image_files(src_images):
        if not src_image.name.startswith(prefix):
            continue
        link_or_copy(src_image, dst_images / src_image.name, copy_images)
        src_label = src_labels / f"{src_image.stem}.txt"
        dst_label = dst_labels / f"{src_image.stem}.txt"
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        else:
            dst_label.write_text("", encoding="utf-8")
        count += 1

    data = {
        "path": str(output.resolve()),
        "train": "images/val",
        "val": "images/val",
        "test": "images/val",
        "names": {0: "fire_smoke"},
    }
    with (output / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return count


def main() -> None:
    args = parse_args()
    dataset = args.dataset or default_dataset()
    if not (dataset / "data.yaml").exists():
        raise SystemExit(f"Detect dataset not found: {dataset}")
    if not args.model.exists():
        raise SystemExit(f"Checkpoint not found: {args.model}")
    if args.work_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Work directory exists: {args.work_dir}. Pass --overwrite.")
        shutil.rmtree(args.work_dir)

    from ultralytics import YOLO

    for group in args.groups:
        subset_dir = args.work_dir / f"{args.split}_g{group}"
        image_count = prepare_subset(
            dataset=dataset,
            split=args.split,
            group=group,
            output=subset_dir,
            copy_images=args.copy_images,
        )
        if image_count == 0:
            print(f"group {group}: no images found for split {args.split}; skipping")
            continue
        print(f"group {group}: validating {image_count} images")
        YOLO(str(args.model)).val(
            data=str(subset_dir / "data.yaml"),
            split="val",
            imgsz=args.imgsz,
            device=args.device,
            augment=args.tta,
        )


if __name__ == "__main__":
    main()
