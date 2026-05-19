#!/usr/bin/env python3
"""Convert the prepared YOLO11-seg dataset into 1-class YOLO detect format."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import stat
import time
from collections import Counter
from pathlib import Path

import yaml


SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO segmentation labels to 1-class YOLO detection labels."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/fire_vn_yolo11seg_v1"),
        help="Prepared YOLO11-seg dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/fire_vn_yolo11det_fire_smoke_v2"),
        help="Output YOLO detection dataset directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory.")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy image files instead of hard-linking where possible.",
    )
    parser.add_argument("--make-zip", action="store_true", help="Create output .zip archive.")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(message)


def ensure_output(output: Path, overwrite: bool) -> None:
    resolved = output.resolve()
    cwd = Path.cwd().resolve()
    unsafe = {cwd, cwd.parent, Path(resolved.anchor)}
    if output.exists():
        if not overwrite:
            fail(f"Output already exists: {output}. Pass --overwrite to replace it.")
        if resolved in unsafe:
            fail(f"Refusing to overwrite unsafe output path: {resolved}")
        remove_tree(output)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output / "metadata").mkdir(parents=True, exist_ok=True)


def remove_tree(path: Path) -> None:
    def onerror(func, failed_path, _exc_info):
        try:
            os.chmod(failed_path, stat.S_IWRITE)
            func(failed_path)
        except OSError:
            pass

    for attempt in range(5):
        try:
            shutil.rmtree(path, onerror=onerror)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.5)


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> str:
    if copy_images:
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def seg_line_to_bbox(line: str) -> tuple[str | None, str | None]:
    parts = line.strip().split()
    if not parts:
        return None, None
    if len(parts) < 5:
        return None, "too_few_values"

    try:
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return None, "non_numeric"

    if len(coords) == 4:
        x_center, y_center, width, height = [clamp01(value) for value in coords]
        if width <= 0.0 or height <= 0.0:
            return None, "invalid_bbox"
        return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}", None

    if len(coords) < 6 or len(coords) % 2 != 0:
        return None, "invalid_polygon"

    xs = [clamp01(value) for value in coords[0::2]]
    ys = [clamp01(value) for value in coords[1::2]]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0.0 or height <= 0.0:
        return None, "invalid_bbox"

    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}", None


def convert_label(src: Path, dst: Path) -> Counter:
    stats: Counter = Counter()
    lines: list[str] = []
    if src.exists():
        for raw_line in src.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            converted, error = seg_line_to_bbox(raw_line)
            if converted:
                lines.append(converted)
                stats["objects"] += 1
            else:
                stats[f"skipped_{error or 'unknown'}"] += 1
    if not lines:
        stats["background_images"] += 1
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return stats


def write_data_yaml(output: Path) -> None:
    data = {
        "path": ".",
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {0: "fire_smoke"},
    }
    with (output / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def validate_output(output: Path) -> Counter:
    stats: Counter = Counter()
    for split in SPLITS:
        for label_path in sorted((output / "labels" / split).glob("*.txt")):
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    fail(f"Invalid detect label width: {label_path}:{line_number}")
                if parts[0] != "0":
                    fail(f"Unexpected class id in {label_path}:{line_number}: {parts[0]}")
                try:
                    x_center, y_center, width, height = [float(value) for value in parts[1:]]
                except ValueError:
                    fail(f"Non-numeric detect label: {label_path}:{line_number}")
                values = (x_center, y_center, width, height)
                if not all(0.0 <= value <= 1.0 for value in values):
                    fail(f"Out-of-range bbox in {label_path}:{line_number}: {values}")
                if width <= 0.0 or height <= 0.0:
                    fail(f"Invalid bbox size in {label_path}:{line_number}: {values}")
                stats["validated_objects"] += 1
    return stats


def convert_dataset(input_dir: Path, output_dir: Path, overwrite: bool, copy_images: bool) -> Counter:
    if not (input_dir / "data.yaml").exists():
        fail(f"Missing source data.yaml: {input_dir / 'data.yaml'}")

    ensure_output(output_dir, overwrite)
    summary_rows: list[dict[str, int | str]] = []
    total: Counter = Counter()

    for split in SPLITS:
        src_images_dir = input_dir / "images" / split
        src_labels_dir = input_dir / "labels" / split
        if not src_images_dir.exists():
            fail(f"Missing source images directory: {src_images_dir}")
        if not src_labels_dir.exists():
            fail(f"Missing source labels directory: {src_labels_dir}")

        split_stats: Counter = Counter()
        for src_image in image_files(src_images_dir):
            dst_image = output_dir / "images" / split / src_image.name
            mode = link_or_copy(src_image, dst_image, copy_images)
            split_stats[f"images_{mode}"] += 1
            split_stats["images"] += 1

            src_label = src_labels_dir / f"{src_image.stem}.txt"
            dst_label = output_dir / "labels" / split / f"{src_image.stem}.txt"
            split_stats.update(convert_label(src_label, dst_label))

        summary_rows.append(
            {
                "split": split,
                "images": split_stats["images"],
                "objects": split_stats["objects"],
                "background_images": split_stats["background_images"],
                "images_hardlinked": split_stats["images_hardlink"],
                "images_copied": split_stats["images_copy"],
                "skipped_too_few_values": split_stats["skipped_too_few_values"],
                "skipped_non_numeric": split_stats["skipped_non_numeric"],
                "skipped_invalid_polygon": split_stats["skipped_invalid_polygon"],
                "skipped_invalid_bbox": split_stats["skipped_invalid_bbox"],
            }
        )
        total.update(split_stats)

    write_data_yaml(output_dir)
    validation_stats = validate_output(output_dir)
    total.update(validation_stats)
    write_summary(output_dir, summary_rows)
    copy_source_metadata(input_dir, output_dir)
    return total


def write_summary(output: Path, rows: list[dict[str, int | str]]) -> None:
    path = output / "metadata" / "conversion_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def copy_source_metadata(input_dir: Path, output_dir: Path) -> None:
    src_metadata = input_dir / "metadata"
    dst_metadata = output_dir / "metadata" / "source_seg_metadata"
    if src_metadata.exists():
        shutil.copytree(src_metadata, dst_metadata, dirs_exist_ok=True)


def make_zip(output: Path) -> Path:
    archive_base = output.with_suffix("")
    zip_path = shutil.make_archive(str(archive_base), "zip", root_dir=output)
    return Path(zip_path)


def main() -> None:
    args = parse_args()
    stats = convert_dataset(
        input_dir=args.input,
        output_dir=args.output,
        overwrite=args.overwrite,
        copy_images=args.copy_images,
    )
    print(f"converted dataset: {args.output}")
    print(f"images: {stats['images']}")
    print(f"objects: {stats['objects']}")
    print(f"background images: {stats['background_images']}")
    print(f"validated objects: {stats['validated_objects']}")
    if args.make_zip:
        zip_path = make_zip(args.output)
        print(f"zip: {zip_path}")


if __name__ == "__main__":
    main()
