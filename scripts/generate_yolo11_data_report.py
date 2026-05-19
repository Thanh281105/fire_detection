#!/usr/bin/env python3
"""Generate visual data audit artifacts for the prepared YOLO11 dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from PIL import Image, ImageDraw, ImageFont


SPLITS = ("train", "valid", "test")
GROUPS = ("01", "02", "03", "04", "05")
GROUP_NAMES = {
    "01": "positive_standard",
    "02": "alley_context",
    "03": "hard_negative",
    "04": "sahi_small_objects",
    "05": "ambient_null",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create charts, CSV tables, and sample images for YOLO11 data audit."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/fire_vn_yolo11seg_v1"),
        help="Prepared YOLO dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/data_audit"),
        help="Output report directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-per-group", type=int, default=8)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=int,
        default=2,
        help="64-bit dHash Hamming distance for near-duplicate audit.",
    )
    parser.add_argument(
        "--max-duplicate-sheets",
        type=int,
        default=12,
        help="Maximum near-duplicate contact sheets to render.",
    )
    parser.add_argument(
        "--skip-near-duplicates",
        action="store_true",
        help="Skip perceptual near-duplicate audit.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(message)


def ensure_dirs(output: Path) -> dict[str, Path]:
    dirs = {
        "root": output,
        "figures": output / "figures",
        "tables": output / "tables",
        "samples": output / "samples",
        "duplicates": output / "near_duplicates",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_names(dataset: Path) -> dict[int, str]:
    data_yaml = dataset / "data.yaml"
    if not data_yaml.exists():
        fail(f"Missing data.yaml: {data_yaml}")
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {}) if isinstance(data, dict) else {}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    fail(f"Unsupported names format in {data_yaml}")


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def group_from_name(name: str) -> str:
    stem = Path(name).stem
    if len(stem) >= 3 and stem.startswith("g") and stem[1:3].isdigit():
        return stem[1:3]
    return "unknown"


def parse_label(label_path: Path) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    if not label_path.exists():
        return objects
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        if len(coords) == 4:
            x_center, y_center, width, height = coords
            xs = [x_center - width / 2.0, x_center + width / 2.0]
            ys = [y_center - height / 2.0, y_center + height / 2.0]
            polygon = None
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs = coords[0::2]
            ys = coords[1::2]
            polygon = coords
        else:
            continue
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        if width <= 0.0 or height <= 0.0:
            continue
        objects.append(
            {
                "class_id": class_id,
                "line_number": line_number,
                "x_min": clamp01(min(xs)),
                "y_min": clamp01(min(ys)),
                "x_max": clamp01(max(xs)),
                "y_max": clamp01(max(ys)),
                "width": clamp01(width),
                "height": clamp01(height),
                "area": clamp01(width) * clamp01(height),
                "aspect_ratio": width / height if height > 0 else 0.0,
                "polygon": polygon,
            }
        )
    return objects


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def collect_dataset_stats(dataset: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    image_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        images_dir = dataset / "images" / split
        labels_dir = dataset / "labels" / split
        for image_path in image_files(images_dir):
            group = group_from_name(image_path.name)
            label_path = labels_dir / f"{image_path.stem}.txt"
            objects = parse_label(label_path)
            width, height = read_image_size(image_path)
            image_rows.append(
                {
                    "split": split,
                    "group": group,
                    "image": str(image_path),
                    "file_name": image_path.name,
                    "width_px": width,
                    "height_px": height,
                    "objects": len(objects),
                    "is_background": int(len(objects) == 0),
                }
            )
            for obj in objects:
                obj_row = dict(obj)
                obj_row.update(
                    {
                        "split": split,
                        "group": group,
                        "image": str(image_path),
                        "file_name": image_path.name,
                        "width_px": width,
                        "height_px": height,
                    }
                )
                object_rows.append(obj_row)
    return image_rows, object_rows


def read_image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return 0, 0


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_images(image_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        grouped[(row["split"], row["group"])].append(row)
    summary: list[dict[str, Any]] = []
    for split in SPLITS:
        for group in GROUPS:
            rows = grouped.get((split, group), [])
            objects = sum(int(row["objects"]) for row in rows)
            backgrounds = sum(int(row["is_background"]) for row in rows)
            summary.append(
                {
                    "split": split,
                    "group": group,
                    "group_name": GROUP_NAMES.get(group, group),
                    "images": len(rows),
                    "objects": objects,
                    "background_images": backgrounds,
                    "background_ratio": round(backgrounds / len(rows), 4) if rows else 0.0,
                    "objects_per_image": round(objects / len(rows), 4) if rows else 0.0,
                }
            )
    return summary


def plot_grouped_bars(
    rows: list[dict[str, Any]],
    value_key: str,
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    x = list(range(len(GROUPS)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 5.8))
    colors = {"train": "#2f6fbb", "valid": "#5ba85b", "test": "#d9902f"}
    for idx, split in enumerate(SPLITS):
        values = [
            next(
                (
                    float(row[value_key])
                    for row in rows
                    if row["split"] == split and row["group"] == group
                ),
                0.0,
            )
            for group in GROUPS
        ]
        offsets = [pos + (idx - 1) * width for pos in x]
        ax.bar(offsets, values, width=width, label=split, color=colors[split])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([f"g{group}\n{GROUP_NAMES[group]}" for group in GROUPS])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_class_distribution(
    object_rows: list[dict[str, Any]], names: dict[int, str], output: Path
) -> None:
    counts = Counter(int(row["class_id"]) for row in object_rows)
    class_ids = sorted(names)
    labels = [names[class_id] for class_id in class_ids]
    values = [counts[class_id] for class_id in class_ids]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=["#5f7f95", "#c94f45", "#7d9851"][: len(labels)])
    ax.set_title("Class Distribution")
    ax.set_ylabel("Objects")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value, str(value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_object_size_hist(object_rows: list[dict[str, Any]], output: Path) -> None:
    areas = [max(float(row["area"]), 1e-8) * 100.0 for row in object_rows]
    if not areas:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(areas, bins=60, color="#4d7c8a", edgecolor="white")
    ax.set_xscale("log")
    ax.set_title("Object Area Distribution")
    ax.set_xlabel("BBox area (% of image, log scale)")
    ax.set_ylabel("Objects")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_width_height_scatter(object_rows: list[dict[str, Any]], output: Path) -> None:
    if not object_rows:
        return
    colors = {"train": "#2f6fbb", "valid": "#5ba85b", "test": "#d9902f"}
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for split in SPLITS:
        rows = [row for row in object_rows if row["split"] == split]
        ax.scatter(
            [float(row["width"]) for row in rows],
            [float(row["height"]) for row in rows],
            s=7,
            alpha=0.35,
            label=split,
            color=colors[split],
        )
    ax.set_title("Normalized Object Width vs Height")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(markerscale=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_resolution_distribution(image_rows: list[dict[str, Any]], output: Path) -> None:
    rows = [row for row in image_rows if int(row["width_px"]) > 0 and int(row["height_px"]) > 0]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for split, color in zip(SPLITS, ["#2f6fbb", "#5ba85b", "#d9902f"]):
        subset = [row for row in rows if row["split"] == split]
        ax.scatter(
            [int(row["width_px"]) for row in subset],
            [int(row["height_px"]) for row in subset],
            s=10,
            alpha=0.35,
            label=split,
            color=color,
        )
    ax.set_title("Image Resolution Distribution")
    ax.set_xlabel("Width px")
    ax.set_ylabel("Height px")
    ax.legend(markerscale=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_background_ratio(summary_rows: list[dict[str, Any]], output: Path) -> None:
    plot_grouped_bars(
        summary_rows,
        "background_ratio",
        "Background Image Ratio by Group",
        "Background ratio",
        output,
    )


def render_sample_montages(
    dataset: Path,
    image_rows: list[dict[str, Any]],
    names: dict[int, str],
    output: Path,
    per_group: int,
    seed: int,
) -> list[Path]:
    rng = random.Random(seed)
    rendered: list[Path] = []
    train_rows = [row for row in image_rows if row["split"] == "train"]
    for group in GROUPS:
        rows = [row for row in train_rows if row["group"] == group]
        if not rows:
            continue
        positives = [row for row in rows if int(row["objects"]) > 0]
        backgrounds = [row for row in rows if int(row["objects"]) == 0]
        selected = sample_rows(positives, max(1, per_group - 2), rng)
        selected.extend(sample_rows(backgrounds, per_group - len(selected), rng))
        selected = selected[:per_group]
        if not selected:
            continue
        out_path = output / f"train_g{group}_{GROUP_NAMES[group]}_samples.jpg"
        render_contact_sheet(dataset, selected, names, out_path)
        rendered.append(out_path)
    return rendered


def sample_rows(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:count]


def render_contact_sheet(
    dataset: Path,
    rows: list[dict[str, Any]],
    names: dict[int, str],
    output: Path,
    tile_size: tuple[int, int] = (320, 240),
) -> None:
    cols = min(4, max(1, len(rows)))
    rows_count = math.ceil(len(rows) / cols)
    caption_h = 46
    sheet = Image.new("RGB", (cols * tile_size[0], rows_count * (tile_size[1] + caption_h)), "white")
    for idx, row in enumerate(rows):
        image_path = Path(row["image"])
        tile = draw_annotated_tile(dataset, image_path, names, tile_size)
        x = (idx % cols) * tile_size[0]
        y = (idx // cols) * (tile_size[1] + caption_h)
        sheet.paste(tile, (x, y))
        draw = ImageDraw.Draw(sheet)
        caption = f"{row['split']} g{row['group']} | objs={row['objects']}\n{row['file_name'][:42]}"
        draw.text((x + 6, y + tile_size[1] + 4), caption, fill=(20, 20, 20))
    sheet.save(output, quality=92)


def draw_annotated_tile(
    dataset: Path, image_path: Path, names: dict[int, str], tile_size: tuple[int, int]
) -> Image.Image:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original_w, original_h = image.size
        scale = min(tile_size[0] / original_w, tile_size[1] / original_h)
        resized = image.resize((max(1, int(original_w * scale)), max(1, int(original_h * scale))))
    tile = Image.new("RGB", tile_size, (235, 235, 235))
    offset_x = (tile_size[0] - resized.width) // 2
    offset_y = (tile_size[1] - resized.height) // 2
    tile.paste(resized, (offset_x, offset_y))

    split = image_path.parent.name
    label_path = dataset / "labels" / split / f"{image_path.stem}.txt"
    objects = parse_label(label_path)
    draw = ImageDraw.Draw(tile)
    colors = [(208, 67, 57), (70, 120, 172), (89, 143, 74)]
    for obj in objects:
        color = colors[int(obj["class_id"]) % len(colors)]
        x0 = offset_x + obj["x_min"] * resized.width
        y0 = offset_y + obj["y_min"] * resized.height
        x1 = offset_x + obj["x_max"] * resized.width
        y1 = offset_y + obj["y_max"] * resized.height
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        label = names.get(int(obj["class_id"]), str(obj["class_id"]))
        draw.rectangle((x0, max(0, y0 - 16), x0 + 8 * len(label) + 6, y0), fill=color)
        draw.text((x0 + 3, max(0, y0 - 15)), label, fill="white")
    return tile


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dhash(path: Path) -> int:
    with Image.open(path) as image:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        pixels = list(image.convert("L").resize((9, 8), resampling).getdata())
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return value


def collect_raw_sources(dataset: Path) -> list[dict[str, Any]]:
    raw_root = dataset / "_raw_from_zip"
    if not raw_root.exists():
        return []
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import prepare_yolo11_seg_dataset as prep

    sources: list[dict[str, Any]] = []
    for group in GROUPS:
        group_dir = raw_root / f"group_{group}"
        if not group_dir.exists():
            continue
        group_sources, _ = prep.collect_sources(group, group_dir)
        for source in group_sources:
            sources.append(
                {
                    "group": source.group,
                    "source_key": source.source_key,
                    "file_name": source.original_file_name,
                    "path": str(source.image_path),
                    "annotations": len(source.annotations),
                    "content_hash": source.content_hash,
                    "dhash": image_dhash(source.image_path),
                }
            )
    return sources


def near_duplicate_audit(
    sources: list[dict[str, Any]],
    threshold: int,
    tables_dir: Path,
    figures_dir: Path,
    sheets_dir: Path,
    max_sheets: int,
) -> dict[str, Any]:
    if threshold < 0:
        fail("--near-duplicate-threshold must be >= 0")
    duplicate_sets = find_near_duplicate_sets(sources, threshold)
    rows: list[dict[str, Any]] = []
    removed_by_group = Counter()
    for cluster_id, cluster in enumerate(duplicate_sets, 1):
        keep = choose_keep_source(cluster)
        for source in sorted(cluster, key=lambda row: (row["source_key"] != keep["source_key"], row["group"], row["file_name"])):
            action = "kept" if source["source_key"] == keep["source_key"] else "would_remove"
            if action == "would_remove":
                removed_by_group[source["group"]] += 1
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "action": action,
                    "group": source["group"],
                    "source_key": source["source_key"],
                    "file_name": source["file_name"],
                    "path": source["path"],
                    "annotations": source["annotations"],
                    "content_hash": source["content_hash"],
                    "dhash": f"{source['dhash']:016x}",
                    "kept_source_key": keep["source_key"],
                    "hamming_to_keep": (int(source["dhash"]) ^ int(keep["dhash"])).bit_count(),
                }
            )
    write_csv(
        tables_dir / "near_duplicate_sources.csv",
        rows,
        [
            "cluster_id",
            "action",
            "group",
            "source_key",
            "file_name",
            "path",
            "annotations",
            "content_hash",
            "dhash",
            "kept_source_key",
            "hamming_to_keep",
        ],
    )
    plot_duplicate_counts(removed_by_group, figures_dir / "near_duplicate_would_remove_by_group.png")
    render_duplicate_sheets(duplicate_sets, sheets_dir, max_sheets)
    return {
        "clusters": len(duplicate_sets),
        "would_remove": sum(len(cluster) - 1 for cluster in duplicate_sets),
        "removed_by_group": dict(sorted(removed_by_group.items())),
    }


def find_near_duplicate_sets(sources: list[dict[str, Any]], threshold: int) -> list[list[dict[str, Any]]]:
    parent = {source["source_key"]: source["source_key"] for source in sources}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    by_content_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        by_content_hash[str(source["content_hash"])].append(source)
    for duplicate_group in by_content_hash.values():
        for source in duplicate_group[1:]:
            union(duplicate_group[0]["source_key"], source["source_key"])

    for idx, source_a in enumerate(sources):
        hash_a = int(source_a["dhash"])
        for source_b in sources[idx + 1 :]:
            if (hash_a ^ int(source_b["dhash"])).bit_count() <= threshold:
                union(source_a["source_key"], source_b["source_key"])

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        clusters[find(source["source_key"])].append(source)
    return sorted(
        [cluster for cluster in clusters.values() if len(cluster) > 1],
        key=lambda cluster: (-len(cluster), choose_keep_source(cluster)["group"]),
    )


def choose_keep_source(cluster: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        cluster,
        key=lambda row: (
            -int(row["annotations"]),
            str(row["group"]),
            str(row["file_name"]).lower(),
            str(row["source_key"]),
        ),
    )[0]


def plot_duplicate_counts(removed_by_group: Counter, output: Path) -> None:
    values = [removed_by_group.get(group, 0) for group in GROUPS]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([f"g{group}" for group in GROUPS], values, color="#8a5a44")
    ax.set_title("Near-Duplicate Candidates by Group")
    ax.set_ylabel("Images that would be removed")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value, str(value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def render_duplicate_sheets(
    duplicate_sets: list[list[dict[str, Any]]], output_dir: Path, max_sheets: int
) -> None:
    for idx, cluster in enumerate(duplicate_sets[:max_sheets], 1):
        keep = choose_keep_source(cluster)
        selected = [keep] + [source for source in cluster if source["source_key"] != keep["source_key"]]
        selected = selected[:8]
        sheet = Image.new("RGB", (4 * 260, 2 * 245), "white")
        for item_idx, source in enumerate(selected):
            image_path = Path(source["path"])
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((250, 185))
            x = (item_idx % 4) * 260 + 5
            y = (item_idx // 4) * 245 + 5
            sheet.paste(image, (x, y))
            draw = ImageDraw.Draw(sheet)
            action = "KEEP" if source["source_key"] == keep["source_key"] else "DUP"
            caption = f"{action} g{source['group']} anns={source['annotations']}\n{source['file_name'][:34]}"
            draw.text((x, y + 190), caption, fill=(20, 20, 20))
        sheet.save(output_dir / f"near_duplicate_cluster_{idx:03d}.jpg", quality=92)


def write_markdown_summary(
    output: Path,
    dataset: Path,
    image_rows: list[dict[str, Any]],
    object_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    preprocessing_duplicate_summary: dict[str, Any] | None,
    duplicate_summary: dict[str, Any] | None,
    names: dict[int, str],
) -> None:
    total_images = len(image_rows)
    total_objects = len(object_rows)
    backgrounds = sum(int(row["is_background"]) for row in image_rows)
    class_counts = Counter(int(row["class_id"]) for row in object_rows)
    lines = [
        "# YOLO11 Data Audit",
        "",
        f"Dataset: `{dataset}`",
        "",
        "## Overview",
        "",
        f"- Images: `{total_images}`",
        f"- Objects: `{total_objects}`",
        f"- Background images: `{backgrounds}`",
        f"- Background ratio: `{backgrounds / total_images:.2%}`" if total_images else "- Background ratio: `0%`",
        "",
        "## Class Counts",
        "",
    ]
    for class_id, name in names.items():
        lines.append(f"- `{name}`: `{class_counts[class_id]}`")
    lines.extend(["", "## Split And Group Counts", ""])
    for row in summary_rows:
        lines.append(
            f"- `{row['split']} g{row['group']}`: "
            f"{row['images']} images, {row['objects']} objects, "
            f"{row['background_images']} background"
        )
    if preprocessing_duplicate_summary is not None:
        lines.extend(
            [
                "",
                "## Duplicate Preprocessing Applied",
                "",
                f"- Duplicate clusters: `{preprocessing_duplicate_summary['clusters']}`",
                f"- Removed source images: `{preprocessing_duplicate_summary['removed']}`",
                f"- Kept source images in duplicate clusters: `{preprocessing_duplicate_summary['kept']}`",
                f"- Removed by group: `{preprocessing_duplicate_summary['removed_by_group']}`",
                "",
                "Audit file: `../datasets/fire_vn_yolo11seg_v1/metadata/duplicate_sources.csv`",
            ]
        )
    if duplicate_summary is not None:
        lines.extend(
            [
                "",
                "## Near-Duplicate Visual Audit",
                "",
                f"- Clusters: `{duplicate_summary['clusters']}`",
                f"- Images that would be removed under this audit: `{duplicate_summary['would_remove']}`",
                f"- Removed by group: `{duplicate_summary['removed_by_group']}`",
                "",
                "Review contact sheets in `near_duplicates/` before enabling near-duplicate removal.",
            ]
        )
    lines.extend(
        [
            "",
            "## Generated Artifacts",
            "",
            "- `figures/`: charts for report writing.",
            "- `samples/`: annotated image montages.",
            "- `near_duplicates/`: visual duplicate audit sheets.",
            "- `tables/`: CSV tables for appendix or further analysis.",
        ]
    )
    (output / "data_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_preprocessing_duplicate_summary(dataset: Path) -> dict[str, Any] | None:
    path = dataset / "metadata" / "duplicate_sources.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    removed = [row for row in rows if row.get("action") == "removed"]
    kept = [row for row in rows if row.get("action") == "kept"]
    removed_by_group = Counter(row.get("group", "unknown") for row in removed)
    kept_keys = {row.get("kept_source_key") for row in rows if row.get("kept_source_key")}
    return {
        "clusters": len(kept_keys),
        "removed": len(removed),
        "kept": len(kept),
        "removed_by_group": dict(sorted(removed_by_group.items())),
    }


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        fail(f"Dataset not found: {args.dataset}")
    dirs = ensure_dirs(args.output)
    names = load_names(args.dataset)
    image_rows, object_rows = collect_dataset_stats(args.dataset)
    if not image_rows:
        fail(f"No images found under {args.dataset}")

    summary_rows = summarize_images(image_rows)
    write_csv(dirs["tables"] / "image_level_summary.csv", image_rows)
    write_csv(dirs["tables"] / "object_level_summary.csv", object_rows)
    write_csv(dirs["tables"] / "split_group_summary.csv", summary_rows)

    plot_grouped_bars(
        summary_rows,
        "images",
        "Images by Split and Group",
        "Images",
        dirs["figures"] / "images_by_split_group.png",
    )
    plot_grouped_bars(
        summary_rows,
        "objects",
        "Objects by Split and Group",
        "Objects",
        dirs["figures"] / "objects_by_split_group.png",
    )
    plot_background_ratio(summary_rows, dirs["figures"] / "background_ratio_by_group.png")
    plot_class_distribution(object_rows, names, dirs["figures"] / "class_distribution.png")
    plot_object_size_hist(object_rows, dirs["figures"] / "object_area_distribution.png")
    plot_width_height_scatter(object_rows, dirs["figures"] / "object_width_height_scatter.png")
    plot_resolution_distribution(image_rows, dirs["figures"] / "image_resolution_distribution.png")
    render_sample_montages(
        args.dataset,
        image_rows,
        names,
        dirs["samples"],
        args.sample_per_group,
        args.seed,
    )

    preprocessing_duplicate_summary = load_preprocessing_duplicate_summary(args.dataset)
    duplicate_summary = None
    if not args.skip_near_duplicates:
        raw_sources = collect_raw_sources(args.dataset)
        if raw_sources:
            write_csv(dirs["tables"] / "raw_source_images.csv", raw_sources)
            duplicate_summary = near_duplicate_audit(
                raw_sources,
                args.near_duplicate_threshold,
                dirs["tables"],
                dirs["figures"],
                dirs["duplicates"],
                args.max_duplicate_sheets,
            )
        else:
            print("near duplicate audit skipped: raw source images not found")

    write_markdown_summary(
        dirs["root"],
        args.dataset,
        image_rows,
        object_rows,
        summary_rows,
        preprocessing_duplicate_summary,
        duplicate_summary,
        names,
    )
    print(f"report written to: {args.output}")
    print(f"summary: {args.output / 'data_audit_summary.md'}")


if __name__ == "__main__":
    main()
