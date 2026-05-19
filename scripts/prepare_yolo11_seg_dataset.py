#!/usr/bin/env python3
"""Prepare the 5-group fire/smoke segmentation dataset for YOLO11-seg.

Input:
  - Five Roboflow COCO-seg exports, either flat or with train/valid/test folders.

Output:
  - YOLO segmentation dataset:
      output/images/{train,valid,test}
      output/labels/{train,valid,test}
      output/data.yaml
  - COCO mirrors and metadata for auditing.

This script intentionally splits original images before any slicing. That keeps
all crops from a source image in train only and prevents train/valid/test leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ANNOTATION_FILE_NAMES = (
    "_annotations.coco.json",
    "_annotations.coco-segmentation.json",
    "_annotations.json",
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPLITS = ("train", "valid", "test")

GROUPS: dict[str, dict[str, Any]] = {
    "01": {
        "name": "01_positive_standard",
        "aliases": ("01_positive_standard", "01_Positive_Standard"),
        "ratios": (0.87, 0.07, 0.06),
        "slice": None,
    },
    "02": {
        "name": "02_Alley_Context",
        "aliases": ("02_Alley_Context", "02_alley_context"),
        "ratios": (0.80, 0.10, 0.10),
        "slice": {
            "size": 640,
            "overlap": 0.40,
            "min_area_ratio": 0.10,
            "ignore_negative_samples": False,
            "max_empty_slices_per_source": 6,
        },
    },
    "03": {
        "name": "03_Negative_Hard_Samples",
        "aliases": ("03_Negative_Hard_Samples", "03_negative_hard_samples"),
        "ratios": (100 / 140, 20 / 140, 20 / 140),
        "slice": {
            "size": 640,
            "overlap": 0.40,
            "min_area_ratio": 0.10,
            "ignore_negative_samples": False,
            "max_empty_slices_per_source": 8,
        },
    },
    "04": {
        "name": "04_SAHI_Small_Objects",
        "aliases": ("04_SAHI_Small_Objects", "04_sahi_small_objects"),
        "ratios": (70 / 110, 20 / 110, 20 / 110),
        "slice": {
            "size": 416,
            "overlap": 0.50,
            "min_area_ratio": 0.05,
            "ignore_negative_samples": True,
            "max_empty_slices_per_source": 0,
        },
    },
    "05": {
        "name": "05_Ambient_Context_Null",
        "aliases": (
            "05_Ambient_Context_Null",
            "05_ambient_context_null",
            "05_Real_Situation",
        ),
        "ratios": (0.87, 0.07, 0.06),
        "slice": None,
    },
}


@dataclass(frozen=True)
class SourceImage:
    group: str
    source_index: int
    image_path: Path
    original_file_name: str
    width: int
    height: int
    annotations: tuple[dict[str, Any], ...]
    source_key: str
    content_hash: str

    @property
    def stable_stem(self) -> str:
        return f"g{self.group}_{self.source_index:06d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a balanced YOLO11-seg dataset from 5 COCO-seg exports."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("datasets/raw"),
        help="Folder containing five group export folders or zip files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/fire_vn_yolo11seg_v1"),
        help="Output dataset folder.",
    )
    parser.add_argument("--group-01", type=Path, help="Explicit folder or zip path for group 01.")
    parser.add_argument("--group-02", type=Path, help="Explicit folder or zip path for group 02.")
    parser.add_argument("--group-03", type=Path, help="Explicit folder or zip path for group 03.")
    parser.add_argument("--group-04", type=Path, help="Explicit folder or zip path for group 04.")
    parser.add_argument("--group-05", type=Path, help="Explicit folder or zip path for group 05.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument(
        "--skip-slicing",
        action="store_true",
        help="Only split/merge and convert; do not create SAHI training crops.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate source images instead of removing them before split.",
    )
    parser.add_argument(
        "--dedupe-near-duplicates",
        action="store_true",
        help="Also remove perceptual near-duplicate source images before split.",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=int,
        default=2,
        help="Maximum 64-bit dHash Hamming distance for --dedupe-near-duplicates.",
    )
    parser.add_argument(
        "--make-zip",
        action="store_true",
        help="Create output.zip after dataset preparation.",
    )
    parser.add_argument(
        "--hf-repo-id",
        help="Optional Hugging Face dataset repo id, e.g. username/fire-vn-yolo11seg.",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Create the Hugging Face dataset repo as private when uploading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output folder first if it already exists.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def ensure_clean_output(output: Path, overwrite: bool) -> None:
    resolved = output.resolve()
    cwd = Path.cwd().resolve()
    if overwrite and output.exists() and resolved in {cwd, cwd.parent, Path(resolved.anchor)}:
        fail(f"Refusing to overwrite unsafe output path: {resolved}")
    if output.exists():
        if not overwrite:
            fail(f"Output already exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output / "coco" / split).mkdir(parents=True, exist_ok=True)
    (output / "metadata").mkdir(parents=True, exist_ok=True)
    (output / "_work").mkdir(parents=True, exist_ok=True)


def find_group_dirs(args: argparse.Namespace, extract_root: Path) -> dict[str, Path]:
    explicit = {
        "01": args.group_01,
        "02": args.group_02,
        "03": args.group_03,
        "04": args.group_04,
        "05": args.group_05,
    }
    group_dirs: dict[str, Path] = {}

    for group, explicit_path in explicit.items():
        if explicit_path:
            path = explicit_path.resolve()
            if not path.exists():
                fail(f"Group {group} path does not exist: {path}")
            group_dirs[group] = prepare_group_input(group, path, extract_root)

    if len(group_dirs) == len(GROUPS):
        return group_dirs

    if not args.raw_root.exists():
        fail(
            f"Raw root not found: {args.raw_root}. Create it or pass --group-01 ... --group-05."
        )

    children = [
        p
        for p in args.raw_root.iterdir()
        if p.is_dir() or p.suffix.lower() == ".zip"
    ]
    for group, cfg in GROUPS.items():
        if group in group_dirs:
            continue
        candidates = [
            p
            for p in children
            if p.name.startswith(group) or p.name in set(cfg["aliases"])
        ]
        if len(candidates) != 1:
            names = ", ".join(p.name for p in candidates) or "none"
            fail(
                f"Expected exactly one folder or zip for group {group} under {args.raw_root}; found {names}."
            )
        group_dirs[group] = prepare_group_input(group, candidates[0].resolve(), extract_root)

    return group_dirs


def prepare_group_input(group: str, path: Path, extract_root: Path) -> Path:
    if path.is_dir():
        return path
    if path.suffix.lower() != ".zip":
        fail(f"Group {group} input must be a folder or .zip file: {path}")
    return extract_group_zip(group, path, extract_root / f"group_{group}")


def zip_text(zf: zipfile.ZipFile, member: str) -> str:
    return zf.read(member).decode("utf-8-sig")


def zip_norm(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def zip_parent(path: str) -> str:
    normalized = zip_norm(path)
    if "/" not in normalized:
        return ""
    return normalized.rsplit("/", 1)[0]


def split_from_zip_path(path: str) -> str | None:
    parts = zip_norm(path).split("/")
    for part in reversed(parts):
        if part in SPLITS:
            return part
    return None


def find_zip_image_member(
    members: set[str], base_dir: str, file_name: str, basename_index: dict[str, list[str]]
) -> str | None:
    normalized_name = zip_norm(file_name)
    candidates = [
        f"{base_dir}/{normalized_name}" if base_dir else normalized_name,
        f"{base_dir}/images/{Path(normalized_name).name}" if base_dir else f"images/{Path(normalized_name).name}",
        normalized_name,
        Path(normalized_name).name,
    ]
    for candidate in candidates:
        candidate = zip_norm(candidate)
        if candidate in members:
            return candidate

    basename = Path(normalized_name).name
    matches = basename_index.get(basename, [])
    if len(matches) == 1:
        return matches[0]
    base_matches = [m for m in matches if m.startswith(base_dir + "/")] if base_dir else matches
    if len(base_matches) == 1:
        return base_matches[0]
    return None


def extract_group_zip(group: str, zip_path: Path, output_dir: Path) -> Path:
    """Extract a Roboflow COCO zip with short filenames to avoid Windows MAX_PATH."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        all_members = [zip_norm(info.filename) for info in zf.infolist() if not info.is_dir()]
        member_set = set(all_members)
        basename_index: dict[str, list[str]] = defaultdict(list)
        for member in all_members:
            basename_index[Path(member).name].append(member)

        annotation_members = [
            member
            for member in all_members
            if Path(member).name in ANNOTATION_FILE_NAMES
        ]
        if not annotation_members:
            fail(f"No COCO annotation file found inside zip for group {group}: {zip_path}")

        used_output_dirs: set[Path] = set()
        for ann_idx, annotation_member in enumerate(sorted(annotation_members), start=1):
            coco = json.loads(zip_text(zf, annotation_member))
            base_dir = zip_parent(annotation_member)
            split = split_from_zip_path(annotation_member)
            target_dir = output_dir / split if split else output_dir
            if target_dir in used_output_dirs:
                target_dir = output_dir / f"part_{ann_idx:02d}"
            used_output_dirs.add(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)

            for image_idx, image in enumerate(coco.get("images", []), start=1):
                file_name = str(image.get("file_name", ""))
                member = find_zip_image_member(member_set, base_dir, file_name, basename_index)
                if member is None:
                    print(
                        f"WARNING: missing image in {zip_path}: {file_name}",
                        file=sys.stderr,
                    )
                    continue
                suffix = Path(file_name).suffix.lower()
                if suffix not in IMAGE_EXTS:
                    suffix = Path(member).suffix.lower()
                if suffix not in IMAGE_EXTS:
                    suffix = ".jpg"
                short_name = f"img_{ann_idx:02d}_{image_idx:06d}{suffix}"
                with zf.open(member) as src, (target_dir / short_name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                image["original_file_name"] = file_name
                image["file_name"] = short_name

            dump_json(target_dir / "_annotations.coco.json", coco)

    print(f"group {group}: extracted zip with short filenames -> {output_dir}")
    return output_dir


def find_annotation_files(group_dir: Path) -> list[tuple[Path, Path]]:
    """Return (annotation_json, image_dir) pairs for flat or split Roboflow exports."""
    pairs: list[tuple[Path, Path]] = []

    for name in ANNOTATION_FILE_NAMES:
        annotation_path = group_dir / name
        if annotation_path.exists():
            pairs.append((annotation_path, group_dir))

    for split in SPLITS:
        split_dir = group_dir / split
        if not split_dir.is_dir():
            continue
        for name in ANNOTATION_FILE_NAMES:
            annotation_path = split_dir / name
            if annotation_path.exists():
                pairs.append((annotation_path, split_dir))

    # Roboflow sometimes exports images in a nested "images" folder.
    normalized_pairs: list[tuple[Path, Path]] = []
    for annotation_path, image_dir in pairs:
        if (image_dir / "images").is_dir():
            normalized_pairs.append((annotation_path, image_dir / "images"))
        else:
            normalized_pairs.append((annotation_path, image_dir))

    if not normalized_pairs:
        fail(
            f"No COCO annotation file found in {group_dir}. Expected _annotations.coco.json."
        )
    return normalized_pairs


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def image_path_for(image_dir: Path, file_name: str) -> Path | None:
    candidates = [
        image_dir / file_name,
        image_dir / Path(file_name).name,
        image_dir.parent / file_name,
        image_dir.parent / Path(file_name).name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def source_hash(group: str, annotation_path: Path, image_id: Any, file_name: str) -> str:
    raw = f"{group}|{annotation_path}|{image_id}|{file_name}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_sources(group: str, group_dir: Path) -> tuple[list[SourceImage], list[dict[str, Any]]]:
    sources: list[SourceImage] = []
    categories_by_name: dict[str, dict[str, Any]] = {}
    source_index = 0

    for annotation_path, image_dir in find_annotation_files(group_dir):
        coco = load_json(annotation_path)
        category_name_by_id: dict[int, str] = {}
        for category in coco.get("categories", []):
            name = str(category.get("name", "")).strip()
            if name:
                categories_by_name.setdefault(name, category)
                category_name_by_id[int(category["id"])] = name

        annotations_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            ann_copy = dict(ann)
            ann_copy["_category_name"] = category_name_by_id.get(int(ann.get("category_id", -1)))
            annotations_by_image[ann.get("image_id")].append(ann_copy)

        for image in coco.get("images", []):
            file_name = image.get("file_name")
            if not file_name:
                continue
            resolved = image_path_for(image_dir, file_name)
            if resolved is None:
                print(
                    f"WARNING: missing image for {annotation_path}: {file_name}",
                    file=sys.stderr,
                )
                continue

            width = int(image.get("width") or 0)
            height = int(image.get("height") or 0)
            if width <= 0 or height <= 0:
                try:
                    from PIL import Image

                    with Image.open(resolved) as im:
                        width, height = im.size
                except Exception as exc:  # pragma: no cover - depends on user files.
                    print(
                        f"WARNING: cannot read image size for {resolved}: {exc}",
                        file=sys.stderr,
                    )
                    continue

            source_index += 1
            sources.append(
                SourceImage(
                    group=group,
                    source_index=source_index,
                    image_path=resolved,
                    original_file_name=str(file_name),
                    width=width,
                    height=height,
                    annotations=tuple(annotations_by_image.get(image.get("id"), [])),
                    source_key=source_hash(group, annotation_path, image.get("id"), str(file_name)),
                    content_hash=file_sha1(resolved),
                )
            )

    return sources, sorted(categories_by_name.values(), key=lambda c: str(c.get("name", "")))


def deduplicate_sources(
    sources_by_group: dict[str, list[SourceImage]],
    output: Path,
    near_duplicate_threshold: int | None,
) -> dict[str, list[SourceImage]]:
    all_sources = [source for sources in sources_by_group.values() for source in sources]
    duplicate_sets = find_duplicate_sets(all_sources, near_duplicate_threshold)
    if not duplicate_sets:
        write_duplicate_report(output, [])
        if near_duplicate_threshold is None:
            print("dedupe: no exact duplicate source images found")
        else:
            print("dedupe: no exact or near-duplicate source images found")
        return sources_by_group

    keep_keys: set[str] = set()
    report_rows: list[dict[str, Any]] = []
    removed = 0
    for duplicate_sources in duplicate_sets:
        keep = choose_duplicate_source(duplicate_sources)
        keep_keys.add(keep.source_key)
        for source in sorted(
            duplicate_sources,
            key=lambda s: (s.source_key != keep.source_key, s.group, s.original_file_name, s.source_key),
        ):
            kept = source.source_key == keep.source_key
            if not kept:
                removed += 1
            report_rows.append(
                {
                    "content_hash": source.content_hash,
                    "perceptual_hash": image_dhash(source.image_path)
                    if near_duplicate_threshold is not None
                    else "",
                    "action": "kept" if kept else "removed",
                    "group": source.group,
                    "source_key": source.source_key,
                    "original_file_name": source.original_file_name,
                    "path": str(source.image_path),
                    "annotations": len(source.annotations),
                    "width": source.width,
                    "height": source.height,
                    "kept_source_key": keep.source_key,
                }
            )

    deduped: dict[str, list[SourceImage]] = {}
    duplicate_keys = {source.source_key for sources in duplicate_sets for source in sources}
    for group, sources in sources_by_group.items():
        deduped[group] = [
            source
            for source in sources
            if source.source_key not in duplicate_keys or source.source_key in keep_keys
        ]

    write_duplicate_report(output, report_rows)
    if near_duplicate_threshold is None:
        print(f"dedupe: removed {removed} exact duplicate source images")
    else:
        print(
            "dedupe: removed "
            f"{removed} exact/near-duplicate source images "
            f"(dHash threshold <= {near_duplicate_threshold})"
        )
    return deduped


def find_duplicate_sets(
    sources: list[SourceImage], near_duplicate_threshold: int | None
) -> list[list[SourceImage]]:
    parent = {source.source_key: source.source_key for source in sources}

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

    by_content_hash: dict[str, list[SourceImage]] = defaultdict(list)
    for source in sources:
        by_content_hash[source.content_hash].append(source)
    for duplicates in by_content_hash.values():
        for source in duplicates[1:]:
            union(duplicates[0].source_key, source.source_key)

    if near_duplicate_threshold is not None:
        hashes = [(source, int(image_dhash(source.image_path), 16)) for source in sources]
        for idx, (source_a, hash_a) in enumerate(hashes):
            for source_b, hash_b in hashes[idx + 1 :]:
                if (hash_a ^ hash_b).bit_count() <= near_duplicate_threshold:
                    union(source_a.source_key, source_b.source_key)

    by_root: dict[str, list[SourceImage]] = defaultdict(list)
    for source in sources:
        by_root[find(source.source_key)].append(source)
    return [group for group in by_root.values() if len(group) > 1]


def image_dhash(path: Path) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        fail("Pillow is required for perceptual dedupe. Install it with: pip install pillow")
        raise exc

    with Image.open(path) as image:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        pixels = list(image.convert("L").resize((9, 8), resampling).getdata())

    value = 0
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            value = (value << 1) | int(left > right)
    return f"{value:016x}"


def choose_duplicate_source(sources: list[SourceImage]) -> SourceImage:
    return sorted(
        sources,
        key=lambda s: (
            -len(s.annotations),
            s.group,
            s.original_file_name.lower(),
            s.source_key,
        ),
    )[0]


def write_duplicate_report(output: Path, rows: list[dict[str, Any]]) -> None:
    path = output / "metadata" / "duplicate_sources.csv"
    fieldnames = [
        "content_hash",
        "perceptual_hash",
        "action",
        "group",
        "source_key",
        "original_file_name",
        "path",
        "annotations",
        "width",
        "height",
        "kept_source_key",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_counts(n: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    if n <= 0:
        return {"train": 0, "valid": 0, "test": 0}
    if n == 1:
        return {"train": 1, "valid": 0, "test": 0}
    if n == 2:
        return {"train": 1, "valid": 1, "test": 0}

    valid = max(1, int(round(n * ratios[1])))
    test = max(1, int(round(n * ratios[2])))
    train = n - valid - test

    if train < 1:
        train = 1
        if valid >= test and valid > 1:
            valid -= 1
        elif test > 1:
            test -= 1

    return {"train": train, "valid": valid, "test": test}


def assign_splits(
    sources_by_group: dict[str, list[SourceImage]], seed: int
) -> dict[str, list[SourceImage]]:
    assigned = {split: [] for split in SPLITS}
    rng = random.Random(seed)

    for group, sources in sources_by_group.items():
        items = sorted(sources, key=lambda s: (s.original_file_name, s.source_key))
        rng.shuffle(items)
        counts = split_counts(len(items), GROUPS[group]["ratios"])

        train_end = counts["train"]
        valid_end = train_end + counts["valid"]
        assigned["train"].extend(items[:train_end])
        assigned["valid"].extend(items[train_end:valid_end])
        assigned["test"].extend(items[valid_end:])

    for split in SPLITS:
        assigned[split].sort(key=lambda s: (s.group, s.source_index))
    return assigned


def category_maps(categories: Iterable[dict[str, Any]]) -> tuple[dict[str, int], list[str]]:
    names: list[str] = []
    for category in categories:
        name = str(category.get("name", "")).strip()
        if not name:
            continue
        if name not in names:
            names.append(name)

    if not names:
        fail("No categories found. COCO-seg exports must include categories.")

    name_to_yolo = {name: yolo_id for yolo_id, name in enumerate(names)}
    return name_to_yolo, names


def observed_categories(sources_by_group: dict[str, list[SourceImage]]) -> list[dict[str, str]]:
    names: list[str] = []
    for group in sorted(sources_by_group):
        for source in sources_by_group[group]:
            for ann in source.annotations:
                name = ann.get("_category_name")
                if name and name not in names:
                    names.append(str(name))
    return [{"name": name} for name in names]


def copy_image(source: SourceImage, destination_dir: Path) -> str:
    suffix = source.image_path.suffix.lower()
    if suffix not in IMAGE_EXTS:
        suffix = ".jpg"
    file_name = f"{source.stable_stem}{suffix}"
    shutil.copy2(source.image_path, destination_dir / file_name)
    return file_name


def remap_annotation(
    ann: dict[str, Any], image_id: int, category_name_map: dict[str, int]
) -> dict[str, Any] | None:
    category_name = ann.get("_category_name")
    if not category_name or category_name not in category_name_map:
        return None
    new_ann = dict(ann)
    new_ann["image_id"] = image_id
    new_ann["category_id"] = category_name_map[category_name]
    new_ann.pop("_category_name", None)
    new_ann.pop("id", None)
    return new_ann


def build_split_coco(
    split: str,
    sources: list[SourceImage],
    output: Path,
    category_name_map: dict[str, int],
    category_names: list[str],
    include_groups: set[str] | None = None,
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    image_id = 1

    image_dir = output / "coco" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        if include_groups is not None and source.group not in include_groups:
            continue
        file_name = copy_image(source, image_dir)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": source.width,
                "height": source.height,
                "group": source.group,
                "source_key": source.source_key,
                "original_file_name": source.original_file_name,
            }
        )
        for ann in source.annotations:
            new_ann = remap_annotation(ann, image_id, category_name_map)
            if new_ann is None:
                continue
            new_ann["id"] = ann_id
            annotations.append(new_ann)
            ann_id += 1
        image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": idx, "name": name, "supercategory": "fire_smoke"}
            for idx, name in enumerate(category_names)
        ],
    }
    dump_json(image_dir / "_annotations.coco.json", coco)
    return coco


def run_sahi_slice(
    group: str,
    split_sources: list[SourceImage],
    output: Path,
    category_name_map: dict[str, int],
    category_names: list[str],
) -> dict[str, Any]:
    cfg = GROUPS[group]["slice"]
    if cfg is None:
        fail(f"Group {group} has no slicing config.")

    work_dir = output / "_work" / f"group_{group}_train"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    group_coco = build_split_coco(
        split=f"_work/group_{group}_train_input",
        sources=split_sources,
        output=output,
        category_name_map=category_name_map,
        category_names=category_names,
        include_groups={group},
    )
    input_dir = output / "coco" / f"_work/group_{group}_train_input"
    input_json = input_dir / "_annotations.coco.json"
    sliced_dir = work_dir / "sliced"
    sliced_dir.mkdir(parents=True, exist_ok=True)

    if not group_coco["images"]:
        return {"images": [], "annotations": [], "categories": group_coco["categories"]}

    sliced_coco = slice_coco_locally(
        coco=group_coco,
        image_dir=input_dir,
        output_dir=sliced_dir,
        slice_size=cfg["size"],
        overlap=cfg["overlap"],
        min_area_ratio=cfg["min_area_ratio"],
        ignore_negative_samples=cfg["ignore_negative_samples"],
    )
    dump_json(sliced_dir / "_annotations.coco.json", sliced_coco)
    return cap_empty_slices(group, sliced_coco, sliced_dir, cfg["max_empty_slices_per_source"])


def slice_coco_locally(
    coco: dict[str, Any],
    image_dir: Path,
    output_dir: Path,
    slice_size: int,
    overlap: float,
    min_area_ratio: float,
    ignore_negative_samples: bool,
) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError:
        fail("Pillow is required for local slicing. Install it with: pip install pillow")

    output_dir.mkdir(parents=True, exist_ok=True)
    anns_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann.get("image_id")].append(ann)

    sliced_images: list[dict[str, Any]] = []
    sliced_annotations: list[dict[str, Any]] = []
    next_image_id = 1
    next_ann_id = 1
    step = max(1, int(round(slice_size * (1.0 - overlap))))

    for image in coco.get("images", []):
        src_path = image_dir / str(image["file_name"])
        if not src_path.exists():
            print(f"WARNING: missing image for slicing: {src_path}", file=sys.stderr)
            continue
        width = int(image["width"])
        height = int(image["height"])
        windows = slice_windows(width, height, slice_size, step)

        with Image.open(src_path) as img:
            for window_idx, (x0, y0, x1, y1) in enumerate(windows):
                window_annotations: list[dict[str, Any]] = []
                for ann in anns_by_image.get(image["id"], []):
                    clipped_ann = clip_annotation_to_window(
                        ann=ann,
                        window=(x0, y0, x1, y1),
                        min_area_ratio=min_area_ratio,
                    )
                    if clipped_ann is not None:
                        clipped_ann["id"] = next_ann_id + len(window_annotations)
                        clipped_ann["image_id"] = next_image_id
                        window_annotations.append(clipped_ann)

                if ignore_negative_samples and not window_annotations:
                    continue

                out_name = f"{Path(image['file_name']).stem}_slice_{window_idx:04d}{src_path.suffix.lower()}"
                img.crop((x0, y0, x1, y1)).save(output_dir / out_name)
                sliced_images.append(
                    {
                        "id": next_image_id,
                        "file_name": out_name,
                        "width": x1 - x0,
                        "height": y1 - y0,
                    }
                )
                sliced_annotations.extend(window_annotations)
                next_image_id += 1
                next_ann_id += len(window_annotations)

    return {
        "images": sliced_images,
        "annotations": sliced_annotations,
        "categories": coco.get("categories", []),
    }


def slice_windows(width: int, height: int, slice_size: int, step: int) -> list[tuple[int, int, int, int]]:
    xs = axis_starts(width, slice_size, step)
    ys = axis_starts(height, slice_size, step)
    return [(x, y, min(x + slice_size, width), min(y + slice_size, height)) for y in ys for x in xs]


def axis_starts(length: int, slice_size: int, step: int) -> list[int]:
    if length <= slice_size:
        return [0]
    starts = list(range(0, max(1, length - slice_size + 1), step))
    last = length - slice_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def clip_annotation_to_window(
    ann: dict[str, Any], window: tuple[int, int, int, int], min_area_ratio: float
) -> dict[str, Any] | None:
    polygons = annotation_to_polygons(ann)
    if not polygons:
        return None

    x0, y0, x1, y1 = window
    clipped_polygons: list[list[float]] = []
    original_area = 0.0
    clipped_area = 0.0

    for polygon in polygons:
        original_area += polygon_area(polygon)
        clipped = clip_polygon_rect(polygon, x0, y0, x1, y1)
        if len(clipped) >= 6:
            area = polygon_area(clipped)
            if area > 1.0:
                clipped_area += area
                shifted: list[float] = []
                for px, py in zip(clipped[0::2], clipped[1::2]):
                    shifted.extend([px - x0, py - y0])
                clipped_polygons.append(shifted)

    if not clipped_polygons or original_area <= 0:
        return None
    if clipped_area / original_area < min_area_ratio:
        return None

    clipped_width = x1 - x0
    clipped_height = y1 - y0
    bbox = bbox_from_polygons(clipped_polygons, clipped_width, clipped_height)
    if bbox[2] <= 1 or bbox[3] <= 1:
        return None

    new_ann = dict(ann)
    new_ann["segmentation"] = clipped_polygons
    new_ann["bbox"] = bbox
    new_ann["area"] = clipped_area
    return new_ann


def annotation_to_polygons(ann: dict[str, Any]) -> list[list[float]]:
    segmentation = ann.get("segmentation")
    if isinstance(segmentation, dict):
        bbox = ann.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            x, y, w, h = [float(v) for v in bbox[:4]]
            if w > 1 and h > 1:
                return [[x, y, x + w, y, x + w, y + h, x, y + h]]
        return []
    return annotation_polygons(segmentation)


def annotation_polygons(segmentation: Any) -> list[list[float]]:
    if not isinstance(segmentation, list):
        return []
    if all(isinstance(value, (int, float)) for value in segmentation):
        return [[float(v) for v in segmentation]]
    return [
        [float(v) for v in polygon]
        for polygon in segmentation
        if isinstance(polygon, list) and len(polygon) >= 6
    ]


def clip_polygon_rect(
    polygon: list[float], x0: float, y0: float, x1: float, y1: float
) -> list[float]:
    points = [(float(x), float(y)) for x, y in zip(polygon[0::2], polygon[1::2])]
    for edge in ("left", "right", "top", "bottom"):
        points = clip_points_edge(points, edge, x0, y0, x1, y1)
        if not points:
            return []
    flattened: list[float] = []
    for x, y in points:
        flattened.extend([min(max(x, x0), x1), min(max(y, y0), y1)])
    return flattened


def clip_points_edge(
    points: list[tuple[float, float]],
    edge: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> list[tuple[float, float]]:
    if not points:
        return []
    output: list[tuple[float, float]] = []
    previous = points[-1]
    previous_inside = point_inside_edge(previous, edge, x0, y0, x1, y1)
    for current in points:
        current_inside = point_inside_edge(current, edge, x0, y0, x1, y1)
        if current_inside:
            if not previous_inside:
                output.append(edge_intersection(previous, current, edge, x0, y0, x1, y1))
            output.append(current)
        elif previous_inside:
            output.append(edge_intersection(previous, current, edge, x0, y0, x1, y1))
        previous = current
        previous_inside = current_inside
    return output


def point_inside_edge(
    point: tuple[float, float], edge: str, x0: float, y0: float, x1: float, y1: float
) -> bool:
    x, y = point
    if edge == "left":
        return x >= x0
    if edge == "right":
        return x <= x1
    if edge == "top":
        return y >= y0
    if edge == "bottom":
        return y <= y1
    raise ValueError(edge)


def edge_intersection(
    a: tuple[float, float],
    b: tuple[float, float],
    edge: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> tuple[float, float]:
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    if edge in {"left", "right"}:
        x = x0 if edge == "left" else x1
        if abs(dx) < 1e-9:
            return (x, ay)
        t = (x - ax) / dx
        return (x, ay + t * dy)
    y = y0 if edge == "top" else y1
    if abs(dy) < 1e-9:
        return (ax, y)
    t = (y - ay) / dy
    return (ax + t * dx, y)


def bbox_from_polygons(
    polygons: list[list[float]], width: int, height: int
) -> list[float]:
    xs: list[float] = []
    ys: list[float] = []
    for polygon in polygons:
        xs.extend(polygon[0::2])
        ys.extend(polygon[1::2])
    min_x = min(max(min(xs), 0.0), float(width))
    max_x = min(max(max(xs), 0.0), float(width))
    min_y = min(max(min(ys), 0.0), float(height))
    max_y = min(max(max(ys), 0.0), float(height))
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def source_stem_from_sahi_file(file_name: str) -> str | None:
    match = re.match(r"^(g\d{2}_\d{6})", Path(file_name).stem)
    return match.group(1) if match else None


def cap_empty_slices(
    group: str, coco: dict[str, Any], image_dir: Path, max_empty_per_source: int | None
) -> dict[str, Any]:
    if max_empty_per_source is None:
        return coco

    anns_by_image = Counter(ann.get("image_id") for ann in coco.get("annotations", []))
    keep_image_ids: set[int] = set()
    empty_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for image in sorted(coco.get("images", []), key=lambda img: img.get("file_name", "")):
        image_id = image.get("id")
        if anns_by_image[image_id] > 0:
            keep_image_ids.add(image_id)
            continue
        source_stem = source_stem_from_sahi_file(str(image.get("file_name", "")))
        empty_by_source[source_stem or "unknown"].append(image)

    for images in empty_by_source.values():
        for image in images[:max_empty_per_source]:
            keep_image_ids.add(image.get("id"))
        for image in images[max_empty_per_source:]:
            file_path = image_dir / str(image.get("file_name", ""))
            if file_path.exists():
                file_path.unlink()

    filtered = dict(coco)
    filtered["images"] = [
        image for image in coco.get("images", []) if image.get("id") in keep_image_ids
    ]
    filtered["annotations"] = [
        ann for ann in coco.get("annotations", []) if ann.get("image_id") in keep_image_ids
    ]
    removed = len(coco.get("images", [])) - len(filtered["images"])
    print(f"group {group}: removed {removed} extra empty training slices")
    return filtered


def merge_cocos(cocos: list[tuple[dict[str, Any], Path]], output_json: Path) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    categories = cocos[0][0].get("categories", []) if cocos else []
    next_image_id = 1
    next_ann_id = 1

    destination_dir = output_json.parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    existing_names: set[str] = set()

    for coco, image_dir in cocos:
        image_id_map: dict[Any, int] = {}
        for image in coco.get("images", []):
            src_name = str(image["file_name"])
            src_path = image_dir / src_name
            if not src_path.exists():
                print(f"WARNING: missing generated image: {src_path}", file=sys.stderr)
                continue
            new_name = unique_name(src_name, existing_names)
            existing_names.add(new_name)
            shutil.copy2(src_path, destination_dir / new_name)

            new_image = dict(image)
            image_id_map[image["id"]] = next_image_id
            new_image["id"] = next_image_id
            new_image["file_name"] = new_name
            images.append(new_image)
            next_image_id += 1

        for ann in coco.get("annotations", []):
            if ann.get("image_id") not in image_id_map:
                continue
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = image_id_map[ann["image_id"]]
            annotations.append(new_ann)
            next_ann_id += 1

    merged = {"images": images, "annotations": annotations, "categories": categories}
    dump_json(output_json, merged)
    return merged


def unique_name(file_name: str, existing: set[str]) -> str:
    if file_name not in existing:
        return file_name
    path = Path(file_name)
    counter = 1
    while True:
        candidate = f"{path.stem}_{counter}{path.suffix}"
        if candidate not in existing:
            return candidate
        counter += 1


def normalize(value: float, size: int) -> float:
    if size <= 0:
        return 0.0
    return min(max(value / size, 0.0), 1.0)


def polygon_area(points: list[float]) -> float:
    if len(points) < 6:
        return 0.0
    area = 0.0
    pairs = list(zip(points[0::2], points[1::2]))
    for idx, (x1, y1) in enumerate(pairs):
        x2, y2 = pairs[(idx + 1) % len(pairs)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def coco_to_yolo_labels(coco: dict[str, Any], output: Path, split: str) -> Counter:
    anns_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    for ann in coco.get("annotations", []):
        anns_by_image[ann.get("image_id")].append(ann)

    labels_dir = output / "labels" / split
    images_dir = output / "images" / split
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    for image in coco.get("images", []):
        src_path = output / "coco" / split / image["file_name"]
        dst_path = images_dir / image["file_name"]
        shutil.copy2(src_path, dst_path)

        label_path = labels_dir / f"{Path(image['file_name']).stem}.txt"
        lines: list[str] = []
        for ann in anns_by_image.get(image["id"], []):
            polygons = annotation_to_polygons(ann)
            if not polygons:
                skipped["rle_or_missing_segmentation"] += 1
                continue

            # YOLO labels do not support separate rings in one object. Keeping the
            # largest polygon is stable and avoids duplicating one instance.
            polygon = max(polygons, key=polygon_area)
            coords: list[str] = []
            for x, y in zip(polygon[0::2], polygon[1::2]):
                coords.append(f"{normalize(x, int(image['width'])):.6f}")
                coords.append(f"{normalize(y, int(image['height'])):.6f}")
            lines.append(f"{int(ann['category_id'])} " + " ".join(coords))

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return skipped


def write_data_yaml(output: Path, category_names: list[str]) -> None:
    lines = [
        "path: .",
        "train: images/train",
        "val: images/valid",
        "test: images/test",
        "names:",
    ]
    for idx, name in enumerate(category_names):
        lines.append(f"  {idx}: {json.dumps(name, ensure_ascii=False)}")
    with (output / "data.yaml").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_manifest(
    output: Path, assigned: dict[str, list[SourceImage]], final_cocos: dict[str, dict[str, Any]]
) -> None:
    manifest_path = output / "metadata" / "source_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "group",
                "source_key",
                "content_hash",
                "original_file_name",
                "stable_stem",
                "path",
            ]
        )
        for split in SPLITS:
            for source in assigned[split]:
                writer.writerow(
                    [
                        split,
                        source.group,
                        source.source_key,
                        source.content_hash,
                        source.original_file_name,
                        source.stable_stem,
                        str(source.image_path),
                    ]
                )

    summary_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        group_counts = Counter(source.group for source in assigned[split])
        image_count = len(final_cocos[split].get("images", []))
        ann_count = len(final_cocos[split].get("annotations", []))
        for group in GROUPS:
            summary_rows.append(
                {
                    "split": split,
                    "group": group,
                    "source_images": group_counts[group],
                    "final_images_after_slicing": count_final_images(final_cocos[split], group),
                    "annotations": count_final_annotations(final_cocos[split], group),
                    "total_final_images_in_split": image_count,
                    "total_annotations_in_split": ann_count,
                }
            )

    summary_path = output / "metadata" / "split_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def count_final_images(coco: dict[str, Any], group: str) -> int:
    prefix = f"g{group}_"
    return sum(1 for image in coco.get("images", []) if str(image.get("file_name", "")).startswith(prefix))


def count_final_annotations(coco: dict[str, Any], group: str) -> int:
    image_ids = {
        image["id"]
        for image in coco.get("images", [])
        if str(image.get("file_name", "")).startswith(f"g{group}_")
    }
    return sum(1 for ann in coco.get("annotations", []) if ann.get("image_id") in image_ids)


def validate_no_leakage(assigned: dict[str, list[SourceImage]]) -> None:
    seen: dict[str, str] = {}
    for split in SPLITS:
        for source in assigned[split]:
            previous = seen.get(source.content_hash)
            if previous and previous != split:
                fail(
                    f"Leakage detected: {source.original_file_name} appears in {previous} and {split}."
                )
            seen[source.content_hash] = split


def validate_all_groups_present(assigned: dict[str, list[SourceImage]]) -> None:
    missing: list[str] = []
    for split in SPLITS:
        counts = Counter(source.group for source in assigned[split])
        for group in GROUPS:
            if counts[group] == 0:
                missing.append(f"{split}:{group}")
    if missing:
        fail(
            "Every split must contain all 5 groups. Missing groups: " + ", ".join(missing)
        )


def make_zip(output: Path) -> Path:
    archive_base = output.with_suffix("")
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", output))
    return archive_path


def upload_to_hf(output: Path, repo_id: str, private: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        fail("huggingface_hub is required for upload. Install it with: pip install huggingface_hub")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(folder_path=str(output), repo_id=repo_id, repo_type="dataset")
    else:
        api.upload_folder(folder_path=str(output), repo_id=repo_id, repo_type="dataset")


def main() -> None:
    args = parse_args()
    if args.near_duplicate_threshold < 0:
        fail("--near-duplicate-threshold must be >= 0")
    ensure_clean_output(args.output, args.overwrite)
    group_dirs = find_group_dirs(args, args.output / "_raw_from_zip")

    sources_by_group: dict[str, list[SourceImage]] = {}
    for group, group_dir in group_dirs.items():
        sources, _categories = collect_sources(group, group_dir)
        if not sources:
            fail(f"Group {group} has no usable images: {group_dir}")
        sources_by_group[group] = sources
        print(f"group {group}: loaded {len(sources)} source images from {group_dir}")

    if not args.keep_duplicates:
        near_threshold = (
            args.near_duplicate_threshold if args.dedupe_near_duplicates else None
        )
        sources_by_group = deduplicate_sources(
            sources_by_group, args.output, near_threshold
        )
        for group, sources in sources_by_group.items():
            print(f"group {group}: kept {len(sources)} source images after dedupe")
    else:
        write_duplicate_report(args.output, [])
        print("dedupe: skipped because --keep-duplicates was provided")

    category_name_map, category_names = category_maps(observed_categories(sources_by_group))
    assigned = assign_splits(sources_by_group, args.seed)
    validate_no_leakage(assigned)
    validate_all_groups_present(assigned)

    final_cocos: dict[str, dict[str, Any]] = {}
    skipped_total = Counter()

    for split in SPLITS:
        base_coco = build_split_coco(
            split=split,
            sources=assigned[split],
            output=args.output,
            category_name_map=category_name_map,
            category_names=category_names,
        )

        if split == "train" and not args.skip_slicing:
            coco_parts: list[tuple[dict[str, Any], Path]] = [
                (base_coco, args.output / "coco" / "train")
            ]
            for group, cfg in GROUPS.items():
                if cfg["slice"] is None:
                    continue
                sliced_coco = run_sahi_slice(
                    group=group,
                    split_sources=assigned["train"],
                    output=args.output,
                    category_name_map=category_name_map,
                    category_names=category_names,
                )
                sliced_dir = args.output / "_work" / f"group_{group}_train" / "sliced"
                coco_parts.append((sliced_coco, sliced_dir))

            merged_dir = args.output / "coco" / "train_merged"
            merged_coco = merge_cocos(coco_parts, merged_dir / "_annotations.coco.json")
            shutil.rmtree(args.output / "coco" / "train")
            merged_dir.rename(args.output / "coco" / "train")
            final_cocos[split] = merged_coco
        else:
            final_cocos[split] = base_coco

        skipped = coco_to_yolo_labels(final_cocos[split], args.output, split)
        skipped_total.update(skipped)

    write_data_yaml(args.output, category_names)
    write_manifest(args.output, assigned, final_cocos)

    if skipped_total:
        print(f"WARNING: skipped segmentation items during YOLO conversion: {dict(skipped_total)}")

    if args.make_zip:
        archive_path = make_zip(args.output)
        print(f"created zip: {archive_path}")

    if args.hf_repo_id:
        upload_to_hf(args.output, args.hf_repo_id, args.hf_private)
        print(f"uploaded dataset to HF: {args.hf_repo_id}")

    print("done")
    print(f"data yaml: {args.output / 'data.yaml'}")
    print(f"summary: {args.output / 'metadata' / 'split_summary.csv'}")


if __name__ == "__main__":
    main()
