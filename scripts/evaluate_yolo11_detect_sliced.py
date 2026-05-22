#!/usr/bin/env python3
"""Evaluate a YOLO detect checkpoint with sliced inference on full images.

This keeps the test labels in full-image coordinates. Images are sliced only at
inference time, predictions are mapped back to the original image, merged with
NMS, then scored against the original YOLO labels. This is meant for small-object
fire/smoke evaluation where training contains sliced images but valid/test are
kept as full images.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GROUP_NAMES = {
    "01": "positive_standard",
    "02": "alley_context",
    "03": "hard_negative",
    "04": "sahi_small_objects",
    "05": "ambient_null",
}


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    image_path: Path
    label_path: Path
    width: int
    height: int
    group: str


@dataclass(frozen=True)
class Detection:
    image_id: str
    score: float
    box: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sliced inference and compute full-image detection mAP."
    )
    parser.add_argument("--model", type=Path, required=True, help="YOLO checkpoint path.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Prepared YOLO detect dataset. Defaults to work/..., then datasets/....",
    )
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=1024, help="YOLO inference image size.")
    parser.add_argument("--device", default=0)
    parser.add_argument("--batch", type=int, default=16, help="Tile prediction batch size.")
    parser.add_argument("--conf", type=float, default=0.001, help="Low score threshold for AP curves.")
    parser.add_argument("--tile-iou", type=float, default=0.70, help="YOLO per-tile NMS IoU.")
    parser.add_argument("--merge-iou", type=float, default=0.55, help="Full-image merge NMS IoU.")
    parser.add_argument("--max-det", type=int, default=300, help="Max merged predictions per image.")
    parser.add_argument("--slice-size", type=int, default=768, help="Square tile size in original pixels.")
    parser.add_argument("--overlap", type=float, default=0.25, help="Tile overlap ratio in [0, 0.9).")
    parser.add_argument(
        "--include-full-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also predict on the original full image before merging with sliced predictions.",
    )
    parser.add_argument("--tta", action="store_true", help="Use Ultralytics test-time augmentation.")
    parser.add_argument("--report-conf", type=float, default=0.25, help="Confidence for P/R/F1 report.")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit for number of images.")
    parser.add_argument("--save-samples", type=int, default=24, help="Number of prediction images to draw.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/sliced_eval"),
        help="Directory for metrics, plots, and sample images.",
    )
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


def group_from_name(name: str) -> str:
    if len(name) >= 3 and name[0] == "g" and name[1:3].isdigit():
        return name[1:3]
    return "unknown"


def load_records(dataset: Path, split: str, limit: int) -> list[ImageRecord]:
    image_dir = dataset / "images" / split
    label_dir = dataset / "labels" / split
    if not image_dir.exists():
        raise SystemExit(f"Missing image directory: {image_dir}")
    if not label_dir.exists():
        raise SystemExit(f"Missing label directory: {label_dir}")

    records: list[ImageRecord] = []
    for image_path in image_files(image_dir):
        with Image.open(image_path) as image:
            width, height = image.size
        records.append(
            ImageRecord(
                image_id=image_path.stem,
                image_path=image_path,
                label_path=label_dir / f"{image_path.stem}.txt",
                width=width,
                height=height,
                group=group_from_name(image_path.name),
            )
        )
        if limit and len(records) >= limit:
            break
    return records


def load_yolo_boxes(label_path: Path, width: int, height: int) -> np.ndarray:
    boxes: list[list[float]] = []
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue
        try:
            x_center, y_center, box_width, box_height = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        x1 = (x_center - box_width / 2.0) * width
        y1 = (y_center - box_height / 2.0) * height
        x2 = (x_center + box_width / 2.0) * width
        y2 = (y_center + box_height / 2.0) * height
        boxes.append(
            [
                max(0.0, min(float(width), x1)),
                max(0.0, min(float(height), y1)),
                max(0.0, min(float(width), x2)),
                max(0.0, min(float(height), y2)),
            ]
        )
    return np.asarray(boxes, dtype=np.float32)


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return sorted(set(starts))


def tile_windows(width: int, height: int, tile_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    windows: list[tuple[int, int, int, int]] = []
    for y1 in tile_starts(height, tile_size, overlap):
        for x1 in tile_starts(width, tile_size, overlap):
            x2 = min(width, x1 + tile_size)
            y2 = min(height, y1 + tile_size)
            windows.append((x1, y1, x2, y2))
    return windows


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(width))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(height))
    return boxes


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = np.maximum(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_det: int) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0 and len(keep) < max_det:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        ious = box_iou_matrix(boxes[[current]], boxes[remaining])[0]
        order = remaining[ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def predict_record(model: Any, record: ImageRecord, args: argparse.Namespace) -> list[Detection]:
    image = Image.open(record.image_path).convert("RGB")
    arrays: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []

    if args.include_full_image:
        arrays.append(np.asarray(image))
        offsets.append((0, 0))

    for x1, y1, x2, y2 in tile_windows(record.width, record.height, args.slice_size, args.overlap):
        arrays.append(np.asarray(image.crop((x1, y1, x2, y2))))
        offsets.append((x1, y1))

    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for start in range(0, len(arrays), args.batch):
        batch_arrays = arrays[start : start + args.batch]
        batch_offsets = offsets[start : start + args.batch]
        results = model.predict(
            batch_arrays,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.tile_iou,
            device=args.device,
            augment=args.tta,
            max_det=args.max_det,
            verbose=False,
        )
        for result, (offset_x, offset_y) in zip(results, batch_offsets):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32, copy=False)
            scores = result.boxes.conf.cpu().numpy().astype(np.float32, copy=False)
            boxes[:, [0, 2]] += float(offset_x)
            boxes[:, [1, 3]] += float(offset_y)
            boxes = clip_boxes(boxes, record.width, record.height)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if np.any(valid):
                all_boxes.append(boxes[valid])
                all_scores.append(scores[valid])

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    keep = nms(boxes, scores, args.merge_iou, args.max_det)
    boxes = boxes[keep]
    scores = scores[keep]
    return [
        Detection(
            image_id=record.image_id,
            score=float(score),
            box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
        )
        for box, score in zip(boxes, scores)
    ]


def ap_from_curve(recall: np.ndarray, precision: np.ndarray) -> float:
    if len(recall) == 0:
        return float("nan")
    recall_points = np.linspace(0.0, 1.0, 101)
    values: list[float] = []
    for recall_point in recall_points:
        mask = recall >= recall_point
        values.append(float(np.max(precision[mask])) if np.any(mask) else 0.0)
    return float(np.mean(values))


def evaluate_ap(
    image_ids: set[str],
    gt_by_image: dict[str, np.ndarray],
    detections: list[Detection],
    iou_threshold: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    gt_subset = {image_id: gt_by_image[image_id] for image_id in image_ids}
    gt_count = sum(len(boxes) for boxes in gt_subset.values())
    if gt_count == 0:
        return float("nan"), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    matched = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in gt_subset.items()}
    det_subset = [det for det in detections if det.image_id in image_ids]
    det_subset.sort(key=lambda det: det.score, reverse=True)

    true_positive = np.zeros(len(det_subset), dtype=np.float32)
    false_positive = np.zeros(len(det_subset), dtype=np.float32)
    for index, det in enumerate(det_subset):
        gt_boxes = gt_subset[det.image_id]
        if len(gt_boxes) == 0:
            false_positive[index] = 1.0
            continue

        det_box = np.asarray(det.box, dtype=np.float32).reshape(1, 4)
        ious = box_iou_matrix(det_box, gt_boxes)[0]
        best_index = int(np.argmax(ious)) if len(ious) else -1
        best_iou = float(ious[best_index]) if best_index >= 0 else 0.0
        if best_iou >= iou_threshold and not matched[det.image_id][best_index]:
            true_positive[index] = 1.0
            matched[det.image_id][best_index] = True
        else:
            false_positive[index] = 1.0

    if len(det_subset) == 0:
        return 0.0, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    tp_cum = np.cumsum(true_positive)
    fp_cum = np.cumsum(false_positive)
    recall = tp_cum / max(gt_count, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return ap_from_curve(recall, precision), recall, precision


def precision_recall_at_conf(
    image_ids: set[str],
    gt_by_image: dict[str, np.ndarray],
    detections: list[Detection],
    conf_threshold: float,
    iou_threshold: float = 0.50,
) -> dict[str, float | int]:
    gt_subset = {image_id: gt_by_image[image_id] for image_id in image_ids}
    gt_count = sum(len(boxes) for boxes in gt_subset.values())
    matched = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in gt_subset.items()}
    det_subset = [
        det for det in detections if det.image_id in image_ids and det.score >= conf_threshold
    ]
    det_subset.sort(key=lambda det: det.score, reverse=True)

    tp = 0
    fp = 0
    for det in det_subset:
        gt_boxes = gt_subset[det.image_id]
        if len(gt_boxes) == 0:
            fp += 1
            continue
        ious = box_iou_matrix(np.asarray(det.box, dtype=np.float32).reshape(1, 4), gt_boxes)[0]
        best_index = int(np.argmax(ious)) if len(ious) else -1
        best_iou = float(ious[best_index]) if best_index >= 0 else 0.0
        if best_iou >= iou_threshold and not matched[det.image_id][best_index]:
            tp += 1
            matched[det.image_id][best_index] = True
        else:
            fp += 1

    fn = gt_count - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / gt_count if gt_count else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_at_conf": precision,
        "recall_at_conf": recall,
        "f1_at_conf": f1,
    }


def evaluate_subset(
    name: str,
    image_ids: set[str],
    gt_by_image: dict[str, np.ndarray],
    detections: list[Detection],
    report_conf: float,
) -> dict[str, Any]:
    thresholds = np.arange(0.50, 0.96, 0.05)
    aps: list[float] = []
    pr50_recall = np.zeros((0,), dtype=np.float32)
    pr50_precision = np.zeros((0,), dtype=np.float32)
    for threshold in thresholds:
        ap, recall, precision = evaluate_ap(image_ids, gt_by_image, detections, float(threshold))
        aps.append(ap)
        if math.isclose(float(threshold), 0.50):
            pr50_recall = recall
            pr50_precision = precision

    valid_aps = [ap for ap in aps if not math.isnan(ap)]
    pr_report = precision_recall_at_conf(image_ids, gt_by_image, detections, report_conf)
    gt_count = sum(len(gt_by_image[image_id]) for image_id in image_ids)
    det_count = sum(1 for det in detections if det.image_id in image_ids)
    return {
        "name": name,
        "images": len(image_ids),
        "gt_objects": gt_count,
        "detections": det_count,
        "mAP50": aps[0] if aps and not math.isnan(aps[0]) else None,
        "mAP50_95": float(np.mean(valid_aps)) if valid_aps else None,
        "ap_by_iou": {
            f"{threshold:.2f}": (None if math.isnan(ap) else ap)
            for threshold, ap in zip(thresholds, aps)
        },
        **pr_report,
        "_pr50_recall": pr50_recall,
        "_pr50_precision": pr50_precision,
    }


def clean_metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def write_outputs(
    output_dir: Path,
    metrics: list[dict[str, Any]],
    detections: list[Detection],
    records: list[ImageRecord],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_metrics = [clean_metric_row(item) for item in metrics]
    payload = {
        "config": {
            "model": str(args.model),
            "dataset": str(args.dataset or default_dataset()),
            "split": args.split,
            "imgsz": args.imgsz,
            "slice_size": args.slice_size,
            "overlap": args.overlap,
            "include_full_image": args.include_full_image,
            "conf": args.conf,
            "tile_iou": args.tile_iou,
            "merge_iou": args.merge_iou,
            "max_det": args.max_det,
            "tta": args.tta,
            "report_conf": args.report_conf,
        },
        "metrics": serializable_metrics,
    }
    (output_dir / "sliced_eval_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    with (output_dir / "sliced_eval_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "name",
            "images",
            "gt_objects",
            "detections",
            "mAP50",
            "mAP50_95",
            "tp",
            "fp",
            "fn",
            "precision_at_conf",
            "recall_at_conf",
            "f1_at_conf",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in serializable_metrics:
            writer.writerow({key: row.get(key) for key in fieldnames})

    with (output_dir / "sliced_eval_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "score", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        for det in detections:
            writer.writerow(
                {
                    "image_id": det.image_id,
                    "score": det.score,
                    "x1": det.box[0],
                    "y1": det.box[1],
                    "x2": det.box[2],
                    "y2": det.box[3],
                }
            )

    plot_curves(output_dir, metrics)
    draw_samples(output_dir / "samples", records, detections, args.save_samples, args.report_conf)


def plot_curves(output_dir: Path, metrics: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping PR curve plot")
        return

    overall = next((item for item in metrics if item["name"] == "overall"), None)
    if not overall:
        return
    recall = overall["_pr50_recall"]
    precision = overall["_pr50_precision"]
    if len(recall) == 0:
        return

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label="Sliced inference AP@50")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve @ IoU 0.50")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "sliced_pr_curve_iou50.png", dpi=200)
    plt.close()


def draw_samples(
    output_dir: Path,
    records: list[ImageRecord],
    detections: list[Detection],
    sample_count: int,
    conf_threshold: float,
) -> None:
    if sample_count <= 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    detections_by_image: dict[str, list[Detection]] = {}
    for det in detections:
        if det.score >= conf_threshold:
            detections_by_image.setdefault(det.image_id, []).append(det)

    font = ImageFont.load_default()
    drawn = 0
    for record in records:
        preds = detections_by_image.get(record.image_id, [])
        gt_boxes = load_yolo_boxes(record.label_path, record.width, record.height)
        if len(preds) == 0 and len(gt_boxes) == 0:
            continue
        image = Image.open(record.image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in gt_boxes:
            draw.rectangle(tuple(box), outline=(0, 200, 0), width=3)
            draw.text((float(box[0]), max(0.0, float(box[1]) - 12.0)), "GT", fill=(0, 200, 0), font=font)
        for det in preds:
            draw.rectangle(det.box, outline=(255, 64, 64), width=3)
            draw.text(
                (det.box[0], max(0.0, det.box[1] - 12.0)),
                f"{det.score:.2f}",
                fill=(255, 64, 64),
                font=font,
            )
        image.save(output_dir / f"{record.image_id}_sliced_pred.jpg", quality=90)
        drawn += 1
        if drawn >= sample_count:
            break


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.overlap < 0.9:
        raise SystemExit("--overlap must be in [0, 0.9)")
    if args.slice_size <= 0:
        raise SystemExit("--slice-size must be positive")
    if args.batch <= 0:
        raise SystemExit("--batch must be positive")

    dataset = args.dataset or default_dataset()
    if not (dataset / "data.yaml").exists():
        raise SystemExit(f"Detect dataset not found: {dataset}")
    if not args.model.exists():
        raise SystemExit(f"Checkpoint not found: {args.model}")

    records = load_records(dataset, args.split, args.limit)
    if not records:
        raise SystemExit(f"No images found for split {args.split} in {dataset}")

    print(
        "sliced eval:",
        f"images={len(records)}",
        f"slice={args.slice_size}",
        f"overlap={args.overlap}",
        f"imgsz={args.imgsz}",
        f"include_full={args.include_full_image}",
    )

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    gt_by_image = {
        record.image_id: load_yolo_boxes(record.label_path, record.width, record.height)
        for record in records
    }

    detections: list[Detection] = []
    for index, record in enumerate(records, start=1):
        detections.extend(predict_record(model, record, args))
        if index == 1 or index % 25 == 0 or index == len(records):
            print(f"processed {index}/{len(records)} images; detections={len(detections)}")

    all_ids = {record.image_id for record in records}
    metrics = [evaluate_subset("overall", all_ids, gt_by_image, detections, args.report_conf)]
    for group, group_name in GROUP_NAMES.items():
        group_ids = {record.image_id for record in records if record.group == group}
        if group_ids:
            metrics.append(
                evaluate_subset(f"g{group}_{group_name}", group_ids, gt_by_image, detections, args.report_conf)
            )

    write_outputs(args.output, metrics, detections, records, args)
    print("\nSliced inference metrics")
    for row in metrics:
        clean = clean_metric_row(row)
        map50 = clean["mAP50"]
        map5095 = clean["mAP50_95"]
        print(
            f"{clean['name']}: images={clean['images']} gt={clean['gt_objects']} "
            f"mAP50={map50 if map50 is not None else 'NA'} "
            f"mAP50-95={map5095 if map5095 is not None else 'NA'} "
            f"P@{args.report_conf}={clean['precision_at_conf']:.4f} "
            f"R@{args.report_conf}={clean['recall_at_conf']:.4f}"
        )
    print(f"\noutputs: {args.output}")


if __name__ == "__main__":
    main()
