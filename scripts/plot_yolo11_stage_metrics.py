#!/usr/bin/env python3
"""Plot YOLO11 Stage 1 and Stage 2 training metrics from Ultralytics results.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


STAGE1_DEFAULT = Path("runs/final/yolo11x_detect_fire_smoke_l4_final/results.csv")
STAGE2_DEFAULT = Path("runs/final/yolo11x_detect_fire_smoke_l4_finetune/results.csv")
OUTPUT_DEFAULT = Path("reports/figures/stage1_stage2_training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Stage 1 and Stage 2 YOLO11 training curves."
    )
    parser.add_argument("--stage1", type=Path, default=STAGE1_DEFAULT)
    parser.add_argument("--stage2", type=Path, default=STAGE2_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args()


def load_results(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise SystemExit(f"Missing results.csv: {path}")

    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row: dict[str, float] = {}
            for key, value in raw_row.items():
                clean_key = (key or "").strip()
                clean_value = (value or "").strip()
                if not clean_key:
                    continue
                try:
                    row[clean_key] = float(clean_value)
                except ValueError:
                    continue
            rows.append(row)

    if not rows:
        raise SystemExit(f"No numeric rows found in {path}")
    return rows


def first_column(row: dict[str, float], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in row:
            return candidate
    return None


def values(rows: list[dict[str, float]], candidates: Iterable[str]) -> tuple[str | None, list[float]]:
    column = first_column(rows[0], candidates)
    if column is None:
        return None, []
    return column, [row[column] for row in rows if column in row]


def epochs(rows: list[dict[str, float]]) -> list[float]:
    column, vals = values(rows, ("epoch",))
    if column:
        return vals
    return [float(index + 1) for index in range(len(rows))]


def plot_stage(stage_name: str, rows: list[dict[str, float]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    x = epochs(rows)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{stage_name} Training Curves", fontsize=15)

    plot_lines(
        axes[0, 0],
        x,
        rows,
        {
            "train/box_loss": ("train/box_loss",),
            "train/cls_loss": ("train/cls_loss",),
            "train/dfl_loss": ("train/dfl_loss",),
        },
        "Train Loss",
    )
    plot_lines(
        axes[0, 1],
        x,
        rows,
        {
            "val/box_loss": ("val/box_loss",),
            "val/cls_loss": ("val/cls_loss",),
            "val/dfl_loss": ("val/dfl_loss",),
        },
        "Validation Loss",
    )
    plot_lines(
        axes[1, 0],
        x,
        rows,
        {
            "precision": ("metrics/precision(B)", "metrics/precision"),
            "recall": ("metrics/recall(B)", "metrics/recall"),
        },
        "Precision / Recall",
    )
    plot_lines(
        axes[1, 1],
        x,
        rows,
        {
            "mAP50": ("metrics/mAP50(B)", "metrics/mAP50"),
            "mAP50-95": ("metrics/mAP50-95(B)", "metrics/mAP50-95"),
        },
        "mAP",
    )

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.tight_layout()
    stage_slug = stage_name.lower().replace(" ", "")
    fig.savefig(output_dir / f"{stage_slug}_training_curves.png", dpi=200)
    plt.close(fig)


def plot_lines(
    ax,
    x: list[float],
    rows: list[dict[str, float]],
    series: dict[str, tuple[str, ...]],
    title: str,
) -> None:
    for label, candidates in series.items():
        _column, y = values(rows, candidates)
        if y:
            ax.plot(x[: len(y)], y, label=label, linewidth=2)
    ax.set_title(title)


def best_metrics(stage_name: str, rows: list[dict[str, float]]) -> dict[str, float | str]:
    map50_column = first_column(rows[0], ("metrics/mAP50(B)", "metrics/mAP50"))
    if map50_column is None:
        raise SystemExit(f"{stage_name}: missing mAP50 column in results.csv")

    best_index = max(range(len(rows)), key=lambda index: rows[index].get(map50_column, float("-inf")))
    best = rows[best_index]
    result: dict[str, float | str] = {
        "stage": stage_name,
        "best_epoch": best.get("epoch", float(best_index + 1)),
    }

    for output_name, candidates in {
        "precision": ("metrics/precision(B)", "metrics/precision"),
        "recall": ("metrics/recall(B)", "metrics/recall"),
        "mAP50": ("metrics/mAP50(B)", "metrics/mAP50"),
        "mAP50_95": ("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }.items():
        column = first_column(best, candidates)
        result[output_name] = best[column] if column else float("nan")
    return result


def plot_comparison(
    stage1_rows: list[dict[str, float]],
    stage2_rows: list[dict[str, float]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for stage_name, rows in (("Stage 1", stage1_rows), ("Stage 2", stage2_rows)):
        x = epochs(rows)
        _column, map50 = values(rows, ("metrics/mAP50(B)", "metrics/mAP50"))
        _column, map5095 = values(rows, ("metrics/mAP50-95(B)", "metrics/mAP50-95"))
        if map50:
            ax.plot(x[: len(map50)], map50, label=f"{stage_name} mAP50", linewidth=2)
        if map5095:
            ax.plot(x[: len(map5095)], map5095, label=f"{stage_name} mAP50-95", linewidth=2)

    ax.set_title("Stage 1 vs Stage 2 mAP")
    ax.set_xlabel("Epoch Within Stage")
    ax.set_ylabel("Metric")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_stage2_map_comparison.png", dpi=200)
    plt.close(fig)


def plot_best_bars(summary: list[dict[str, float | str]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    metric_names = ("precision", "recall", "mAP50", "mAP50_95")
    x = range(len(metric_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 6))
    for offset, row in ((-width / 2, summary[0]), (width / 2, summary[1])):
        values_for_row = [float(row[name]) for name in metric_names]
        positions = [index + offset for index in x]
        ax.bar(positions, values_for_row, width=width, label=str(row["stage"]))

    ax.set_title("Best Validation Metrics By Stage")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_stage2_best_metrics.png", dpi=200)
    plt.close(fig)


def write_summary(summary: list[dict[str, float | str]], output_dir: Path) -> None:
    fieldnames = ["stage", "best_epoch", "precision", "recall", "mAP50", "mAP50_95"]
    with (output_dir / "stage1_stage2_best_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise SystemExit("matplotlib is required: pip install matplotlib") from exc

    stage1_rows = load_results(args.stage1)
    stage2_rows = load_results(args.stage2)
    summary = [best_metrics("Stage 1", stage1_rows), best_metrics("Stage 2", stage2_rows)]

    plot_stage("Stage 1", stage1_rows, args.output)
    plot_stage("Stage 2", stage2_rows, args.output)
    plot_comparison(stage1_rows, stage2_rows, args.output)
    plot_best_bars(summary, args.output)
    write_summary(summary, args.output)

    print(f"wrote plots to: {args.output}")
    for row in summary:
        print(
            f"{row['stage']}: best_epoch={row['best_epoch']} "
            f"mAP50={float(row['mAP50']):.4f} "
            f"mAP50-95={float(row['mAP50_95']):.4f}"
        )


if __name__ == "__main__":
    main()
