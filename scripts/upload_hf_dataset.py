#!/usr/bin/env python3
"""Upload the prepared YOLO11-seg dataset zip to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload prepared dataset zip to HF.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face dataset repo id, e.g. username/fire-vn-yolo11seg-v1.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("datasets/fire_vn_yolo11seg_v1.zip"),
        help="Prepared dataset zip produced by prepare_yolo11_seg_dataset.py.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private if it does not exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.zip_path.exists():
        raise SystemExit(f"Zip file not found: {args.zip_path}")

    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import HfApi, HfFolder
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    token = token or HfFolder.get_token()
    if not token:
        raise SystemExit("HF token missing. Set HF_TOKEN or run huggingface-cli login.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj=str(args.zip_path),
        path_in_repo=args.zip_path.name,
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"uploaded {args.zip_path} to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
