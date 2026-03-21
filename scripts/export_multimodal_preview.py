#!/usr/bin/env python3
"""Export small JSONL previews for local multimodal datasets.

The goal is readability, not lossless round-tripping:
- write a few sample rows to JSONL
- save images as separate files
- replace raw image bytes/PIL objects with image metadata + relative paths
"""

from __future__ import annotations

import argparse
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_mllmu_image(image_field: dict[str, Any], image_path: Path) -> dict[str, Any]:
    raw_bytes = image_field.get("bytes")
    original_path = image_field.get("path")
    if raw_bytes is None:
        return {
            "saved_path": None,
            "original_path": original_path,
            "bytes_len": 0,
        }
    image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    image.save(image_path)
    return {
        "saved_path": str(image_path),
        "original_path": original_path,
        "bytes_len": len(raw_bytes),
        "size": list(image.size),
        "mode": image.mode,
    }


def save_clear_image(image: Image.Image, image_path: Path) -> dict[str, Any]:
    image = image.convert("RGB")
    image.save(image_path)
    return {
        "saved_path": str(image_path),
        "size": list(image.size),
        "mode": image.mode,
    }


def export_mllmu_preview(mllmu_file: Path, output_dir: Path, max_rows: int) -> Path:
    df = pd.read_parquet(mllmu_file)
    records = []
    images_dir = output_dir / "images" / "mllmu"
    images_dir.mkdir(parents=True, exist_ok=True)

    for idx, (_, row) in enumerate(df.head(max_rows).iterrows()):
        record = row.to_dict()
        image_meta = save_mllmu_image(record.pop("image"), images_dir / f"sample_{idx:03d}.png")
        record["image"] = image_meta
        records.append(to_jsonable(record))

    out_file = output_dir / f"{mllmu_file.parent.name}_preview.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_file


def export_clear_preview(clear_dir: Path, output_dir: Path, max_rows: int, hf_home: Path) -> Path:
    images_dir = output_dir / "images" / "clear"
    images_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")

    dataset = load_dataset(str(clear_dir), split="train")
    records = []
    for idx in range(min(max_rows, len(dataset))):
        sample = dict(dataset[idx])
        image_meta = save_clear_image(sample.pop("image"), images_dir / f"sample_{idx:03d}.png")
        sample["image"] = image_meta
        records.append(to_jsonable(sample))

    out_file = output_dir / f"{clear_dir.name}_preview.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export JSONL previews for multimodal datasets.")
    parser.add_argument(
        "--mllmu-file",
        type=Path,
        default=Path("data/MLLMU-Bench/forget_5/train-00000-of-00001.parquet"),
        help="Path to a local MLLMU parquet file.",
    )
    parser.add_argument(
        "--clear-dir",
        type=Path,
        default=Path("data/CLEAR/forget05"),
        help="Path to a local CLEAR split directory loadable by datasets.load_dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/_exports/json_preview"),
        help="Directory where JSONL previews and extracted images are written.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5,
        help="Number of rows to export for each dataset.",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path("data/.cache/huggingface"),
        help="Writable Hugging Face cache directory for local CLEAR loading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mllmu_file = args.mllmu_file.resolve()
    clear_dir = args.clear_dir.resolve()
    hf_home = args.hf_home.resolve()
    hf_home.mkdir(parents=True, exist_ok=True)

    mllmu_out = export_mllmu_preview(mllmu_file, output_dir, args.max_rows)
    clear_out = export_clear_preview(clear_dir, output_dir, args.max_rows, hf_home)

    print(f"MLLMU preview: {mllmu_out}")
    print(f"CLEAR preview: {clear_out}")


if __name__ == "__main__":
    main()
