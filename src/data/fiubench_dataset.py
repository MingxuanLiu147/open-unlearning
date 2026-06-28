"""
FIUBench (Fictitious Identity Unlearning Benchmark) data adapter.

Reference: ICLR 2025 — "Benchmarking Vision Language Model Unlearning
           via Fictitious Facial Identity Dataset"
           arXiv:2411.03554; GitHub: SaFoLab-WISC/FIUBench
           Dataset: HuggingFace gray311/FIUBench

FIUBench contains 400 synthetic face identities with 20 privacy-oriented
VQA pairs each (8,000 total).  Data format:
  - image: PIL Image or bytes
  - question: str
  - answer: str
  - person_id: str (identity grouping)

The dataset follows the same collate pattern as MLLMU / CLEAR via
``mm_collate_fn`` so all existing MM trainers work unchanged.
"""

import glob
import logging
import os
from io import BytesIO
from typing import Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from data.multimodal import mm_collate_fn

logger = logging.getLogger(__name__)


def _load_frame(path_or_dir: str) -> pd.DataFrame:
    if not os.path.exists(path_or_dir):
        raise FileNotFoundError(f"FIUBench data path not found: {path_or_dir}")
    if os.path.isfile(path_or_dir):
        return pd.read_parquet(path_or_dir)
    parquet_files = sorted(glob.glob(os.path.join(path_or_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {path_or_dir}")
    return pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)


def _maybe_limit(dataset: Dataset, max_samples: Optional[int]) -> Dataset:
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, range(max_samples))


class FIUBenchDataset(Dataset):
    """FIUBench VQA dataset: image + question -> answer."""

    def __init__(self, parquet_path: str, *, text_only: bool = False):
        super().__init__()
        self.text_only = text_only
        df = _load_frame(parquet_path)
        self.samples = self._flatten(df)
        logger.info(
            "FIUBenchDataset: %d samples from %s (text_only=%s)",
            len(self.samples), parquet_path, text_only,
        )

    def _flatten(self, df: pd.DataFrame) -> List[Dict]:
        data = []
        for _, row in df.iterrows():
            image = None
            if not self.text_only:
                raw = row.get("image")
                if raw is not None:
                    try:
                        if isinstance(raw, dict) and "bytes" in raw:
                            image = Image.open(BytesIO(raw["bytes"])).convert("RGB")
                        elif isinstance(raw, str) and os.path.exists(raw):
                            image = Image.open(raw).convert("RGB")
                        elif isinstance(raw, Image.Image):
                            image = raw.convert("RGB")
                    except Exception as e:
                        logger.warning("Skipping image decode: %s", e)
                        continue

            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            if q and a:
                data.append({"image": image, "question": q, "answer": a})
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _make_loader(dataset, processor, *, batch_size, ans_only, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: mm_collate_fn(x, processor, ans_only=ans_only),
    )


def get_fiubench_data(data_cfg: DictConfig, processor):
    """Build FIUBench forget/retain data loaders.

    Config keys:
        data_dir: root directory with forget/ and retain/ subdirectories
        forget_split_ratio: percentage of identities to forget (default 10)
        batch_size, ans_only, max_samples: standard controls
    """
    data_dir = str(data_cfg.data_dir)
    batch_size = int(data_cfg.get("batch_size", 4))
    ans_only = bool(data_cfg.get("ans_only", False))
    max_samples = data_cfg.get("max_samples")
    max_samples = int(max_samples) if max_samples is not None else None

    forget_ratio = int(data_cfg.get("forget_split_ratio", 10))
    retain_ratio = 100 - forget_ratio

    forget_path = os.path.join(data_dir, f"forget_{forget_ratio}")
    retain_path = os.path.join(data_dir, f"retain_{retain_ratio}")

    if not os.path.exists(forget_path):
        forget_path = os.path.join(data_dir, "forget")
    if not os.path.exists(retain_path):
        retain_path = os.path.join(data_dir, "retain")

    forget_dataset = _maybe_limit(FIUBenchDataset(forget_path), max_samples)
    forget_loader = _make_loader(
        forget_dataset, processor, batch_size=batch_size, ans_only=ans_only, shuffle=True
    )

    retain_loader = None
    if os.path.exists(retain_path):
        retain_dataset = _maybe_limit(FIUBenchDataset(retain_path), max_samples)
        retain_loader = _make_loader(
            retain_dataset, processor, batch_size=batch_size, ans_only=ans_only, shuffle=True
        )

    return forget_loader, retain_loader
