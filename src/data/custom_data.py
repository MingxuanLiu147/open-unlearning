"""Custom data utilities for user-uploaded datasets.

Provides format detection, validation, and train/val splitting for the three
task modes: inject, unlearn, and edit.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

INJECT_REQUIRED = {"instruction": str, "output": str}
INJECT_OPTIONAL = {"input": str}
SHAREGPT_REQUIRED = {"conversations": list}
UNLEARN_REQUIRED = {"question": str, "answer": str}
EDIT_REQUIRED = {"prompt": str, "subject": str, "target_new": str}
EDIT_OPTIONAL = {
    "target_old": str,
    "rephrase_prompts": list,
    "locality_inputs": dict,
    "portability_inputs": dict,
}


@dataclass
class ValidationResult:
    total: int = 0
    passed: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def load_json_records(data_path: str) -> List[Dict[str, Any]]:
    """Load records from a JSON or JSONL file."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Line {i} is not valid JSON: {e}") from e
        else:
            raw = json.load(f)
            records = raw if isinstance(raw, list) else [raw]
    return records


def detect_format(data_path: str) -> str:
    """Auto-detect the data format by inspecting the first record.

    Returns one of: ``"alpaca"``, ``"sharegpt"``, ``"qa"``, ``"edit"``,
    or ``"unknown"``.
    """
    records = load_json_records(data_path)
    if not records:
        return "unknown"

    sample = records[0]
    if not isinstance(sample, dict):
        return "unknown"

    if "conversations" in sample:
        return "sharegpt"
    if "instruction" in sample and "output" in sample:
        return "alpaca"
    if "prompt" in sample and "subject" in sample and "target_new" in sample:
        return "edit"
    if "question" in sample and "answer" in sample:
        return "qa"
    return "unknown"


def _check_fields(
    record: Dict[str, Any],
    idx: int,
    required: Dict[str, type],
) -> List[str]:
    errors: List[str] = []
    for fld, expected_type in required.items():
        if fld not in record:
            errors.append(f"[{idx}] missing required field '{fld}'")
        elif not isinstance(record[fld], expected_type):
            errors.append(
                f"[{idx}] '{fld}' expected {expected_type.__name__}, "
                f"got {type(record[fld]).__name__}"
            )
        elif isinstance(record[fld], str) and not record[fld].strip():
            errors.append(f"[{idx}] field '{fld}' is empty")
    return errors


def validate(data_path: str, mode: str) -> ValidationResult:
    """Validate a data file against the expected schema for *mode*.

    Args:
        data_path: Path to JSON / JSONL file.
        mode: One of ``"inject"``, ``"unlearn"``, ``"edit"``.

    Returns:
        A :class:`ValidationResult` with per-record error messages.
    """
    records = load_json_records(data_path)
    result = ValidationResult(total=len(records))
    bad_indices: set = set()

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            result.errors.append(f"[{idx}] record is not a JSON object")
            bad_indices.add(idx)
            continue

        if mode == "inject":
            is_sharegpt = "conversations" in record
            if is_sharegpt:
                errs = _check_fields(record, idx, SHAREGPT_REQUIRED)
                if not errs:
                    convs = record["conversations"]
                    if not isinstance(convs, list) or len(convs) == 0:
                        errs.append(f"[{idx}] 'conversations' must be a non-empty list")
            else:
                errs = _check_fields(record, idx, INJECT_REQUIRED)
        elif mode == "unlearn":
            errs = _check_fields(record, idx, UNLEARN_REQUIRED)
        elif mode == "edit":
            errs = _check_fields(record, idx, EDIT_REQUIRED)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if errs:
            result.errors.extend(errs)
            bad_indices.add(idx)

    result.passed = result.total - len(bad_indices)
    return result


def split_train_val(
    data_path: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """Split a JSON/JSONL file into train and val files.

    Args:
        data_path: Path to the source file.
        val_ratio: Fraction of data to use for validation.
        seed: Random seed for reproducibility.
        output_dir: Directory for output files; defaults to same directory as
            the source file.

    Returns:
        ``(train_path, val_path)`` tuple of the written file paths.
    """
    records = load_json_records(data_path)
    if not records:
        raise ValueError(f"No records found in {data_path}")

    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    val_size = max(1, int(len(records) * val_ratio))
    val_indices = set(indices[:val_size])

    train_records = [r for i, r in enumerate(records) if i not in val_indices]
    val_records = [r for i, r in enumerate(records) if i in val_indices]

    src = Path(data_path)
    out_dir = Path(output_dir) if output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = src.stem
    train_path = out_dir / f"{stem}_train.jsonl"
    val_path = out_dir / f"{stem}_val.jsonl"

    def _write_jsonl(path: Path, recs: list):
        with open(path, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _write_jsonl(train_path, train_records)
    _write_jsonl(val_path, val_records)

    logger.info(
        "Split %d records -> %d train + %d val (%s, %s)",
        len(records), len(train_records), len(val_records),
        train_path, val_path,
    )
    return str(train_path), str(val_path)


def convert_csv_to_jsonl(
    csv_path: str,
    output_path: Optional[str] = None,
) -> str:
    """Convert a CSV file to JSONL format.

    Args:
        csv_path: Path to input CSV.
        output_path: Path for the output JSONL. Defaults to replacing the
            extension of *csv_path*.

    Returns:
        Path to the written JSONL file.
    """
    import csv

    src = Path(csv_path)
    out = Path(output_path) if output_path else src.with_suffix(".jsonl")

    with open(src, "r", encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            fout.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    logger.info("Converted %s -> %s", csv_path, out)
    return str(out)
