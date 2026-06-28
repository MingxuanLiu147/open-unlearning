#!/usr/bin/env python3
"""Validate user-provided data files against the expected JSON schema.

Supports JSON, JSONL, and plain-text (.txt) files.
Plain-text files are split by blank lines into paragraphs and auto-converted
to the structured format required by each mode.

Usage
-----
    python scripts/validate_data.py --mode inject  --data /path/to/train.jsonl
    python scripts/validate_data.py --mode unlearn --data /path/to/forget.jsonl
    python scripts/validate_data.py --mode edit    --data /path/to/edits.jsonl
    python scripts/validate_data.py --mode inject  --data /path/to/knowledge.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCHEMAS: Dict[str, Dict[str, Any]] = {
    "inject": {
        "description": "Alpaca format for knowledge injection",
        "required": {"instruction": str, "output": str},
        "optional": {"input": str},
    },
    "unlearn": {
        "description": "QA format for knowledge unlearning",
        "required": {"question": str, "answer": str},
        "optional": {},
    },
    "edit": {
        "description": "Edit triple for knowledge editing",
        "required": {"prompt": str, "subject": str, "target_new": str},
        "optional": {
            "target_old": str,
            "rephrase_prompts": list,
            "locality_inputs": dict,
            "portability_inputs": dict,
        },
    },
}


def _split_paragraphs(text: str) -> List[str]:
    """将纯文本按空行分割为段落，过滤空段落。"""
    paragraphs: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            paragraphs.append("\n".join(current))
            current = []
    if current:
        paragraphs.append("\n".join(current))
    return paragraphs


# 匹配 "X是Y" / "X 是 Y" / "X为Y" / "X：Y" 等中文模式，以及 "X is Y" 英文模式
_EDIT_PATTERN = re.compile(
    r"^(.{2,40}?)\s*(?:是|为|：|:|\s+is\s+)\s*(.+)$", re.DOTALL
)


def _auto_convert_freetext(
    paragraphs: List[str], mode: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """将纯文本段落自动转换为对应 mode 的结构化记录。

    返回 (records, conversion_errors)。
    """
    records: List[Dict[str, Any]] = []
    errors: List[str] = []
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
        if mode == "inject":
            records.append({"instruction": "请学习以下知识", "output": para})
        elif mode == "unlearn":
            records.append({"question": "关于以下内容你知道什么？", "answer": para})
        elif mode == "edit":
            m = _EDIT_PATTERN.match(para)
            if m:
                subject = m.group(1).strip()
                target_new = m.group(2).strip()
                records.append({
                    "prompt": f"What is {subject}?",
                    "subject": subject,
                    "target_new": target_new,
                })
            else:
                errors.append(
                    f"[{i}] edit 模式无法从段落自动提取 subject/target_new，"
                    "请使用 'X是Y' 格式或改用 JSON 结构化数据"
                )
    return records, errors


def _load_records(data_path: str, mode: str = "") -> List[Dict[str, Any]]:
    path = Path(data_path)
    if not path.exists():
        print(f"ERROR: file not found: {data_path}")
        sys.exit(1)

    # 纯文本文件：按段落分割后自动转换
    if path.suffix == ".txt":
        text = path.read_text(encoding="utf-8")
        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            print("ERROR: .txt file contains no paragraphs")
            sys.exit(1)
        records, conv_errors = _auto_convert_freetext(paragraphs, mode)
        if conv_errors and not records:
            for e in conv_errors:
                print(f"  {e}")
            sys.exit(1)
        return records

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
                    print(f"ERROR: line {i} is not valid JSON: {e}")
                    sys.exit(1)
        else:
            try:
                raw = json.load(f)
            except json.JSONDecodeError as e:
                print(f"ERROR: file is not valid JSON: {e}")
                sys.exit(1)
            records = raw if isinstance(raw, list) else [raw]

    return records


def _check_inject_record(record: Dict[str, Any], idx: int) -> List[str]:
    """Inject supports Alpaca and ShareGPT formats."""
    errors: List[str] = []

    is_alpaca = "instruction" in record or "output" in record
    is_sharegpt = "conversations" in record

    if not is_alpaca and not is_sharegpt:
        errors.append(
            f"[{idx}] needs 'instruction'+'output' (Alpaca) "
            "or 'conversations' (ShareGPT)"
        )
        return errors

    if is_sharegpt:
        convs = record.get("conversations")
        if not isinstance(convs, list) or len(convs) == 0:
            errors.append(f"[{idx}] 'conversations' must be a non-empty list")
        else:
            for ti, turn in enumerate(convs):
                if not isinstance(turn, dict):
                    errors.append(f"[{idx}] conversations[{ti}] must be a dict")
                    continue
                if "from" not in turn and "role" not in turn:
                    errors.append(
                        f"[{idx}] conversations[{ti}] missing 'from'/'role'"
                    )
                if "value" not in turn and "content" not in turn:
                    errors.append(
                        f"[{idx}] conversations[{ti}] missing 'value'/'content'"
                    )
    else:
        schema = SCHEMAS["inject"]
        for field, expected_type in schema["required"].items():
            if field not in record:
                errors.append(f"[{idx}] missing required field '{field}'")
            elif not isinstance(record[field], expected_type):
                errors.append(
                    f"[{idx}] '{field}' expected {expected_type.__name__}, "
                    f"got {type(record[field]).__name__}"
                )
            elif isinstance(record[field], str) and not record[field].strip():
                errors.append(f"[{idx}] field '{field}' is empty")

    return errors


def _check_record(record: Dict[str, Any], idx: int, mode: str) -> List[str]:
    if mode == "inject":
        return _check_inject_record(record, idx)

    errors: List[str] = []
    schema = SCHEMAS[mode]

    for field, expected_type in schema["required"].items():
        if field not in record:
            errors.append(f"[{idx}] missing required field '{field}'")
        elif not isinstance(record[field], expected_type):
            errors.append(
                f"[{idx}] '{field}' expected {expected_type.__name__}, "
                f"got {type(record[field]).__name__}"
            )
        elif isinstance(record[field], str) and not record[field].strip():
            errors.append(f"[{idx}] field '{field}' is empty")

    return errors


def validate(data_path: str, mode: str) -> Tuple[int, int, List[str]]:
    if mode not in SCHEMAS:
        print(f"ERROR: unknown mode '{mode}'. Choose from: {list(SCHEMAS.keys())}")
        sys.exit(1)

    path = Path(data_path)
    # .txt 文件需要先转换再校验
    if path.suffix == ".txt":
        text = path.read_text(encoding="utf-8")
        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            return 0, 0, ["文件不包含任何段落"]
        records, conv_errors = _auto_convert_freetext(paragraphs, mode)
        if conv_errors:
            return len(paragraphs), len(records), conv_errors
        # 转换成功后走正常校验流程
        total = len(records)
        all_errors: List[str] = []
        for idx, record in enumerate(records):
            all_errors.extend(_check_record(record, idx, mode))
        bad_indices = set()
        for e in all_errors:
            try:
                bad_indices.add(int(e.split("]")[0][1:]))
            except (ValueError, IndexError):
                pass
        passed = total - len(bad_indices)
        return total, passed, all_errors

    records = _load_records(data_path, mode)
    total = len(records)
    all_errors: List[str] = []

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            all_errors.append(f"[{idx}] record is not a JSON object")
            continue
        all_errors.extend(_check_record(record, idx, mode))

    bad_indices = set()
    for e in all_errors:
        try:
            bad_indices.add(int(e.split("]")[0][1:]))
        except (ValueError, IndexError):
            pass
    passed = total - len(bad_indices)
    return total, passed, all_errors


def convert_freetext_to_jsonl(
    txt_path: str, mode: str, output_path: str
) -> Tuple[int, List[str]]:
    """将 .txt 自由文本转换为 .jsonl 文件，返回 (记录数, 转换错误列表)。"""
    text = Path(txt_path).read_text(encoding="utf-8")
    paragraphs = _split_paragraphs(text)
    records, errors = _auto_convert_freetext(paragraphs, mode)
    if records:
        with open(output_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records), errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate data files for open-unlearning"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["inject", "unlearn", "edit"],
        help="Task mode: inject, unlearn, or edit",
    )
    parser.add_argument(
        "--data", required=True, help="Path to JSON or JSONL data file"
    )
    args = parser.parse_args()

    schema = SCHEMAS[args.mode]
    print(f"Validating for mode '{args.mode}': {schema['description']}")
    print(f"Required fields: {list(schema['required'].keys())}")
    print(f"File: {args.data}")
    print("-" * 60)

    total, passed, errors = validate(args.data, args.mode)

    if errors:
        print(f"\nFOUND {len(errors)} ERROR(S) in {total} records:")
        for err in errors[:20]:
            print(f"  {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        print(f"\n{passed}/{total} records passed validation")
        sys.exit(1)
    else:
        print(f"\nALL {total} records passed validation")
        sys.exit(0)


if __name__ == "__main__":
    main()
