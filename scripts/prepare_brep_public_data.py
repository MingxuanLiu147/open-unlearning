#!/usr/bin/env python3
"""Download and normalize directly downloadable BREP datasets.

This script only handles datasets that can be downloaded from public sources
without recreating a mixed training recipe. The following datasets are
explicitly excluded and must be rebuilt separately:

- math10k
- commonsense
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "inject" / "brep"
DEFAULT_RAW_ROOT = DEFAULT_OUTPUT_ROOT / "_raw"
DEFAULT_HF_HOME = PROJECT_ROOT / ".cache" / "huggingface"

os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_HOME / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))

from datasets import Dataset, load_dataset

RECIPE_ONLY_DATASETS = {"math10k", "commonsense"}
DEFAULT_DATASETS = [
    "prm800k",
    "ultrafeedback",
    "gsm8k",
    "hellaswag",
    "svamp",
    "mathqa",
    "math500",
    "amc23",
    "truthfulqa",
]


def _print(message: str) -> None:
    print(message, file=sys.stderr)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: Any) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _iter_dataset(dataset: Dataset, max_rows: int | None = None) -> Iterable[dict[str, Any]]:
    total = len(dataset)
    limit = total if max_rows is None else min(total, max_rows)
    for idx in range(limit):
        yield dataset[idx]


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _first_non_empty(mapping: Mapping[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in mapping:
            value = _stringify(mapping[key])
            if value:
                return value
    return ""


def _split_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_stringify(item) for item in value if _stringify(item)]
    text = _stringify(value)
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def _extract_numeric_leaves(value: Any) -> list[float]:
    values: list[float] = []
    if isinstance(value, bool):
        return values
    if isinstance(value, (int, float)):
        values.append(float(value))
        return values
    if isinstance(value, Mapping):
        for nested in value.values():
            values.extend(_extract_numeric_leaves(nested))
        return values
    if isinstance(value, list):
        for nested in value:
            values.extend(_extract_numeric_leaves(nested))
        return values
    return values


def _download_file(urls: list[str], destination: Path, force: bool) -> Path:
    if destination.exists() and not force:
        return destination

    _ensure_parent(destination)
    headers = {"User-Agent": "open-unlearning-brep-data-prep/1.0"}
    last_error: Exception | None = None

    for url in urls:
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
                handle.write(response.read())
            return destination
        except Exception as exc:  # pragma: no cover - network depends on runtime
            last_error = exc
            continue

    raise RuntimeError(
        f"Failed to download {destination.name} from all configured URLs."
    ) from last_error


def _is_git_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        prefix = handle.read(200)
    return prefix.startswith("version https://git-lfs.github.com/spec/v1")


def _extract_final_gsm8k_answer(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip()


def _build_hellaswag_instruction(row: Mapping[str, Any]) -> str:
    ctx = _first_non_empty(row, ["ctx"])
    if ctx:
        return ctx

    activity = _first_non_empty(row, ["activity_label"])
    ctx_a = _first_non_empty(row, ["ctx_a"])
    ctx_b = _first_non_empty(row, ["ctx_b"])
    if activity and ctx_a:
        prefix = f"{activity}: {ctx_a}"
        return f"{prefix} {ctx_b}".strip()
    return f"{ctx_a} {ctx_b}".strip()


def _parse_mathqa_options(options: str) -> dict[str, str]:
    matches = re.findall(r"([a-z])\s*\)\s*([^,]+?)(?=(?:,\s*[a-z]\s*\)|$))", options, flags=re.I)
    parsed: dict[str, str] = {}
    for label, text in matches:
        parsed[label.lower()] = text.strip()
    return parsed


def _select_best_ultrafeedback_completion(row: Mapping[str, Any]) -> tuple[str, float | None]:
    direct_choice = row.get("chosen", None)
    if isinstance(direct_choice, Mapping):
        text = _first_non_empty(direct_choice, ["response", "text", "content", "output"])
        if text:
            scores = _extract_numeric_leaves(direct_choice)
            score = sum(scores) / len(scores) if scores else None
            return text, score
    if isinstance(direct_choice, str) and direct_choice.strip():
        return direct_choice.strip(), None

    candidates = row.get("completions", row.get("responses", row.get("outputs", [])))
    best_text = ""
    best_score = float("-inf")
    found_scored = False

    if not isinstance(candidates, list):
        return "", None

    for candidate in candidates:
        if isinstance(candidate, str):
            text = candidate.strip()
            score = None
        elif isinstance(candidate, Mapping):
            text = _first_non_empty(candidate, ["response", "text", "content", "output"])
            numbers = _extract_numeric_leaves(candidate)
            score = sum(numbers) / len(numbers) if numbers else None
        else:
            continue

        if not text:
            continue

        effective_score = score if score is not None else 0.0
        if not best_text or effective_score > best_score:
            best_text = text
            best_score = effective_score
            found_scored = score is not None

    if not best_text:
        return "", None
    return best_text, best_score if found_scored else None


def _load_split_with_fallback(dataset_name: str, preferred_splits: list[str], **kwargs: Any) -> tuple[Dataset, str]:
    last_error: Exception | None = None
    for split_name in preferred_splits:
        try:
            dataset = load_dataset(dataset_name, split=split_name, **kwargs)
            return dataset, split_name
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(
        f"Unable to load any configured split for dataset `{dataset_name}`: {preferred_splits}"
    ) from last_error


def _load_named_split_with_fallback(
    dataset_names: list[str], preferred_splits: list[str], **kwargs: Any
) -> tuple[Dataset, str, str]:
    last_error: Exception | None = None
    for dataset_name in dataset_names:
        try:
            dataset, split_name = _load_split_with_fallback(
                dataset_name, preferred_splits, **kwargs
            )
            return dataset, split_name, dataset_name
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(
        "Unable to load any configured dataset id from "
        f"{dataset_names} with splits {preferred_splits}."
    ) from last_error


def _normalize_prm800k(output_root: Path, raw_root: Path, max_rows: int | None, force: bool) -> dict[str, Any]:
    raw_path = raw_root / "prm800k" / "phase2_train.jsonl"
    _download_file(
        [
            "https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/data/phase2_train.jsonl",
            "https://raw.githubusercontent.com/openai/prm800k/main/prm800k/data/phase2_train.jsonl",
            "https://raw.githubusercontent.com/openai/prm800k/main/data/phase2_train.jsonl",
            "https://media.githubusercontent.com/media/openai/prm800k/main/data/phase2_train.jsonl",
            "https://github.com/openai/prm800k/raw/main/prm800k/data/phase2_train.jsonl",
            "https://github.com/openai/prm800k/raw/main/data/phase2_train.jsonl",
        ],
        raw_path,
        force=force,
    )
    if _is_git_lfs_pointer(raw_path):
        _download_file(
            [
                "https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/data/phase2_train.jsonl",
                "https://media.githubusercontent.com/media/openai/prm800k/main/data/phase2_train.jsonl",
            ],
            raw_path,
            force=True,
        )
    if _is_git_lfs_pointer(raw_path):
        raise RuntimeError(
            "Downloaded PRM800K file is still a Git LFS pointer after retrying media URLs."
        )

    rows: list[dict[str, Any]] = []
    with raw_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if max_rows is not None and len(rows) >= max_rows:
                break
            payload = json.loads(line)
            question = payload.get("question", {})
            instruction = _stringify(question.get("problem"))
            output = _stringify(question.get("ground_truth_solution"))
            answer = _stringify(question.get("ground_truth_answer"))
            if not instruction or not output:
                continue
            rows.append(
                {
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                    "answer": answer,
                    "source_dataset": "prm800k",
                    "index": line_idx,
                }
            )

    target = output_root / "train" / "prm800k" / "train.json"
    _save_json(target, rows)
    return {"dataset": "prm800k", "rows": len(rows), "path": str(target)}


def _normalize_ultrafeedback(output_root: Path, max_rows: int | None) -> dict[str, Any]:
    dataset = load_dataset("openbmb/UltraFeedback", split="train")
    rows: list[dict[str, Any]] = []

    for row_idx, row in enumerate(_iter_dataset(dataset, max_rows=max_rows)):
        instruction = _first_non_empty(row, ["instruction", "prompt", "question"])
        output, score = _select_best_ultrafeedback_completion(row)
        if not instruction or not output:
            continue
        item = {
            "instruction": instruction,
            "input": "",
            "output": output,
            "source_dataset": "ultrafeedback",
            "index": row_idx,
        }
        if score is not None:
            item["response_score"] = round(float(score), 6)
        rows.append(item)

    target = output_root / "train" / "ultrafeedback" / "train.json"
    _save_json(target, rows)
    return {"dataset": "ultrafeedback", "rows": len(rows), "path": str(target)}


def _normalize_gsm8k(output_root: Path, max_rows: int | None) -> dict[str, Any]:
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    rows: list[dict[str, Any]] = []

    for row_idx, row in enumerate(_iter_dataset(dataset, max_rows=max_rows)):
        question = _first_non_empty(row, ["question"])
        output = _first_non_empty(row, ["answer"])
        answer = _extract_final_gsm8k_answer(output)
        if not question or not output:
            continue
        rows.append(
            {
                "instruction": question,
                "output": output,
                "answer": answer,
                "index": row_idx,
            }
        )

    target = output_root / "eval" / "gsm8k" / "test.json"
    _save_json(target, rows)
    return {"dataset": "gsm8k", "rows": len(rows), "path": str(target)}


def _normalize_hellaswag(output_root: Path, max_rows: int | None) -> dict[str, Any]:
    dataset, split_name = _load_split_with_fallback("Rowan/hellaswag", ["validation", "train"])
    rows: list[dict[str, Any]] = []

    for row_idx, row in enumerate(_iter_dataset(dataset, max_rows=max_rows)):
        instruction = _build_hellaswag_instruction(row)
        endings = row.get("endings", [])
        label_text = _stringify(row.get("label"))
        if not instruction or not isinstance(endings, list) or not label_text:
            continue
        try:
            label = int(label_text)
            answer = _stringify(endings[label])
        except (ValueError, IndexError, TypeError):
            continue
        rows.append(
            {
                "instruction": instruction,
                "output": answer,
                "answer": answer,
                "index": row_idx,
                "source_split": split_name,
                "choices": [_stringify(item) for item in endings],
            }
        )

    target = output_root / "eval" / "hellaswag" / "test.json"
    _save_json(target, rows)
    return {"dataset": "hellaswag", "rows": len(rows), "path": str(target), "split": split_name}


def _normalize_svamp(output_root: Path, max_rows: int | None) -> dict[str, Any]:
    dataset, split_name = _load_split_with_fallback("ChilleD/SVAMP", ["test", "validation", "train"])
    rows: list[dict[str, Any]] = []

    for row_idx, row in enumerate(_iter_dataset(dataset, max_rows=max_rows)):
        instruction = _first_non_empty(row, ["question_concat"])
        if not instruction:
            body = _first_non_empty(row, ["Body", "body"])
            question = _first_non_empty(row, ["Question", "question"])
            instruction = " ".join(part for part in [body, question] if part).strip()
        answer = _first_non_empty(row, ["Answer", "answer"])
        equation = _first_non_empty(row, ["Equation", "equation"])
        if not instruction or not answer:
            continue
        item = {
            "instruction": instruction,
            "output": answer,
            "answer": answer,
            "index": row_idx,
            "source_split": split_name,
        }
        if equation:
            item["equation"] = equation
        rows.append(item)

    target = output_root / "eval" / "svamp" / "test.json"
    _save_json(target, rows)
    return {"dataset": "svamp", "rows": len(rows), "path": str(target), "split": split_name}


def _normalize_mathqa(output_root: Path, max_rows: int | None) -> dict[str, Any]:
    dataset, split_name = _load_split_with_fallback(
        "allenai/math_qa",
        ["test", "validation", "train"],
        trust_remote_code=True,
    )
    rows: list[dict[str, Any]] = []

    for row_idx, row in enumerate(_iter_dataset(dataset, max_rows=max_rows)):
        instruction = _first_non_empty(row, ["Problem", "problem"])
        rationale = _first_non_empty(row, ["Rationale", "rationale"])
        options = _first_non_empty(row, ["options", "Options"])
        correct = _first_non_empty(row, ["correct", "Correct"])
        if not instruction or not correct:
            continue

        parsed = _parse_mathqa_options(options)
        answer = parsed.get(correct.lower().strip(), correct)
        output = rationale or answer
        rows.append(
            {
                "instruction": instruction,
                "output": output,
                "answer": answer,
                "index": row_idx,
                "source_split": split_name,
            }
        )

    target = output_root / "eval" / "mathqa" / "test.json"
    _save_json(target, rows)
    return {"dataset": "mathqa", "rows": len(rows), "path": str(target), "split": split_name}


def _normalize_math500(output_root: Path, max_rows: int | None) -> dict[str, Any]:
    dataset, split_name = _load_split_with_fallback("HuggingFaceH4/MATH-500", ["test", "train"])
    rows: list[dict[str, Any]] = []

    for row_idx, row in enumerate(_iter_dataset(dataset, max_rows=max_rows)):
        instruction = _first_non_empty(row, ["problem"])
        output = _first_non_empty(row, ["solution"])
        answer = _first_non_empty(row, ["answer"])
        if not instruction or not answer:
            continue
        item = {
            "instruction": instruction,
            "output": output or answer,
            "answer": answer,
            "index": row_idx,
            "source_split": split_name,
        }
        subject = _first_non_empty(row, ["subject"])
        level = _first_non_empty(row, ["level"])
        if subject:
            item["subject"] = subject
        if level:
            item["level"] = level
        rows.append(item)

    target = output_root / "eval" / "math500" / "test.json"
    _save_json(target, rows)
    return {"dataset": "math500", "rows": len(rows), "path": str(target), "split": split_name}


def _normalize_amc23(output_root: Path, max_rows: int | None) -> dict[str, Any]:
    dataset, split_name, dataset_name = _load_named_split_with_fallback(
        ["ScaleFrontierData/amc23", "baohao/amc23"],
        ["train"],
    )
    rows: list[dict[str, Any]] = []

    for row_idx, row in enumerate(_iter_dataset(dataset, max_rows=max_rows)):
        instruction = _first_non_empty(row, ["question"])
        answer = _first_non_empty(row, ["answer"])
        metadata = row.get("metadata", {})
        if not instruction and isinstance(metadata, Mapping):
            instruction = _first_non_empty(metadata, ["problem", "question"])
        if not instruction or not answer:
            continue
        item = {
            "instruction": instruction,
            "output": answer,
            "answer": answer,
            "index": row_idx,
            "source_split": split_name,
        }
        if isinstance(metadata, Mapping):
            url = _first_non_empty(metadata, ["url"])
            if url:
                item["source_url"] = url
        rows.append(item)

    target = output_root / "eval" / "amc23" / "test.json"
    _save_json(target, rows)
    return {
        "dataset": "amc23",
        "rows": len(rows),
        "path": str(target),
        "split": split_name,
        "source_dataset_id": dataset_name,
    }


def _normalize_truthfulqa(output_root: Path, max_rows: int | None) -> dict[str, Any]:
    dataset, split_name, dataset_name = _load_named_split_with_fallback(
        ["domenicrosati/TruthfulQA", "truthfulqa/truthful_qa"],
        ["train", "validation"],
    )
    rows: list[dict[str, Any]] = []

    for row_idx, row in enumerate(_iter_dataset(dataset, max_rows=max_rows)):
        question = _first_non_empty(row, ["Question", "question"])
        correct_answers = _split_answers(row.get("Correct Answers", row.get("correct_answers")))
        incorrect_answers = _split_answers(row.get("Incorrect Answers", row.get("incorrect_answers")))
        best_answer = _first_non_empty(row, ["Best Answer", "best_answer"])
        if not question or not correct_answers:
            continue
        item = {
            "question": question,
            "correct_answers": correct_answers,
            "incorrect_answers": incorrect_answers,
            "index": row_idx,
            "source_split": split_name,
        }
        if best_answer:
            item["best_answer"] = best_answer
        rows.append(item)

    target = output_root / "analysis" / "TruthfulQA" / "truthful_qa.jsonl"
    _save_jsonl(target, rows)
    return {
        "dataset": "truthfulqa",
        "rows": len(rows),
        "path": str(target),
        "split": split_name,
        "source_dataset_id": dataset_name,
    }


NORMALIZERS = {
    "prm800k": _normalize_prm800k,
    "ultrafeedback": _normalize_ultrafeedback,
    "gsm8k": _normalize_gsm8k,
    "hellaswag": _normalize_hellaswag,
    "svamp": _normalize_svamp,
    "mathqa": _normalize_mathqa,
    "math500": _normalize_math500,
    "amc23": _normalize_amc23,
    "truthfulqa": _normalize_truthfulqa,
}


def _resolve_requested_datasets(requested: list[str]) -> list[str]:
    if not requested or requested == ["all-direct"]:
        return list(DEFAULT_DATASETS)

    resolved: list[str] = []
    for name in requested:
        normalized = name.lower()
        if normalized == "all-direct":
            resolved.extend(DEFAULT_DATASETS)
        else:
            resolved.append(normalized)

    deduped: list[str] = []
    seen: set[str] = set()
    for name in resolved:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def _write_manifest(output_root: Path, results: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> None:
    manifest = {
        "output_root": str(output_root),
        "datasets_processed": results,
        "datasets_skipped": skipped,
    }
    _save_json(output_root / "manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and normalize directly downloadable BREP datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all-direct"],
        help=(
            "Datasets to prepare. Use `all-direct` for the default public batch. "
            "Recipe-only datasets such as math10k and commonsense are skipped."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for normalized BREP data outputs.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Cache directory for raw downloads such as PRM800K jsonl files.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for quick smoke preparation.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download raw remote files such as PRM800K even if cached locally.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print supported dataset names and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_datasets:
        print("Directly downloadable datasets:")
        for name in DEFAULT_DATASETS:
            print(f"- {name}")
        print("Recipe-only datasets:")
        for name in sorted(RECIPE_ONLY_DATASETS):
            print(f"- {name}")
        return 0

    requested = _resolve_requested_datasets(args.datasets)
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for dataset_name in requested:
        if dataset_name in RECIPE_ONLY_DATASETS:
            message = (
                f"Skipping `{dataset_name}`: this dataset is a reconstruction recipe, "
                "not a single direct download."
            )
            _print(message)
            skipped.append({"dataset": dataset_name, "reason": "recipe-only"})
            continue

        normalizer = NORMALIZERS.get(dataset_name)
        if normalizer is None:
            message = f"Skipping `{dataset_name}`: unsupported dataset name."
            _print(message)
            skipped.append({"dataset": dataset_name, "reason": "unsupported"})
            continue

        _print(f"[prepare] {dataset_name}")
        if dataset_name == "prm800k":
            result = normalizer(
                output_root=args.output_root,
                raw_root=args.raw_root,
                max_rows=args.max_rows,
                force=args.force_download,
            )
        else:
            result = normalizer(output_root=args.output_root, max_rows=args.max_rows)
        results.append(result)

    _write_manifest(args.output_root, results, skipped)
    _print(f"Prepared {len(results)} dataset(s). Manifest: {args.output_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
