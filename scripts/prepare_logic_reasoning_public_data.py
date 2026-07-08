#!/usr/bin/env python3
"""Download and normalize public logic-reasoning benchmarks."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "inject" / "logic"
DEFAULT_RAW_ROOT = DEFAULT_OUTPUT_ROOT / "_raw"
DEFAULT_HF_HOME = PROJECT_ROOT / ".cache" / "huggingface"

os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_HOME / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))

DEFAULT_DATASETS = [
    "clutrr",
    "ruletaker",
    "proofwriter",
    "folio",
    "logicbench",
    "rulearena",
]

CLUTRR_BASE_URL = "https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/main/"
CLUTRR_CONFIGS = [
    "gen_train23_test2to10",
    "gen_train234_test2to10",
    "rob_train_clean_23_test_all_23",
    "rob_train_disc_23_test_all_23",
    "rob_train_irr_23_test_all_23",
    "rob_train_sup_23_test_all_23",
]

LOGICBENCH_REPO = "https://huggingface.co/datasets/cogint/LogicBench-v1.0"
RULEARENA_REPO = "https://github.com/SkyRiver-2000/RuleArena"

AIRLINE_PROMPT_TEMPLATE = """
The policies of American Airlines are as follows:

{reference_rules}

{question_prompt} Compute the total cost for the passenger step by step (don't omit any bag) and end your response with "The total cost is $xxx." (xxx is a number)
""".strip()

NBA_PROMPT_TEMPLATE = """
You are given rules in NBA Collective Bargaining Agreement and the information about some teams and players. Then you will be given a list of operations, each of which describes how some teams conduct some transaction. You should determine whether each operation complies with the given rules.

Assume:
* the Salary Cap for the prior (2023-24) Salary Cap Year is $136,000,000;
* the Average Player Salary for the prior (2023-24) Salary Cap Year is $9,700,000;
* the Salary Cap for the current (2024-25) NBA Salary Cap Year is $140,588,000;
* the Luxury Tax is $170,814,000;
* the First Apron Level is $178,132,000;
* the Second Apron Level is $188,931,000;
* the Team Salary of each team listed under "Team Situations:" do not include the amount of contracts that expire at the end of 2023-2024 Salary Cap Year.

Reference Rules in NBA Collective Bargaining Agreement:

{reference_rules}

Decide whether any operation by any team violates the rules:

{question_prompt}

Analyze the described operations and explicitly state the type of Salary Cap Exceptions if you think the exception should be involved. Conclude your response with:
* "Answer: False." if there is no violation to the rules;
* "Answer: True. Illegal Operation: X. Problematic Team: Y." if Team Y in Operation X violates the rules. Both X and Y should be a single capital letter as A/B/C/...
""".strip()

TAX_PROMPT_TEMPLATE = """
You are given several forms used to report US income tax and the instructions or rules about how to fill the forms. Then you will be given the income and/or payment information about a tax payer According to the given information. You should calculate the income tax owed by this payer.

IRS Forms for the tax payer:
{forms}

Calculate the tax owed by the payer step-by-step according to the information provided by the forms. You should calculate all fields marked with [__]. DO NOT round numbers without explicit instructions. End your response with:
1. "The total tax owed is $xxx." (xxx is a number) if there is tax owed.
2. "The total tax overpaid is $xxx." (xxx is a number) if there is tax overpaid (and should be refunded).
""".strip()

TBD_MARK = "[__]"

COMPLEMENTARY_FIRST = {
    "China",
    "Hong Kong",
    "Japan",
    "South Korea",
    "India",
    "Qatar",
    "Haiti",
    "Cuba",
    "Panama",
    "Colombia",
    "Ecuador",
    "Peru",
    "South America",
    "Israel",
}


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


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _iter_limited(rows: Iterable[Any], max_rows: int | None) -> Iterator[Any]:
    for idx, row in enumerate(rows):
        if max_rows is not None and idx >= max_rows:
            break
        yield row


def _download_file(urls: list[str], destination: Path, force: bool) -> Path:
    if destination.exists() and not force:
        return destination

    _ensure_parent(destination)
    headers = {"User-Agent": "open-unlearning-logic-data-prep/1.0"}
    last_error: Exception | None = None

    for url in urls:
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
                handle.write(response.read())
            return destination
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc

    raise RuntimeError(
        f"Failed to download {destination.name} from all configured URLs."
    ) from last_error


def _ensure_git_repo(remote_url: str, destination: Path, force: bool) -> Path:
    if destination.exists():
        if not force:
            return destination
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", remote_url, str(destination)],
        check=True,
    )
    return destination


def _format_currency(value: float) -> str:
    rounded = round(float(value), 2)
    if rounded.is_integer():
        return f"{int(rounded):,}"
    return f"{rounded:,.2f}"


def _parse_clutrr_query(query_text: str) -> tuple[str, str]:
    text = _stringify(query_text)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            return _stringify(parsed[0]), _stringify(parsed[1])
    except (SyntaxError, ValueError):
        pass
    return text, ""


def _build_clutrr_instruction(story: str, query_text: str) -> str:
    left, right = _parse_clutrr_query(query_text)
    if left and right:
        question = f"What is the relationship between {left} and {right}?"
    else:
        question = f"What relationship is asked by the query {query_text}?"
    return (
        f"Story:\n{story}\n\nQuestion:\n{question}\n"
        "Answer with a single kinship term."
    )


def _normalize_clutrr(
    output_root: Path,
    raw_root: Path,
    max_rows: int | None,
    force: bool,
) -> list[dict[str, Any]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for config_name in CLUTRR_CONFIGS:
        for split_name in ("train", "test"):
            raw_path = raw_root / "clutrr" / config_name / f"{split_name}.csv"
            _download_file(
                [f"{CLUTRR_BASE_URL}{config_name}/{split_name}.csv"],
                raw_path,
                force=force,
            )
            with raw_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row_idx, row in enumerate(_iter_limited(reader, max_rows)):
                    story = _stringify(row.get("story"))
                    target_text = _stringify(row.get("target_text"))
                    if not story or not target_text:
                        continue
                    item = {
                        "instruction": _build_clutrr_instruction(
                            story, _stringify(row.get("query"))
                        ),
                        "input": "",
                        "output": target_text,
                        "answer": target_text,
                        "source_dataset": "clutrr",
                        "source_config": config_name,
                        "source_split": split_name,
                        "task_name": _stringify(row.get("task_name")),
                        "row_id": _stringify(row.get("id")),
                        "row_index": row_idx,
                    }
                    if split_name == "train":
                        train_rows.append(item)
                    else:
                        eval_rows.append(item)

    train_path = output_root / "train" / "clutrr" / "train.json"
    eval_path = output_root / "eval" / "clutrr" / "test.json"
    _save_json(train_path, train_rows)
    _save_json(eval_path, eval_rows)
    return [
        {
            "dataset": "clutrr",
            "split": "train",
            "rows": len(train_rows),
            "path": str(train_path),
            "source_configs": CLUTRR_CONFIGS,
        },
        {
            "dataset": "clutrr",
            "split": "test",
            "rows": len(eval_rows),
            "path": str(eval_path),
            "source_configs": CLUTRR_CONFIGS,
        },
    ]


def _normalize_ruletaker(output_root: Path, max_rows: int | None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    split_map = {"train": "train", "test": "eval"}
    for hf_split, local_split in split_map.items():
        rows: list[dict[str, Any]] = []
        dataset = load_dataset("tasksource/ruletaker", split=hf_split, streaming=True)
        for row_idx, row in enumerate(_iter_limited(dataset, max_rows)):
            context = _stringify(row.get("context"))
            question = _stringify(row.get("question"))
            label = _stringify(row.get("label")).lower()
            if not context or not question or not label:
                continue
            rows.append(
                {
                    "instruction": (
                        f"Context:\n{context}\n\nQuestion:\n"
                        f"Is the following statement entailed, contradicted, or unknown given the context?\n"
                        f"{question}\n\nAnswer with one of: entailment, contradiction, unknown."
                    ),
                    "input": "",
                    "output": label,
                    "answer": label,
                    "source_dataset": "ruletaker",
                    "source_split": hf_split,
                    "source_config": _stringify(row.get("config")),
                    "row_index": row_idx,
                }
            )
        target = output_root / local_split / "ruletaker" / (
            "train.json" if local_split == "train" else "test.json"
        )
        _save_json(target, rows)
        results.append(
            {
                "dataset": "ruletaker",
                "split": hf_split,
                "rows": len(rows),
                "path": str(target),
                "source_dataset_id": "tasksource/ruletaker",
            }
        )
    return results


def _normalize_proofwriter(output_root: Path, max_rows: int | None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    split_map = {"train": "train", "test": "eval"}
    for hf_split, local_split in split_map.items():
        rows: list[dict[str, Any]] = []
        dataset = load_dataset("tasksource/proofwriter", split=hf_split, streaming=True)
        for row_idx, row in enumerate(_iter_limited(dataset, max_rows)):
            theory = _stringify(row.get("theory"))
            question = _stringify(row.get("question"))
            answer = _stringify(row.get("answer"))
            if not theory or not question or not answer:
                continue
            item = {
                "instruction": (
                    f"Theory:\n{theory}\n\nQuestion:\n{question}\n\n"
                    "Answer with one of: True, False, Unknown."
                ),
                "input": "",
                "output": answer,
                "answer": answer,
                "source_dataset": "proofwriter",
                "source_split": hf_split,
                "source_config": _stringify(row.get("config")),
                "row_id": _stringify(row.get("id")),
                "row_index": row_idx,
            }
            proofs = _stringify(row.get("allProofs"))
            if proofs:
                item["proof"] = proofs
            rows.append(item)
        target = output_root / local_split / "proofwriter" / (
            "train.json" if local_split == "train" else "test.json"
        )
        _save_json(target, rows)
        results.append(
            {
                "dataset": "proofwriter",
                "split": hf_split,
                "rows": len(rows),
                "path": str(target),
                "source_dataset_id": "tasksource/proofwriter",
            }
        )
    return results


def _normalize_folio(output_root: Path, max_rows: int | None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    split_map = {"train": "train", "validation": "eval"}
    for hf_split, local_split in split_map.items():
        rows: list[dict[str, Any]] = []
        dataset = load_dataset("tasksource/folio", split=hf_split, streaming=True)
        for row_idx, row in enumerate(_iter_limited(dataset, max_rows)):
            premises = _stringify(row.get("premises"))
            conclusion = _stringify(row.get("conclusion"))
            label = _stringify(row.get("label"))
            if not premises or not conclusion or not label:
                continue
            rows.append(
                {
                    "instruction": (
                        f"Premises:\n{premises}\n\nConclusion:\n{conclusion}\n\n"
                        "Question:\nDoes the conclusion logically follow from the premises?\n"
                        "Answer with one of: True, False, Uncertain."
                    ),
                    "input": "",
                    "output": label,
                    "answer": label,
                    "source_dataset": "folio",
                    "source_split": hf_split,
                    "story_id": _stringify(row.get("story_id")),
                    "example_id": _stringify(row.get("example_id")),
                    "row_index": row_idx,
                }
            )
        target = output_root / local_split / "folio" / (
            "train.json" if local_split == "train" else "test.json"
        )
        _save_json(target, rows)
        results.append(
            {
                "dataset": "folio",
                "split": hf_split,
                "rows": len(rows),
                "path": str(target),
                "source_dataset_id": "tasksource/folio",
            }
        )
    return results


def _iter_logicbench_aug(repo_root: Path) -> Iterator[dict[str, Any]]:
    base = repo_root / "data" / "LogicBench(Aug)"
    for path in sorted(base.rglob("data_instances.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        logic_type = path.parent.parent.name
        axiom = path.parent.name
        for sample_idx, sample in enumerate(payload.get("data_samples", [])):
            context = _stringify(sample.get("context"))
            for qa_idx, qa in enumerate(sample.get("qa_pairs", [])):
                question = _stringify(qa.get("question"))
                answer = _stringify(qa.get("answer"))
                if context and question and answer:
                    yield {
                        "instruction": f"Context:\n{context}\n\nQuestion:\n{question}",
                        "input": "",
                        "output": answer,
                        "answer": answer,
                        "source_dataset": "logicbench",
                        "source_split": "train",
                        "logic_type": logic_type,
                        "axiom": axiom,
                        "eval_format": "aug_bqa",
                        "sample_index": sample_idx,
                        "qa_index": qa_idx,
                    }


def _iter_logicbench_eval(repo_root: Path) -> Iterator[dict[str, Any]]:
    base = repo_root / "data" / "LogicBench(Eval)"

    for path in sorted((base / "BQA").rglob("data_instances.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        logic_type = path.parent.parent.name
        axiom = path.parent.name
        for sample_idx, sample in enumerate(payload.get("samples", [])):
            context = _stringify(sample.get("context"))
            for qa_idx, qa in enumerate(sample.get("qa_pairs", [])):
                question = _stringify(qa.get("question"))
                answer = _stringify(qa.get("answer"))
                if context and question and answer:
                    yield {
                        "instruction": f"Context:\n{context}\n\nQuestion:\n{question}",
                        "input": "",
                        "output": answer,
                        "answer": answer,
                        "source_dataset": "logicbench",
                        "source_split": "test",
                        "logic_type": logic_type,
                        "axiom": axiom,
                        "eval_format": "bqa",
                        "sample_index": sample_idx,
                        "qa_index": qa_idx,
                    }

    for path in sorted((base / "MCQA").rglob("MCQ_data_instances.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        logic_type = path.parent.parent.name
        axiom = path.parent.name
        for sample_idx, sample in enumerate(payload.get("samples", [])):
            context = _stringify(sample.get("context"))
            question = _stringify(sample.get("question"))
            choices = sample.get("choices", {})
            answer_key = _stringify(sample.get("answer"))
            answer_text = _stringify(choices.get(answer_key))
            if not context or not question or not answer_key or not answer_text:
                continue
            choice_lines = "\n".join(
                f"- {key}: {_stringify(value)}" for key, value in choices.items()
            )
            yield {
                "instruction": (
                    f"Context:\n{context}\n\nQuestion:\n{question}\n\nChoices:\n{choice_lines}\n\n"
                    "Answer with the correct choice id followed by its text."
                ),
                "input": "",
                "output": f"{answer_key}: {answer_text}",
                "answer": answer_key,
                "answer_text": answer_text,
                "choices": choices,
                "source_dataset": "logicbench",
                "source_split": "test",
                "logic_type": logic_type,
                "axiom": axiom,
                "eval_format": "mcqa",
                "sample_index": sample_idx,
            }


def _normalize_logicbench(
    output_root: Path,
    raw_root: Path,
    max_rows: int | None,
    force: bool,
) -> list[dict[str, Any]]:
    repo_root = _ensure_git_repo(
        LOGICBENCH_REPO, raw_root / "logicbench" / "repo", force=force
    )
    train_rows = list(_iter_limited(_iter_logicbench_aug(repo_root), max_rows))
    eval_rows = list(_iter_limited(_iter_logicbench_eval(repo_root), max_rows))
    train_path = output_root / "train" / "logicbench" / "train.json"
    eval_path = output_root / "eval" / "logicbench" / "test.json"
    _save_json(train_path, train_rows)
    _save_json(eval_path, eval_rows)
    return [
        {
            "dataset": "logicbench",
            "split": "train",
            "rows": len(train_rows),
            "path": str(train_path),
            "source_repo": LOGICBENCH_REPO,
        },
        {
            "dataset": "logicbench",
            "split": "test",
            "rows": len(eval_rows),
            "path": str(eval_path),
            "source_repo": LOGICBENCH_REPO,
        },
    ]


def _invert_order(values: list[Any], order: list[int]) -> list[Any]:
    return [values[i] for i in np.argsort(order)]


def _load_checking_fee_tables(repo_root: Path) -> list[dict[int, pd.DataFrame]]:
    base_dir = repo_root / "airline" / "fee_tables"
    check_base: list[dict[int, pd.DataFrame]] = []
    for bag_num in range(1, 5):
        us_departure = pd.read_csv(base_dir / f"bag_{bag_num}" / "0.csv", index_col=0)
        us_arrival = pd.read_csv(base_dir / f"bag_{bag_num}" / "1.csv", index_col=0)
        check_base.append({0: us_departure, 1: us_arrival})
    return check_base


def _compute_airline_oversize(bag: Mapping[str, Any], routine: str) -> int:
    size_total = sum(bag["size"])
    if size_total <= 62:
        return 0
    if size_total <= 65:
        return 30
    if routine in {
        "Panama",
        "South America",
        "Peru",
        "Colombia",
        "Ecuador",
        "Europe",
        "Israel",
        "Qatar",
    }:
        return 150
    return 200


def _compute_airline_overweight(
    bag: Mapping[str, Any],
    routine: str,
    customer_class: str,
    complementary: bool,
) -> int:
    weight = bag["weight"]
    if routine in {"Australia", "New Zealand"}:
        if complementary:
            return 0 if weight <= 70 else 200
        if weight <= 50:
            return 0
        if weight <= 53:
            return 30
        if weight <= 70:
            return 200 if routine == "Cuba" else 100
        return 450 if routine in COMPLEMENTARY_FIRST else 200
    if complementary and customer_class in {"Business", "First"}:
        if weight <= 70:
            return 0
        return 450 if routine in COMPLEMENTARY_FIRST else 200
    if weight <= 50:
        return 0
    if weight <= 53:
        return 30
    if weight <= 70:
        return 200 if routine == "Cuba" else 100
    return 450 if routine in COMPLEMENTARY_FIRST else 200


def _compute_airline_base(
    bag_list: list[dict[str, Any]],
    direction: int,
    routine: str,
    customer_class: str,
    check_base_tables: list[dict[int, pd.DataFrame]],
) -> list[Any]:
    base_fees = []
    for bag_idx, _ in enumerate(bag_list):
        table_idx = min(3, bag_idx)
        base_fees.append(check_base_tables[table_idx][direction][customer_class][routine])
    return base_fees


def _compute_airline_check_cost(
    bag_list: list[dict[str, Any]],
    direction: int,
    routine: str,
    customer_class: str,
    check_base_tables: list[dict[int, pd.DataFrame]],
) -> tuple[float, dict[str, Any]]:
    oversize_cost = [_compute_airline_oversize(bag, routine) for bag in bag_list]
    overweight_if_comp = [
        _compute_airline_overweight(bag, routine, customer_class, True)
        for bag in bag_list
    ]
    overweight_if_not_comp = [
        _compute_airline_overweight(bag, routine, customer_class, False)
        for bag in bag_list
    ]
    violation_if_comp = np.maximum(oversize_cost, overweight_if_comp)
    violation_if_not_comp = np.maximum(oversize_cost, overweight_if_not_comp)
    complementary_gain = violation_if_not_comp - violation_if_comp
    order = list(np.argsort(-complementary_gain))
    sorted_bags = [bag_list[i] for i in order]
    base_fees = _compute_airline_base(
        sorted_bags, direction, routine, customer_class, check_base_tables
    )
    complementary = [fee == 0 for fee in base_fees]
    oversize_cost = [_compute_airline_oversize(bag, routine) for bag in sorted_bags]
    overweight_cost = [
        _compute_airline_overweight(bag, routine, customer_class, comp)
        for bag, comp in zip(sorted_bags, complementary)
    ]
    violation_cost = np.maximum(oversize_cost, overweight_cost).sum()
    total_cost = float(np.sum(base_fees) + violation_cost)
    info = {
        "base": _invert_order(list(base_fees), order),
        "oversize": _invert_order(list(oversize_cost), order),
        "overweight": _invert_order(list(overweight_cost), order),
    }
    return total_cost, info


def _compute_airline_total_cost(
    base_price: int,
    direction: int,
    routine: str,
    customer_class: str,
    bag_list: list[dict[str, Any]],
    check_base_tables: list[dict[int, pd.DataFrame]],
) -> tuple[float, dict[str, Any]]:
    extra_cost, info = _compute_airline_check_cost(
        bag_list[1:], direction, routine, customer_class, check_base_tables
    )
    total_cost = float(base_price + extra_cost)
    info.update(
        {
            "ticket_price": base_price,
            "customer_class": customer_class,
            "routine": routine,
            "direction": direction,
            "bag_list": bag_list[1:],
            "total_cost": total_cost,
        }
    )
    return total_cost, info


def _build_nba_query_prompt(problem: Mapping[str, Any]) -> str:
    team_info = "Team Situations:\n" + "\n".join(problem["team_situations"])
    player_info = "Player Situations:\n" + "\n".join(problem["player_situations"])
    operations = "Operations:\n" + "\n".join(problem["operations"])
    return team_info + "\n\n" + player_info + "\n\n" + operations


def _load_tax_modules(repo_root: Path):
    tax_dir = repo_root / "tax"
    if str(tax_dir) not in sys.path:
        sys.path.insert(0, str(tax_dir))
    prompt_module = importlib.import_module("prompt")
    structured_forms = importlib.import_module("structured_forms")
    micro_eval = importlib.import_module("micro_evaluation")
    return prompt_module, structured_forms, micro_eval


def _build_tax_forms(prompt_module: Any, tax_payer: dict[str, Any]) -> str:
    forms = [prompt_module.basic_forms]
    if tax_payer["itemized"]:
        forms.append(prompt_module.itemized_forms)
    if tax_payer["self_employed"]:
        forms.append(prompt_module.self_employ_forms)
    if tax_payer["has_student_loans_or_education_expenses"]:
        forms.append(prompt_module.edu_forms)
    if tax_payer["child_and_dependent"]:
        forms.append(prompt_module.schedule_8812)
    rendered = "".join(forms)

    payload = dict(tax_payer["data"])
    for key, value in payload.items():
        replacement = f"{value:,}" if not isinstance(value, str) else value
        rendered = rendered.replace("$" + key, "$" + replacement)
    rendered = rendered.replace("$TBD", TBD_MARK)
    rendered = rendered.replace("$name", tax_payer["name"])
    rendered = rendered.replace("$age", str(tax_payer["age"]))
    rendered = rendered.replace("$spouse_age", str(tax_payer["spouse_age"]))
    rendered = rendered.replace("$blind", str(tax_payer["blind"]))
    rendered = rendered.replace("$spouse_blind", str(tax_payer["spouse_blind"]))
    rendered = rendered.replace("$filing_status", tax_payer["filing_status"])
    rendered = rendered.replace("$itemized", str(tax_payer["itemized"]))
    rendered = rendered.replace(
        "$num_qualifying_children", str(tax_payer["num_qualifying_children"])
    )
    rendered = rendered.replace(
        "$num_other_dependents", str(tax_payer["num_other_dependents"])
    )
    return rendered


def _format_tax_answer(amount: float) -> str:
    if amount < 0:
        return f"The total tax overpaid is ${_format_currency(abs(amount))}."
    return f"The total tax owed is ${_format_currency(amount)}."


def _normalize_rulearena(
    output_root: Path,
    raw_root: Path,
    max_rows: int | None,
    force: bool,
) -> list[dict[str, Any]]:
    repo_root = _ensure_git_repo(
        RULEARENA_REPO, raw_root / "rulearena" / "repo", force=force
    )
    airline_rules = (
        repo_root / "airline" / "reference_rules_textual.txt"
    ).read_text(encoding="utf-8")
    nba_rules = (repo_root / "nba" / "reference_rules.txt").read_text(encoding="utf-8")
    airline_fee_tables = _load_checking_fee_tables(repo_root)
    prompt_module, structured_forms, micro_eval = _load_tax_modules(repo_root)

    rows: list[dict[str, Any]] = []

    for path in sorted((repo_root / "airline" / "synthesized_problems").glob("comp_*.jsonl")):
        complexity = path.stem.split("_")[-1]
        with path.open("r", encoding="utf-8") as handle:
            for row_idx, line in enumerate(_iter_limited(handle, max_rows)):
                problem = json.loads(line)
                total_cost, _ = _compute_airline_total_cost(
                    check_base_tables=airline_fee_tables, **problem["info"]
                )
                rows.append(
                    {
                        "instruction": AIRLINE_PROMPT_TEMPLATE.format(
                            reference_rules=airline_rules,
                            question_prompt=problem["prompt"],
                        ),
                        "input": "",
                        "output": f"The total cost is ${_format_currency(total_cost)}.",
                        "answer": _format_currency(total_cost),
                        "source_dataset": "rulearena",
                        "domain": "airline",
                        "complexity": int(complexity),
                        "source_split": "test",
                        "row_index": row_idx,
                    }
                )

    for path in sorted((repo_root / "nba" / "annotated_problems").glob("comp_*.json")):
        complexity = path.stem.split("_")[-1]
        problems = json.loads(path.read_text(encoding="utf-8"))
        for row_idx, problem in enumerate(_iter_limited(problems, max_rows)):
            query_prompt = _build_nba_query_prompt(problem)
            if problem["answer"]:
                output = (
                    f"Answer: True. Illegal Operation: {problem['illegal_operation']}. "
                    f"Problematic Team: {problem['problematic_team']}."
                )
            else:
                output = "Answer: False."
            rows.append(
                {
                    "instruction": NBA_PROMPT_TEMPLATE.format(
                        reference_rules=nba_rules,
                        question_prompt=query_prompt,
                    ),
                    "input": "",
                    "output": output,
                    "answer": str(problem["answer"]).lower(),
                    "source_dataset": "rulearena",
                    "domain": "nba",
                    "complexity": int(complexity),
                    "source_split": "test",
                    "row_index": row_idx,
                    "relevant_rules": problem.get("relevant_rules", []),
                }
            )

    for path in sorted((repo_root / "tax" / "synthesized_problems").glob("comp_*.json")):
        complexity = path.stem.split("_")[-1]
        problems = json.loads(path.read_text(encoding="utf-8"))
        for row_idx, problem in enumerate(_iter_limited(problems, max_rows)):
            tax_payer = structured_forms.TaxPayer.model_validate(problem["pydantic"])
            amount, _ = micro_eval.compute_answer(tax_payer)
            forms = _build_tax_forms(prompt_module, problem["dict"])
            rows.append(
                {
                    "instruction": TAX_PROMPT_TEMPLATE.format(forms=forms),
                    "input": "",
                    "output": _format_tax_answer(amount),
                    "answer": _format_currency(abs(amount)),
                    "source_dataset": "rulearena",
                    "domain": "tax",
                    "complexity": int(complexity),
                    "source_split": "test",
                    "row_index": row_idx,
                }
            )

    target = output_root / "eval" / "rulearena" / "test.json"
    _save_json(target, rows)
    return [
        {
            "dataset": "rulearena",
            "split": "test",
            "rows": len(rows),
            "path": str(target),
            "source_repo": RULEARENA_REPO,
        }
    ]


NORMALIZERS = {
    "clutrr": _normalize_clutrr,
    "ruletaker": _normalize_ruletaker,
    "proofwriter": _normalize_proofwriter,
    "folio": _normalize_folio,
    "logicbench": _normalize_logicbench,
    "rulearena": _normalize_rulearena,
}


def _resolve_requested_datasets(requested: list[str]) -> list[str]:
    if not requested or requested == ["all"]:
        return list(DEFAULT_DATASETS)
    resolved: list[str] = []
    for name in requested:
        lowered = name.lower()
        if lowered == "all":
            resolved.extend(DEFAULT_DATASETS)
        else:
            resolved.append(lowered)
    deduped: list[str] = []
    seen: set[str] = set()
    for name in resolved:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def _write_manifest(output_root: Path, results: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> None:
    _save_json(
        output_root / "manifest.json",
        {
            "output_root": str(output_root),
            "datasets_processed": results,
            "datasets_skipped": skipped,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and normalize public logic-reasoning benchmarks."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Datasets to prepare. Use `all` for the default batch.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for normalized logic benchmark outputs.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Cache directory for raw downloads and cloned benchmark repos.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit per split/file for smoke preparation.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download raw files and re-clone source repos.",
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
        for name in DEFAULT_DATASETS:
            print(f"- {name}")
        return 0

    requested = _resolve_requested_datasets(args.datasets)
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for dataset_name in requested:
        normalizer = NORMALIZERS.get(dataset_name)
        if normalizer is None:
            _print(f"Skipping `{dataset_name}`: unsupported dataset name.")
            skipped.append({"dataset": dataset_name, "reason": "unsupported"})
            continue

        _print(f"[prepare] {dataset_name}")
        if dataset_name in {"clutrr", "logicbench", "rulearena"}:
            normalized = normalizer(
                output_root=args.output_root,
                raw_root=args.raw_root,
                max_rows=args.max_rows,
                force=args.force_download,
            )
        else:
            normalized = normalizer(
                output_root=args.output_root,
                max_rows=args.max_rows,
            )
        results.extend(normalized)

    _write_manifest(args.output_root, results, skipped)
    _print(
        f"Prepared {len(results)} output split(s). Manifest: {args.output_root / 'manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
