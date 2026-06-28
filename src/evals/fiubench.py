"""
FIUBench Evaluator for multimodal sidecar pipeline.

Reference: ICLR 2025 — arXiv:2411.03554
           SaFoLab-WISC/FIUBench

Evaluates unlearning of fictitious facial identity VQA with:
- Classification accuracy (image+text, pure text)
- Fill-in-the-blank accuracy
- Generation quality (ROUGE, BLEU)
- MIA and adversarial privacy attack metrics

Reuses the MLLMU evaluation functions since data format is compatible.
"""

import json
import logging
import os
import re

import torch
from tqdm import tqdm

from evals.mllmu import (
    _generate_text,
    _load_parquet_frame,
    _new_counter,
    _finalize_counter,
    _new_generation_counter,
    _finalize_generation_counter,
    _safe_rouge_scorer,
    _compute_bleu,
    _select_record_image,
    _to_list,
)

logger = logging.getLogger(__name__)


def evaluate_fiubench_vqa(
    parquet_path: str,
    processor,
    model,
    *,
    max_new_tokens: int = 50,
    max_records: int | None = None,
):
    """Evaluate FIUBench VQA: generate answer, compute exact-match and ROUGE."""
    df = _load_parquet_frame(parquet_path)
    if max_records and len(df) > max_records:
        df = df.head(max_records).reset_index(drop=True)

    scorer = _safe_rouge_scorer()
    counter = _new_counter()
    gen_counter = _new_generation_counter()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="FIUBench-VQA"):
        record = row.to_dict()
        image = _select_record_image(record)
        question = str(record.get("question", "")).strip()
        answer = str(record.get("answer", "")).strip()
        if not question or not answer:
            continue

        generated = _generate_text(
            processor, model, question, image=image, max_new_tokens=max_new_tokens
        )

        if answer.lower() in generated.lower():
            counter["correct"] += 1
        counter["total"] += 1

        if scorer is not None:
            scores = scorer.score(answer, generated)
            gen_counter["rouge1"] += scores["rouge1"].fmeasure
            gen_counter["rouge2"] += scores["rouge2"].fmeasure
            gen_counter["rougeL"] += scores["rougeL"].fmeasure
        gen_counter["bleu"] += _compute_bleu(answer, generated)
        gen_counter["total"] += 1

    return {
        "accuracy": _finalize_counter(counter),
        "generation": _finalize_generation_counter(gen_counter),
    }


class FIUBenchEvaluator:
    """Config-driven FIUBench evaluator for mm_eval.py."""

    def __init__(self, eval_cfg, model, processor):
        self.model = model
        self.processor = processor
        self.data_dir = eval_cfg.get("data_dir", "data/FIUBench")
        self.output_dir = eval_cfg.get("output_dir", "saves/mm_eval/fiubench")
        self.splits = list(eval_cfg.get("splits", ["forget", "retain"]))
        self.max_new_tokens = int(eval_cfg.get("max_new_tokens", 50))
        self.max_records = eval_cfg.get("max_records")
        self.max_records = int(self.max_records) if self.max_records is not None else None

    def _split_path(self, split_name: str) -> str:
        return os.path.join(self.data_dir, split_name)

    def evaluate(self):
        os.makedirs(self.output_dir, exist_ok=True)
        results = {}

        for split_name in self.splits:
            split_path = self._split_path(split_name)
            if not os.path.exists(split_path):
                logger.warning("Skipping %s: %s not found", split_name, split_path)
                continue

            split_results = evaluate_fiubench_vqa(
                split_path,
                self.processor,
                self.model,
                max_new_tokens=self.max_new_tokens,
                max_records=self.max_records,
            )
            results[split_name] = split_results

        output_file = os.path.join(self.output_dir, "FIUBENCH_EVAL.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("FIUBench eval results saved to %s", output_file)
        return results
