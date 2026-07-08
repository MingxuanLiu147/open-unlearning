"""CLEAR benchmark evaluator for multimodal sidecar integration."""

import json
import logging
import os
import random
import re

import torch

from data.clear_dataset import (
    CAPTION_MODE,
    TEXT_MODE,
    CLEARDataset,
    decode_clear_image,
    load_clear_dataframe,
)

logger = logging.getLogger(__name__)

random.seed(42)


def _maybe_limit_dataframe(df, max_records: int | None):
    if max_records is None or max_records <= 0 or len(df) <= max_records:
        return df
    return df.head(max_records).reset_index(drop=True)


def _safe_rouge_scorer():
    try:
        from rouge_score import rouge_scorer

        return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    except ImportError:
        logger.warning("rouge_score is not installed; CLEAR generation ROUGE metrics will be zero.")
        return None


def _compute_bleu(reference: str, prediction: str) -> float:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        return sentence_bleu(
            [reference.split()],
            prediction.split(),
            smoothing_function=SmoothingFunction().method1,
        )
    except ImportError:
        logger.warning("nltk is not installed; CLEAR BLEU metrics will be zero.")
        return 0.0


def _formulate_prompt_with_options(question: str, options: list[str], answer: str) -> tuple[str, str]:
    choices = list(options)
    insert_at = random.randint(0, len(choices))
    choices.insert(insert_at, answer)
    prompt = "\n".join([question] + [f"{chr(ord('A') + idx)}. {option}" for idx, option in enumerate(choices)])
    return prompt, chr(ord("A") + insert_at)


def _prepare_inputs(processor, model, question: str, image, image_text: bool):
    content = [{"type": "text", "text": question}]
    images = None
    if image_text and image is not None:
        content.insert(0, {"type": "image"})
        images = [image]

    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt")
    return {key: value.to(model.device) for key, value in inputs.items()}


def _generate_text(processor, model, question: str, image=None, *, max_new_tokens: int = 50):
    inputs = _prepare_inputs(processor, model, question, image, image is not None)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    output_tokens = outputs[:, inputs["input_ids"].shape[-1] :]
    return processor.tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()


def evaluate_classification(
    model,
    processor,
    data_path: str,
    with_options: bool,
    max_new_tokens: int = 50,
    max_records: int | None = None,
):
    df = _maybe_limit_dataframe(load_clear_dataframe(data_path), max_records)
    correct = 0
    total = 0

    for record in df.to_dict("records"):
        image = decode_clear_image(record.get("image"))
        question = str(record.get("question") or "What is the name of the person in the image?").strip()
        answer = str(record.get("answer") or record.get("name") or "").strip()
        if not answer:
            continue

        prompt = question
        expected = answer
        options = []

        if with_options:
            raw_options = record.get("perturbed_names")
            if raw_options is None:
                raw_options = record.get("options")
            options = (
                [str(option) for option in list(raw_options) if str(option).strip()]
                if raw_options is not None
                else []
            )
            if not options:
                continue
            prompt, expected = _formulate_prompt_with_options(question, options, answer)

        generated = _generate_text(
            processor,
            model,
            prompt,
            image=image,
            max_new_tokens=max_new_tokens,
        )
        cleaned = re.sub(r"[^a-zA-Z0-9]", "", generated)

        if with_options:
            predicted = cleaned[0].upper() if cleaned else None
            if predicted == expected:
                correct += 1
        else:
            if answer.lower() in generated.lower():
                correct += 1
        total += 1

    accuracy = correct / total if total else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def _evaluate_generation_dataset(
    dataset: CLEARDataset,
    processor,
    model,
    scorer,
    *,
    label: str,
    max_new_tokens: int,
):
    records = []
    total = 0
    rouge1 = 0.0
    rouge2 = 0.0
    rouge_l = 0.0
    bleu = 0.0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        generated = _generate_text(
            processor,
            model,
            sample["question"],
            image=sample["image"],
            max_new_tokens=max_new_tokens,
        )
        answer = sample["answer"]

        if scorer is not None:
            rouge_scores = scorer.score(answer, generated)
            rouge1 += rouge_scores["rouge1"].fmeasure
            rouge2 += rouge_scores["rouge2"].fmeasure
            rouge_l += rouge_scores["rougeL"].fmeasure
        bleu += _compute_bleu(answer, generated)
        total += 1

        records.append(
            {
                "idx": idx,
                "type": label,
                "question": sample["question"],
                "generated_answer": generated,
                "ground_truth": answer,
            }
        )

    if total == 0:
        return {"total": 0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0}, records

    return (
        {
            "total": total,
            "rouge1": rouge1 / total,
            "rouge2": rouge2 / total,
            "rougeL": rouge_l / total,
            "bleu": bleu / total,
        },
        records,
    )


def evaluate_generation(
    model,
    processor,
    data_path: str,
    *,
    output_dir: str,
    split_name: str,
    max_new_tokens: int = 50,
    max_records: int | None = None,
):
    df = _maybe_limit_dataframe(load_clear_dataframe(data_path), max_records)
    scorer = _safe_rouge_scorer()
    vqa_dataset = CLEARDataset(df, mode=CAPTION_MODE)
    qa_dataset = CLEARDataset(df, mode=TEXT_MODE)

    vqa_metrics, vqa_records = _evaluate_generation_dataset(
        vqa_dataset,
        processor,
        model,
        scorer,
        label="vqa",
        max_new_tokens=max_new_tokens,
    )
    qa_metrics, qa_records = _evaluate_generation_dataset(
        qa_dataset,
        processor,
        model,
        scorer,
        label="qa",
        max_new_tokens=max_new_tokens,
    )

    os.makedirs(output_dir, exist_ok=True)
    detail_path = os.path.join(output_dir, f"{split_name}_generation_results.json")
    with open(detail_path, "w", encoding="utf-8") as handle:
        json.dump(vqa_records + qa_records, handle, ensure_ascii=False, indent=2)

    return {"vqa": vqa_metrics, "qa": qa_metrics}


class CLEAREvaluator:
    """CLEAR benchmark evaluator aligned to sidecar config-driven entrypoints."""

    def __init__(self, eval_cfg, model, processor):
        self.eval_cfg = eval_cfg
        self.model = model
        self.processor = processor
        self.data_dir = eval_cfg.get("data_dir", "data/CLEAR")
        self.output_dir = eval_cfg.get("output_dir", "saves/mm_eval/clear")
        self.forget_ratio = int(eval_cfg.get("forget_split_ratio", 5))
        self.tasks = list(eval_cfg.get("tasks", ["classification", "generation"]))
        self.splits = list(eval_cfg.get("splits", ["forget", "retain", "realface", "realworld"]))
        self.max_new_tokens = int(eval_cfg.get("max_new_tokens", 50))
        self.max_records = eval_cfg.get("max_records")
        self.max_records = int(self.max_records) if self.max_records is not None else None

    def _folder_name(self, key: str) -> str:
        retain_ratio = 100 - self.forget_ratio
        defaults = {
            "forget_classification": f"forget{self.forget_ratio:02d}_perturbed",
            "forget_generation": f"forget{self.forget_ratio:02d}+tofu",
            "retain_classification": "retain_perturbed",
            "retain_generation": f"retain{retain_ratio}+tofu",
            "realface": "real_faces",
            "realworld": "real_world",
        }
        override = self.eval_cfg.get(f"{key}_folder")
        return str(override or defaults[key])

    def evaluate(self):
        os.makedirs(self.output_dir, exist_ok=True)
        results = {}

        for split in self.splits:
            split_results = {}

            if split == "forget":
                cls_path = os.path.join(self.data_dir, self._folder_name("forget_classification"))
                gen_path = os.path.join(self.data_dir, self._folder_name("forget_generation"))
                if "classification" in self.tasks and os.path.exists(cls_path):
                    split_results["classification"] = evaluate_classification(
                        self.model,
                        self.processor,
                        cls_path,
                        with_options=True,
                        max_new_tokens=self.max_new_tokens,
                        max_records=self.max_records,
                    )
                if "generation" in self.tasks and os.path.exists(gen_path):
                    split_results["generation"] = evaluate_generation(
                        self.model,
                        self.processor,
                        gen_path,
                        output_dir=self.output_dir,
                        split_name="forget",
                        max_new_tokens=self.max_new_tokens,
                        max_records=self.max_records,
                    )

            elif split == "retain":
                cls_path = os.path.join(self.data_dir, self._folder_name("retain_classification"))
                gen_path = os.path.join(self.data_dir, self._folder_name("retain_generation"))
                if "classification" in self.tasks and os.path.exists(cls_path):
                    split_results["classification"] = evaluate_classification(
                        self.model,
                        self.processor,
                        cls_path,
                        with_options=True,
                        max_new_tokens=self.max_new_tokens,
                        max_records=self.max_records,
                    )
                if "generation" in self.tasks and os.path.exists(gen_path):
                    split_results["generation"] = evaluate_generation(
                        self.model,
                        self.processor,
                        gen_path,
                        output_dir=self.output_dir,
                        split_name="retain",
                        max_new_tokens=self.max_new_tokens,
                        max_records=self.max_records,
                    )

            elif split == "realface":
                realface_path = os.path.join(self.data_dir, self._folder_name("realface"))
                if os.path.exists(realface_path):
                    split_results["classification"] = evaluate_classification(
                        self.model,
                        self.processor,
                        realface_path,
                        with_options=True,
                        max_new_tokens=self.max_new_tokens,
                        max_records=self.max_records,
                    )

            elif split == "realworld":
                realworld_path = os.path.join(self.data_dir, self._folder_name("realworld"))
                if os.path.exists(realworld_path):
                    split_results["classification"] = evaluate_classification(
                        self.model,
                        self.processor,
                        realworld_path,
                        with_options=True,
                        max_new_tokens=self.max_new_tokens,
                        max_records=self.max_records,
                    )

            if split_results:
                results[split] = split_results

        output_file = os.path.join(self.output_dir, "CLEAR_EVAL.json")
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        logger.info("CLEAR eval results saved to %s", output_file)
        return results
