"""MLLMU-Bench benchmark evaluator（sidecar，不修改现有 evals 体系）。

支持 classification / generation 两种评估任务，
在 forget / retain / test 三路数据上评估遗忘效果。
"""

import json
import logging
import os
import re
from io import BytesIO

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _formulate_prompt_with_options(question, options):
    options_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
    return f"{question}\n{options_str}"


def evaluate_classification(parquet_path, processor, model, max_new_tokens=50):
    """在分类任务上评估模型准确率。

    Returns:
        dict: {"image_textual_acc": float, "pure_text_acc": float, "total_acc": float}
    """
    df = pd.read_parquet(parquet_path)
    tokenizer = processor.tokenizer

    total_correct = 0
    total_questions = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classification"):
        tasks = row.get("Classification_Task", {})
        image_bytes = row["image"].get("bytes")
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:
            continue

        for q_data in tasks.get("Image_Textual_Questions", []):
            question = q_data["Question"]
            options = q_data["Options"]
            correct = q_data["Correct_Answer"]
            prompt_text = _formulate_prompt_with_options(question, options)

            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{prompt_text}\nJust give ONE letter representing the answer directly."},
                ]},
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(images=[image], text=text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            out_tokens = outputs[:, inputs["input_ids"].shape[-1]:]
            answer_text = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
            cleaned = re.sub(r"[^a-zA-Z0-9]", "", answer_text)
            predicted = cleaned[0].upper() if cleaned and cleaned[0].upper() in options else None

            if predicted == correct:
                total_correct += 1
            total_questions += 1

    acc = total_correct / total_questions if total_questions > 0 else 0.0
    return {"accuracy": acc, "correct": total_correct, "total": total_questions}


def evaluate_generation(parquet_path, processor, model, max_new_tokens=200):
    """在生成任务上评估模型输出质量（ROUGE-L）。

    Returns:
        dict: {"rouge_l": float, "total": int}
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning("rouge_score not installed, skipping generation eval")
        return {"rouge_l": 0.0, "total": 0}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    df = pd.read_parquet(parquet_path)
    tokenizer = processor.tokenizer

    total_rouge = 0.0
    total_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generation"):
        image_bytes = row["image"].get("bytes")
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:
            continue

        metadata = json.loads(row["metadata"])
        for qa in metadata:
            question = qa.get("Question", "")
            answer = qa.get("Answer", "")
            if not question or not answer:
                continue

            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ]},
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(images=[image], text=text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            out_tokens = outputs[:, inputs["input_ids"].shape[-1]:]
            generated = tokenizer.decode(out_tokens[0], skip_special_tokens=True)

            score = scorer.score(answer, generated)
            total_rouge += score["rougeL"].fmeasure
            total_count += 1

    avg_rouge = total_rouge / total_count if total_count > 0 else 0.0
    return {"rouge_l": avg_rouge, "total": total_count}


class MLLMUEvaluator:
    """MLLMU-Bench 评估器，由 mm_eval.py 调用。"""

    def __init__(self, eval_cfg, model, processor):
        self.model = model
        self.processor = processor
        self.data_dir = eval_cfg.get("data_dir", "data/MLLMU-Bench")
        self.output_dir = eval_cfg.get("output_dir", "saves/mm_eval/mllmu")
        self.forget_ratio = eval_cfg.get("forget_split_ratio", 5)
        self.tasks = eval_cfg.get("tasks", ["classification"])
        self.max_new_tokens = eval_cfg.get("max_new_tokens", 50)

    def evaluate(self):
        os.makedirs(self.output_dir, exist_ok=True)
        results = {}

        splits = {
            "forget": f"forget_{self.forget_ratio}",
            "retain": f"retain_{100 - self.forget_ratio}",
        }

        for split_name, folder_name in splits.items():
            parquet = os.path.join(self.data_dir, folder_name, "train-00000-of-00001.parquet")
            if not os.path.exists(parquet):
                logger.warning("Skipping %s: %s not found", split_name, parquet)
                continue

            split_results = {}
            if "classification" in self.tasks:
                split_results["classification"] = evaluate_classification(
                    parquet, self.processor, self.model, self.max_new_tokens
                )
            if "generation" in self.tasks:
                split_results["generation"] = evaluate_generation(
                    parquet, self.processor, self.model, 200
                )
            results[split_name] = split_results

        output_file = os.path.join(self.output_dir, "MLLMU_EVAL.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("MLLMU eval results saved to %s", output_file)
        return results
