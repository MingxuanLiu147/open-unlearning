"""
Knowledge Injection 评估器
=========================

实现知识注入（微调）的评估指标：
- Task Accuracy（任务准确率）：微调任务的性能
- Knowledge Retention（知识保持）：原有知识的保持率
"""

import logging
from typing import Any, Dict, List, Tuple

import torch
from evals.base import Evaluator

logger = logging.getLogger(__name__)


class InjectEvaluator(Evaluator):
    """知识注入评估器基类"""

    def __init__(self, eval_cfg, **kwargs):
        self.name = "inject"
        super().__init__(self.name, eval_cfg, **kwargs)

    def _get_model_device(self, model) -> torch.device:
        if hasattr(model, "device"):
            return model.device
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _normalize_text(self, value: Any) -> str:
        while isinstance(value, list):
            if not value:
                return ""
            value = value[0]
        if value is None:
            return ""
        return str(value).strip()

    def _join_prompt_target(self, prompt: str, target: str) -> str:
        if not prompt:
            return target
        if not target:
            return prompt
        if prompt[-1].isspace() or target[0].isspace():
            return prompt + target
        return f"{prompt} {target}"

    def _get_pad_token_id(self, tokenizer):
        if tokenizer.pad_token_id is not None:
            return tokenizer.pad_token_id
        return tokenizer.eos_token_id

    def _load_cached_result(self, output_dir=None):
        if self.eval_cfg.get("overwrite", False):
            return None
        target_output_dir = output_dir if output_dir else self.eval_cfg.get("output_dir")
        if target_output_dir is None:
            return None
        logs_file = self.get_logs_file_path(target_output_dir)
        cached = self.load_logs_from_file(logs_file)
        if cached:
            logger.info("Skipping `%s`, loaded cached result from %s", self.name, logs_file)
            return cached
        return None

    def _summary_from_result(self, result: Dict[str, Any]) -> Dict[str, float]:
        if "task_accuracy" in result:
            return {"task_accuracy": float(result["task_accuracy"])}
        if "knowledge_retention" in result:
            return {"knowledge_retention": float(result["knowledge_retention"])}
        if "perplexity" in result:
            return {"perplexity": float(result["perplexity"])}
        return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}

    def _persist_result(self, result: Dict[str, Any], output_dir=None):
        target_output_dir = output_dir if output_dir else self.eval_cfg.get("output_dir")
        if target_output_dir is None:
            logger.warning(
                "No output_dir configured for `%s`; skip persisting results.",
                self.name,
            )
            return
        logs_file = self.get_logs_file_path(target_output_dir)
        summary_file = self.get_logs_file_path(target_output_dir, suffix="SUMMARY")
        self.save_logs(result, logs_file)
        self.save_logs(self._summary_from_result(result), summary_file)
        logger.info("Saved `%s` evaluation to %s", self.name, logs_file)


class InjectAccuracyEvaluator(InjectEvaluator):
    """微调任务准确率评估器"""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "inject_accuracy"

    def evaluate(self, model, output_dir=None, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        eval_data = kwargs.get("eval_data", [])
        eval_method = self.eval_cfg.get("args", {}).get("eval_method", "generation")

        cached = self._load_cached_result(output_dir=output_dir)
        if cached is not None:
            return cached

        if tokenizer is None:
            result = {"task_accuracy": 0.0, "correct": 0, "total": 0}
            self._persist_result(result, output_dir=output_dir)
            return result

        model = self.prepare_model(model)
        if eval_method == "generation":
            result = self._eval_generation(model, tokenizer, eval_data)
        elif eval_method == "perplexity":
            result = self._eval_perplexity(model, tokenizer, eval_data)
        else:
            raise ValueError(
                f"Unsupported eval_method `{eval_method}` for InjectAccuracyEvaluator."
            )

        self._persist_result(result, output_dir=output_dir)
        return result

    def _eval_generation(
        self, model, tokenizer, eval_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        correct = 0
        total = 0
        device = self._get_model_device(model)
        pad_token_id = self._get_pad_token_id(tokenizer)
        max_new_tokens = int(self.eval_cfg.get("args", {}).get("max_new_tokens", 100))

        for item in eval_data:
            prompt = self._normalize_text(item.get("prompt", item.get("instruction", "")))
            expected = self._normalize_text(
                item.get("expected", item.get("output", item.get("answer", "")))
            )
            if not prompt or not expected:
                continue

            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prompt_token_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )[0]

            generated = tokenizer.decode(
                output_ids[prompt_token_len:], skip_special_tokens=True
            ).strip()
            if expected.lower() in generated.lower():
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        logger.info("Inject Task Accuracy: %.4f (%d/%d)", accuracy, correct, total)
        return {
            "task_accuracy": accuracy,
            "correct": correct,
            "total": total,
            "eval_method": "generation",
        }

    def _eval_perplexity(
        self, model, tokenizer, eval_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        total_loss = 0.0
        total_tokens = 0
        device = self._get_model_device(model)
        max_length = int(self.eval_cfg.get("args", {}).get("max_length", 2048))

        for item in eval_data:
            prompt = self._normalize_text(item.get("prompt", item.get("instruction", "")))
            expected = self._normalize_text(
                item.get("expected", item.get("output", item.get("answer", "")))
            )
            text = self._normalize_text(item.get("text", ""))

            prompt_token_len = 0
            if text and not prompt and not expected:
                full_text = text
            else:
                if not prompt or not expected:
                    continue
                full_text = self._join_prompt_target(prompt, expected)
                prompt_ids = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=max_length
                )["input_ids"]
                prompt_token_len = prompt_ids.shape[1]

            inputs = tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=max_length
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]
            if prompt_token_len >= input_ids.shape[1]:
                continue

            labels = input_ids.clone()
            if prompt_token_len > 0:
                labels[:, :prompt_token_len] = -100
            if "attention_mask" in inputs:
                labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

            num_tokens = int((labels != -100).sum().item())
            if num_tokens <= 0:
                continue

            with torch.no_grad():
                loss = model(**inputs, labels=labels).loss

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = (
            torch.exp(torch.tensor(avg_loss)).item() if total_tokens > 0 else 0.0
        )
        logger.info("Inject Perplexity: %.4f", perplexity)
        return {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "total_tokens": total_tokens,
            "eval_method": "perplexity",
        }


class InjectRetentionEvaluator(InjectEvaluator):
    """知识保持评估器"""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "inject_retention"

    def evaluate(self, model, output_dir=None, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        benchmark_data = kwargs.get("benchmark_data", [])
        original_model = kwargs.get("original_model")

        cached = self._load_cached_result(output_dir=output_dir)
        if cached is not None:
            return cached

        if tokenizer is None or not benchmark_data:
            result = {
                "knowledge_retention": 0.0,
                "current_score": 0.0,
                "current_correct": 0,
                "current_total": 0,
            }
            self._persist_result(result, output_dir=output_dir)
            return result

        model = self.prepare_model(model)
        current_score, current_correct, current_total = self._compute_scores(
            model, tokenizer, benchmark_data
        )

        result = {
            "current_score": current_score,
            "current_correct": current_correct,
            "current_total": current_total,
        }

        if original_model is not None:
            original_model = self.prepare_model(original_model)
            original_score, original_correct, original_total = self._compute_scores(
                original_model, tokenizer, benchmark_data
            )
            if original_score > 0:
                retention = current_score / original_score
            elif current_score == 0:
                retention = 1.0
            else:
                retention = 0.0
            result.update(
                {
                    "knowledge_retention": retention,
                    "original_score": original_score,
                    "original_correct": original_correct,
                    "original_total": original_total,
                }
            )
        else:
            result["knowledge_retention"] = current_score

        logger.info("Knowledge Retention: %.4f", result["knowledge_retention"])
        self._persist_result(result, output_dir=output_dir)
        return result

    def _compute_scores(
        self, model, tokenizer, data: List[Dict[str, Any]]
    ) -> Tuple[float, int, int]:
        correct = 0
        total = 0
        device = self._get_model_device(model)
        pad_token_id = self._get_pad_token_id(tokenizer)
        max_new_tokens = int(self.eval_cfg.get("args", {}).get("max_new_tokens", 50))

        for item in data:
            prompt = self._normalize_text(item.get("prompt", item.get("question", "")))
            expected = self._normalize_text(
                item.get("expected", item.get("output", item.get("answer", "")))
            )
            if not prompt or not expected:
                continue

            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prompt_token_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )[0]

            generated = tokenizer.decode(
                output_ids[prompt_token_len:], skip_special_tokens=True
            ).strip()
            if expected.lower() in generated.lower():
                correct += 1
            total += 1

        score = correct / total if total > 0 else 0.0
        return score, correct, total
