"""
Knowledge Injection 评估器
=========================

实现知识注入（微调）的评估指标：
- Task Accuracy（任务准确率）：微调任务的性能
- Knowledge Retention（知识保持）：原有知识的保持率

"""

import logging
from typing import Dict, Any, Optional, List

import torch
import torch.nn.functional as F
from evals.base import Evaluator

logger = logging.getLogger(__name__)


class InjectEvaluator(Evaluator):
    """知识注入评估器基类"""

    def __init__(self, eval_cfg, **kwargs):
        self.name = "inject"
        super().__init__(self.name, eval_cfg, **kwargs)


class InjectAccuracyEvaluator(InjectEvaluator):
    """微调任务准确率评估器

    评估微调后模型在目标任务上的性能。
    """

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "inject_accuracy"

    def evaluate(self, model, output_dir=None, **kwargs):
        """执行任务准确率评估

        支持两种评估方式：
        1. generation: 生成式评估
        2. perplexity: 困惑度评估
        """
        tokenizer = kwargs.get("tokenizer")
        eval_data = kwargs.get("eval_data", [])
        eval_method = self.eval_cfg.get("args", {}).get("eval_method", "generation")

        model = self.prepare_model(model)

        if eval_method == "generation":
            return self._eval_generation(model, tokenizer, eval_data)
        else:
            return self._eval_perplexity(model, tokenizer, eval_data)

    def _eval_generation(
        self, model, tokenizer, eval_data: List[Dict]
    ) -> Dict[str, Any]:
        """生成式评估"""
        correct = 0
        total = 0

        for item in eval_data:
            prompt = item.get("prompt", item.get("instruction", ""))
            expected = item.get("expected", item.get("output", ""))

            if not prompt or not expected:
                continue

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = generated[len(prompt) :].strip()

            # 简单的精确匹配或包含匹配
            if expected.lower() in generated.lower():
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0

        result = {
            "task_accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

        logger.info(f"Inject Task Accuracy: {accuracy:.4f} ({correct}/{total})")
        return result

    def _eval_perplexity(
        self, model, tokenizer, eval_data: List[Dict]
    ) -> Dict[str, Any]:
        """困惑度评估"""
        total_loss = 0.0
        total_tokens = 0

        for item in eval_data:
            text = item.get("text", "")
            if not text:
                prompt = item.get("prompt", item.get("instruction", ""))
                output = item.get("output", "")
                text = f"{prompt} {output}"

            if not text.strip():
                continue

            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=2048
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

            num_tokens = inputs["input_ids"].shape[1]
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        result = {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "total_tokens": total_tokens,
        }

        logger.info(f"Inject Perplexity: {perplexity:.4f}")
        return result


class InjectRetentionEvaluator(InjectEvaluator):
    """知识保持评估器

    评估微调后模型对原有知识的保持程度。
    通过在基准数据集上测试来衡量。
    """

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "inject_retention"

    def evaluate(self, model, output_dir=None, **kwargs):
        """执行知识保持评估"""
        tokenizer = kwargs.get("tokenizer")
        benchmark_data = kwargs.get("benchmark_data", [])
        original_model = kwargs.get("original_model")

        model = self.prepare_model(model)

        if not benchmark_data:
            logger.warning("No benchmark data provided for retention evaluation")
            return {"knowledge_retention": 1.0}

        # 计算当前模型的性能
        current_scores = self._compute_scores(model, tokenizer, benchmark_data)

        # 如果有原始模型，计算相对保持率
        if original_model is not None:
            original_model = self.prepare_model(original_model)
            original_scores = self._compute_scores(
                original_model, tokenizer, benchmark_data
            )

            # 计算保持率
            retention = current_scores / original_scores if original_scores > 0 else 1.0
        else:
            retention = current_scores

        result = {
            "knowledge_retention": retention,
            "current_score": current_scores,
        }

        if original_model is not None:
            result["original_score"] = original_scores

        logger.info(f"Knowledge Retention: {retention:.4f}")
        return result

    def _compute_scores(self, model, tokenizer, data: List[Dict]) -> float:
        """计算模型在数据集上的得分"""
        correct = 0
        total = 0

        for item in data:
            prompt = item.get("prompt", item.get("question", ""))
            expected = item.get("expected", item.get("answer", ""))

            if not prompt or not expected:
                continue

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = generated[len(prompt) :].strip()

            if expected.lower() in generated.lower():
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0
