"""
Knowledge Editing 评估器
=======================

实现知识编辑的评估指标：
- Reliability（可靠性）：编辑成功率
- Generalization（泛化性）：改写表达的成功率
- Locality（局部性）：无关知识保持率
- Portability（可移植性）：知识迁移能力

参考 EasyEdit 的评估体系设计。
"""

import logging
from typing import Dict, Any, Optional, List

import torch
import torch.nn.functional as F
from evals.base import Evaluator

logger = logging.getLogger(__name__)


class EditEvaluator(Evaluator):
    """知识编辑评估器基类"""

    def __init__(self, eval_cfg, **kwargs):
        self.name = "edit"
        super().__init__(self.name, eval_cfg, **kwargs)

    def compute_edit_success(
        self, model, tokenizer, prompt: str, target: str, **kwargs
    ) -> float:
        """计算单个编辑的成功率

        Args:
            model: 模型实例
            tokenizer: 分词器
            prompt: 输入提示
            target: 期望输出

        Returns:
            成功率 (0 或 1)
        """
        # 生成输出
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

        # 检查目标是否在生成中
        success = target.lower() in generated.lower()
        return 1.0 if success else 0.0

    def compute_target_probability(
        self, model, tokenizer, prompt: str, target: str, **kwargs
    ) -> float:
        """计算目标输出的概率

        Args:
            model: 模型实例
            tokenizer: 分词器
            prompt: 输入提示
            target: 期望输出

        Returns:
            目标 token 的平均概率
        """
        # 编码完整序列
        full_text = f"{prompt} {target}"
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(model.device)

        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits

            # 只计算目标部分的概率
            target_logits = logits[0, prompt_len - 1 : -1, :]  # 预测目标 token
            target_ids = full_ids[0, prompt_len:]  # 实际目标 token

            # 计算每个 token 的概率
            probs = F.softmax(target_logits, dim=-1)
            target_probs = probs.gather(1, target_ids.unsqueeze(1)).squeeze()

            # 返回平均概率
            if target_probs.dim() == 0:
                return target_probs.item()
            return target_probs.mean().item()

    def compute_perplexity(
        self, model, tokenizer, prompt: str, target: str, **kwargs
    ) -> float:
        """计算目标输出的困惑度

        Args:
            model: 模型实例
            tokenizer: 分词器
            prompt: 输入提示
            target: 期望输出

        Returns:
            困惑度
        """
        full_text = f"{prompt} {target}"
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        return torch.exp(loss).item()


class EditReliabilityEvaluator(EditEvaluator):
    """编辑可靠性评估器

    测试编辑后的知识是否能正确输出。
    """

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "edit_reliability"

    def evaluate(self, model, output_dir=None, **kwargs):
        """执行可靠性评估"""
        tokenizer = kwargs.get("tokenizer")
        edit_data = kwargs.get("edit_data", [])

        if not edit_data:
            logger.warning("No edit data provided for reliability evaluation")
            return {"reliability": 0.0}

        model = self.prepare_model(model)

        successes = []
        for item in edit_data:
            prompt = item.get("prompt", "")
            target_new = item.get("target_new", "")

            if prompt and target_new:
                success = self.compute_edit_success(
                    model, tokenizer, prompt, target_new
                )
                successes.append(success)

        reliability = sum(successes) / len(successes) if successes else 0.0

        result = {
            "reliability": reliability,
            "total_samples": len(successes),
            "successful_edits": int(sum(successes)),
        }

        logger.info(f"Edit Reliability: {reliability:.4f}")
        return result


class EditGeneralizationEvaluator(EditEvaluator):
    """编辑泛化性评估器

    测试改写表达是否也能正确输出。
    """

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "edit_generalization"

    def evaluate(self, model, output_dir=None, **kwargs):
        """执行泛化性评估"""
        tokenizer = kwargs.get("tokenizer")
        edit_data = kwargs.get("edit_data", [])

        model = self.prepare_model(model)

        successes = []
        for item in edit_data:
            rephrase_prompts = item.get("rephrase_prompts", [])
            target_new = item.get("target_new", "")

            for rephrase in rephrase_prompts:
                if rephrase and target_new:
                    success = self.compute_edit_success(
                        model, tokenizer, rephrase, target_new
                    )
                    successes.append(success)

        generalization = sum(successes) / len(successes) if successes else 0.0

        result = {
            "generalization": generalization,
            "total_rephrase_samples": len(successes),
        }

        logger.info(f"Edit Generalization: {generalization:.4f}")
        return result


class EditLocalityEvaluator(EditEvaluator):
    """编辑局部性评估器

    测试无关知识是否保持不变。
    """

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "edit_locality"

    def evaluate(self, model, output_dir=None, **kwargs):
        """执行局部性评估"""
        tokenizer = kwargs.get("tokenizer")
        edit_data = kwargs.get("edit_data", [])
        original_model = kwargs.get("original_model")

        model = self.prepare_model(model)

        if original_model is None:
            logger.warning("Original model not provided, skipping locality evaluation")
            return {"locality": 1.0}

        preserved = []
        for item in edit_data:
            locality_inputs = item.get("locality_inputs", [])

            for loc_item in locality_inputs:
                prompt = loc_item.get("prompt", "")
                expected = loc_item.get("expected", "")

                if prompt:
                    # 检查原始模型的输出是否保持
                    success = self.compute_edit_success(
                        model, tokenizer, prompt, expected
                    )
                    preserved.append(success)

        locality = sum(preserved) / len(preserved) if preserved else 1.0

        result = {
            "locality": locality,
            "total_locality_samples": len(preserved),
        }

        logger.info(f"Edit Locality: {locality:.4f}")
        return result


class EditPortabilityEvaluator(EditEvaluator):
    """编辑可移植性评估器

    测试编辑后的知识是否能迁移到相关推理。
    """

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "edit_portability"

    def evaluate(self, model, output_dir=None, **kwargs):
        """执行可移植性评估"""
        tokenizer = kwargs.get("tokenizer")
        edit_data = kwargs.get("edit_data", [])

        model = self.prepare_model(model)

        ported = []
        for item in edit_data:
            portability_inputs = item.get("portability_inputs", [])

            for port_item in portability_inputs:
                prompt = port_item.get("prompt", "")
                expected = port_item.get("expected", "")

                if prompt and expected:
                    success = self.compute_edit_success(
                        model, tokenizer, prompt, expected
                    )
                    ported.append(success)

        portability = sum(ported) / len(ported) if ported else 0.0

        result = {
            "portability": portability,
            "total_portability_samples": len(ported),
        }

        logger.info(f"Edit Portability: {portability:.4f}")
        return result


class EditComprehensiveEvaluator(EditEvaluator):
    """知识编辑综合评估器

    整合所有四个指标进行一站式评估：
    - Reliability: 编辑成功率
    - Generalization: 改写泛化率
    - Locality: 局部性保持率
    - Portability: 知识可移植性

    并计算综合得分。
    """

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "edit_comprehensive"
        # 从配置获取各指标权重
        args = eval_cfg.get("args", {})
        self.reliability_weight = args.get("reliability_weight", 0.25)
        self.generalization_weight = args.get("generalization_weight", 0.25)
        self.locality_weight = args.get("locality_weight", 0.25)
        self.portability_weight = args.get("portability_weight", 0.25)
        self.use_probability = args.get("use_probability", False)

    def evaluate(self, model, output_dir=None, **kwargs):
        """执行综合评估"""
        tokenizer = kwargs.get("tokenizer")
        edit_data = kwargs.get("edit_data", [])
        original_model = kwargs.get("original_model")

        if not edit_data:
            logger.warning("No edit data provided for comprehensive evaluation")
            return self._empty_result()

        model = self.prepare_model(model)

        # 评估各指标
        reliability_scores = []
        generalization_scores = []
        locality_scores = []
        portability_scores = []

        for item in edit_data:
            prompt = item.get("prompt", "")
            target_new = item.get("target_new", "")
            rephrase_prompts = item.get("rephrase_prompts", [])
            locality_inputs = item.get("locality_inputs", [])
            portability_inputs = item.get("portability_inputs", [])

            # 1. Reliability
            if prompt and target_new:
                if self.use_probability:
                    score = self.compute_target_probability(
                        model, tokenizer, prompt, target_new
                    )
                else:
                    score = self.compute_edit_success(
                        model, tokenizer, prompt, target_new
                    )
                reliability_scores.append(score)

            # 2. Generalization
            for rephrase in rephrase_prompts:
                if rephrase and target_new:
                    if self.use_probability:
                        score = self.compute_target_probability(
                            model, tokenizer, rephrase, target_new
                        )
                    else:
                        score = self.compute_edit_success(
                            model, tokenizer, rephrase, target_new
                        )
                    generalization_scores.append(score)

            # 3. Locality
            for loc_item in locality_inputs:
                loc_prompt = loc_item.get("prompt", "")
                loc_expected = loc_item.get("expected", "")
                if loc_prompt and loc_expected:
                    score = self.compute_edit_success(
                        model, tokenizer, loc_prompt, loc_expected
                    )
                    locality_scores.append(score)

            # 4. Portability
            for port_item in portability_inputs:
                port_prompt = port_item.get("prompt", "")
                port_expected = port_item.get("expected", "")
                if port_prompt and port_expected:
                    score = self.compute_edit_success(
                        model, tokenizer, port_prompt, port_expected
                    )
                    portability_scores.append(score)

        # 计算各指标平均值
        reliability = (
            sum(reliability_scores) / len(reliability_scores)
            if reliability_scores
            else 0.0
        )
        generalization = (
            sum(generalization_scores) / len(generalization_scores)
            if generalization_scores
            else 0.0
        )
        locality = (
            sum(locality_scores) / len(locality_scores) if locality_scores else 1.0
        )
        portability = (
            sum(portability_scores) / len(portability_scores)
            if portability_scores
            else 0.0
        )

        # 计算综合得分
        overall_score = (
            self.reliability_weight * reliability
            + self.generalization_weight * generalization
            + self.locality_weight * locality
            + self.portability_weight * portability
        )

        result = {
            "reliability": reliability,
            "generalization": generalization,
            "locality": locality,
            "portability": portability,
            "overall_score": overall_score,
            "total_edits": len(edit_data),
            "reliability_samples": len(reliability_scores),
            "generalization_samples": len(generalization_scores),
            "locality_samples": len(locality_scores),
            "portability_samples": len(portability_scores),
        }

        logger.info(
            f"Edit Comprehensive: reliability={reliability:.4f}, "
            f"generalization={generalization:.4f}, locality={locality:.4f}, "
            f"portability={portability:.4f}, overall={overall_score:.4f}"
        )

        return result

    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            "reliability": 0.0,
            "generalization": 0.0,
            "locality": 1.0,
            "portability": 0.0,
            "overall_score": 0.0,
            "total_edits": 0,
        }
