"""
Multimodal Knowledge Editing evaluators
========================================

Extend the text-only ``EditEvaluator`` family with image-aware metrics:

- ``MMEditReliabilityEvaluator``:  rewrite accuracy using multimodal forward
- ``MMEditGeneralizationEvaluator``: text rephrase_acc + image_rephrase_acc
- ``MMEditLocalityEvaluator``: text locality + multimodal locality (OK-VQA)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch

from evals.edit import EditEvaluator
from trainer.edit.mm_mixin import MMEditMixin

logger = logging.getLogger(__name__)


class _MMEditEvalHelper(EditEvaluator):
    """Shared helpers for multimodal edit evaluators."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self._mm = MMEditMixin()

    def _init_mm(self, processor):
        self._mm.init_mm(processor)

    def _mm_token_accuracy(self, model, processor, prompt, target, image=None):
        if image is not None:
            inputs = self._mm.mm_tokenize(prompt, image, target)
        else:
            inputs = self._mm.mm_tokenize_text_only(prompt, target)

        with torch.no_grad():
            outputs = self._mm.mm_forward(model, inputs)

        logits = outputs.logits
        labels = inputs["labels"]

        preds = logits.argmax(dim=-1)
        shift_preds = preds[:, :-1]
        shift_labels = labels[:, 1:]
        shift_mask = shift_labels != -100

        if not shift_mask.any():
            return 0.0

        correct = ((shift_preds == shift_labels) & shift_mask).sum().item()
        total = shift_mask.sum().item()
        return correct / total if total > 0 else 0.0

    def _mm_pred_tokens(self, model, processor, prompt, target, image=None):
        if image is not None:
            inputs = self._mm.mm_tokenize(prompt, image, target)
        else:
            inputs = self._mm.mm_tokenize_text_only(prompt, target)

        with torch.no_grad():
            outputs = self._mm.mm_forward(model, inputs)

        logits = outputs.logits
        labels = inputs["labels"]
        preds = logits.argmax(dim=-1)
        shift_preds = preds[:, :-1]
        shift_labels = labels[:, 1:]
        shift_mask = shift_labels != -100

        if not shift_mask.any():
            return []

        return shift_preds[0][shift_mask[0]].detach().cpu().tolist()


class MMEditReliabilityEvaluator(_MMEditEvalHelper):
    """Multimodal reliability: rewrite accuracy with image input."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "mm_edit_reliability"

    def evaluate(self, model, output_dir=None, **kwargs):
        processor = kwargs.get("processor")
        edit_data = kwargs.get("edit_data", [])
        if processor is None or not edit_data:
            return {"mm_reliability": 0.0}

        self._init_mm(processor)
        model = self.prepare_model(model)
        self._mm.model = model

        scores = []
        for item in edit_data:
            prompt = str(item.get("prompt", "")).strip()
            target_new = str(item.get("target_new", "")).strip()
            image = item.get("image")
            if prompt and target_new:
                scores.append(
                    self._mm_token_accuracy(model, processor, prompt, target_new, image)
                )

        mm_reliability = self._mean(scores, 0.0)
        result = {
            "mm_reliability": mm_reliability,
            "total_samples": len(scores),
        }
        logger.info("MM Edit Reliability: %.4f", mm_reliability)
        return result


class MMEditGeneralizationEvaluator(_MMEditEvalHelper):
    """Multimodal generalization: text rephrase_acc + image_rephrase_acc."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "mm_edit_generalization"

    def evaluate(self, model, output_dir=None, **kwargs):
        processor = kwargs.get("processor")
        edit_data = kwargs.get("edit_data", [])
        if processor is None:
            return {
                "mm_generalization": 0.0,
                "text_rephrase_acc": 0.0,
                "image_rephrase_acc": 0.0,
            }

        self._init_mm(processor)
        model = self.prepare_model(model)
        self._mm.model = model

        text_rephrase_scores = []
        image_rephrase_scores = []

        for item in edit_data:
            target_new = str(item.get("target_new", "")).strip()
            image = item.get("image")

            for rephrase in self._normalize_rephrase_prompts(item):
                rephrase = rephrase.strip()
                if rephrase and target_new:
                    text_rephrase_scores.append(
                        self._mm_token_accuracy(
                            model, processor, rephrase, target_new, image
                        )
                    )

            image_rephrase = item.get("image_rephrase")
            if image_rephrase is not None:
                prompt = str(item.get("prompt", "")).strip()
                if prompt and target_new:
                    image_rephrase_scores.append(
                        self._mm_token_accuracy(
                            model, processor, prompt, target_new, image_rephrase
                        )
                    )

        text_rephrase_acc = self._mean(text_rephrase_scores, 0.0)
        image_rephrase_acc = self._mean(image_rephrase_scores, 0.0)
        all_scores = text_rephrase_scores + image_rephrase_scores
        mm_generalization = self._mean(all_scores, 0.0)

        result = {
            "mm_generalization": mm_generalization,
            "text_rephrase_acc": text_rephrase_acc,
            "image_rephrase_acc": image_rephrase_acc,
            "text_rephrase_samples": len(text_rephrase_scores),
            "image_rephrase_samples": len(image_rephrase_scores),
        }
        logger.info(
            "MM Edit Generalization: %.4f (text=%.4f, image=%.4f)",
            mm_generalization,
            text_rephrase_acc,
            image_rephrase_acc,
        )
        return result


class MMEditLocalityEvaluator(_MMEditEvalHelper):
    """Multimodal locality: text locality + multimodal locality (OK-VQA)."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "mm_edit_locality"

    def evaluate(self, model, output_dir=None, **kwargs):
        processor = kwargs.get("processor")
        edit_data = kwargs.get("edit_data", [])
        original_model = kwargs.get("original_model")
        if processor is None:
            return {"mm_locality": 1.0}
        if original_model is None:
            return {"mm_locality": 1.0, "total_samples": 0}

        self._init_mm(processor)
        model = self.prepare_model(model)
        original_model = self.prepare_model(original_model)
        self._mm.model = model

        text_loc_scores = []
        mm_loc_scores = []

        for item in edit_data:
            for _group, prompt, expected in self._iter_eval_pairs(
                item.get("locality_inputs", item.get("locality")),
                "text_locality",
            ):
                self._mm.model = original_model
                pre = self._mm_pred_tokens(original_model, processor, prompt, expected)
                self._mm.model = model
                post = self._mm_pred_tokens(model, processor, prompt, expected)
                text_loc_scores.append(self.token_match_ratio(pre, post))

            mm_loc = item.get("multimodal_locality_inputs")
            if mm_loc and isinstance(mm_loc, dict):
                mm_prompt = str(mm_loc.get("prompt", "")).strip()
                mm_gt = str(mm_loc.get("ground_truth", "")).strip()
                mm_image = mm_loc.get("image")
                if mm_prompt and mm_gt:
                    self._mm.model = original_model
                    pre = self._mm_pred_tokens(
                        original_model, processor, mm_prompt, mm_gt, mm_image
                    )
                    self._mm.model = model
                    post = self._mm_pred_tokens(
                        model, processor, mm_prompt, mm_gt, mm_image
                    )
                    mm_loc_scores.append(self.token_match_ratio(pre, post))

        text_locality = self._mean(text_loc_scores, 1.0)
        multimodal_locality = self._mean(mm_loc_scores, 1.0)
        all_scores = text_loc_scores + mm_loc_scores
        mm_locality = self._mean(all_scores, 1.0)

        result = {
            "mm_locality": mm_locality,
            "text_locality_acc": text_locality,
            "multimodal_locality_acc": multimodal_locality,
            "text_locality_samples": len(text_loc_scores),
            "multimodal_locality_samples": len(mm_loc_scores),
        }
        logger.info(
            "MM Edit Locality: %.4f (text=%.4f, mm=%.4f)",
            mm_locality,
            text_locality,
            multimodal_locality,
        )
        return result
