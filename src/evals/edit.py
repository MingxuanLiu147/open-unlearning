"""
Knowledge Editing evaluator:
- Reliability: rewrite accuracy
- Generalization: rephrase accuracy
- Locality: token-level invariance before/after edit
- Portability: grouped transfer accuracy
"""

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from evals.base import Evaluator

logger = logging.getLogger(__name__)


class EditEvaluator(Evaluator):
    """Base helper for edit evaluators."""

    def __init__(self, eval_cfg, **kwargs):
        self.name = "edit"
        super().__init__(self.name, eval_cfg, **kwargs)

    def _get_model_device(self, model) -> torch.device:
        if hasattr(model, "device"):
            return model.device
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _mean(self, values: List[float], default: float) -> float:
        return sum(values) / len(values) if values else default

    def _flatten_target(self, target: Any) -> str:
        while isinstance(target, list):
            if not target:
                return ""
            target = target[0]
        return "" if target is None else str(target)

    def _normalize_prompt_target(self, prompt: Any, target: Any) -> Tuple[str, str]:
        return ("" if prompt is None else str(prompt).strip(), self._flatten_target(target).strip())

    def _normalize_rephrase_prompts(self, item: Dict[str, Any]) -> List[str]:
        raw = item.get(
            "rephrase_prompts",
            item.get("rephrase_prompt", item.get("rephrase", [])),
        )
        if raw is None:
            return []
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        return [str(v).strip() for v in raw if v is not None and str(v).strip()]

    def _extract_expected(self, item: Dict[str, Any]) -> str:
        return self._flatten_target(
            item.get(
                "expected",
                item.get("ground_truth", item.get("answer", item.get("target", ""))),
            )
        )

    def _iter_eval_pairs(self, raw_inputs: Any, default_group: str) -> List[Tuple[str, str, str]]:
        """Normalize locality/portability inputs into (group, prompt, expected)."""
        if raw_inputs is None:
            return []

        pairs: List[Tuple[str, str, str]] = []

        if isinstance(raw_inputs, dict):
            # single item dict: {"prompt": ..., "ground_truth": ...}
            if any(k in raw_inputs for k in ("prompt", "expected", "ground_truth", "answer", "target")):
                raw_inputs = [raw_inputs]
            else:
                for group_key, payload in raw_inputs.items():
                    group = str(group_key)
                    if payload is None:
                        continue

                    # {group: {"prompt": [...], "ground_truth": [...]}}
                    if isinstance(payload, dict):
                        prompts = payload.get("prompt", [])
                        targets = payload.get(
                            "ground_truth",
                            payload.get("expected", payload.get("answer", payload.get("target", []))),
                        )

                        if isinstance(prompts, str):
                            prompts = [prompts]
                        elif not isinstance(prompts, list):
                            prompts = []

                        if isinstance(targets, str):
                            targets = [targets]
                        elif not isinstance(targets, list):
                            targets = [targets]

                        for prompt, target in zip(prompts, targets):
                            p, t = self._normalize_prompt_target(prompt, target)
                            if p and t:
                                pairs.append((group, p, t))
                        continue

                    # {group: [{"prompt": ..., "ground_truth": ...}, ...]}
                    if isinstance(payload, list):
                        for item in payload:
                            if not isinstance(item, dict):
                                continue
                            p, t = self._normalize_prompt_target(item.get("prompt", ""), self._extract_expected(item))
                            if p and t:
                                pairs.append((group, p, t))
                        continue
                return pairs

        if isinstance(raw_inputs, list):
            for item in raw_inputs:
                if not isinstance(item, dict):
                    continue
                group = str(item.get("type") or item.get("category") or item.get("key") or default_group)
                p, t = self._normalize_prompt_target(item.get("prompt", ""), self._extract_expected(item))
                if p and t:
                    pairs.append((group, p, t))

        return pairs

    def _teacher_forcing_prediction_and_labels(
        self, model, tokenizer, prompt: str, target: str
    ) -> Tuple[List[int], List[int]]:
        device = self._get_model_device(model)

        if getattr(model.config, "is_encoder_decoder", False):
            src = tokenizer(prompt, return_tensors="pt").to(device)
            trg = tokenizer(target, return_tensors="pt").to(device)
            target_ids = trg["input_ids"][0]
            if target_ids.numel() <= 1:
                return [], []
            with torch.no_grad():
                outputs = model(
                    input_ids=src["input_ids"],
                    attention_mask=src.get("attention_mask", None),
                    decoder_input_ids=trg["input_ids"],
                    decoder_attention_mask=trg.get("attention_mask", None),
                )
                logits = outputs.logits
            pred = torch.argmax(logits[0, :-1, :], dim=-1)
            gold = target_ids[:-1]
            return pred.detach().cpu().tolist(), gold.detach().cpu().tolist()

        prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
        full_inputs = tokenizer(f"{prompt} {target}", return_tensors="pt").to(device)
        full_ids = full_inputs["input_ids"]
        prompt_len = prompt_ids.shape[1]
        if prompt_len >= full_ids.shape[1]:
            return [], []

        with torch.no_grad():
            logits = model(**full_inputs).logits

        pred = torch.argmax(logits[0, prompt_len - 1 : -1, :], dim=-1)
        gold = full_ids[0, prompt_len:]
        return pred.detach().cpu().tolist(), gold.detach().cpu().tolist()

    def compute_token_accuracy(self, model, tokenizer, prompt: str, target: str, **kwargs) -> float:
        pred_tokens, gold_tokens = self._teacher_forcing_prediction_and_labels(model, tokenizer, prompt, target)
        if not gold_tokens:
            return 0.0
        overlap = min(len(pred_tokens), len(gold_tokens))
        if overlap == 0:
            return 0.0
        correct = sum(1 for i in range(overlap) if pred_tokens[i] == gold_tokens[i])
        return correct / len(gold_tokens)

    def token_match_ratio(self, pre_tokens: List[int], post_tokens: List[int]) -> float:
        if not pre_tokens and not post_tokens:
            return 1.0
        if not pre_tokens or not post_tokens:
            return 0.0
        overlap = min(len(pre_tokens), len(post_tokens))
        same = sum(1 for i in range(overlap) if pre_tokens[i] == post_tokens[i])
        # Locality compares against pre-edit output length.
        return same / len(pre_tokens)

    def compute_target_probability(self, model, tokenizer, prompt: str, target: str, **kwargs) -> float:
        device = self._get_model_device(model)

        if getattr(model.config, "is_encoder_decoder", False):
            src = tokenizer(prompt, return_tensors="pt").to(device)
            trg = tokenizer(target, return_tensors="pt").to(device)
            target_ids = trg["input_ids"][0]
            if target_ids.numel() <= 1:
                return 0.0
            with torch.no_grad():
                logits = model(
                    input_ids=src["input_ids"],
                    attention_mask=src.get("attention_mask", None),
                    decoder_input_ids=trg["input_ids"],
                    decoder_attention_mask=trg.get("attention_mask", None),
                ).logits
            probs = F.softmax(logits[0, :-1, :], dim=-1)
            gold = target_ids[:-1]
            target_probs = probs.gather(1, gold.unsqueeze(1)).squeeze()
            return float(target_probs.item()) if target_probs.dim() == 0 else float(target_probs.mean().item())

        full_ids = tokenizer(f"{prompt} {target}", return_tensors="pt")["input_ids"].to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        prompt_len = prompt_ids.shape[1]
        if prompt_len >= full_ids.shape[1]:
            return 0.0

        with torch.no_grad():
            logits = model(full_ids).logits
        target_logits = logits[0, prompt_len - 1 : -1, :]
        target_ids = full_ids[0, prompt_len:]
        probs = F.softmax(target_logits, dim=-1)
        target_probs = probs.gather(1, target_ids.unsqueeze(1)).squeeze()
        return float(target_probs.item()) if target_probs.dim() == 0 else float(target_probs.mean().item())

    def _score_prompt_target(
        self, model, tokenizer, prompt: str, target: str, use_probability: bool = False
    ) -> float:
        if use_probability:
            return self.compute_target_probability(model, tokenizer, prompt, target)
        return self.compute_token_accuracy(model, tokenizer, prompt, target)

    def _aggregate_case_group_scores(
        self,
        raw_inputs: Any,
        default_group: str,
        score_fn: Callable[[str, str], float],
    ) -> Tuple[Optional[float], int, Dict[str, float]]:
        grouped_scores = defaultdict(list)
        for group, prompt, expected in self._iter_eval_pairs(raw_inputs, default_group):
            grouped_scores[group].append(float(score_fn(prompt, expected)))

        case_group_means = {group: self._mean(scores, 0.0) for group, scores in grouped_scores.items() if scores}
        case_score = self._mean(list(case_group_means.values()), 0.0) if case_group_means else None
        sample_count = sum(len(scores) for scores in grouped_scores.values())
        return case_score, sample_count, case_group_means

    def _merge_group_case_means(
        self, dst: Dict[str, List[float]], src: Dict[str, float]
    ) -> None:
        for key, value in src.items():
            dst[key].append(value)

    def _finalize_group_acc(self, grouped_case_means: Dict[str, List[float]]) -> Dict[str, float]:
        return {f"{key}_acc": self._mean(values, 0.0) for key, values in grouped_case_means.items()}


class EditSuiteEvaluator(Evaluator):
    """Suite-style edit evaluator (single handler + multiple metric evaluators).

    This mirrors unlearning eval flow (`tofu`/`muse`): one evaluator node orchestrates
    multiple metric entries configured under `eval.edit.metrics.*`.
    """

    PRIMARY_VALUE_KEYS = {
        "reliability": "reliability",
        "generalization": "generalization",
        "locality": "locality",
        "portability": "portability",
        "comprehensive": "overall_score",
    }

    def __init__(self, eval_cfg, **kwargs):
        self.name = "edit"
        self.eval_cfg = eval_cfg
        self.metrics_cfg = self.eval_cfg.get("metrics", {}) or {}
        self.metrics = {}
        output_dir = self.eval_cfg.get("output_dir", None)
        if output_dir is not None:
            logger.info("Evaluations stored in the experiment directory: %s", output_dir)
        else:
            logger.info(
                "Evaluator `%s` initialized without output_dir "
                "(standalone evaluator mode).",
                self.name,
            )

    def _handler_registry(self):
        return {
            "EditReliabilityEvaluator": EditReliabilityEvaluator,
            "EditGeneralizationEvaluator": EditGeneralizationEvaluator,
            "EditLocalityEvaluator": EditLocalityEvaluator,
            "EditPortabilityEvaluator": EditPortabilityEvaluator,
            "EditComprehensiveEvaluator": EditComprehensiveEvaluator,
        }

    def summarize(self, logs):
        summary = {}
        for metric_name, metric_result in logs.items():
            if not isinstance(metric_result, dict):
                summary[metric_name] = metric_result
                continue
            key = self.PRIMARY_VALUE_KEYS.get(metric_name, metric_name)
            if key in metric_result:
                summary[metric_name] = metric_result[key]
            elif metric_name in metric_result:
                summary[metric_name] = metric_result[metric_name]
        return summary

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        overwrite = self.eval_cfg.get("overwrite", False) if overwrite is None else overwrite
        model = self.prepare_model(model)

        output_dir = output_dir if output_dir else self.eval_cfg.get("output_dir", None)
        logs_file_path = None
        summary_file_path = None
        logs = {}
        if output_dir:
            logs_file_path = self.get_logs_file_path(output_dir)
            summary_file_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")
            logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}

        if not self.metrics_cfg:
            logger.warning("No edit metrics configured under `eval.edit.metrics`.")
            return {}

        handler_registry = self._handler_registry()
        tokenizer = kwargs.get("tokenizer", None)
        edit_data = kwargs.get("edit_data", [])
        original_model = kwargs.get("original_model", None)

        for metric_name, metric_cfg in self.metrics_cfg.items():
            if not overwrite and metric_name in logs and logs[metric_name]:
                logger.info("Skipping %s, already evaluated.", metric_name)
                continue

            handler_name = metric_cfg.get("handler", None)
            evaluator_cls = handler_registry.get(handler_name)
            if evaluator_cls is None:
                raise NotImplementedError(
                    f"{handler_name} not implemented or not registered for edit suite"
                )

            metric_evaluator = evaluator_cls(metric_cfg)
            metric_result = metric_evaluator.evaluate(
                model=model,
                tokenizer=tokenizer,
                edit_data=edit_data,
                original_model=original_model,
                output_dir=output_dir,
            )
            logs[metric_name] = metric_result

            if output_dir:
                self.save_logs(logs, logs_file_path)
                self.save_logs(self.summarize(logs), summary_file_path)

        return self.summarize(logs)


class EditReliabilityEvaluator(EditEvaluator):
    """Reliability: mean rewrite accuracy."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "edit_reliability"
        args = eval_cfg.get("args", {})
        self.success_threshold = float(args.get("success_threshold", 1.0))

    def evaluate(self, model, output_dir=None, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        edit_data = kwargs.get("edit_data", [])
        if tokenizer is None or not edit_data:
            return {"reliability": 0.0}

        model = self.prepare_model(model)
        scores = []
        for item in edit_data:
            prompt, target_new = self._normalize_prompt_target(item.get("prompt", ""), item.get("target_new", ""))
            if prompt and target_new:
                scores.append(self.compute_token_accuracy(model, tokenizer, prompt, target_new))

        reliability = self._mean(scores, 0.0)
        result = {
            "reliability": reliability,
            "total_samples": len(scores),
            "successful_edits": int(sum(1 for score in scores if score >= self.success_threshold)),
        }
        logger.info("Edit Reliability: %.4f", reliability)
        return result


class EditGeneralizationEvaluator(EditEvaluator):
    """Generalization: mean rephrase accuracy (case-level macro)."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "edit_generalization"

    def evaluate(self, model, output_dir=None, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        edit_data = kwargs.get("edit_data", [])
        if tokenizer is None:
            return {"generalization": 0.0}

        model = self.prepare_model(model)
        case_scores = []
        total_rephrase_samples = 0
        for item in edit_data:
            _, target_new = self._normalize_prompt_target(item.get("prompt", ""), item.get("target_new", ""))
            per_case = []
            for rephrase in self._normalize_rephrase_prompts(item):
                prompt, target = self._normalize_prompt_target(rephrase, target_new)
                if prompt and target:
                    per_case.append(self.compute_token_accuracy(model, tokenizer, prompt, target))
                    total_rephrase_samples += 1
            if per_case:
                case_scores.append(self._mean(per_case, 0.0))

        generalization = self._mean(case_scores, 0.0)
        result = {
            "generalization": generalization,
            "total_rephrase_samples": total_rephrase_samples,
            "evaluated_cases": len(case_scores),
        }
        logger.info("Edit Generalization: %.4f", generalization)
        return result


class EditLocalityEvaluator(EditEvaluator):
    """Locality: pre/post token invariance (case-level macro + by_key)."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "edit_locality"

    def evaluate(self, model, output_dir=None, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        edit_data = kwargs.get("edit_data", [])
        original_model = kwargs.get("original_model")
        if tokenizer is None:
            return {"locality": 1.0}
        if original_model is None:
            return {"locality": 1.0, "total_locality_samples": 0}

        model = self.prepare_model(model)
        original_model = self.prepare_model(original_model)

        def _loc_score(prompt: str, expected: str) -> float:
            pre_tokens, _ = self._teacher_forcing_prediction_and_labels(original_model, tokenizer, prompt, expected)
            post_tokens, _ = self._teacher_forcing_prediction_and_labels(model, tokenizer, prompt, expected)
            return self.token_match_ratio(pre_tokens, post_tokens)

        case_scores = []
        total_locality_samples = 0
        grouped_case_means = defaultdict(list)
        for item in edit_data:
            case_score, sample_count, case_group_means = self._aggregate_case_group_scores(
                item.get("locality_inputs", item.get("locality", None)),
                "locality",
                _loc_score,
            )
            total_locality_samples += sample_count
            self._merge_group_case_means(grouped_case_means, case_group_means)
            if case_score is not None:
                case_scores.append(case_score)

        locality = self._mean(case_scores, 1.0)
        result = {
            "locality": locality,
            "total_locality_samples": total_locality_samples,
            "evaluated_cases": len(case_scores),
            "locality_by_key": self._finalize_group_acc(grouped_case_means),
        }
        logger.info("Edit Locality: %.4f", locality)
        return result


class EditPortabilityEvaluator(EditEvaluator):
    """Portability: grouped transfer accuracy (case-level macro + by_key)."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "edit_portability"

    def evaluate(self, model, output_dir=None, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        edit_data = kwargs.get("edit_data", [])
        if tokenizer is None:
            return {"portability": 0.0}

        model = self.prepare_model(model)

        def _port_score(prompt: str, expected: str) -> float:
            return self.compute_token_accuracy(model, tokenizer, prompt, expected)

        case_scores = []
        total_portability_samples = 0
        grouped_case_means = defaultdict(list)
        for item in edit_data:
            case_score, sample_count, case_group_means = self._aggregate_case_group_scores(
                item.get("portability_inputs", item.get("portability", None)),
                "portability",
                _port_score,
            )
            total_portability_samples += sample_count
            self._merge_group_case_means(grouped_case_means, case_group_means)
            if case_score is not None:
                case_scores.append(case_score)

        portability = self._mean(case_scores, 0.0)
        result = {
            "portability": portability,
            "total_portability_samples": total_portability_samples,
            "evaluated_cases": len(case_scores),
            "portability_by_key": self._finalize_group_acc(grouped_case_means),
        }
        logger.info("Edit Portability: %.4f", portability)
        return result


class EditComprehensiveEvaluator(EditEvaluator):
    """Comprehensive scorer for all four metrics."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "edit_comprehensive"
        args = eval_cfg.get("args", {})
        self.reliability_weight = float(args.get("reliability_weight", 0.25))
        self.generalization_weight = float(args.get("generalization_weight", 0.25))
        self.locality_weight = float(args.get("locality_weight", 0.25))
        self.portability_weight = float(args.get("portability_weight", 0.25))
        self.use_probability = bool(args.get("use_probability", False))

    def evaluate(self, model, output_dir=None, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        edit_data = kwargs.get("edit_data", [])
        original_model = kwargs.get("original_model")
        if tokenizer is None or not edit_data:
            return self._empty_result()

        model = self.prepare_model(model)
        if original_model is not None:
            original_model = self.prepare_model(original_model)

        reliability_case_scores = []
        generalization_case_scores = []
        locality_case_scores = []
        portability_case_scores = []

        reliability_samples = 0
        generalization_samples = 0
        locality_samples = 0
        portability_samples = 0

        grouped_locality_case_means = defaultdict(list)
        grouped_portability_case_means = defaultdict(list)

        def _loc_score(prompt: str, expected: str) -> float:
            pre_tokens, _ = self._teacher_forcing_prediction_and_labels(original_model, tokenizer, prompt, expected)
            post_tokens, _ = self._teacher_forcing_prediction_and_labels(model, tokenizer, prompt, expected)
            return self.token_match_ratio(pre_tokens, post_tokens)

        def _port_score(prompt: str, expected: str) -> float:
            return self.compute_token_accuracy(model, tokenizer, prompt, expected)

        for item in edit_data:
            prompt, target_new = self._normalize_prompt_target(item.get("prompt", ""), item.get("target_new", ""))

            if prompt and target_new:
                reliability_case_scores.append(
                    self._score_prompt_target(model, tokenizer, prompt, target_new, self.use_probability)
                )
                reliability_samples += 1

            rephrase_scores = []
            for rephrase in self._normalize_rephrase_prompts(item):
                rp, target = self._normalize_prompt_target(rephrase, target_new)
                if rp and target:
                    rephrase_scores.append(
                        self._score_prompt_target(model, tokenizer, rp, target, self.use_probability)
                    )
                    generalization_samples += 1
            if rephrase_scores:
                generalization_case_scores.append(self._mean(rephrase_scores, 0.0))

            if original_model is not None:
                loc_case_score, loc_samples, loc_case_group_means = self._aggregate_case_group_scores(
                    item.get("locality_inputs", item.get("locality", None)),
                    "locality",
                    _loc_score,
                )
                locality_samples += loc_samples
                self._merge_group_case_means(grouped_locality_case_means, loc_case_group_means)
                if loc_case_score is not None:
                    locality_case_scores.append(loc_case_score)

            port_case_score, port_samples, port_case_group_means = self._aggregate_case_group_scores(
                item.get("portability_inputs", item.get("portability", None)),
                "portability",
                _port_score,
            )
            portability_samples += port_samples
            self._merge_group_case_means(grouped_portability_case_means, port_case_group_means)
            if port_case_score is not None:
                portability_case_scores.append(port_case_score)

        reliability = self._mean(reliability_case_scores, 0.0)
        generalization = self._mean(generalization_case_scores, 0.0)
        locality = self._mean(locality_case_scores, 1.0)
        portability = self._mean(portability_case_scores, 0.0)

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
            "reliability_samples": reliability_samples,
            "generalization_samples": generalization_samples,
            "locality_samples": locality_samples,
            "portability_samples": portability_samples,
            "reliability_cases": len(reliability_case_scores),
            "generalization_cases": len(generalization_case_scores),
            "locality_cases": len(locality_case_scores),
            "portability_cases": len(portability_case_scores),
            "locality_by_key": self._finalize_group_acc(grouped_locality_case_means),
            "portability_by_key": self._finalize_group_acc(grouped_portability_case_means),
        }

        logger.info(
            "Edit Comprehensive: reliability=%.4f, generalization=%.4f, locality=%.4f, portability=%.4f, overall=%.4f",
            reliability,
            generalization,
            locality,
            portability,
            overall_score,
        )
        return result

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "reliability": 0.0,
            "generalization": 0.0,
            "locality": 1.0,
            "portability": 0.0,
            "overall_score": 0.0,
            "total_edits": 0,
            "reliability_samples": 0,
            "generalization_samples": 0,
            "locality_samples": 0,
            "portability_samples": 0,
            "reliability_cases": 0,
            "generalization_cases": 0,
            "locality_cases": 0,
            "portability_cases": 0,
            "locality_by_key": {},
            "portability_by_key": {},
        }
