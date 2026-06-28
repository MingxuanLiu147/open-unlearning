"""
MQuAKE Evaluator: Multi-hop reasoning evaluation for knowledge editing
======================================================================

Beyond standard edit metrics (reliability / generalization / locality /
portability), MQuAKE requires *chain-level* multi-hop accuracy: all hops
in the reasoning chain must resolve correctly for the final answer to be
right.

Metrics reported:
  - multi_hop_accuracy: fraction of multi-hop questions answered correctly
  - single_hop_accuracy: fraction of individual hop questions correct
  - chain_accuracy: fraction of instances where ALL hops are correct
  - instance_accuracy: fraction of instances where final answer is correct

Paper: https://arxiv.org/abs/2305.14795
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

from evals.edit import EditEvaluator

logger = logging.getLogger(__name__)


class MQuAKEMultiHopEvaluator(EditEvaluator):
    """Evaluator specialised for MQuAKE multi-hop reasoning."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "mquake_multihop"
        args = eval_cfg.get("args", {})
        self.max_new_tokens = int(args.get("max_new_tokens", 32))
        self.use_exact_match = bool(args.get("use_exact_match", True))
        self.use_token_accuracy = bool(args.get("use_token_accuracy", False))

    def evaluate(self, model, output_dir=None, **kwargs) -> Dict[str, Any]:
        tokenizer = kwargs.get("tokenizer")
        edit_data = kwargs.get("edit_data", [])
        raw_records = kwargs.get("raw_records", [])

        if tokenizer is None or not edit_data:
            return self._empty_result()

        model = self.prepare_model(model)

        multihop_scores = []
        singlehop_scores = []
        chain_scores = []
        instance_scores = []

        for idx, item in enumerate(edit_data):
            raw = raw_records[idx] if idx < len(raw_records) else {}

            mh_correct, mh_total = self._eval_multihop(
                model, tokenizer, item, raw
            )
            if mh_total > 0:
                multihop_scores.append(mh_correct / mh_total)
                instance_scores.append(1.0 if mh_correct == mh_total else 0.0)

            sh_correct, sh_total, all_sh_ok = self._eval_single_hops(
                model, tokenizer, raw
            )
            if sh_total > 0:
                singlehop_scores.append(sh_correct / sh_total)
                chain_scores.append(1.0 if all_sh_ok else 0.0)

        result = {
            "multi_hop_accuracy": self._mean(multihop_scores, 0.0),
            "single_hop_accuracy": self._mean(singlehop_scores, 0.0),
            "chain_accuracy": self._mean(chain_scores, 0.0),
            "instance_accuracy": self._mean(instance_scores, 0.0),
            "total_instances": len(edit_data),
            "evaluated_multihop": len(multihop_scores),
            "evaluated_singlehop": len(singlehop_scores),
        }

        logger.info(
            "MQuAKE: multi_hop=%.4f, single_hop=%.4f, chain=%.4f, instance=%.4f",
            result["multi_hop_accuracy"],
            result["single_hop_accuracy"],
            result["chain_accuracy"],
            result["instance_accuracy"],
        )
        return result

    def _eval_multihop(
        self,
        model,
        tokenizer,
        item: Dict[str, Any],
        raw: Dict[str, Any],
    ) -> Tuple[int, int]:
        questions = raw.get("questions", [])
        new_answer = str(raw.get("new_answer", "")).strip()
        aliases = raw.get("new_answer_alias", [])

        if not questions or not new_answer:
            port = item.get("portability_inputs", item.get("portability", {}))
            if isinstance(port, dict):
                mh_entries = port.get("MultiHop", [])
                correct = total = 0
                for entry in mh_entries:
                    prompt = str(entry.get("prompt", "")).strip()
                    expected = str(entry.get("ground_truth", "")).strip()
                    if not prompt or not expected:
                        continue
                    total += 1
                    if self._check_answer(model, tokenizer, prompt, expected, []):
                        correct += 1
                return correct, total
            return 0, 0

        correct = total = 0
        for q in questions:
            q_text = str(q).strip()
            if not q_text:
                continue
            total += 1
            if self._check_answer(model, tokenizer, q_text, new_answer, aliases):
                correct += 1
        return correct, total

    def _eval_single_hops(
        self,
        model,
        tokenizer,
        raw: Dict[str, Any],
    ) -> Tuple[int, int, bool]:
        single_hops = raw.get("single_hops", [])
        if not single_hops:
            return 0, 0, True

        correct = total = 0
        for hop in single_hops:
            if not isinstance(hop, dict):
                continue
            prompt = str(hop.get("question", hop.get("prompt", ""))).strip()
            answer = str(hop.get("answer", hop.get("target_new", ""))).strip()
            if not prompt or not answer:
                continue
            total += 1
            if self._check_answer(model, tokenizer, prompt, answer, []):
                correct += 1

        all_ok = (correct == total) if total > 0 else True
        return correct, total, all_ok

    def _check_answer(
        self,
        model,
        tokenizer,
        prompt: str,
        expected: str,
        aliases: List[str],
    ) -> bool:
        if self.use_token_accuracy:
            acc = self.compute_token_accuracy(model, tokenizer, prompt, expected)
            return acc >= 0.5

        generated = self._generate(model, tokenizer, prompt)
        candidates = [expected] + [str(a) for a in (aliases or [])]
        return self._exact_match(generated, candidates)

    def _generate(self, model, tokenizer, prompt: str) -> str:
        device = self._get_model_device(model)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0, prompt_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def _exact_match(generated: str, candidates: List[str]) -> bool:
        gen_lower = generated.lower().strip()
        gen_normed = re.sub(r"\s+", " ", gen_lower)
        for cand in candidates:
            cand_lower = cand.lower().strip()
            if not cand_lower:
                continue
            if cand_lower in gen_normed or gen_normed.startswith(cand_lower):
                return True
        return False

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "multi_hop_accuracy": 0.0,
            "single_hop_accuracy": 0.0,
            "chain_accuracy": 0.0,
            "instance_accuracy": 0.0,
            "total_instances": 0,
            "evaluated_multihop": 0,
            "evaluated_singlehop": 0,
        }
