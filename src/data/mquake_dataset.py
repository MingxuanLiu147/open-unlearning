"""
MQuAKE: Multi-hop Question Answering for Knowledge Editing
===========================================================

EMNLP 2023, 300+ citations.
Evaluates whether knowledge edits propagate through multi-hop reasoning chains.

Each sample contains one or more atomic edits (requested_rewrite) plus multi-hop
questions whose answers change as a result of those edits.

Supports three variants:
  - MQuAKE-CF  (counterfact-based, 2-hop)
  - MQuAKE-T   (temporal updates, 2-4 hop)
  - MQuAKE-3k  (3000 hard cases, 2-4 hop)

Paper: https://arxiv.org/abs/2305.14795
Code:  https://github.com/Zce1112zslx/MQuAKE
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from data.editing import EditingDataset, EditingSample

logger = logging.getLogger(__name__)


class MQuAKEDataset(EditingDataset):
    """MQuAKE multi-hop knowledge editing benchmark.

    Each record carries ``requested_rewrite`` (list of atomic edits) and
    ``questions`` (multi-hop queries whose ground-truth changes after edits).
    We expose multi-hop questions as *portability_inputs* so that the standard
    ``EditPortabilityEvaluator`` and ``EditComprehensiveEvaluator`` can score
    them, while the dedicated ``MQuAKEMultiHopEvaluator`` provides richer
    chain-level analysis.
    """

    VARIANT_FILES = {
        "cf": "data/edit/mquake/MQuAKE-CF.json",
        "t": "data/edit/mquake/MQuAKE-T.json",
        "3k": "data/edit/mquake/MQuAKE-3k.json",
    }

    HF_ARGS = {
        "cf": {"path": "Zce1112zslx/MQuAKE", "name": "MQuAKE-CF", "split": "test"},
        "t": {"path": "Zce1112zslx/MQuAKE", "name": "MQuAKE-T", "split": "test"},
        "3k": {"path": "Zce1112zslx/MQuAKE", "name": "MQuAKE-3k", "split": "test"},
    }

    def __init__(
        self,
        tokenizer=None,
        variant: str = "cf",
        data_path: Optional[str] = None,
        hf_args: Optional[Dict[str, Any]] = None,
        max_hops: Optional[int] = None,
        **kwargs,
    ):
        self.variant = variant.lower()
        self.max_hops = max_hops
        default_hf = self.HF_ARGS.get(self.variant)
        super().__init__(
            hf_args=hf_args or default_hf,
            data_path=data_path,
            tokenizer=tokenizer,
            prompt_key="prompt",
            subject_key="subject",
            target_new_key="target_new",
            **kwargs,
        )

    def _default_data_path(self, split: Optional[str]) -> Optional[str]:
        return self.VARIANT_FILES.get(self.variant)

    def normalize_record(
        self, item: Dict[str, Any], index: int
    ) -> Optional[Union[EditingSample, List[EditingSample]]]:
        rewrites = item.get("requested_rewrite", [])
        if not rewrites:
            return None

        questions = item.get("questions", [])
        new_answer = self._flatten_text(item.get("new_answer", "")).strip()
        new_answer_alias = item.get("new_answer_alias", [])
        single_hops = item.get("single_hops", [])

        if self.max_hops is not None:
            num_hops = len(single_hops) if single_hops else len(rewrites)
            if num_hops > self.max_hops:
                return None

        multihop_entries = []
        for q in questions:
            q_text = self._flatten_text(q).strip()
            if q_text and new_answer:
                multihop_entries.append(
                    {"prompt": q_text, "ground_truth": new_answer}
                )

        portability_inputs = {"MultiHop": multihop_entries} if multihop_entries else None

        single_hop_entries = []
        for hop in single_hops:
            if isinstance(hop, dict):
                hp = self._flatten_text(hop.get("question", hop.get("prompt", ""))).strip()
                ha = self._flatten_text(hop.get("answer", hop.get("target_new", ""))).strip()
                if hp and ha:
                    single_hop_entries.append({"prompt": hp, "ground_truth": ha})

        if single_hop_entries:
            portability_inputs = portability_inputs or {}
            portability_inputs["SingleHop"] = single_hop_entries

        primary = rewrites[0] if isinstance(rewrites, list) else rewrites
        subject = self._flatten_text(primary.get("subject", "")).strip()
        prompt_template = self._flatten_text(primary.get("prompt", "")).strip()
        prompt = self._format_subject_prompt(prompt_template, subject) if prompt_template else ""
        target_new = self._flatten_text(primary.get("target_new", "")).strip()
        target_old = self._flatten_text(
            primary.get("target_true", primary.get("target_old", ""))
        ).strip()

        rephrase_prompts = None
        if len(rewrites) > 1:
            extra_prompts = []
            for rw in rewrites[1:]:
                rw_subj = self._flatten_text(rw.get("subject", "")).strip()
                rw_tmpl = self._flatten_text(rw.get("prompt", "")).strip()
                rw_prompt = self._format_subject_prompt(rw_tmpl, rw_subj) if rw_tmpl else ""
                if rw_prompt:
                    extra_prompts.append(rw_prompt)
            rephrase_prompts = extra_prompts or None

        return self._build_sample(
            prompt=prompt,
            subject=subject,
            target_new=target_new,
            target_old=target_old,
            rephrase_prompts=rephrase_prompts,
            portability_inputs=portability_inputs,
        )

    @property
    def raw_records(self) -> List[Dict[str, Any]]:
        """Expose raw JSON records for chain-level evaluation."""
        if hasattr(self, "_raw_records"):
            return self._raw_records
        return []

    def _load_data(self, hf_args):
        raw_data = self._load_raw_data(hf_args)
        self._raw_records = raw_data
        samples = []
        for index, item in enumerate(raw_data):
            normalized = self.normalize_record(item, index)
            if normalized is None:
                continue
            if isinstance(normalized, list):
                samples.extend(s for s in normalized if s is not None)
            else:
                samples.append(normalized)
        return samples
