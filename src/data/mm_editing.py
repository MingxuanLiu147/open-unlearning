"""
Multimodal Knowledge Editing Datasets
======================================

Adapts multimodal editing benchmarks (MMEdit E-VQA / E-IC, MMKE-Bench) into
the ``EditingSample`` / ``EditRequest`` pipeline used by open-unlearning.

Image fields are stored as ``PIL.Image`` inside ``EditingSample`` so that
``MMEditMixin.mm_tokenize`` can consume them directly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from data.editing import EditingDataset, EditingSample

logger = logging.getLogger(__name__)


def _safe_load_image(path: str) -> Optional[Image.Image]:
    """Load an image from *path*, returning ``None`` on failure."""
    if not path or not os.path.isfile(path):
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:
        logger.warning("Failed to load image %s: %s", path, exc)
        return None


class MMEditingDataset(EditingDataset):
    """Base class for multimodal editing datasets.

    Subclasses override ``normalize_record`` to produce ``EditingSample``
    instances with the ``image`` / ``image_rephrase`` /
    ``multimodal_locality_inputs`` fields populated.
    """

    def __init__(
        self,
        image_dir: Optional[str] = None,
        rephrase_image_dir: Optional[str] = None,
        **kwargs,
    ):
        self.image_dir = self._to_path(image_dir)
        self.rephrase_image_dir = self._to_path(rephrase_image_dir)
        super().__init__(**kwargs)

    def _to_path(self, p: Optional[str]) -> Optional[Path]:
        if p is None:
            return None
        path = Path(p).expanduser()
        if not path.is_absolute():
            path = self.project_root() / path
        return path

    def _resolve_image(self, name: str, base_dir: Optional[Path] = None) -> Optional[Image.Image]:
        base = base_dir or self.image_dir
        if base is None:
            return None
        return _safe_load_image(str(base / name))


class MMEditVQADataset(MMEditingDataset):
    """MMEdit E-VQA dataset (EMNLP 2023).

    Expected annotation format (JSON list)::

        {
          "image": "relative/path.jpg",
          "src": "What sport ...",
          "pred": "surfing",
          "alt": "skateboarding",
          "rephrase": "Which sport ...",
          "image_rephrase": "rephrase_img.jpg",
          "loc": "locality question",
          "loc_ans": "locality answer",
          "m_loc": "multimodal_locality_img.jpg",
          "m_loc_q": "multimodal locality question",
          "m_loc_a": "multimodal locality answer"
        }
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        rephrase_image_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            data_path=data_path,
            image_dir=image_dir,
            rephrase_image_dir=rephrase_image_dir,
            **kwargs,
        )

    def _default_data_path(self, split: Optional[str]) -> Optional[str]:
        return "data/edit/mmedit/vqa.json"

    def normalize_record(
        self, item: Dict[str, Any], index: int
    ) -> Optional[Union[EditingSample, List[EditingSample]]]:
        prompt = self._flatten_text(item.get("src", item.get("prompt", ""))).strip()
        target_new = self._flatten_text(item.get("alt", item.get("target_new", ""))).strip()
        if not prompt or not target_new:
            return None

        subject = self._flatten_text(item.get("subject", "")).strip() or prompt
        target_old = self._flatten_text(item.get("pred", "")).strip() or None

        image = self._resolve_image(item.get("image", ""))
        rephrase_image = self._resolve_image(
            item.get("image_rephrase", ""),
            self.rephrase_image_dir,
        )

        rephrase_text = self._flatten_text(item.get("rephrase", "")).strip()
        rephrase_prompts = [rephrase_text] if rephrase_text else None

        locality_inputs = None
        loc_prompt = self._flatten_text(item.get("loc", "")).strip()
        loc_answer = self._flatten_text(item.get("loc_ans", "")).strip()
        if loc_prompt and loc_answer:
            locality_inputs = {
                "text_locality": [{"prompt": loc_prompt, "ground_truth": loc_answer}]
            }

        mm_locality = None
        m_loc_img_name = item.get("m_loc", "")
        m_loc_q = self._flatten_text(item.get("m_loc_q", "")).strip()
        m_loc_a = self._flatten_text(item.get("m_loc_a", "")).strip()
        if m_loc_q and m_loc_a:
            mm_locality = {
                "image": self._resolve_image(m_loc_img_name) if m_loc_img_name else None,
                "prompt": m_loc_q,
                "ground_truth": m_loc_a,
            }

        return EditingSample(
            prompt=prompt,
            subject=subject,
            target_new=target_new,
            target_old=target_old,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            image=image,
            image_rephrase=rephrase_image,
            multimodal_locality_inputs=mm_locality,
        )


class MMEditCaptionDataset(MMEditingDataset):
    """MMEdit E-IC (Image Captioning) dataset (EMNLP 2023).

    Annotation format is identical to E-VQA; the only difference is that the
    prompt is a captioning instruction rather than a visual question.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        rephrase_image_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            data_path=data_path,
            image_dir=image_dir,
            rephrase_image_dir=rephrase_image_dir,
            **kwargs,
        )

    def _default_data_path(self, split: Optional[str]) -> Optional[str]:
        return "data/edit/mmedit/caption.json"

    def normalize_record(
        self, item: Dict[str, Any], index: int
    ) -> Optional[Union[EditingSample, List[EditingSample]]]:
        prompt = self._flatten_text(item.get("src", item.get("prompt", ""))).strip()
        target_new = self._flatten_text(item.get("alt", item.get("target_new", ""))).strip()
        if not prompt or not target_new:
            return None

        subject = self._flatten_text(item.get("subject", "")).strip() or prompt
        target_old = self._flatten_text(item.get("pred", "")).strip() or None

        image = self._resolve_image(item.get("image", ""))
        rephrase_image = self._resolve_image(
            item.get("image_rephrase", ""),
            self.rephrase_image_dir,
        )

        rephrase_text = self._flatten_text(item.get("rephrase", "")).strip()
        rephrase_prompts = [rephrase_text] if rephrase_text else None

        locality_inputs = None
        loc_prompt = self._flatten_text(item.get("loc", "")).strip()
        loc_answer = self._flatten_text(item.get("loc_ans", "")).strip()
        if loc_prompt and loc_answer:
            locality_inputs = {
                "text_locality": [{"prompt": loc_prompt, "ground_truth": loc_answer}]
            }

        mm_locality = None
        m_loc_img_name = item.get("m_loc", "")
        m_loc_q = self._flatten_text(item.get("m_loc_q", "")).strip()
        m_loc_a = self._flatten_text(item.get("m_loc_a", "")).strip()
        if m_loc_q and m_loc_a:
            mm_locality = {
                "image": self._resolve_image(m_loc_img_name) if m_loc_img_name else None,
                "prompt": m_loc_q,
                "ground_truth": m_loc_a,
            }

        return EditingSample(
            prompt=prompt,
            subject=subject,
            target_new=target_new,
            target_old=target_old,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            image=image,
            image_rephrase=rephrase_image,
            multimodal_locality_inputs=mm_locality,
        )


class MMKEBenchDataset(MMEditingDataset):
    """MMKE-Bench dataset (ICLR 2025).

    Three editing task subsets:
      - visual_entity: editing visual entity knowledge
      - visual_semantic: editing visual semantic knowledge
      - user_specific: user-specific editing

    Data can be loaded from a local JSON file or via HuggingFace datasets.
    """

    SUBSET_FILES = {
        "visual_entity": "data/edit/mmke_bench/visual_entity.json",
        "visual_semantic": "data/edit/mmke_bench/visual_semantic.json",
        "user_specific": "data/edit/mmke_bench/user_specific.json",
    }

    def __init__(
        self,
        subset: str = "visual_entity",
        data_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        hf_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.subset = subset.lower()
        default_hf = None
        if hf_args is None and data_path is None:
            default_hf = {
                "path": "MMKE-Bench-dataset",
                "name": self.subset,
                "split": "test",
            }
        super().__init__(
            data_path=data_path,
            image_dir=image_dir,
            hf_args=hf_args or default_hf,
            **kwargs,
        )

    def _default_data_path(self, split: Optional[str]) -> Optional[str]:
        return self.SUBSET_FILES.get(self.subset)

    def normalize_record(
        self, item: Dict[str, Any], index: int
    ) -> Optional[Union[EditingSample, List[EditingSample]]]:
        prompt = self._flatten_text(
            item.get("prompt", item.get("question", item.get("src", "")))
        ).strip()
        target_new = self._flatten_text(
            item.get("target_new", item.get("alt", item.get("answer", "")))
        ).strip()
        if not prompt or not target_new:
            return None

        subject = self._flatten_text(item.get("subject", "")).strip() or prompt
        target_old = self._flatten_text(
            item.get("target_old", item.get("ground_truth", ""))
        ).strip() or None

        image_field = item.get("image", item.get("image_path", ""))
        image = None
        if isinstance(image_field, str) and image_field:
            image = self._resolve_image(image_field)
        elif hasattr(image_field, "convert"):
            image = image_field.convert("RGB")

        rephrase_text = self._flatten_text(
            item.get("rephrase", item.get("rephrase_prompt", ""))
        ).strip()
        rephrase_prompts = [rephrase_text] if rephrase_text else None

        locality_inputs = self._normalize_eval_groups(
            item.get("locality_inputs", item.get("locality")), "locality"
        )
        portability_inputs = self._normalize_eval_groups(
            item.get("portability_inputs", item.get("portability")), "portability"
        )

        return EditingSample(
            prompt=prompt,
            subject=subject,
            target_new=target_new,
            target_old=target_old,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            image=image,
        )
