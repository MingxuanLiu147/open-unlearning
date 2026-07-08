"""
Multimodal Editing Mixin
========================

Provides multimodal tokenization and forward-pass utilities for MM editing
methods.  All MM editors (MM-GRACE, MM-WISE, etc.) inherit from this mixin
alongside their text-only parent to gain image processing capabilities.

Target models: Qwen2-VL / Qwen2.5-VL / LLaVA-OneVision (HF Processor path).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class MMEditMixin:
    """Mixin that injects multimodal capabilities into text-only editors.

    Expects the concrete class to expose ``self.model`` and sets
    ``self.processor`` during ``init_mm()``.
    """

    processor: Any = None

    def init_mm(self, processor: Any) -> None:
        """Attach the multimodal processor (call once after __init__)."""
        self.processor = processor

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def mm_tokenize(
        self,
        prompts: Union[str, List[str]],
        images: Union[Any, List[Any]],
        targets: Union[str, List[str]],
        *,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build model inputs from prompt + image + target.

        Uses ``processor.apply_chat_template`` to construct the chat text,
        appends the target answer, then calls the processor to encode
        everything.  Labels are masked so that only the target (answer)
        tokens contribute to the loss.

        Returns a dict with at least ``input_ids``, ``attention_mask``,
        ``pixel_values``, and ``labels``.
        """
        if self.processor is None:
            raise RuntimeError("Call init_mm(processor) before mm_tokenize")

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(targets, str):
            targets = [targets]
        if not isinstance(images, list):
            images = [images]

        if device is None:
            device = self._mm_device()

        texts: List[str] = []
        valid_images: List[Any] = []

        for prompt, image, target in zip(prompts, images, targets):
            user_content: list = []
            if image is not None:
                user_content.append({"type": "image"})
                valid_images.append(self._load_image(image))
            user_content.append({"type": "text", "text": prompt})

            messages = [
                {"role": "user", "content": user_content},
            ]
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            texts.append(text + target)

        batch = self.processor(
            text=texts,
            images=valid_images if valid_images else None,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        labels = self._build_answer_labels(batch["input_ids"], targets)
        batch["labels"] = labels

        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def mm_tokenize_text_only(
        self,
        prompts: Union[str, List[str]],
        targets: Union[str, List[str]],
        *,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text-only inputs through the processor (for locality tests)."""
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(targets, str):
            targets = [targets]
        if device is None:
            device = self._mm_device()

        texts = []
        for prompt, target in zip(prompts, targets):
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            texts.append(text + target)

        batch = self.processor(
            text=texts,
            images=None,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        labels = self._build_answer_labels(batch["input_ids"], targets)
        batch["labels"] = labels

        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def mm_forward(
        self,
        model: Any,
        inputs: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Any:
        """Run a multimodal forward pass.  Filters out non-tensor keys."""
        tensor_inputs = {
            k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)
        }
        return model(**tensor_inputs, **kwargs)

    def mm_loss(
        self,
        model: Any,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute cross-entropy loss on the answer tokens."""
        outputs = self.mm_forward(model, inputs)
        return outputs.loss

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image: Any) -> Image.Image:
        """Accept PIL.Image, file path, or bytes; return RGB PIL.Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, bytes)):
            return Image.open(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mm_device(self) -> torch.device:
        """Best-effort device detection from ``self.model``."""
        model = getattr(self, "model", None)
        if model is None:
            return torch.device("cpu")
        if hasattr(model, "device"):
            return model.device
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _build_answer_labels(
        self,
        input_ids: torch.Tensor,
        targets: List[str],
    ) -> torch.Tensor:
        """Create labels that only supervise the answer span.

        Encodes each target string, finds its token ids at the *end* of the
        corresponding input_ids row, and masks everything else with -100.
        """
        labels = torch.full_like(input_ids, -100)

        for i, target in enumerate(targets):
            target_ids = self.processor.tokenizer(
                target, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]

            seq_len = input_ids.size(1)
            tgt_len = len(target_ids)

            if tgt_len > seq_len:
                tgt_len = seq_len
                target_ids = target_ids[:tgt_len]

            matched = False
            search_start = max(0, seq_len - tgt_len - 10)
            for start in range(seq_len - tgt_len, search_start - 1, -1):
                if start < 0:
                    break
                if torch.equal(input_ids[i, start : start + tgt_len], target_ids.to(input_ids.device)):
                    labels[i, start : start + tgt_len] = input_ids[i, start : start + tgt_len]
                    matched = True
                    break

            if not matched:
                labels[i, -tgt_len:] = input_ids[i, -tgt_len:]

        return labels
