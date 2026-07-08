"""
MM-IKE: Multimodal In-Context Knowledge Editing
================================================

Extends IKEEditor with multimodal capabilities.  Does not modify model
weights -- instead, stores multimodal edit facts and retrieves them as
ICL demonstrations during inference.

Reference:
  - IKE: https://arxiv.org/abs/2305.12740
  - MMEdit: https://arxiv.org/abs/2310.08475
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from trainer.edit.base import EditRequest
from trainer.edit.ike import IKEEditor
from trainer.edit.mm_mixin import MMEditMixin

logger = logging.getLogger(__name__)


class MMIKEEditor(IKEEditor, MMEditMixin):
    """Multimodal IKE editor.

    Stores multimodal edit facts (text descriptions of the image context)
    and retrieves them as ICL demonstrations.  The sentence-transformer
    encodes only the textual part of each fact for retrieval.
    """

    def __init__(
        self,
        processor=None,
        image_description_template: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if processor is not None:
            self.init_mm(processor)
        self.image_description_template = image_description_template or (
            "[Image context] {prompt}"
        )

    def edit(
        self,
        requests: Union[EditRequest, List[EditRequest]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Store multimodal edits.

        For each request that carries an image, the fact string is
        augmented with an image description prefix so that retrieval can
        distinguish visual from text-only edits.
        """
        if isinstance(requests, EditRequest):
            requests = [requests]

        results: Dict[str, Any] = {
            "success": True,
            "edited_count": 0,
            "icl_examples": [],
        }

        for request in requests:
            try:
                if request.image is not None:
                    enriched_prompt = self.image_description_template.format(
                        prompt=request.prompt
                    )
                    import dataclasses
                    enriched_request = dataclasses.replace(
                        request, prompt=enriched_prompt
                    )
                    self._store_edit(enriched_request)
                else:
                    self._store_edit(request)

                results["edited_count"] += 1
                icl = (
                    self.get_icl_prompt(request.prompt)
                    if self.knowledge_store
                    else ""
                )
                results["icl_examples"].append(icl)
            except Exception as e:
                logger.error("MM-IKE edit failed: %s", e)
                results["success"] = False

        return results

    def get_mm_icl_prompt(self, query_prompt: str, image=None) -> str:
        """Retrieve ICL prompt, optionally enriched with image context."""
        if image is not None:
            query = self.image_description_template.format(prompt=query_prompt)
        else:
            query = query_prompt
        return self.get_icl_prompt(query)
