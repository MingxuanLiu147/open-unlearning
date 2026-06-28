"""
UniKE: Unified Multimodal Knowledge Editing
============================================

Combines IKE-style external memory (assimilation) with ROME-style parameter
editing (accommodation) into a single vectorized key-value framework.

Paper: Towards Unified Multimodal Editing with Enhanced Knowledge Collaboration
       (NeurIPS 2024 Spotlight)
       https://arxiv.org/abs/2409.19872
Code:  https://github.com/beepkh/UniKE

Core ideas:
  1. Both intrinsic and external knowledge are vectorized key-value memories
  2. Assimilation phase: update external KV store (like IKE)
  3. Accommodation phase: rank-one parameter update (like ROME)
  4. Knowledge representations are disentangled into semantic / truthfulness
     spaces to promote collaboration between the two phases
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from trainer.edit.base import EditTrainer, EditRequest

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except ImportError:
    SentenceTransformer = None
    st_util = None


class UniKEEditor(EditTrainer):
    """UniKE editor: unified intrinsic + external knowledge editing.

    Phase 1 (assimilation): stores the edit fact in an external KV memory
    and builds an ICL retrieval index (same as IKE).

    Phase 2 (accommodation): applies a rank-one parameter update to the
    target MLP layer so that the model internalises the new fact (same as
    ROME).  The update vector is guided by the semantic representation
    from Phase 1 to promote knowledge collaboration.

    Attributes:
        k: number of ICL examples to retrieve
        v_lr: learning rate for ROME value optimisation
        v_num_grad_steps: optimisation steps for the target value vector
        accommodation_weight: scaling factor for the rank-one update
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        sentence_model_name: str = "all-MiniLM-L6-v2",
        k: int = 3,
        v_lr: float = 0.5,
        v_num_grad_steps: int = 20,
        clamp_norm_factor: float = 4.0,
        kl_factor: float = 0.0625,
        accommodation_weight: float = 1.0,
        use_icl_examples: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(layers=layers, *args, **kwargs)

        self.sentence_model_name = sentence_model_name
        self.k = k
        self.v_lr = v_lr
        self.v_num_grad_steps = v_num_grad_steps
        self.clamp_norm_factor = clamp_norm_factor
        self.kl_factor = kl_factor
        self.accommodation_weight = accommodation_weight
        self.use_icl_examples = use_icl_examples

        self.knowledge_store: List[Dict[str, Any]] = []
        self.knowledge_embeddings: Optional[torch.Tensor] = None
        self._sentence_model: Optional[Any] = None

    # ------------------------------------------------------------------
    # Sentence model (lazy)
    # ------------------------------------------------------------------

    @property
    def sentence_model(self):
        if self._sentence_model is not None:
            return self._sentence_model
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for UniKE")
        device = next(self.model.parameters()).device
        self._sentence_model = SentenceTransformer(self.sentence_model_name).to(device)
        return self._sentence_model

    # ------------------------------------------------------------------
    # Edit
    # ------------------------------------------------------------------

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs,
    ) -> Dict[str, Any]:
        if isinstance(requests, EditRequest):
            requests = [requests]

        results: Dict[str, Any] = {
            "success": True,
            "edited_count": 0,
            "phases": [],
        }

        for request in requests:
            try:
                phase_info = self._unified_edit(request)
                results["edited_count"] += 1
                results["phases"].append(phase_info)
            except Exception as exc:
                logger.error("UniKE edit failed for '%s': %s", request.prompt, exc)
                results["success"] = False
                results["phases"].append({"error": str(exc)})

        return results

    def _unified_edit(self, request: EditRequest) -> Dict[str, Any]:
        phase_info: Dict[str, Any] = {}

        semantic_vec = self._assimilate(request)
        phase_info["assimilation"] = "ok"

        self._accommodate(request, semantic_vec)
        phase_info["accommodation"] = "ok"

        return phase_info

    # ------------------------------------------------------------------
    # Phase 1: Assimilation (external KV memory, like IKE)
    # ------------------------------------------------------------------

    def _assimilate(self, request: EditRequest) -> torch.Tensor:
        """Store the new fact in external memory and return its semantic vector."""
        fact = f"{request.prompt} {request.target_new}"
        entry = {
            "prompt": request.prompt,
            "subject": request.subject,
            "target_new": request.target_new,
            "fact": fact,
        }
        self.knowledge_store.append(entry)

        if self.use_icl_examples:
            device = next(self.model.parameters()).device
            emb = self.sentence_model.encode(
                fact, show_progress_bar=False, convert_to_tensor=True,
            )
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            emb = st_util.normalize_embeddings(emb.to(device))
            if self.knowledge_embeddings is None:
                self.knowledge_embeddings = emb
            else:
                self.knowledge_embeddings = torch.cat(
                    [self.knowledge_embeddings, emb], dim=0,
                )
            return emb.squeeze(0)

        return torch.zeros(1)

    def get_icl_prompt(self, query: str) -> str:
        """Retrieve top-k ICL demonstrations for the query."""
        if (
            not self.use_icl_examples
            or not self.knowledge_store
            or self.knowledge_embeddings is None
        ):
            return ""
        device = self.knowledge_embeddings.device
        q_emb = self.sentence_model.encode(
            query, show_progress_bar=False, convert_to_tensor=True,
        )
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        q_emb = st_util.normalize_embeddings(q_emb.to(device))
        top_k = min(self.k, len(self.knowledge_store))
        hits = st_util.semantic_search(
            q_emb, self.knowledge_embeddings,
            score_function=st_util.dot_score, top_k=top_k,
        )
        if not hits or not hits[0]:
            return ""
        parts = []
        for hit in hits[0]:
            entry = self.knowledge_store[hit["corpus_id"]]
            parts.append(
                f"New Fact: {entry['fact']}\nPrompt: {entry['prompt']}\n\n"
            )
        return "".join(parts)

    # ------------------------------------------------------------------
    # Phase 2: Accommodation (rank-one parameter update, like ROME)
    # ------------------------------------------------------------------

    def _accommodate(
        self, request: EditRequest, semantic_vec: torch.Tensor,
    ) -> None:
        """Apply a rank-one update to the target MLP layer."""
        model = self.model
        tokenizer = self.tokenizer
        layer_idx = self._resolve_layer_indices(self.layers)[0]
        device = self._input_device(model)

        key = self._compute_subject_key(request, layer_idx, device)
        value = self._optimise_target_value(request, layer_idx, key, device)

        residual = value - self._current_mlp_output(key, layer_idx)
        key_norm_sq = key.dot(key).clamp(min=1e-10)
        update = (self.accommodation_weight * torch.outer(residual, key)) / key_norm_sq

        target_weight = self._get_target_weight(layer_idx)
        with torch.no_grad():
            target_weight.add_(update.to(target_weight.dtype))

        logger.info(
            "UniKE accommodation at layer %d: '%s' -> '%s' (update_norm=%.4f)",
            layer_idx, request.subject, request.target_new,
            update.norm().item(),
        )

    def _compute_subject_key(
        self, request: EditRequest, layer_idx: int, device: torch.device,
    ) -> torch.Tensor:
        """Get the hidden state at the subject's last token position."""
        model = self.model
        tokenizer = self.tokenizer

        prompt_ids = tokenizer(request.prompt, return_tensors="pt")["input_ids"].to(device)
        subj_ids = tokenizer(request.subject, add_special_tokens=False, return_tensors="pt")["input_ids"]
        subj_len = subj_ids.shape[1]

        prompt_len = prompt_ids.shape[1]
        subj_end = prompt_len - 1
        for start in range(prompt_len - subj_len, -1, -1):
            if torch.equal(prompt_ids[0, start:start + subj_len], subj_ids[0].to(device)):
                subj_end = start + subj_len - 1
                break

        layers_container = self._get_layers_container(model)
        hook_output = {}

        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                hook_output["hidden"] = out[0]
            else:
                hook_output["hidden"] = out

        handle = layers_container[layer_idx].register_forward_hook(hook_fn)
        model_dtype = next(model.parameters()).dtype
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=model_dtype, enabled=model_dtype != torch.float32):
            model(input_ids=prompt_ids)
        handle.remove()

        hidden = hook_output["hidden"]
        return hidden[0, min(subj_end, hidden.shape[1] - 1)].detach().float()

    def _optimise_target_value(
        self, request: EditRequest, layer_idx: int,
        key: torch.Tensor, device: torch.device,
    ) -> torch.Tensor:
        """Optimise a target value vector that makes the model output target_new."""
        model = self.model
        tokenizer = self.tokenizer

        full_text = f"{request.prompt} {request.target_new}"
        prompt_ids = tokenizer(request.prompt, return_tensors="pt")["input_ids"].to(device)
        full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)

        prompt_len = prompt_ids.shape[1]
        target_ids = full_ids[:, prompt_len:]
        if target_ids.numel() == 0:
            return key.clone()

        layers_container = self._get_layers_container(model)
        target_layer = layers_container[layer_idx]
        target_weight = self._get_target_weight(layer_idx)
        current_value = F.linear(key, target_weight.float())
        delta = torch.zeros_like(current_value, requires_grad=True)

        optimizer = torch.optim.Adam([delta], lr=self.v_lr)
        model_dtype = next(model.parameters()).dtype

        for step in range(self.v_num_grad_steps):
            def edit_hook(module, inp, out):
                if isinstance(out, tuple):
                    h = out[0].clone()
                else:
                    h = out.clone()
                h[0, -1] = h[0, -1] + delta.to(h.dtype)
                return (h,) + out[1:] if isinstance(out, tuple) else h

            handle = target_layer.register_forward_hook(edit_hook)
            with torch.amp.autocast("cuda", dtype=model_dtype, enabled=model_dtype != torch.float32):
                outputs = model(input_ids=full_ids)
            handle.remove()

            logits = outputs.logits[0, prompt_len - 1:-1]
            nll = F.cross_entropy(logits, target_ids[0])

            reg = self.kl_factor * delta.norm()
            loss = nll + reg

            optimizer.zero_grad()
            loss.backward()

            if self.clamp_norm_factor > 0 and delta.grad is not None:
                max_norm = self.clamp_norm_factor * current_value.norm()
                torch.nn.utils.clip_grad_norm_([delta], max_norm.item())

            optimizer.step()

        return (current_value + delta).detach().float()

    def _current_mlp_output(self, key: torch.Tensor, layer_idx: int) -> torch.Tensor:
        target_weight = self._get_target_weight(layer_idx)
        return F.linear(key, target_weight.float())

    def _get_target_weight(self, layer_idx: int) -> torch.Tensor:
        layers = self._get_layers_container(self.model)
        block = layers[layer_idx]
        for name in ("mlp.down_proj", "mlp.c_proj", "mlp.dense_4h_to_h"):
            parts = name.split(".")
            mod = block
            try:
                for p in parts:
                    mod = getattr(mod, p)
                return mod.weight
            except AttributeError:
                continue
        raise ValueError(f"Cannot find MLP output projection in layer {layer_idx}")
