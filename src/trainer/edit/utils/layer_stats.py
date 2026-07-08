"""
Compute second-moment (covariance) statistics for a model layer.

Used by MEMIT / AlphaEdit to estimate the key distribution and construct
the update matrix.  The statistics are cached to disk to avoid redundant
re-computation.

Ported from:
- https://github.com/kmeng01/memit  (MEMIT, MIT License)
- https://github.com/TrustedLLM/UnKE  (UnKE / AnyEdit)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from trainer.edit.utils import nethook

logger = logging.getLogger(__name__)


def _resolve_max_length(model_config) -> int:
    """Heuristic to determine the maximum sequence length."""
    for attr in (
        "max_position_embeddings",
        "n_positions",
        "max_sequence_length",
        "seq_length",
    ):
        val = getattr(model_config, attr, None)
        if val is not None:
            return int(val)
    return 2048


def layer_stats(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_name: str,
    stats_dir: str | Path,
    ds_name: str = "wikipedia",
    to_collect: list[str] | None = None,
    *,
    sample_size: int = 100_000,
    batch_tokens: int | None = None,
    precision: str = "float32",
    force_recompute: bool = False,
) -> dict[str, torch.Tensor]:
    """Load or compute cached second-moment statistics for *layer_name*.

    Returns a dict mapping stat name (e.g. ``"mom2"``) to the accumulated
    tensor.  Results are saved as ``.npz`` files under *stats_dir*.
    """
    to_collect = to_collect or ["mom2"]
    dtype = getattr(torch, precision)
    stats_dir = Path(stats_dir)

    model_tag = model.config._name_or_path.rsplit("/")[-1]
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    file_name = (
        stats_dir
        / model_tag
        / f"{ds_name}_stats"
        / f"{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    )

    if file_name.exists() and not force_recompute:
        logger.info("Loading cached layer stats from %s", file_name)
        data = dict(**torch.load(file_name, map_location="cpu", weights_only=True))
        return data

    logger.info("Computing layer stats for %s (this may take a while) ...", layer_name)

    maxlen = _resolve_max_length(model.config)
    if batch_tokens is not None and batch_tokens < maxlen:
        maxlen = batch_tokens
    if batch_tokens is None:
        batch_tokens = maxlen * 3

    raw_ds = load_dataset(
        ds_name,
        {"wikitext": "wikitext-103-raw-v1", "wikipedia": "20220301.en"}.get(ds_name, ds_name),
    )

    text_key = "text"
    if text_key not in raw_ds["train"].column_names:
        text_key = raw_ds["train"].column_names[0]

    device = next(model.parameters()).device
    mom2_accum: Optional[torch.Tensor] = None
    count = 0

    nethook.set_requires_grad(False, model)

    texts = list(raw_ds["train"][text_key])
    if sample_size is not None:
        texts = texts[:sample_size]

    batch_size = 8
    for start in tqdm(range(0, len(texts), batch_size), desc="layer_stats"):
        batch_texts = [t for t in texts[start : start + batch_size] if t and t.strip()]
        if not batch_texts:
            continue
        tok_batch = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=maxlen,
        ).to(device)

        with torch.no_grad():
            with nethook.Trace(
                model, layer_name, retain_input=True, retain_output=False, stop=True
            ) as tr:
                model(**tok_batch)

            feats = tr.input
            if isinstance(feats, tuple):
                feats = feats[0]

            mask = tok_batch["attention_mask"].unsqueeze(-1).to(feats.dtype)
            feats = feats * mask
            flat = feats.reshape(-1, feats.shape[-1])
            valid = mask.reshape(-1).bool().squeeze(-1)
            flat = flat[valid].to(dtype=dtype)

            if mom2_accum is None:
                mom2_accum = flat.T @ flat
            else:
                mom2_accum += flat.T @ flat
            count += flat.shape[0]

    if mom2_accum is not None:
        mom2_accum /= count

    result = {"mom2": mom2_accum, "count": torch.tensor(count)}
    file_name.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, file_name)
    logger.info("Saved layer stats to %s (%d tokens)", file_name, count)
    return result
