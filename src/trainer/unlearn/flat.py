"""
FLAT (Forget data only Loss AdjustmenT) Unlearning
====================================================

Reference: ICLR 2025 — "FLAT: Forgetting via f-Divergence Maximization
           Between Template and Forget Answers"
           OpenReview id=6ESRicalFE; arXiv:2410.11143
           Code: https://github.com/UCSC-REAL/FLAT

Core idea:
    Maximise an f-divergence between the model's prediction on the forget
    answer y_f and a safe template answer y_e, using ONLY forget data (no
    retain set, no reference model).

    With Pearson chi^2 divergence the variational loss becomes:
        L = -P(y_e|x) + 0.25 * P(y_f|x)^2 + 0.5 * P(y_f|x)

    where P(y|x; theta) is the geometric-mean token probability.

    In practice the paper implements this via two forward passes per sample:
    one with the original answer labels, one with template labels.
"""

import torch
import torch.nn.functional as F
from trainer.unlearn.base import UnlearnTrainer


def _avg_token_prob(logits, labels):
    """Geometric-mean token probability = exp(-avg_NLL)."""
    shifted = logits[..., :-1, :].contiguous()
    targets = labels[..., 1:].contiguous()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    per_token = loss_fn(shifted.transpose(-1, -2), targets)
    mask = targets != -100
    avg_nll = (per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    return torch.exp(-avg_nll)


class FLAT(UnlearnTrainer):
    """FLAT: f-divergence based unlearning using only forget data.

    This trainer expects the data pipeline to provide *two* label variants
    inside ``inputs["forget"]``:

    - ``labels``: original forget answer y_f
    - ``template_labels``: safe template answer y_e (e.g. "I don't know.")

    If ``template_labels`` is absent, a fixed IDK response is tokenised on
    the fly using ``self.tokenizer``.
    """

    def __init__(self, idk_text: str = "I don't know.", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idk_text = idk_text

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_inputs = inputs["forget"]
        input_ids = forget_inputs["input_ids"]
        attention_mask = forget_inputs["attention_mask"]
        forget_labels = forget_inputs["labels"]

        template_labels = forget_inputs.get("template_labels", None)
        if template_labels is None:
            template_labels = self._make_template_labels(
                input_ids, attention_mask, forget_labels
            )

        base = {"input_ids": input_ids, "attention_mask": attention_mask}

        out_f = model(**base, labels=forget_labels)
        p_f = _avg_token_prob(out_f.logits, forget_labels)

        out_e = model(**base, labels=template_labels)
        p_e = _avg_token_prob(out_e.logits, template_labels)

        loss = -p_e + 0.25 * p_f.pow(2) + 0.5 * p_f
        loss = loss.mean()

        return (loss, out_f) if return_outputs else loss

    def _make_template_labels(self, input_ids, attention_mask, forget_labels):
        """Build template labels by replacing answer tokens with IDK tokens."""
        idk_ids = self.tokenizer(
            self.idk_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0].to(input_ids.device)

        tpl = forget_labels.clone()
        for i in range(tpl.size(0)):
            ans_mask = tpl[i] != -100
            ans_positions = ans_mask.nonzero(as_tuple=True)[0]
            if len(ans_positions) == 0:
                continue
            start = ans_positions[0].item()
            end = min(start + len(idk_ids), tpl.size(1))
            tpl[i, start:end] = -100
            fill_len = min(len(idk_ids), end - start)
            tpl[i, start : start + fill_len] = idk_ids[:fill_len]
            tpl[i, start + fill_len : ] = -100
        return tpl
