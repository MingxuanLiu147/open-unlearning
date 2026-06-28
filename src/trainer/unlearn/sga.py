"""
SGA (Smoothed Gradient Ascent) Unlearning
==========================================

Reference: "Label Smoothing Improves Gradient Ascent in LLM Unlearning"
           arXiv:2510.22376

Core idea:
    Combine forget data (gradient ascent direction) with K-1 "normal"
    completions using a smoothing rate r, stabilizing the GA process.

Loss:
    L_SGA = (1 - r + r/K) * L_forget + (r/K) * sum(L_normal_k)

    where r=0 degenerates to standard GA.
    Normal data comes from the retain split.
"""

from trainer.unlearn.grad_diff import GradDiff


class SGA(GradDiff):
    """Smoothed Gradient Ascent.

    Inherits GradDiff so that both forget and retain loaders are available.
    The smoothing rate ``r`` and number of normal sequences ``K`` control
    the gradient mixture.  When r=0 this reduces to GradAscent.
    """

    def __init__(self, r: float = 0.8, K: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = r
        self.K = K

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_outputs = model(**forget_inputs)
        forget_ce = forget_outputs.loss

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_outputs = model(**retain_inputs)
        retain_ce = retain_outputs.loss

        w_forget = 1.0 - self.r + self.r / self.K
        w_retain = self.r / self.K

        loss = -w_forget * forget_ce + w_retain * retain_ce

        return (loss, forget_outputs) if return_outputs else loss
