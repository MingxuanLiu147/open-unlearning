"""
SOUL (Second-Order UnLearning) for LLMs
========================================

Reference: EMNLP 2024 — "SOUL: Unlocking the Power of Second-Order
           Optimization for LLM Unlearning"
           arXiv:2404.18239; Code: https://github.com/OPTML-Group/SOUL

Core idea:
    Wrap any existing unlearning objective (GA, GradDiff, PO, NPO) with
    Sophia-style second-order preconditioning.  The diagonal Hessian
    estimate rescales per-parameter gradients, enabling more effective
    unlearning with controlled divergence.

    Update rule (per-element):
        m_t = beta1 * m_{t-1} + (1-beta1) * g_t
        h_t = beta2 * h_{t-1} + (1-beta2) * diag_hessian_t
        theta_{t+1} = theta_t - lr * clip(m_t / max(rho*h_t, eps), 1)

    For the forget objective the sign is flipped (ascent), while for
    retain it stays descent — controlled via the sign of gamma in
    GradDiff.

Implementation:
    We inherit GradDiff (which already implements the GA+retain loss
    pattern) and override the optimizer creation to inject Sophia.
    The compute_loss stays the same as GradDiff by default, but can
    be combined with any loss variant via Hydra config inheritance.
"""

import torch
from torch.optim import Optimizer
from trainer.unlearn.grad_diff import GradDiff


class SophiaG(Optimizer):
    """Minimal Sophia-G optimizer (Gauss-Newton diagonal Hessian estimate).

    Faithfully re-implements the core Sophia algorithm from
    Liu et al. 2023 ("Sophia: A Scalable Stochastic Second-order
    Optimizer for Language Model Pre-training") and used in the SOUL paper.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas=(0.9, 0.95),
        rho: float = 0.04,
        eps: float = 1e-5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["h"] = torch.zeros_like(p)

                m, h = state["m"], state["h"]
                state["step"] += 1

                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                if "hessian" in state:
                    h.mul_(beta2).add_(state["hessian"], alpha=1 - beta2)

                if wd > 0:
                    p.data.mul_(1 - lr * wd)

                update = m / torch.clamp(rho * h, min=eps)
                update.clamp_(-1.0, 1.0)
                p.data.add_(update, alpha=-lr)

        return loss

    def update_hessian(self, params):
        """Store per-parameter diagonal Hessian estimate (Gauss-Newton)."""
        for p in params:
            if p.grad is not None:
                self.state[p]["hessian"] = p.grad.detach().pow(2)


class SOUL(GradDiff):
    """SOUL: Second-Order UnLearning.

    Replaces the default AdamW optimizer with SophiaG. Every
    ``hessian_update_freq`` steps a second backward pass estimates
    the diagonal Hessian on a mini-batch from the forget set.

    All other behaviour (loss, data) is inherited from GradDiff.
    """

    def __init__(
        self,
        sophia_lr: float = 1e-4,
        sophia_betas: tuple = (0.9, 0.95),
        sophia_rho: float = 0.04,
        sophia_eps: float = 1e-5,
        hessian_update_freq: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sophia_lr = sophia_lr
        self.sophia_betas = sophia_betas
        self.sophia_rho = sophia_rho
        self.sophia_eps = sophia_eps
        self.hessian_update_freq = hessian_update_freq

    def create_optimizer(self):
        """Override HF Trainer's optimizer creation to use SophiaG."""
        if self.optimizer is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = SophiaG(
                params,
                lr=self.sophia_lr,
                betas=self.sophia_betas,
                rho=self.sophia_rho,
                eps=self.sophia_eps,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Standard training step + periodic Hessian estimation."""
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        if (
            hasattr(self, "state")
            and self.state.global_step % self.hessian_update_freq == 0
        ):
            self._update_hessian(model, inputs)

        return loss

    def _update_hessian(self, model, inputs):
        """Compute diagonal Hessian estimate on forget mini-batch."""
        model.zero_grad()
        forget_inputs = inputs.get("forget", inputs)
        fi = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        outputs = model(**fi)
        outputs.loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer.update_hessian(trainable)
        model.zero_grad()
