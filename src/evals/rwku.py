"""
RWKU (Real-World Knowledge Unlearning) Evaluator
==================================================

Reference: NeurIPS 2024 D&B — "RWKU: Benchmarking Real-World Knowledge
           Unlearning for Large Language Models"
           GitHub: jinzhuoran/RWKU
           Dataset: HuggingFace jinzhuoran/RWKU

Evaluates unlearning on 200 real-world famous people with:
- 4 MIA methods (loss-based, zlib, min-k, min-k++)
- 9 adversarial attack probes (prefix injection, affirmative suffix,
  role playing, reverse query, etc.)
- Locality (neighbour knowledge retention)
- Utility (general, reasoning, truthfulness, factuality, fluency)

This evaluator wraps the standard Evaluator base class so it plugs
into the existing eval.py / Hydra pipeline unchanged.
"""

from evals.base import Evaluator


class RWKUEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("RWKU", eval_cfg, **kwargs)
