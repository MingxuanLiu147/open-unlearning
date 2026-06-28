"""
RWKU (Real-World Knowledge Unlearning) Dataset
================================================

Reference: NeurIPS 2024 D&B — "RWKU: Benchmarking Real-World Knowledge
           Unlearning for Large Language Models"
           GitHub: jinzhuoran/RWKU
           HuggingFace: jinzhuoran/RWKU

200 real-world famous people as unlearning targets.
13,131 forget probes: fill-in-the-blank (3,268), QA (2,879), adversarial (6,984).
4 MIA methods + 9 adversarial attack types.
Utility evaluation across general, reasoning, truthfulness, factuality, fluency.
"""

from data.qa import QADataset


class RWKUForgetDataset(QADataset):
    """RWKU forget probes dataset.

    Wraps QADataset with RWKU-specific defaults.
    Supports forget_level1 (fill-in-the-blank + QA) and
    forget_level2 (adversarial probes) configs from HuggingFace.
    """
    pass


class RWKUNeighborDataset(QADataset):
    """RWKU neighbor (locality) dataset.

    Used for evaluating whether unlearning affects related but
    non-target knowledge (locality preservation).
    """
    pass


class RWKUMIADataset(QADataset):
    """RWKU MIA (Membership Inference Attack) dataset.

    Contains forget and retain splits for MIA evaluation.
    """
    pass
