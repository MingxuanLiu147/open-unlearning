"""
Know-Surgery 评估器注册表
========================

统一管理所有评估器：
- Unlearning 评估：TOFUEvaluator, MUSEEvaluator
- Editing 评估：EditReliabilityEvaluator, EditLocalityEvaluator 等
- Injection 评估：InjectAccuracyEvaluator, InjectRetentionEvaluator
"""

from typing import Dict, Any
from omegaconf import DictConfig
from evals.tofu import TOFUEvaluator
from evals.muse import MUSEEvaluator
from evals.lm_eval import LMEvalEvaluator

# Knowledge Editing 评估器
from evals.edit import (
    EditEvaluator,
    EditReliabilityEvaluator,
    EditGeneralizationEvaluator,
    EditLocalityEvaluator,
    EditPortabilityEvaluator,
    EditComprehensiveEvaluator,
)

# Knowledge Injection 评估器
from evals.inject import (
    InjectEvaluator,
    InjectAccuracyEvaluator,
    InjectRetentionEvaluator,
)

EVALUATOR_REGISTRY: Dict[str, Any] = {}


def _register_evaluator(evaluator_class):
    EVALUATOR_REGISTRY[evaluator_class.__name__] = evaluator_class


def get_evaluator(name: str, eval_cfg: DictConfig, **kwargs):
    evaluator_handler_name = eval_cfg.get("handler")
    assert evaluator_handler_name is not None, ValueError(f"{name} handler not set")
    eval_handler = EVALUATOR_REGISTRY.get(evaluator_handler_name)
    if eval_handler is None:
        raise NotImplementedError(
            f"{evaluator_handler_name} not implemented or not registered"
        )
    return eval_handler(eval_cfg, **kwargs)


def get_evaluators(eval_cfgs: DictConfig, **kwargs):
    evaluators = {}
    for eval_name, eval_cfg in eval_cfgs.items():
        evaluators[eval_name] = get_evaluator(eval_name, eval_cfg, **kwargs)
    return evaluators


# Register Unlearning benchmark evaluators
_register_evaluator(TOFUEvaluator)
_register_evaluator(MUSEEvaluator)
_register_evaluator(LMEvalEvaluator)

# Register Knowledge Editing evaluators
_register_evaluator(EditEvaluator)
_register_evaluator(EditReliabilityEvaluator)
_register_evaluator(EditGeneralizationEvaluator)
_register_evaluator(EditLocalityEvaluator)
_register_evaluator(EditPortabilityEvaluator)

# Register Knowledge Injection evaluators
_register_evaluator(InjectEvaluator)
_register_evaluator(InjectAccuracyEvaluator)
_register_evaluator(InjectRetentionEvaluator)
