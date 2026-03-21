"""CLEAR benchmark evaluator 占位（Phase 5 实现，当前仅提供接口）。"""

import logging

logger = logging.getLogger(__name__)


class CLEAREvaluator:
    """CLEAR benchmark 评估器占位。Phase 5 补齐完整评估逻辑。"""

    def __init__(self, eval_cfg, model, processor):
        self.model = model
        self.processor = processor
        self.eval_cfg = eval_cfg

    def evaluate(self):
        logger.warning("CLEAREvaluator is a placeholder. Full implementation in Phase 5.")
        return {}
