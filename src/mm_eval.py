"""多模态 unlearning 评估入口（sidecar，不修改现有 eval.py）。

根据 benchmark 配置选择 MLLMU 或 CLEAR evaluator，执行评估并输出 JSON 结果。

使用示例：
    python src/mm_eval.py \
        model.model_path=saves/mm_unlearn/MMGradAscent/latest \
        eval=mllmu
"""

import logging
import sys

import hydra
from omegaconf import DictConfig

sys.path.insert(0, "src")

logger = logging.getLogger(__name__)

EVALUATOR_REGISTRY = {
    "mllmu": "evals.mllmu.MLLMUEvaluator",
    "clear": "evals.clear.CLEAREvaluator",
}


def _import_cls(dotpath: str):
    module_path, cls_name = dotpath.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


@hydra.main(version_base=None, config_path="../configs", config_name="mm_eval.yaml")
def main(cfg: DictConfig):
    from model.multimodal import get_mm_model

    logger.info("Loading model for evaluation...")
    model, processor = get_mm_model(cfg.model)
    model.eval()

    benchmark = cfg.eval.get("benchmark", "mllmu")
    eval_dotpath = EVALUATOR_REGISTRY.get(benchmark)
    if eval_dotpath is None:
        raise ValueError(
            f"Unknown benchmark: {benchmark}. Available: {list(EVALUATOR_REGISTRY.keys())}"
        )

    eval_cls = _import_cls(eval_dotpath)
    evaluator = eval_cls(cfg.eval, model, processor)

    logger.info("Running %s evaluation...", benchmark)
    results = evaluator.evaluate()
    logger.info("Evaluation complete. Results: %s", results)


if __name__ == "__main__":
    main()
