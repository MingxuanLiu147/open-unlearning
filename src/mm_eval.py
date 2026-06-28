"""多模态 unlearning 评估入口（sidecar，不修改现有 eval.py）。

根据 benchmark 配置选择 MLLMU 或 CLEAR evaluator，执行评估并输出 JSON 结果。

注意：
    当前环境下需要在导入 Hydra 前先导入 torch，否则直接执行
    `python src/mm_eval.py ...` 可能导致 CUDA 不可用。

使用示例：
    python src/mm_eval.py \
        model.model_path=saves/mm_unlearn/MMGradAscent/latest \
        eval=mllmu
"""

import logging
import sys

import hydra
import torch
from omegaconf import DictConfig

sys.path.insert(0, "src")

logger = logging.getLogger(__name__)

EVALUATOR_REGISTRY = {
    "mllmu": "evals.mllmu.MLLMUEvaluator",
    "clear": "evals.clear.CLEAREvaluator",
    "fiubench": "evals.fiubench.FIUBenchEvaluator",
}


def _import_cls(dotpath: str):
    module_path, cls_name = dotpath.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _resolve_runtime_device(cfg: DictConfig) -> str:
    device = str(cfg.get("device", "cuda"))
    require_cuda = bool(cfg.get("require_cuda", device.startswith("cuda")))
    cuda_available = torch.cuda.is_available()

    logger.info(
        "Runtime device check: device=%s, require_cuda=%s, cuda_available=%s, cuda_device_count=%s",
        device,
        require_cuda,
        cuda_available,
        torch.cuda.device_count(),
    )

    if device.startswith("cuda") and not cuda_available:
        hint = (
            "CUDA is not available for mm_eval. "
            "In this environment avoid `python -u`, stdout/stderr redirection, "
            "`CUDA_VISIBLE_DEVICES=*`, and `PYTORCH_ALLOC_CONF=expandable_segments:True`. "
            "If you intentionally want CPU, override with `device=cpu require_cuda=false`."
        )
        if require_cuda:
            raise RuntimeError(hint)
        logger.warning("%s", hint)
        return "cpu"

    return device


@hydra.main(version_base=None, config_path="../configs", config_name="mm_eval.yaml")
def main(cfg: DictConfig):
    from model.multimodal import get_mm_model

    device = _resolve_runtime_device(cfg)

    logger.info("Loading model for evaluation...")
    model, processor = get_mm_model(cfg.model)
    if device.startswith("cuda"):
        model = model.to(device)
        logger.info("Moved evaluation model to %s", device)
    else:
        logger.warning("Evaluation will run on CPU")
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
