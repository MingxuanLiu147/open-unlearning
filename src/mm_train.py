"""多模态 unlearning 训练入口（sidecar，不修改现有 train.py）。

通过 Hydra 配置驱动，加载多模态模型 + MLLMU 数据 + 遗忘方法，执行训练并保存 checkpoint。

注意：
    当前环境下需要在导入 Hydra 前先导入 torch，否则直接执行
    `python src/mm_train.py ...` 可能导致 CUDA 不可用。

使用示例：
    python src/mm_train.py \
        model=Qwen2VL-2B \
        data=mllmu \
        trainer=MMGradAscent
"""

import logging
import sys

import torch
import hydra
from omegaconf import DictConfig

sys.path.insert(0, "src")

logger = logging.getLogger(__name__)

MM_TRAINER_REGISTRY = {
    "MMGradAscent": "trainer.unlearn.mm_grad_ascent.MMGradAscent",
    "MMGradDiff": "trainer.unlearn.mm_grad_diff.MMGradDiff",
    "MMKLMin": "trainer.unlearn.mm_kl_min.MMKLMin",
    "MMNPO": "trainer.unlearn.mm_npo.MMNPO",
    "MMRetainFT": "trainer.unlearn.mm_retain_ft.MMRetainFT",
    "MMUnlearner": "trainer.unlearn.mmunlearner.MMUnlearner",
}


def _import_trainer_cls(dotpath: str):
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
            "CUDA is not available for mm_train. "
            "In this environment avoid `python -u`, stdout/stderr redirection, "
            "`CUDA_VISIBLE_DEVICES=*`, and `PYTORCH_ALLOC_CONF=expandable_segments:True`. "
            "If you intentionally want CPU, override with `device=cpu require_cuda=false`."
        )
        if require_cuda:
            raise RuntimeError(hint)
        logger.warning("%s", hint)
        return "cpu"

    return device


def _get_data_loaders(cfg: DictConfig, processor):
    benchmark = str(cfg.data.get("benchmark", "mllmu")).lower()
    if benchmark == "clear":
        from data.clear_dataset import get_clear_data

        logger.info("Loading CLEAR data...")
        return get_clear_data(cfg.data, processor)

    from data.multimodal import get_mm_data

    logger.info("Loading MLLMU data...")
    return get_mm_data(cfg.data, processor)


@hydra.main(version_base=None, config_path="../configs", config_name="mm_train.yaml")
def main(cfg: DictConfig):
    from model.multimodal import get_mm_model

    device = _resolve_runtime_device(cfg)
    trainer_name = cfg.trainer.handler

    if trainer_name == "MMUnlearner" and bool(cfg.model.get("lora", {}).get("enabled", False)):
        raise ValueError(
            "MMUnlearner should run with `model.lora.enabled=false` so grad masks and "
            "trainable parameters stay in the same parameter space."
        )

    logger.info("Loading multimodal model...")
    model, processor = get_mm_model(cfg.model)
    if device.startswith("cuda"):
        model = model.to(device)
        logger.info("Moved training model to %s", device)

    forget_loader, retain_loader = _get_data_loaders(cfg, processor)

    trainer_dotpath = MM_TRAINER_REGISTRY.get(trainer_name)
    if trainer_dotpath is None:
        raise ValueError(
            f"Unknown mm trainer: {trainer_name}. "
            f"Available: {list(MM_TRAINER_REGISTRY.keys())}"
        )
    trainer_cls = _import_trainer_cls(trainer_dotpath)

    method_args = dict(cfg.trainer.get("method_args", {}))
    trainer_args = dict(cfg.trainer.get("args", {}))

    output_dir = cfg.get("output_dir", "saves/mm_unlearn")

    if trainer_name == "MMNPO":
        oracle_path = method_args.pop("oracle_model_path", None)
        if oracle_path is None:
            raise ValueError("MMNPO requires method_args.oracle_model_path")
        from model.multimodal import get_mm_model as _load
        from omegaconf import OmegaConf
        oracle_cfg = OmegaConf.create({
            "model_path": oracle_path,
            "processor_path": cfg.model.get("processor_path", cfg.model.model_path),
            "model_family": cfg.model.get("model_family", "qwen2vl"),
            "torch_dtype": cfg.model.get("torch_dtype", "bfloat16"),
        })
        oracle_model, _ = _load(oracle_cfg)
        method_args["oracle_model"] = oracle_model

    trainer = trainer_cls(
        model=model,
        processor=processor,
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        lr=trainer_args.get("lr", 5e-4),
        num_epochs=trainer_args.get("num_epochs", 5),
        max_grad_norm=trainer_args.get("max_grad_norm", 1.0),
        gradient_accumulation_steps=trainer_args.get("gradient_accumulation_steps", 1),
        warmup_steps=trainer_args.get("warmup_steps", 0),
        save_dir=output_dir,
        **method_args,
    )

    logger.info("Starting training with %s...", trainer_name)
    trainer.train()

    logger.info("Saving model...")
    trainer.save_model(output_dir)
    logger.info("Done. Model saved to %s", output_dir)


if __name__ == "__main__":
    main()
