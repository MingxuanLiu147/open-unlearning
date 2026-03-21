"""多模态 unlearning 训练入口（sidecar，不修改现有 train.py）。

通过 Hydra 配置驱动，加载多模态模型 + MLLMU 数据 + 遗忘方法，执行训练并保存 checkpoint。

使用示例：
    python src/mm_train.py \
        model=Qwen2VL-2B \
        data=mllmu \
        trainer=MMGradAscent
"""

import logging
import sys

import hydra
from omegaconf import DictConfig

sys.path.insert(0, "src")

logger = logging.getLogger(__name__)

MM_TRAINER_REGISTRY = {
    "MMGradAscent": "trainer.unlearn.mm_grad_ascent.MMGradAscent",
    "MMGradDiff": "trainer.unlearn.mm_grad_diff.MMGradDiff",
    "MMKLMin": "trainer.unlearn.mm_kl_min.MMKLMin",
    "MMNPO": "trainer.unlearn.mm_npo.MMNPO",
    "MMUnlearner": "trainer.unlearn.mmunlearner.MMUnlearner",
}


def _import_trainer_cls(dotpath: str):
    module_path, cls_name = dotpath.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


@hydra.main(version_base=None, config_path="../configs", config_name="mm_train.yaml")
def main(cfg: DictConfig):
    from model.multimodal import get_mm_model
    from data.multimodal import get_mm_data

    logger.info("Loading multimodal model...")
    model, processor = get_mm_model(cfg.model)

    logger.info("Loading MLLMU data...")
    forget_loader, retain_loader = get_mm_data(cfg.data, processor)

    trainer_name = cfg.trainer.handler
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
