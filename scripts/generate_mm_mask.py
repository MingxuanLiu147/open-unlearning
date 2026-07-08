"""SFRon 掩码生成独立入口。

使用 Hydra 配置加载模型和数据，计算 Fisher 信息矩阵，生成权重显著性掩码。

使用示例：
    python scripts/generate_mm_mask.py \
        model=Qwen3-VL-4B \
        data=mllmu \
        +mask=mmunlearner
"""

import os
import sys

sys.path.insert(0, "src")

import logging

import hydra
import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


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
            "CUDA is not available for generate_mm_mask. "
            "In this environment prefer a TTY-backed launch from the repo venv. "
            "If you intentionally want CPU, override with `device=cpu require_cuda=false`."
        )
        if require_cuda:
            raise RuntimeError(hint)
        logger.warning("%s", hint)
        return "cpu"

    return device


def _get_mask_loaders(cfg: DictConfig, processor):
    benchmark = str(cfg.data.get("benchmark", "mllmu")).lower()
    if benchmark == "clear":
        from data.clear_dataset import get_clear_mask_data

        return get_clear_mask_data(cfg.data, processor)

    from data.multimodal import get_mm_mask_data

    return get_mm_mask_data(cfg.data, processor)


def _save_mask_artifact(mask: dict, root_dir: str, filename: str):
    os.makedirs(root_dir, exist_ok=True)
    mask_file = os.path.join(root_dir, filename)
    torch.save({"weight": mask}, mask_file)
    logger.info("Saved mask artifact to %s", mask_file)
    return mask_file


@hydra.main(version_base=None, config_path="../configs", config_name="mm_train.yaml")
def main(cfg: DictConfig):
    from model.multimodal import get_mm_model
    from trainer.unlearn.sfron import generate_saliency_mask

    device = _resolve_runtime_device(cfg)

    if bool(cfg.model.get("lora", {}).get("enabled", False)):
        raise ValueError(
            "Mask generation should run with `model.lora.enabled=false` so the "
            "mask keys match the trainable parameter space."
        )

    logger.info("Loading model for mask generation...")
    model, processor = get_mm_model(cfg.model)
    if device.startswith("cuda"):
        model = model.to(device)

    logger.info("Loading data...")
    mask_loaders = _get_mask_loaders(cfg, processor)

    mask_cfg = cfg.get("mask", {})
    threshold = float(mask_cfg.get("threshold", 1.0))
    save_path = str(
        mask_cfg.get(
            "save_path",
            os.path.join(
                "saves",
                "mm_mask",
                str(cfg.data.get("benchmark", "mllmu")).lower(),
                f"forget{int(cfg.data.get('forget_split_ratio', 5)):02d}",
            ),
        )
    )
    max_batches = mask_cfg.get("max_batches")
    max_batches = int(max_batches) if max_batches is not None else None

    vision_modules = list(
        mask_cfg.get("vision_modules", cfg.data.get("vision_mask_modules", []))
    )
    language_modules = list(
        mask_cfg.get("language_modules", cfg.data.get("language_mask_modules", []))
    )
    if not vision_modules or not language_modules:
        raise ValueError(
            "Mask generation requires both vision_modules and language_modules. "
            "Set them in `+mask=...` or the data config."
        )

    logger.info(
        "Generating vision mask: modules=%s, threshold=%.2f, max_batches=%s",
        vision_modules,
        threshold,
        max_batches,
    )
    vision_dir = os.path.join(save_path, "vision")
    vision_mask = generate_saliency_mask(
        model=model,
        forget_loader=mask_loaders["forget"],
        preserve_loader=mask_loaders["vision_preserve"],
        modules=vision_modules,
        threshold=threshold,
        save_path=vision_dir,
        max_batches=max_batches,
    )

    logger.info(
        "Generating language mask: modules=%s, threshold=%.2f, max_batches=%s",
        language_modules,
        threshold,
        max_batches,
    )
    language_dir = os.path.join(save_path, "language")
    language_mask = generate_saliency_mask(
        model=model,
        forget_loader=mask_loaders["forget"],
        preserve_loader=mask_loaders["language_preserve"],
        modules=language_modules,
        threshold=threshold,
        save_path=language_dir,
        max_batches=max_batches,
    )

    both_mask = dict(language_mask)
    both_mask.update(vision_mask)
    both_dir = os.path.join(save_path, "both")
    os.makedirs(both_dir, exist_ok=True)
    both_file = os.path.join(both_dir, "mask.pt")
    torch.save({"weight": both_mask}, both_file)
    logger.info("Saved merged mask artifact to %s", both_file)

    benchmark = str(cfg.data.get("benchmark", "mllmu")).lower()
    _save_mask_artifact(vision_mask, save_path, f"{benchmark}_vision_mask.pt")
    _save_mask_artifact(language_mask, save_path, f"{benchmark}_language_mask.pt")
    _save_mask_artifact(both_mask, save_path, f"{benchmark}_both_mask.pt")

    logger.info(
        "Done. vision=%d params, language=%d params, both=%d params",
        len(vision_mask),
        len(language_mask),
        len(both_mask),
    )


if __name__ == "__main__":
    main()
