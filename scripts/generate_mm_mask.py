"""SFRon 掩码生成独立入口。

使用 Hydra 配置加载模型和数据，计算 Fisher 信息矩阵，生成权重显著性掩码。

使用示例：
    python scripts/generate_mm_mask.py \
        model=Qwen2VL-2B \
        data=mllmu \
        mask.modules='["model"]' \
        mask.threshold=1.0 \
        mask.save_path=saves/mm_mask/language
"""

import sys
sys.path.insert(0, "src")

import logging
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="mm_train.yaml")
def main(cfg: DictConfig):
    from model.multimodal import get_mm_model
    from data.multimodal import get_mm_data
    from trainer.unlearn.sfron import generate_saliency_mask

    logger.info("Loading model for mask generation...")
    model, processor = get_mm_model(cfg.model)

    logger.info("Loading data...")
    forget_loader, retain_loader = get_mm_data(cfg.data, processor)

    if retain_loader is None:
        raise ValueError("Mask generation requires retain/preserve data")

    mask_cfg = cfg.get("mask", {})
    modules = list(mask_cfg.get("modules", ["model"]))
    threshold = float(mask_cfg.get("threshold", 1.0))
    save_path = str(mask_cfg.get("save_path", "saves/mm_mask"))

    logger.info("Generating mask: modules=%s, threshold=%.2f", modules, threshold)
    mask = generate_saliency_mask(
        model=model,
        forget_loader=forget_loader,
        preserve_loader=retain_loader,
        modules=modules,
        threshold=threshold,
        save_path=save_path,
    )
    logger.info("Done. %d parameter masks generated.", len(mask))


if __name__ == "__main__":
    main()
