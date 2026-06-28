"""SFRon 权重显著性掩码生成模块。

通过 Fisher 信息矩阵计算 forget/preserve 数据对各参数的敏感度，
生成 boolean 掩码标记"对 forget 数据敏感但对 preserve 数据不敏感"的参数位置。
"""

import logging
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def calc_sparsity(tensor):
    num_zero = tensor.numel() - torch.count_nonzero(tensor)
    total = tensor.numel()
    return num_zero.item() / total, total, num_zero.item()


def compute_fisher(model, loader, modules, save_path, tag="forget", max_batches: int | None = None):
    """计算指定模块上的 Fisher 信息（梯度平方的均值）。

    如果 save_path 下已存在缓存则直接加载。

    Args:
        model: 待计算的模型。
        loader: DataLoader。
        modules: 参与计算的模块名关键词列表，如 ["model", "visual"]。
        save_path: Fisher 缓存目录。
        tag: 文件名前缀，"forget" 或 "preserve"。

    Returns:
        dict[str, Tensor]: 参数名 → Fisher 值（CPU tensor）。
    """
    suffix = f"_{max_batches}b" if max_batches is not None else ""
    cache = os.path.join(save_path, f"{tag}{suffix}_fisher.pt")
    if os.path.exists(cache):
        logger.info("Loading cached %s fisher from %s", tag, cache)
        return torch.load(cache, map_location="cpu")

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0)
    gradients = {}
    model.train()

    for name, param in model.named_parameters():
        if any(m in name for m in modules):
            gradients[name] = None

    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    logger.info(
        "Computing %s fisher over %d batches for modules: %s",
        tag,
        total_batches,
        modules,
    )

    for step, batch in enumerate(tqdm(loader, desc=f"{tag} fisher")):
        if max_batches is not None and step >= max_batches:
            break
        device = next(model.parameters()).device
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        outputs = model(**batch)
        loss = -outputs.loss
        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in gradients and param.grad is not None:
                    fisher_val = param.grad.data.cpu() ** 2 / max(total_batches, 1)
                    if gradients[name] is not None:
                        gradients[name] += fisher_val
                    else:
                        gradients[name] = fisher_val

    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(save_path, exist_ok=True)
    torch.save(gradients, cache)
    logger.info("Saved %s fisher to %s", tag, cache)
    return gradients


def generate_saliency_mask(
    model,
    forget_loader: DataLoader,
    preserve_loader: DataLoader,
    modules: list,
    threshold: float = 1.0,
    save_path: str = "saves/mm_mask",
    max_batches: int | None = None,
):
    """生成权重显著性掩码。

    mask[name] = (forget_fisher / preserve_fisher) >= threshold

    Args:
        model: 模型实例。
        forget_loader: forget 数据 DataLoader。
        preserve_loader: preserve/retain 数据 DataLoader。
        modules: 模块名关键词，如 ["model"] (language) 或 ["visual"] (vision)。
        threshold: 显著性阈值。
        save_path: mask 和 fisher 缓存保存目录。

    Returns:
        dict[str, Tensor]: 参数名 → boolean mask。
    """
    forget_fisher = compute_fisher(
        model,
        forget_loader,
        modules,
        save_path,
        "forget",
        max_batches=max_batches,
    )
    preserve_fisher = compute_fisher(
        model,
        preserve_loader,
        modules,
        save_path,
        "preserve",
        max_batches=max_batches,
    )

    mask = {}
    total_cnt = 0
    masked_cnt = 0

    for name in forget_fisher:
        try:
            saliency = (forget_fisher[name] + 1e-15) / (preserve_fisher[name] + 1e-15)
            w = saliency >= threshold
            sparsity, total, zeros = calc_sparsity(w)
            total_cnt += total
            masked_cnt += zeros
            mask[name] = w
        except Exception:
            mask[name] = torch.zeros(1, dtype=torch.bool)

    if total_cnt > 0:
        logger.info(
            "Mask generated: sparsity=%.2f%% (threshold=%.2f, modules=%s)",
            masked_cnt / total_cnt * 100,
            threshold,
            modules,
        )

    mask_file = os.path.join(save_path, "mask.pt")
    torch.save({"weight": mask}, mask_file)
    logger.info("Mask saved to %s", mask_file)
    return mask
