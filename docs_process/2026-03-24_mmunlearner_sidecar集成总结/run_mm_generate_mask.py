"""Generate a multimodal SFRon mask with a CUDA-safe launcher."""

import argparse
import json
import logging
import os
import sys

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/home/liumingxuan/open-unlearning",
    )
    parser.add_argument(
        "--model-family",
        default="qwen3vl",
    )
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-VL-4B-Instruct",
    )
    parser.add_argument(
        "--processor-path",
        default="Qwen/Qwen3-VL-4B-Instruct",
    )
    parser.add_argument(
        "--disable-lora",
        action="store_true",
    )
    parser.add_argument(
        "--forget-ratio",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-forget-samples",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-retain-samples",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--modules",
        default="model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--save-path",
        default="saves/mm_mask/default",
    )
    return parser.parse_args()


def _limit_loader(loader: DataLoader | None, max_samples: int) -> DataLoader | None:
    if loader is None or max_samples <= 0:
        return loader
    if max_samples >= len(loader.dataset):
        return loader
    subset = Subset(loader.dataset, range(max_samples))
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        collate_fn=loader.collate_fn,
        num_workers=0,
    )


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )

    os.chdir(args.repo_root)
    sys.path.insert(0, "src")

    from model.multimodal import get_mm_model
    from data.multimodal import get_mm_data
    from trainer.unlearn.sfron import generate_saliency_mask

    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    model_cfg = OmegaConf.create(
        {
            "model_family": args.model_family,
            "model_path": args.model_path,
            "processor_path": args.processor_path,
            "torch_dtype": "bfloat16",
            "lora": {
                "enabled": not args.disable_lora,
                "r": 16,
                "alpha": 16,
                "dropout": 0.05,
            },
        }
    )
    data_cfg = OmegaConf.create(
        {
            "data_dir": "data/MLLMU-Bench",
            "forget_split_ratio": args.forget_ratio,
            "batch_size": args.batch_size,
            "ans_only": False,
            "full_set_parquet": "data/MLLMU-Bench/Full_Set/train-00000-of-00001.parquet",
        }
    )

    print(
        json.dumps(
            {
                "launcher_cuda_available": torch.cuda.is_available(),
                "launcher_cuda_device_count": torch.cuda.device_count(),
            },
            ensure_ascii=False,
        )
    )

    model, processor = get_mm_model(model_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    forget_loader, retain_loader = get_mm_data(data_cfg, processor)
    forget_loader = _limit_loader(forget_loader, args.max_forget_samples)
    retain_loader = _limit_loader(retain_loader, args.max_retain_samples)

    if retain_loader is None:
        raise ValueError("Mask generation requires retain loader")

    mask = generate_saliency_mask(
        model=model,
        forget_loader=forget_loader,
        preserve_loader=retain_loader,
        modules=modules,
        threshold=args.threshold,
        save_path=args.save_path,
    )
    print(
        json.dumps(
            {
                "mask_path": os.path.join(args.save_path, "mask.pt"),
                "mask_entries": len(mask),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
