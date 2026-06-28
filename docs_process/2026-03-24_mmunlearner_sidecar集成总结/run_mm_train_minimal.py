"""Run minimal multimodal unlearning training experiments for MMUnlearner sidecar.

This launcher intentionally imports torch before adding `src/` to sys.path.
In the current environment, launching `python src/mm_train.py ...` causes CUDA
to become unavailable. This file keeps the experiment reproducible.
"""

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
        "--output-dir",
        default="saves/mm_unlearn/test_ga_f10_bs1_acc8_launcher",
    )
    parser.add_argument(
        "--trainer",
        choices=["MMGradAscent", "MMGradDiff", "MMKLMin", "MMNPO", "MMUnlearner"],
        default="MMGradAscent",
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
        "--grad-accum",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
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
        "--oracle-model-path",
        default=None,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--grad-mask-path",
        default=None,
    )
    parser.add_argument(
        "--forget-alpha",
        type=float,
        default=1.0,
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
    from trainer.unlearn.mm_grad_ascent import MMGradAscent
    from trainer.unlearn.mm_grad_diff import MMGradDiff
    from trainer.unlearn.mm_kl_min import MMKLMin
    from trainer.unlearn.mm_npo import MMNPO
    from trainer.unlearn.mmunlearner import MMUnlearner

    trainer_map = {
        "MMGradAscent": MMGradAscent,
        "MMGradDiff": MMGradDiff,
        "MMKLMin": MMKLMin,
        "MMNPO": MMNPO,
        "MMUnlearner": MMUnlearner,
    }

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
    forget_loader, retain_loader = get_mm_data(data_cfg, processor)
    forget_loader = _limit_loader(forget_loader, args.max_forget_samples)
    retain_loader = _limit_loader(retain_loader, args.max_retain_samples)

    trainer_cls = trainer_map[args.trainer]
    trainer_kwargs = dict(
        model=model,
        processor=processor,
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        max_grad_norm=1.0,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=0,
        save_dir=args.output_dir,
    )

    if args.trainer == "MMNPO":
        oracle_path = args.oracle_model_path or args.model_path
        oracle_cfg = OmegaConf.create(
            {
                "model_family": args.model_family,
                "model_path": oracle_path,
                "processor_path": args.processor_path,
                "torch_dtype": "bfloat16",
                "lora": {"enabled": False},
            }
        )
        oracle_model, _ = get_mm_model(oracle_cfg)
        trainer_kwargs["oracle_model"] = oracle_model
        trainer_kwargs["beta"] = args.beta

    if args.trainer == "MMUnlearner":
        if not args.grad_mask_path:
            raise ValueError("MMUnlearner requires --grad-mask-path")
        trainer_kwargs["grad_mask_path"] = args.grad_mask_path
        trainer_kwargs["forget_alpha"] = args.forget_alpha

    trainer = trainer_cls(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
