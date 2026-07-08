"""CUDA-safe multimodal training launcher for this host environment."""

import argparse
import json
import logging
import os
import sys

import torch
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/home/liumingxuan/open-unlearning")
    parser.add_argument("--benchmark", choices=["mllmu", "clear"], default="mllmu")
    parser.add_argument(
        "--trainer",
        choices=[
            "MMGradAscent",
            "MMGradDiff",
            "MMKLMin",
            "MMNPO",
            "MMRetainFT",
            "MMUnlearner",
        ],
        default="MMGradAscent",
    )
    parser.add_argument("--model-family", default="qwen3vl")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--forget-ratio", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--train-mode", default="caption")
    parser.add_argument("--disable-lora", action="store_true")
    parser.add_argument("--oracle-model-path", default=None)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--grad-mask-path", default=None)
    parser.add_argument("--forget-alpha", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )

    os.chdir(args.repo_root)
    sys.path.insert(0, "src")

    from data.clear_dataset import get_clear_data
    from data.multimodal import get_mm_data
    from model.multimodal import get_mm_model
    from trainer.unlearn.mm_grad_ascent import MMGradAscent
    from trainer.unlearn.mm_grad_diff import MMGradDiff
    from trainer.unlearn.mm_kl_min import MMKLMin
    from trainer.unlearn.mm_npo import MMNPO
    from trainer.unlearn.mm_retain_ft import MMRetainFT
    from trainer.unlearn.mmunlearner import MMUnlearner

    trainer_map = {
        "MMGradAscent": MMGradAscent,
        "MMGradDiff": MMGradDiff,
        "MMKLMin": MMKLMin,
        "MMNPO": MMNPO,
        "MMRetainFT": MMRetainFT,
        "MMUnlearner": MMUnlearner,
    }

    print(
        json.dumps(
            {
                "launcher_cuda_available": torch.cuda.is_available(),
                "launcher_cuda_device_count": torch.cuda.device_count(),
            },
            ensure_ascii=False,
        )
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in run_mm_train.py launcher")
    if args.trainer == "MMUnlearner" and not args.disable_lora:
        raise ValueError("MMUnlearner launcher requires --disable-lora")

    processor_path = args.processor_path or args.model_path
    model_cfg = OmegaConf.create(
        {
            "model_family": args.model_family,
            "model_path": args.model_path,
            "processor_path": processor_path,
            "torch_dtype": "bfloat16",
            "lora": {
                "enabled": not args.disable_lora,
                "r": 16,
                "alpha": 16,
                "dropout": 0.05,
            },
        }
    )

    if args.benchmark == "clear":
        data_cfg = OmegaConf.create(
            {
                "benchmark": "clear",
                "data_dir": "data/CLEAR",
                "forget_split_ratio": args.forget_ratio,
                "batch_size": args.batch_size,
                "ans_only": False,
                "train_mode": args.train_mode,
                "mode": "reference" if args.trainer == "MMRetainFT" else "unlearn",
                "reference_source": "retain",
                "reference_train_mode": args.train_mode,
                "max_samples": args.max_samples if args.max_samples > 0 else None,
                "full_dir": "data/CLEAR/full",
                "full_tofu_dir": "data/CLEAR/full+tofu",
            }
        )
    else:
        data_cfg = OmegaConf.create(
            {
                "benchmark": "mllmu",
                "data_dir": "data/MLLMU-Bench",
                "forget_split_ratio": args.forget_ratio,
                "batch_size": args.batch_size,
                "ans_only": False,
                "mode": "reference" if args.trainer == "MMRetainFT" else "unlearn",
                "reference_source": "retain",
                "reference_text_only": False,
                "max_samples": args.max_samples if args.max_samples > 0 else None,
                "full_set_parquet": "data/MLLMU-Bench/Full_Set/train-00000-of-00001.parquet",
                "retain_set_parquet": "data/MLLMU-Bench/Retain_Set/train-00000-of-00001.parquet",
            }
        )

    model, processor = get_mm_model(model_cfg)
    model = model.to(args.device)

    if args.benchmark == "clear":
        forget_loader, retain_loader = get_clear_data(data_cfg, processor)
    else:
        forget_loader, retain_loader = get_mm_data(data_cfg, processor)

    trainer_cls = trainer_map[args.trainer]
    trainer_kwargs = dict(
        model=model,
        processor=processor,
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        save_dir=args.output_dir,
    )

    if args.trainer == "MMNPO":
        oracle_path = args.oracle_model_path or args.model_path
        oracle_cfg = OmegaConf.create(
            {
                "model_family": args.model_family,
                "model_path": oracle_path,
                "processor_path": processor_path,
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
    print(json.dumps({"output_dir": args.output_dir}, ensure_ascii=False))


if __name__ == "__main__":
    main()
