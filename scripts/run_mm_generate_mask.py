"""CUDA-safe multimodal mask launcher for this host environment."""

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
    parser.add_argument("--model-family", default="qwen3vl")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--forget-ratio", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mask-max-samples", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--save-path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )

    os.chdir(args.repo_root)
    sys.path.insert(0, "src")

    from data.clear_dataset import get_clear_mask_data
    from data.multimodal import get_mm_mask_data
    from model.multimodal import get_mm_model
    from trainer.unlearn.sfron import generate_saliency_mask

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
        raise RuntimeError("CUDA is unavailable in run_mm_generate_mask.py launcher")

    processor_path = args.processor_path or args.model_path
    model_cfg = OmegaConf.create(
        {
            "model_family": args.model_family,
            "model_path": args.model_path,
            "processor_path": processor_path,
            "torch_dtype": "bfloat16",
            "lora": {"enabled": False},
        }
    )

    if args.benchmark == "clear":
        data_cfg = OmegaConf.create(
            {
                "benchmark": "clear",
                "data_dir": "data/CLEAR",
                "forget_split_ratio": args.forget_ratio,
                "batch_size": args.batch_size,
                "mask_batch_size": args.batch_size,
                "mask_ans_only": False,
                "mask_max_samples": args.mask_max_samples if args.mask_max_samples > 0 else None,
                "full_dir": "data/CLEAR/full",
                "full_tofu_dir": "data/CLEAR/full+tofu",
                "vision_mask_modules": ["visual"],
                "language_mask_modules": ["model"],
            }
        )
    else:
        data_cfg = OmegaConf.create(
            {
                "benchmark": "mllmu",
                "data_dir": "data/MLLMU-Bench",
                "forget_split_ratio": args.forget_ratio,
                "batch_size": args.batch_size,
                "mask_batch_size": args.batch_size,
                "mask_ans_only": True,
                "mask_max_samples": args.mask_max_samples if args.mask_max_samples > 0 else None,
                "full_set_parquet": "data/MLLMU-Bench/Full_Set/train-00000-of-00001.parquet",
                "retain_set_parquet": "data/MLLMU-Bench/Retain_Set/train-00000-of-00001.parquet",
                "vision_mask_modules": ["visual"],
                "language_mask_modules": ["model"],
            }
        )

    model, processor = get_mm_model(model_cfg)
    model = model.to(args.device)

    if args.benchmark == "clear":
        loaders = get_clear_mask_data(data_cfg, processor)
    else:
        loaders = get_mm_mask_data(data_cfg, processor)

    max_batches = args.max_batches if args.max_batches > 0 else None
    os.makedirs(args.save_path, exist_ok=True)
    vision_dir = os.path.join(args.save_path, "vision")
    language_dir = os.path.join(args.save_path, "language")
    both_dir = os.path.join(args.save_path, "both")

    vision_mask = generate_saliency_mask(
        model=model,
        forget_loader=loaders["forget"],
        preserve_loader=loaders["vision_preserve"],
        modules=list(data_cfg.vision_mask_modules),
        threshold=args.threshold,
        save_path=vision_dir,
        max_batches=max_batches,
    )
    language_mask = generate_saliency_mask(
        model=model,
        forget_loader=loaders["forget"],
        preserve_loader=loaders["language_preserve"],
        modules=list(data_cfg.language_mask_modules),
        threshold=args.threshold,
        save_path=language_dir,
        max_batches=max_batches,
    )

    both_mask = dict(language_mask)
    both_mask.update(vision_mask)
    os.makedirs(both_dir, exist_ok=True)
    torch.save({"weight": both_mask}, os.path.join(both_dir, "mask.pt"))
    torch.save({"weight": vision_mask}, os.path.join(args.save_path, f"{args.benchmark}_vision_mask.pt"))
    torch.save({"weight": language_mask}, os.path.join(args.save_path, f"{args.benchmark}_language_mask.pt"))
    torch.save({"weight": both_mask}, os.path.join(args.save_path, f"{args.benchmark}_both_mask.pt"))
    print(
        json.dumps(
            {
                "save_path": args.save_path,
                "vision_entries": len(vision_mask),
                "language_entries": len(language_mask),
                "both_entries": len(both_mask),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
