"""Run MLLMU evaluation for an existing multimodal checkpoint.

This launcher intentionally imports torch before adding `src/` to sys.path.
In the current environment, launching `python src/mm_eval.py ...` causes CUDA
to become unavailable. This file keeps the experiment reproducible.
"""

import argparse
import json
import logging
import os
import sys

import torch
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/home/liumingxuan/open-unlearning",
    )
    parser.add_argument(
        "--model-path",
        default="saves/mm_unlearn/test_ga_f10",
    )
    parser.add_argument(
        "--processor-path",
        default="saves/mm_unlearn/test_ga_f10",
    )
    parser.add_argument(
        "--forget-ratio",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output-dir",
        default="saves/mm_eval/mllmu_test_ga_f10_cls_f10_gpu_launcher",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--split",
        choices=["forget", "retain", "both"],
        default="forget",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )

    os.chdir(args.repo_root)
    sys.path.insert(0, "src")

    from model.multimodal import get_mm_model
    from evals.mllmu import MLLMUEvaluator, evaluate_classification

    model_cfg = OmegaConf.create(
        {
            "model_family": "qwen3vl",
            "model_path": args.model_path,
            "processor_path": args.processor_path,
            "torch_dtype": "bfloat16",
            "lora": {"enabled": False},
        }
    )
    eval_cfg = OmegaConf.create(
        {
            "benchmark": "mllmu",
            "data_dir": "data/MLLMU-Bench",
            "forget_split_ratio": args.forget_ratio,
            "output_dir": args.output_dir,
            "tasks": ["classification"],
            "max_new_tokens": args.max_new_tokens,
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
    model.eval()

    if args.split == "both":
        evaluator = MLLMUEvaluator(eval_cfg, model, processor)
        results = evaluator.evaluate()
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        split_name = args.split
        folder_name = f"{split_name}_{args.forget_ratio}" if split_name == "forget" else f"{split_name}_{100 - args.forget_ratio}"
        parquet = os.path.join("data/MLLMU-Bench", folder_name, "train-00000-of-00001.parquet")
        results = {
            split_name: {
                "classification": evaluate_classification(
                    parquet,
                    processor,
                    model,
                    max_new_tokens=args.max_new_tokens,
                )
            }
        }
        with open(os.path.join(args.output_dir, "MLLMU_EVAL.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
