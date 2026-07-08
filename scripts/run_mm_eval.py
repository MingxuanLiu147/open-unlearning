"""CUDA-safe multimodal evaluation launcher for this host environment."""

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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--shot-num", default="zero_shot")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--splits", default="")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--generation-max-new-tokens", type=int, default=200)
    return parser.parse_args()


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _default_tasks(benchmark: str) -> list[str]:
    if benchmark == "clear":
        return ["classification", "generation"]
    return ["classification", "fill_in_the_blank", "generation"]


def _default_splits(benchmark: str) -> list[str]:
    if benchmark == "clear":
        return ["forget", "retain", "realface", "realworld"]
    return ["forget", "retain_shared", "retain_celebrity", "test"]


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )

    os.chdir(args.repo_root)
    sys.path.insert(0, "src")

    from model.multimodal import get_mm_model
    from evals.clear import CLEAREvaluator
    from evals.mllmu import MLLMUEvaluator

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
        raise RuntimeError("CUDA is unavailable in run_mm_eval.py launcher")

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
    model, processor = get_mm_model(model_cfg)
    model = model.to(args.device)
    model.eval()

    tasks = _parse_csv(args.tasks) or _default_tasks(args.benchmark)
    splits = _parse_csv(args.splits) or _default_splits(args.benchmark)
    max_records = args.max_records if args.max_records > 0 else None

    if args.benchmark == "clear":
        eval_cfg = OmegaConf.create(
            {
                "benchmark": "clear",
                "data_dir": "data/CLEAR",
                "forget_split_ratio": args.forget_ratio,
                "output_dir": args.output_dir,
                "tasks": tasks,
                "splits": splits,
                "max_new_tokens": args.max_new_tokens,
                "max_records": max_records,
            }
        )
        evaluator = CLEAREvaluator(eval_cfg, model, processor)
    else:
        eval_cfg = OmegaConf.create(
            {
                "benchmark": "mllmu",
                "data_dir": "data/MLLMU-Bench",
                "forget_split_ratio": args.forget_ratio,
                "output_dir": args.output_dir,
                "tasks": tasks,
                "splits": splits,
                "shot_num": args.shot_num,
                "max_new_tokens": args.max_new_tokens,
                "generation_max_new_tokens": args.generation_max_new_tokens,
                "test_data_dir": "data/MLLMU-Bench/Test_Set",
                "retain_celebrity_data": "data/MLLMU-Bench/Retain_Set/train-00000-of-00001.parquet",
                "few_shot_parquet": "data/MLLMU-Bench/Full_Set/train-00000-of-00001.parquet",
                "max_records": max_records,
            }
        )
        evaluator = MLLMUEvaluator(eval_cfg, model, processor)

    results = evaluator.evaluate()
    print(json.dumps({"output_dir": args.output_dir, "splits": list(results.keys())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
