# Inject Training Modifications

This file records the changes made to the `inject` fine-tuning path.

## Completed

- Updated `[src/data/inject.py](/home/liumingxuan/open-unlearning/src/data/inject.py)`:
  - Alpaca and ShareGPT samples now fail fast when required fields are missing or empty.
  - Chat templates are used when `template_args.apply_chat_template` is enabled.
  - Dataset tokenization no longer pads to `max_length`; dynamic padding is left to the collator.
- Updated `[configs/trainer/inject/base_inject.yaml](/home/liumingxuan/open-unlearning/configs/trainer/inject/base_inject.yaml)`:
  - `output_dir` now uses `${paths.output_dir}` so inject outputs follow `task_name`.
  - `eval_strategy` now uses the current `transformers` argument name and defaults to `'no'` so inject runs are not blocked by a missing `eval_dataset`.
- Added `[configs/experiment/inject/alpaca/adalora.yaml](/home/liumingxuan/open-unlearning/configs/experiment/inject/alpaca/adalora.yaml)`:
  - Provides a ready-to-run AdaLoRA experiment alongside the existing LoRA/DoRA variants.
- Updated shared comments in `[src/data/__init__.py](/home/liumingxuan/open-unlearning/src/data/__init__.py)`, `[src/train.py](/home/liumingxuan/open-unlearning/src/train.py)`, and `[configs/inject.yaml](/home/liumingxuan/open-unlearning/configs/inject.yaml)`:
  - Clarifies that `inject` and `finetune` reuse the standard split-based data loading path.
- Updated inject trainers in `[src/trainer/inject/base.py](/home/liumingxuan/open-unlearning/src/trainer/inject/base.py)`, `[src/trainer/inject/lora.py](/home/liumingxuan/open-unlearning/src/trainer/inject/lora.py)`, `[src/trainer/inject/dora.py](/home/liumingxuan/open-unlearning/src/trainer/inject/dora.py)`, and `[src/trainer/inject/adalora.py](/home/liumingxuan/open-unlearning/src/trainer/inject/adalora.py)`:
  - `compute_loss()` now accepts the extra kwargs passed by the current `transformers` Trainer.
  - Hydra `ListConfig` values are converted to plain Python lists before PEFT save.
  - AdaLoRA now infers `total_step` and handles the current PEFT parameter container layout in orthogonal regularization.

## Verification

```bash
cd /home/liumingxuan/open-unlearning
PYTHONPATH=src .venv/bin/python -m py_compile src/data/inject.py src/data/__init__.py src/train.py
PYTHONPATH=src .venv/bin/python -m py_compile src/trainer/inject/base.py src/trainer/inject/lora.py src/trainer/inject/dora.py src/trainer/inject/adalora.py
```

Smoke tests completed on CPU with a tiny local model and local Alpaca-format dataset:

- LoRA: passed, adapter saved to `/tmp/inject_smoke_lora`
- DoRA: passed, adapter saved to `/tmp/inject_smoke_dora`
- AdaLoRA: passed, adapter saved to `/tmp/inject_smoke_adalora`

## Notes

- The repository already had unrelated local modifications in several files. This change set was kept scoped to the inject path and related comments/configs.
- Inject saves PEFT adapters, not merged full-model weights.
