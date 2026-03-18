# Notes: LoReFT and BREP Integration

## Current Framework Facts
- `src/train.py` already routes `mode: inject` through `get_data()`, `get_collators()`, `load_trainer()`, and the standard `Trainer.train()` flow.
- `src/trainer/__init__.py` registers inject methods by trainer class name.
- `src/data/inject.py` already supports Alpaca, ShareGPT, and custom JSON/JSONL, and is the correct reuse point for the first milestone.
- `src/eval.py` currently reloads inject outputs as if they were full HF model checkpoints; this is insufficient for adapter/intervention style artifacts.

## Reference Repo Facts
- `code/BREP/Prefix/reft_train.py` uses `pyreft.ReftConfig`, `pyreft.get_reft_model`, and `pyreft.ReftTrainerForCausalLM`.
- `code/BREP/model.py` implements activation-level wrappers that replace target projections and save only `activation_*` parameters.
- `code/BREP/Prefix/prefix_train.py` adds weighted loss and PID-style bias control on top of activation-parameter training.

## BREP Dataset Taxonomy
- Training datasets referenced by BREP scripts:
  - `prm800k`
  - `math10k`
  - `commonsense`
  - `ultrafeedback`
- Evaluation datasets referenced by BREP scripts:
  - `gsm8k`
  - `hellaswag`
  - `svamp`
  - `mathqa`
  - `math500`
  - `amc23`
- Auxiliary analysis/probe datasets in the repo:
  - `TruthfulQA`
  - `Faithful`
  - `cal_raw`
  - `cal_4`

## Downloadability Status
- Direct public sources with clear upstream location:
  - `prm800k`
  - `ultrafeedback`
  - `gsm8k`
  - `hellaswag`
  - `svamp`
  - `mathqa`
  - `truthfulqa`
- Direct public source exists but license/source inheritance should be recorded explicitly:
  - `math500`
  - `amc23`
- Not a single direct public file in the BREP repo; requires reconstruction or manual confirmation:
  - `math10k`
  - `commonsense`
- Not directly downloadable benchmark data; generated or locally prepared:
  - `Faithful`
  - `cal_raw`
  - `cal_4`

## Implemented Public-Data Scripts
- `scripts/prepare_brep_public_data.py`
  - Downloads and normalizes the direct-public batch into `data/inject/brep/`.
  - Supports:
    - `prm800k`
    - `ultrafeedback`
    - `gsm8k`
    - `hellaswag`
    - `svamp`
    - `mathqa`
    - `math500`
    - `amc23`
    - `truthfulqa`
  - Explicitly skips:
    - `math10k`
    - `commonsense`
- `scripts/prepare_brep_public_data.sh`
  - Thin wrapper over the Python script with the default direct-public dataset batch.

## Local Verification
- Syntax check passed:
  - `PYTHONPATH=src .venv/bin/python -m py_compile scripts/prepare_brep_public_data.py`
- CLI check passed:
  - `.venv/bin/python scripts/prepare_brep_public_data.py --list-datasets`

## Download Results
- Download root:
  - `/home/liumingxuan/open-unlearning/data/inject/brep`
- Manifest:
  - `/home/liumingxuan/open-unlearning/data/inject/brep/manifest.json`
- Completed datasets:
  - `prm800k`: 93,929 rows
  - `ultrafeedback`: 63,958 rows
  - `gsm8k`: 1,319 rows
  - `hellaswag`: 10,042 rows
  - `svamp`: 300 rows
  - `mathqa`: 2,985 rows
  - `math500`: 500 rows
  - `amc23`: 40 rows
  - `truthfulqa`: 817 rows
- Still excluded from download automation:
  - `math10k`
  - `commonsense`

## BREP Training Data Format
- `Prefix/prefix_train.py` reads `dataset/<name>/train.json`.
- The training pipeline uses `instruction` and `output` as required fields.
- Some source datasets may also carry:
  - `input`
  - `answer`
  - `index`
  - `weight`
- For framework unification, the minimum normalized JSON schema should be:
  - `instruction`: string
  - `input`: string, optional, default `""`
  - `output`: string
  - `source_dataset`: string
  - `index`: integer, optional

## BREP Evaluation Data Format
- `Prefix/make_answer_json.py` reads `dataset/<benchmark>/test.json`.
- The benchmark JSON is expected to contain:
  - `instruction`
  - `output`
  - `answer`
  - `index` (added if missing)
- For the first download/normalization pass, benchmark files should keep that exact structure to stay compatible with the reference scripts.

## Planned Local Download Layout
- Target root: `/home/liumingxuan/open-unlearning/data/inject/brep/`
- Training set layout:
  - `train/prm800k/train.json`
  - `train/math10k/train.json`
  - `train/commonsense/train.json`
  - `train/ultrafeedback/train.json`
- Evaluation set layout:
  - `eval/gsm8k/test.json`
  - `eval/hellaswag/test.json`
  - `eval/svamp/test.json`
  - `eval/mathqa/test.json`
  - `eval/math500/test.json`
  - `eval/amc23/test.json`
- Auxiliary analysis layout:
  - `analysis/TruthfulQA/truthful_qa.jsonl`
  - `analysis/Faithful/faithful.jsonl`
  - `analysis/cal_raw/raw.jsonl`
  - `analysis/cal_4/4.jsonl`

## Chosen Integration Shape
- Add `LoReFTTrainer` under `src/trainer/inject/`.
- Add `BREPTrainer` and a local BREP wrapper module under `src/trainer/inject/`.
- Extend inject data/collator to optionally provide prompt-boundary metadata for ReFT.
- Introduce a base-model-plus-artifact loading contract for evaluation and inference reuse.

## Implemented Minimal Train-Path Integration
- `src/data/inject.py`
  - Adds optional `sample_weight` support via `weight` / `sample_weight`.
  - Returns `prompt_length`, `response_start`, and `train_target_span` for each tokenized sample.
- `src/data/collators.py`
  - Collates optional scalar metadata for inject training.
  - Adds `DataCollatorForReFTDataset`, which emits `unit_locations` for the last prompt token.
- `src/data/__init__.py`
  - Registers `DataCollatorForReFTDataset`.
  - Adds a flat-config compatibility path so `collator=DataCollatorForReFTDataset` works under Hydra.
- `src/trainer/inject/loreft.py`
  - Minimal LoReFT trainer using `pyreft.ReftConfig` and `pyreft.get_reft_model`.
  - Aligns intervention dtype with the base model dtype.
  - Calls the wrapped model with `base={input_ids, attention_mask}` and `unit_locations`.
  - Saves artifacts through `pyreft` native `save()` plus local `inject_artifact.json` metadata.
- `src/trainer/inject/brep.py`
  - Minimal activation-intervention wrapper for decoder-only projection layers.
  - Supports `ffn_down`, `ffn_up`, `attn_q`, `attn_k`, `attn_v`, and `attn_o`.
  - Saves `delta_vector.pth` plus local `inject_artifact.json` metadata.
- `src/trainer/inject/__init__.py` and `src/trainer/__init__.py`
  - Register `LoReFTTrainer` and `BREPTrainer`.
- New configs:
  - `configs/collator/DataCollatorForReFTDataset.yaml`
  - `configs/trainer/inject/LoReFT.yaml`
  - `configs/trainer/inject/BREP.yaml`
  - `configs/experiment/inject/alpaca/loreft.yaml`
  - `configs/experiment/inject/alpaca/brep.yaml`
- Dependency update:
  - `requirements.txt` now records `pyreft==0.0.7`.

## Minimal Smoke-Test Setup
- Tiny local model:
  - `/tmp/inject_smoke_model`
- Tiny local Alpaca-format dataset:
  - `/home/liumingxuan/open-unlearning/data/inject/smoke_alpaca.json`
- Both smoke tests were forced to:
  - CPU execution
  - `max_steps=1`
  - `report_to=none` for LoReFT because `pyreft` config objects are not JSON-serializable by the default Trainer integrations

## Smoke-Test Commands
- BREP:
  - `cd /home/liumingxuan/open-unlearning && CUDA_VISIBLE_DEVICES= HF_HOME=/home/liumingxuan/open-unlearning/.cache/huggingface PYTHONPATH=src .venv/bin/python src/train.py --config-name=inject.yaml trainer=inject/BREP data/datasets@data.train=Custom_inject model=Llama-3.2-1B-Instruct collator=DataCollatorForSupervisedDataset task_name=smoke_brep model.model_args.pretrained_model_name_or_path=/tmp/inject_smoke_model model.model_args.attn_implementation=eager model.model_args.torch_dtype=float32 model.tokenizer_args.pretrained_model_name_or_path=/tmp/inject_smoke_model data.train.train.Custom_train.args.data_path=/home/liumingxuan/open-unlearning/data/inject/smoke_alpaca.json data.train.train.Custom_train.args.format_type=alpaca +trainer.args.max_steps=1 trainer.args.save_steps=1 trainer.args.logging_steps=1 trainer.args.per_device_train_batch_size=1 trainer.args.gradient_accumulation_steps=1 trainer.args.learning_rate=1e-3 +trainer.args.overwrite_output_dir=true trainer.args.bf16=false paths.output_dir=/tmp/inject_smoke_brep`
- LoReFT:
  - `cd /home/liumingxuan/open-unlearning && CUDA_VISIBLE_DEVICES= HF_HOME=/home/liumingxuan/open-unlearning/.cache/huggingface PYTHONPATH=src .venv/bin/python src/train.py --config-name=inject.yaml trainer=inject/LoReFT data/datasets@data.train=Custom_inject model=Llama-3.2-1B-Instruct collator=DataCollatorForReFTDataset task_name=smoke_loreft model.model_args.pretrained_model_name_or_path=/tmp/inject_smoke_model model.model_args.attn_implementation=eager model.model_args.torch_dtype=float32 model.tokenizer_args.pretrained_model_name_or_path=/tmp/inject_smoke_model data.train.train.Custom_train.args.data_path=/home/liumingxuan/open-unlearning/data/inject/smoke_alpaca.json data.train.train.Custom_train.args.format_type=alpaca +trainer.args.max_steps=1 trainer.args.save_steps=1 trainer.args.logging_steps=1 trainer.args.per_device_train_batch_size=1 trainer.args.gradient_accumulation_steps=1 trainer.args.learning_rate=1e-3 +trainer.args.overwrite_output_dir=true trainer.args.bf16=false trainer.method_args.layer=0 +trainer.args.report_to=none paths.output_dir=/tmp/inject_smoke_loreft`

## Smoke-Test Results
- BREP:
  - Status: passed
  - Output dir: `/tmp/inject_smoke_brep`
  - Saved artifacts include:
    - `delta_vector.pth`
    - `brep_config.json`
    - `inject_artifact.json`
    - checkpoint copies under `checkpoint-1/`
- LoReFT:
  - Status: passed
  - Output dir: `/tmp/inject_smoke_loreft`
  - Saved artifacts include:
    - `intkey_layer_0_comp_block_output_unit_pos_nunit_1#0.bin`
    - `config.json` from `pyreft.save()`
    - `loreft_config.json`
    - `inject_artifact.json`
    - checkpoint copies under `checkpoint-1/`

## Current Gaps After Minimal Integration
- Eval-time artifact reload is still not unified under the planned base-model-plus-artifact contract.
- LoReFT currently needs `report_to=none` in practice because of `pyreft` config serialization.
- `pyreft` still emits a non-blocking warning about missing `nnsight`; this did not block local smoke tests.

## Risks
- `pyreft` API is not available locally, so LoReFT will need defensive optional imports and may only be syntax-verified unless the dependency is installed.
- BREP wrapper code must remain model-family aware without breaking existing plain HF loading.
- The current inject evaluation config needs a compatible migration path for existing LoRA outputs.
- The BREP repo does not include direct download scripts for all datasets; some datasets may need manual source mapping before any actual fetch step.
