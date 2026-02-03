## Why

OpenUnlearning provides strong unlearning benchmarks, but it does not yet offer a unified, extensible toolkit for controlled add/delete/edit knowledge updates with reproducible pipelines, future multimodal support, and a user-facing orchestration experience. Know-Surgery fills this gap now while keeping changes minimal and compatible with the existing repo.

## What Changes

- Define Know-Surgery as an umbrella toolkit on top of OpenUnlearning with phased scope: v1 unlearning-only, v2 knowledge editing + initial GUI, v2.5 knowledge injection/PEFT, v3 multimodal.
- Add a minimal Python pipeline wrapper to run one-click unlearning (data -> unlearn -> eval) using existing Hydra configs; no auto-download of data.
- Add model configs for Llama-3.2-1B Instruct and Qwen3-1.7B Base; provide two default TOFU demo split configurations (splits are not fixed).
- Introduce extensible registry metadata and compatibility checks for models, methods, and datasets to support user-defined extensions and future API adapters (e.g., GPT).
- Standardize evaluation artifacts (manifest, metrics, sample traces) and allow graceful degradation when retain logs are missing.
- Specify GUI requirements (Gradio) and an UltraRAG-style wizard for one-click configuration and execution; GUI starts at end of v2 and shares the same config schema as CLI.
- Establish scope for unlearning/editing/injection/multimodal methods, datasets, and metrics as listed in the requirements (TOFU/MUSE/WMDP/RWKU, ZSRE/CounterFact/ELKEN/ConceptEdit/long-text, LoRA/DoRA/REFT, LLaVA/Qwen-VL, etc.).

## Capabilities

### New Capabilities
- `unlearning-pipeline-v1`: Config-driven unlearning pipeline with one-click Python runner, TOFU first, default configs for Llama-3.2-1B Instruct and Qwen3-1.7B Base, and flexible TOFU splits.
- `extensible-registry`: Plugin metadata and compatibility checks for models/methods/datasets; user-extensible and future API adapter ready.
- `evaluation-artifacts`: Standardized evaluation outputs (manifest/metrics/traces) and missing-retain-log handling.
- `knowledge-editing-pipeline`: Editing workflows for single/sequential/batch tasks and editing metrics (reliability/locality/generalization/portability).
- `knowledge-injection-peft`: Knowledge injection workflows using LoRA/DoRA/AdaLoRA/LoReFT/BREP-REFT.
- `multimodal-support`: Multimodal data schema, VLM adapters, and multimodal unlearning/editing benchmarks.
- `gui-orchestrator`: Gradio-based GUI with UltraRAG-style wizard, config export/import, run management, and result visualization (starting v2 end).

### Modified Capabilities
- None.

## Impact

- New configs under `configs/model/`, `configs/experiment/`, and new pipeline wrapper entrypoint in `src/` (minimal change to existing training/eval paths).
- New dependencies planned for later phases (PEFT for v2.5, Gradio for v2 GUI); no v1 breaking changes.
- Additional documentation for CLI/GUI workflows, extension guides, and reproducibility guidance.
- Evaluation artifacts and reporting conventions standardized across unlearning/editing/injection/multimodal.
