# Task Plan: LoReFT and BREP Integration

## Goal
Integrate LoReFT and BREP into the existing `inject` pipeline of `open-unlearning`, with aligned trainer/data/eval interfaces and implementation records stored under `plan/`.

## Phases
- [x] Phase 1: Interface mapping and plan confirmation
- [x] Phase 2: BREP dataset taxonomy and target download layout
- [x] Phase 3: Generate scripts for directly downloadable BREP datasets
- [x] Phase 4: Implement LoReFT and BREP trainers/configs
- [x] Phase 5: Add smoke tests and evaluation wiring (minimal train-path scope)
- [ ] Phase 6: Review, document, and deliver

## Key Questions
1. How should non-full-model inject artifacts be loaded for evaluation?
2. Which existing inject interfaces can be reused directly vs. extended?
3. What is the smallest runnable LoReFT/BREP scope that preserves framework consistency?

## Decisions Made
- Reuse `mode: inject` for both LoReFT and BREP to minimize top-level API churn.
- First milestone prioritizes framework-runnable integration over full reproduction of the BREP repo workflows.
- Use a unified `inject_artifact` runtime contract to load PEFT, LoReFT, and BREP outputs on top of a base model.
- Before any data download, first record BREP dataset categories, file formats, and the target directory layout under `data/inject/`.
- Only generate scripts for directly downloadable datasets; keep `math10k` and `commonsense` out of the automation path and leave them for a dedicated reconstruction workflow.

## Errors Encountered
- `pyreft` is not installed in the local `.venv`; LoReFT runtime verification will require optional-dependency handling or later dependency installation.
- `prm800k` GitHub raw endpoint returned a Git LFS pointer file instead of the real JSONL payload; downloader was updated to use `media.githubusercontent.com`.
- Hugging Face datasets initially tried to use a non-writable `/netcache/...`; downloader now defaults to a local repository cache under `.cache/huggingface/`.
- `allenai/math_qa` requires `trust_remote_code=True`; the loader was updated accordingly.
- Hydra collator overrides flatten single-collator configs; `src/data/__init__.py` needed a compatibility path so `collator=DataCollatorForReFTDataset` works.
- `pyreft` 0.0.7 expects training batches to be passed as `base={...}` rather than `input_ids=...`; the LoReFT trainer had to call the wrapped model accordingly.
- `pyreft` 0.0.7 defaults `LoreftIntervention` parameters to `bfloat16`; on the local float32 tiny model this caused a dtype mismatch and was fixed by explicitly matching the base model dtype.
- `pyreft` 0.0.7 mutates config with non-JSON-serializable objects; smoke tests required `report_to=none` to avoid Trainer integration callbacks crashing on config serialization.
- `pyreft` 0.0.7 `print_trainable_parameters()` raises on the current intervention layout; the LoReFT trainer now treats that call as best-effort logging only.

## Status
**Currently after minimal runnable integration** - Direct-public BREP datasets have been downloaded and normalized into `data/inject/brep/`; `math10k` and `commonsense` remain excluded for later reconstruction. The `inject` pipeline now has minimal `LoReFT` and `BREP` trainers, both verified with 1-step local smoke tests on `/tmp/inject_smoke_model`.
