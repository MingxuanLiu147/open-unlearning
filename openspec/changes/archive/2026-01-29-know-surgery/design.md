## Context

OpenUnlearning already provides a Hydra + registry-based architecture for unlearning benchmarks (train.py/eval.py, model/trainer/data/evals registries). The goal is to extend this into Know-Surgery with minimal changes: keep the existing CLI/config flow, add a thin Python pipeline wrapper, and plan phased expansion (editing, injection/PEFT, multimodal, GUI). v1 focuses on TOFU with Llama-3.2-1B Instruct and Qwen3-1.7B Base and stable runs on 8x3090. GUI and UltraRAG-style one-click orchestration are required but intentionally deferred to end of v2.

## Goals / Non-Goals

**Goals:**
- Preserve the existing registry + Hydra composition model; add new capabilities as additive modules.
- Provide a minimal Python pipeline wrapper that orchestrates data -> unlearn -> eval without changing core training/eval paths.
- Standardize evaluation artifacts (manifest/metrics/traces) and allow missing retain logs to degrade gracefully.
- Define extensibility boundaries for future knowledge editing, injection (LoRA/DoRA/REFT), and multimodal support.
- Keep GUI requirements explicit and aligned with CLI configs (same schema, import/export compatibility), with implementation starting at end of v2.

**Non-Goals:**
- Replacing or re-architecting the existing OpenUnlearning training/eval entrypoints.
- Auto-downloading datasets in v1 (data is prepared manually).
- Implementing GUI, editing, injection, or multimodal functionality in v1.

## Decisions

- **Keep Hydra as the single source of configuration truth.**
  - Rationale: The repo is already Hydra-driven and extensible. A wrapper can generate/consume configs without changing core logic.
  - Alternatives: Build a new config schema or replace Hydra (rejected due to disruption).

- **Introduce a minimal Python pipeline wrapper (not a new training framework).**
  - Rationale: Provides one-click flow while keeping changes minimal and backward compatible.
  - Alternatives: Bash scripts only (harder to extend for GUI), full new pipeline engine (too heavy for v1).

- **Model-first v1 scope with fixed small models (Llama-3.2-1B Instruct, Qwen3-1.7B Base).**
  - Rationale: Ensures stable runs on 8x3090 and reproducibility with minimal resource assumptions.
  - Alternatives: Large models or API models (rejected for v1 due to hardware and extensibility constraints).

- **Capability registry metadata for compatibility checks.**
  - Rationale: Enables safe plugin growth and GUI validation without refactoring registries.
  - Alternatives: Hard-coded compatibility rules (not scalable).

- **GUI deferred to end of v2; CLI remains primary in v1.**
  - Rationale: v1 must stabilize unlearning; GUI depends on config schema maturity.
  - Alternatives: Early GUI (risking rework as pipeline stabilizes).

## Risks / Trade-offs

- **Risk:** Retain-log dependent metrics can block evaluation.
  → **Mitigation:** Provide graceful degradation and flag missing metrics explicitly.
- **Risk:** Adding pipeline wrapper could duplicate logic.
  → **Mitigation:** Wrapper strictly orchestrates existing train/eval entrypoints without logic duplication.
- **Risk:** Future GUI requirements may pressure config changes.
  → **Mitigation:** Define a stable config schema early and use import/export compatibility tests.
- **Risk:** Extending to editing/injection/multimodal may require new data abstractions.
  → **Mitigation:** Reserve schema hooks in data/model registry metadata and plan phased specs.

## Migration Plan

- No breaking changes in v1. Existing training and evaluation scripts remain intact.
- New configs and wrapper are additive. Users can continue using current CLI commands.
- GUI introduced later as an optional layer consuming the same configs.

## Open Questions

- How should capability metadata be stored (YAML next to configs vs. Python registry annotations)?
- What is the minimal schema for evaluation artifacts to support both CLI and future GUI views?
- Which benchmark should be the first to extend beyond TOFU (MUSE vs. WMDP)?
- Should the pipeline wrapper support batch experiment queues in v1 or defer to v2?
