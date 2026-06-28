# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A heavily extended fork of [locuslab/open-unlearning](https://github.com/locuslab/open-unlearning) (internally "Know-Surgery"). The upstream base is a **Hydra-config + registry-pattern** framework for benchmarking **LLM unlearning** (TOFU / MUSE / WMDP). This fork adds three more "knowledge surgery" operations plus multimodal support and a web UI:

- **Unlearn** — remove knowledge (GradAscent, GradDiff, NPO, SimNPO, DPO, RMU, UNDIAL, CEU, SatImp, WGA, PDU, SGA, FLAT, SOUL) — `src/trainer/unlearn/`
- **Inject** — add knowledge via PEFT (LoRA, DoRA, AdaLoRA, LoReFT, BREP) — `src/trainer/inject/`
- **Edit** — surgical knowledge editing (ROME, MEMIT, MEND, SERAC, GRACE, WISE, AlphaEdit, UNKE, IKE, MALMEN, InstructEdit, AnyEdit, UniKE, NMKE, MEMIT-Merge) — `src/trainer/edit/`
- **Multimodal** unlearning + editing for VLMs (Qwen-VL, LLaVA, BLIP-2, InternVL) — the `mm_*` trainers/data/evals
- **Web UI** (Flask + Vue 3) under `new_ui/`, and a LaTeX paper under `paper/`

`docs/` holds the canonical English documentation (`contributing.md`, `evaluation.md`, `experiments.md`, `hydra.md`, plus `EXTENSION_GUIDE.md`, `multimodal_editing_guide.md`, `inject_*.md`, `edit_train_recommendations.md`).

## Entry points & modes (important)

`src/train.py` is the **single entry for finetune / unlearn / edit / inject** — the operation is selected by the top-level config passed via `--config-name`, which sets `cfg.mode`. The trainer is chosen by `cfg.trainer.handler` matching a class `__name__` in the registry. Multimodal is a deliberate **sidecar** with its own entry and its own registry.

| Task | Command entry | `--config-name` | `mode` | Trainer registry |
|------|---------------|-----------------|--------|------------------|
| Finetune | `src/train.py` | `train.yaml` | `train` | `TRAINER_REGISTRY` (`src/trainer/__init__.py`) |
| Unlearn | `src/train.py` | `unlearn.yaml` | `unlearn` | `TRAINER_REGISTRY` |
| Edit | `src/train.py` | `edit.yaml` | `edit` | `TRAINER_REGISTRY` |
| Inject | `src/train.py` | `inject.yaml` | `inject` | `TRAINER_REGISTRY` |
| Eval | `src/eval.py` | `eval.yaml` | `eval` | `EVALUATOR_REGISTRY` (`src/evals/__init__.py`) |
| MM train/unlearn | `src/mm_train.py` | `mm_train.yaml` | — | `MM_TRAINER_REGISTRY` (a dict literal in `src/mm_train.py`, **not** the decorator registry) |
| MM eval | `src/mm_eval.py` | `mm_eval.yaml` | — | — |

Notes:
- `mode` drives data loading in `get_data(...)` (e.g. `unlearn` combines forget+retain into a `ForgetRetainDataset`).
- `mode=edit` takes a **special branch** in `src/train.py` (`build_edit_requests` → `execute_edit_requests` → `save_edit_artifacts`) instead of `trainer.train()`. Edits are applied as requests; evaluate the edited checkpoint separately with `src/eval.py`.
- The multimodal path is independent on purpose: separate registry, separate data loaders (`get_mm_data` / `get_clear_data` / `get_fiubench_data`), separate model loader (`src/model/multimodal.py:get_mm_model`).

## Common commands

```bash
# Setup (Python 3.11)
pip install .[lm_eval]
pip install --no-build-isolation flash-attn==2.6.3   # optional
git submodule update --init                            # pulls code/BREP, code/MIP-Editor, code/MMUnlearner
python setup_data.py --eval                            # downloads eval logs into saves/eval/

# Lint / format (ruff; scope is only: scripts src setup.py setup_data.py)
make quality        # check
make style          # auto-fix

# Tests (CPU-forced)
make test                                              # == CUDA_VISIBLE_DEVICES= pytest tests/
CUDA_VISIBLE_DEVICES= pytest tests/test_edit_pipeline.py            # single file
CUDA_VISIBLE_DEVICES= pytest tests/test_edit_pipeline.py::test_xxx  # single test
```

Run examples (all four text operations go through `src/train.py`):

```bash
# Unlearn
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 trainer=GradAscent task_name=SAMPLE

# Edit
python src/train.py --config-name=edit.yaml trainer=edit/ROME task_name=SAMPLE_EDIT

# Inject
python src/train.py --config-name=inject.yaml trainer=inject/LoRA task_name=SAMPLE_INJECT

# Eval
python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
  model=Llama-3.2-1B-Instruct task_name=SAMPLE_EVAL
```

## Adding a component (registry pattern)

The handler string in a config **must equal the Python class `__name__`**. To add, e.g., a trainer:
1. Implement the class under `src/trainer/{unlearn,inject,edit}/`.
2. Import it and call `_register_trainer(MyClass)` in `src/trainer/__init__.py`.
3. Add a `configs/trainer/...yaml` whose `handler:` is the class name.

Same pattern for evaluators (`src/evals/__init__.py`), datasets (`src/data/__init__.py`), and models (`src/model/__init__.py`). **Multimodal trainers are different**: register them in the `MM_TRAINER_REGISTRY` dict in `src/mm_train.py` (dotted import path), not via the decorator/registry above.

## Multimodal: CUDA caveat (environment-specific)

Running `python src/mm_train.py` / `src/mm_eval.py` **directly can lose CUDA** on this host. `src/mm_train.py` imports `torch` before Hydra and checks device availability; when `require_cuda=true` and CUDA is missing it raises with a hint. Avoid `python -u`, stdout/stderr redirection, `CUDA_VISIBLE_DEVICES=*`, and `PYTORCH_ALLOC_CONF=expandable_segments:True` for these scripts. Prefer the CUDA-safe launchers: `scripts/run_mm_train.py`, `scripts/run_mm_eval.py`, `scripts/run_mm_generate_mask.py`.

## Web UI (`new_ui/`)

Flask backend + Vue 3 / Vite frontend (replaces the old Gradio UI). Backend registers blueprints under `new_ui/backend/api/` (config / runner / results / skills / agent / data) and shares the same `configs/` and `saves/` as the CLI.

```bash
pip install flask flask-cors pyyaml openai
# Backend
python -m new_ui.backend.app --port 18888
# Frontend (dev; proxies /api -> backend)
cd new_ui/frontend && npm install && npm run dev    # http://localhost:3000
# Production: npm run build  (Flask then serves new_ui/frontend/dist/)
```

The Copilot agent panel uses an OpenAI-compatible API (`OPENAI_API_KEY` / `OPENAI_API_BASE` / `OPENAI_MODEL`, or `DEEPSEEK_API_KEY`).

## Submodules

`code/` vendors three research repos as git submodules: `code/BREP` (wrapped by `src/trainer/inject/brep.py`), `code/MIP-Editor`, and `code/MMUnlearner`. Run `git submodule update --init` after cloning.

## Conventions & gotchas

- `community/methods/*` are **documentation + `run.sh` wrappers** that call `src/train.py` — not implementations. The actual algorithms live in `src/trainer/unlearn/`.
- Authoritative docs are in `docs/`. The repo root also contains many working notes (`*_survey.md`, `UNLEARNING_GUIDE_CN.md`, `代码注释总结.md`, `ARCHITECTURE.md`, etc.) and `docs_process/` holds dated experiment run logs — treat these as scratch/history, not API reference.
- `AGENTS.md` documents only the upstream base (it predates the edit/inject/multimodal/UI extensions).
- Most in-code comments and many docstrings are in Chinese; match the surrounding style when editing.
- `paper/` is a self-contained LaTeX project (sections in `paper/sections/`) with its own `uv` venv.
