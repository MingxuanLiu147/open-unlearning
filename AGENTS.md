# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the core Python code with entrypoints in `src/train.py` and `src/eval.py`, plus subpackages like `data/`, `model/`, `trainer/`, and `evals/`.
- `configs/` contains the Hydra configuration tree (`experiment/`, `trainer/`, `data/`, `model/`, `eval/`, etc.).
- `scripts/` provides baseline experiment scripts (for example, `scripts/tofu_unlearn.sh`, `scripts/muse_unlearn.sh`).
- `docs/` and `community/` host extended documentation and benchmark/leaderboard material.
- `data/` and `saves/` are used for datasets, logs, and run artifacts; `assets/` stores figures.

## Build, Test, and Development Commands
- `pip install .[lm_eval]` for runtime deps; `pip install .[dev]` for dev tools (ruff, pre-commit). README uses Python 3.11.
- `pip install --no-build-isolation flash-attn==2.6.3` for optional attention acceleration.
- `python setup_data.py --eval` downloads evaluation logs into `saves/eval/`.
- `python src/train.py --config-name=unlearn.yaml ...` runs unlearning; `python src/eval.py --config-name=eval.yaml ...` runs evaluation.
- `bash scripts/tofu_unlearn.sh` and `bash scripts/muse_unlearn.sh` run baseline experiments.
- `make quality` (ruff lint + format check), `make style` (auto-fix), `make test` (pytest).

## Coding Style & Naming Conventions
- Python formatting and linting are enforced with Ruff (see `make quality` or `pre-commit run --all-files`).
- Keep new Hydra configs under `configs/<area>/...` and reference them via overrides like `experiment=...`, `trainer=...`.
- Add informative docstrings for new methods (per the PR checklist).

## Testing Guidelines
- Test runner: pytest via `make test`.
- There is no tracked `tests/` directory yet; add tests under `tests/` using `test_*.py` names when introducing new behavior.

## Commit & Pull Request Guidelines
- Recent history mixes conventional prefixes (`feat:`, `fix:`) with concise imperative subjects and occasional issue/PR numbers in parentheses. Prefer a short subject, use `feat:`/`fix:` when applicable, and include the issue/PR number if relevant.
- PRs should have a clear title, link related issues, use `[WIP]` for drafts, and ensure lint/tests pass; include docstrings and note config changes where helpful.
