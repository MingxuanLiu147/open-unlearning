# Know-Surgery — Algorithm / Implementation Audit

Date: 2026-06-04. Scope: every registered method across **inject / unlearn / edit** (unimodal + multimodal), the **model** and **dataset** extension entry points, and the consistency between the **code** and the **paper** (`paper/`).

**How this was checked (boundary):** registry ↔ config ↔ paper cross-reference, file inspection, stub-marker grep, and the editor smoke-script's coverage. It is **not** a line-by-line correctness proof and nothing was executed end-to-end (that needs GPUs + local checkpoints). Citations were WebSearch-verified **only for new/changed entries**; the 43 pre-existing `references.bib` entries were taken as-is (offer to verify the rest stands).

---

## 1. TL;DR — what is and isn't fully there

- **Method counts line up with the registry**, with these exceptions/flags:
  - **SFRON** (`src/trainer/unlearn/sfron.py`, 163 lines) is **implemented but orphaned** — not in `TRAINER_REGISTRY`, no config, not in the paper. Either wire it up or treat as dead code.
  - **SERAC** full from-scratch pretraining (`train_serac()`) is a `NotImplementedError` **stub**. SERAC still *works* via a pre-trained `archive` or on-the-fly fine-tuning — so it is functional, but the "train the scope classifier + counterfactual model" pipeline is a TODO.
  - **AnyEdit name collision** — the repo's `anyedit.py` is an **in-house aggregator** over MEMIT/AlphaEdit/UNKE (selected by a `strategy` arg). There is a *separately published* "AnyEdit" (ICML 2025, arXiv 2502.05628) that is a **different** algorithm (autoregressive long-form editing). Needs an author decision (see §5).
- **Inject count mismatch** — paper claims **6** injection methods but only **5** exist (LoRA, DoRA, AdaLoRA, LoReFT, BREP). The total "48 methods" depends on inject=6, so this propagates (see §5).
- **Extension entry points are real**: adding a text model = 1 YAML (everything defaults to `AutoModelForCausalLM`); adding a dataset = class + `_register_data` + YAML. Multimodal model loading is also generic (`AutoModelForImageTextToText`), with prompt-format caveats for genuinely new VLM families.
- **The "smoke tests" are not CI tests** — `tests/test_all_editors_smoke.py` / `test_mm_editors_smoke.py` are manual `main()` scripts (hardcoded `~/.cursor/...` log path, require a local 7B model + GPU). `pytest` collects 0 from them. Real pytest tests: `test_edit_pipeline.py`, `test_edit_eval.py`, `test_inject_dataset.py`.

---

## 2. Method inventory (code ↔ registry ↔ config ↔ paper)

Legend: ✅ present · ⚠️ caveat · ❌ missing.

### 2.1 Unlearning — text (registry: `TRAINER_REGISTRY` in `src/trainer/__init__.py`)

| Method | Impl file | Registered | Config | In paper | Citation key | Status |
|---|---|---|---|---|---|---|
| GradAscent | `unlearn/grad_ascent.py` | ✅ | ✅ | ✅ | `yao2023unlearning` | ✅ |
| GradDiff | `unlearn/grad_diff.py` | ✅ | ✅ | ✅ | `maini2024tofu` | ✅ |
| SGA | `unlearn/sga.py` | ✅ | ✅ | ✅ | `pang2026sga` | ✅ |
| NPO | `unlearn/npo.py` | ✅ | ✅ | ✅ | `zhang2024npo` | ✅ |
| SimNPO | `unlearn/simnpo.py` | ✅ | ✅ | ✅ | `fan2024simnpo` | ✅ |
| DPO | `unlearn/dpo.py` | ✅ | ✅ | ✅ | `rafailov2023dpo` | ✅ |
| RMU | `unlearn/rmu.py` | ✅ | ✅ | ✅ | `li2024wmdp` | ✅ |
| FLAT | `unlearn/flat.py` | ✅ | ✅ | ✅ | `zhang2025flat` | ✅ |
| SOUL | `unlearn/soul.py` | ✅ | ✅ | ✅ | `jia2024soul` | ✅ |
| UNDIAL | `unlearn/undial.py` | ✅ | ✅ | ✅ | `dong2025undial` | ✅ |
| CEU | `unlearn/ceu.py` | ✅ | ✅ | ✅ | `yang2025ceu` | ✅ |
| SatImp | `unlearn/satimp.py` | ✅ | ✅ | ✅ | `yang2025satimp` | ✅ |
| WGA | `unlearn/wga.py` | ✅ | ✅ | ✅ | `geffect2025wga` | ✅ |
| PDU | `unlearn/pdu.py` | ✅ | ✅ | ✅ | `entesari2025pdu` | ✅ |
| **SFRON** | `unlearn/sfron.py` | ❌ | ❌ | ❌ | — | **Orphan** |

14 registered = paper's "Unl. (14)". SFRON is the 15th file, unwired.

### 2.2 Injection — text (PEFT)

| Method | Impl file | Registered | Config | In paper | Citation | Status |
|---|---|---|---|---|---|---|
| LoRA | `inject/lora.py` | ✅ | ✅ (+baichuan/chatglm/gpt2/gptj variants) | ✅ | `hu2022lora` | ✅ |
| DoRA | `inject/dora.py` | ✅ | ✅ | ✅ | `liu2024dora` | ✅ |
| AdaLoRA | `inject/adalora.py` | ✅ | ✅ | ✅ | `zhang2023adalora` | ✅ |
| LoReFT | `inject/loreft.py` | ✅ | ✅ | ✅ | `wu2024loreft` | ✅ |
| BREP | `inject/brep.py` | ✅ | ✅ | ✅ | `liang2025brep` | ✅ |
| (InjectTrainer / full FT) | `inject/base.py` | ✅ | `base_inject` | ⚠️ counted? | — | see §5 count |

**5 PEFT methods** — paper says "6 inject". The 6th is presumably full fine-tuning via `InjectTrainer`; make the table explicit.

### 2.3 Editing — text

| Method | Impl file | Registered | Config | In paper | Citation | Status |
|---|---|---|---|---|---|---|
| ROME | `edit/rome.py` | ✅ | ✅ | ✅ | `meng2022rome` | ✅ |
| MEMIT | `edit/memit.py` | ✅ | ✅ | ✅ | `meng2023memit` | ✅ |
| MEMIT_Merge | `edit/memit_merge.py` | ✅ | ✅ | ✅ | `meng2023memit` | ✅ (variant cite) |
| AlphaEdit | `edit/alphaedit.py` | ✅ | ✅ | ✅ | `fang2024alphaedit` | ✅ |
| MEND | `edit/mend.py` | ✅ | ✅ | ✅ | `mitchell2022mend` | ✅ |
| MALMEN | `edit/malmen.py` | ✅ | ✅ | ✅ | `tan2024malmen` | ✅ |
| InstructEdit | `edit/instructedit.py` | ✅ | ✅ | ✅ | `tian2024instructedit` | ✅ |
| GRACE | `edit/grace.py` | ✅ | ✅ | ✅ | `hartvigsen2023grace` | ✅ |
| WISE | `edit/wise.py` | ✅ | ✅ | ✅ | `wang2024wise` | ✅ |
| IKE | `edit/ike.py` | ✅ | ✅ | ✅ | `zheng2023ike` | ✅ |
| SERAC | `edit/serac.py` | ✅ | ✅ | ✅ | `mitchell2022serac` | ⚠️ pretrain stub |
| UniKE | `edit/unike.py` | ✅ | ✅ | ✅ | `pan2024unike` | ✅ |
| NMKE | `edit/nmke.py` | ✅ | ✅ | ✅ | `liu2025nmke` | ✅ |
| UNKE | `edit/unke_editor.py` | ✅ | ✅ | ✅ | `wang2024unke` | ✅ |
| AnyEdit | `edit/anyedit.py` | ✅ | ✅ | ✅ | `wang2024unke`† | ⚠️ name collision |

15 registered = paper's "Edit (15)".

### 2.4 Multimodal unlearning (registry: `MM_TRAINER_REGISTRY` dict in `src/mm_train.py`)

| Method | Impl file | Registered | Config | In paper | Citation | Status |
|---|---|---|---|---|---|---|
| MMGradAscent | `unlearn/mm_grad_ascent.py` | ✅ | ✅ | ✅ | `yao2023unlearning` | ✅ |
| MMGradDiff | `unlearn/mm_grad_diff.py` | ✅ | ✅ | ✅ | `maini2024tofu` | ✅ |
| MMKLMin | `unlearn/mm_kl_min.py` | ✅ | ✅ | ✅ | — (`--`) | ✅ (generic) |
| MMNPO | `unlearn/mm_npo.py` | ✅ | ✅ | ✅ | `zhang2024npo` | ✅ |
| MMRetainFT | `unlearn/mm_retain_ft.py` | ✅ | ✅ | ✅ | — (`--`) | ✅ (generic) |
| MMUnlearner | `unlearn/mmunlearner.py` | ✅ | ✅ | ✅ | **missing** → add | **fix cite** |
| MMVKD | `unlearn/mm_vkd.py` | ✅ | ✅ | ✅ | `wang2024mmvkd` | ✅ |

7 registered = paper's "MM Unl. (7)". **MMUnlearner** should cite the verified paper (see §5).

### 2.5 Multimodal editing (registry: `TRAINER_REGISTRY`, via `MMEditMixin`)

| Method | Impl file | Registered | Config | In paper | Citation | Status |
|---|---|---|---|---|---|---|
| MM-IKE | `edit/mm_ike.py` | ✅ | ✅ | ✅ | `zheng2023ike` | ✅ |
| MM-GRACE | `edit/mm_grace.py` | ✅ | ✅ | ✅ | `hartvigsen2023grace` | ✅ |
| MM-WISE | `edit/mm_wise.py` | ✅ | ✅ | ✅ | `wang2024wise` | ✅ |
| MM-MEND | `edit/mm_mend.py` | ✅ | ✅ | ✅ | `mitchell2022mend` | ✅ |
| MM-SERAC | `edit/mm_serac.py` | ✅ | ✅ | ✅ | `mitchell2022serac` | ✅ |
| MM-UniKE | `edit/mm_unike.py` | ✅ | ✅ | ✅ | `pan2024unike` | ✅ |

6 registered = paper's "MM Edit (6)".

---

## 3. Implementation-completeness findings

1. **SERAC pretraining stub** — `serac.py:426` `raise NotImplementedError("Full SERAC pre-training is not yet implemented...")`. Usable paths: (a) pass a pre-trained `archive`, (b) on-the-fly fine-tuning (edit without archive, as the smoke script exercises with `archive=None`). Impact: SERAC is *functional* but cannot reproduce the full paper protocol (trained scope classifier + counterfactual model) out of the box.
2. **SFRON orphan** — fully written trainer, never registered/configured. Note `mmunlearner.py` re-implements the same "saliency-mask selective update" idea for the MM path, so SFRON may be the abandoned text-side predecessor.
3. **AnyEdit** — in-house aggregator (`strategy ∈ {memit, alphaedit, alphaedit_are, unke, unke_are}`), not the published ICML-2025 AnyEdit. Functionally fine; only the *naming/citation* is the issue.
4. **Abstract base `NotImplementedError`s are correct** (`edit/base.py:98`, `mia/all_attacks.py`) — these are intended abstract methods, not gaps.
5. **No other stub markers** in `src/` beyond registry "not registered" guards and argument-validation guards (`grad_diff.py:143` retain-loss-type guard, etc.).

---

## 4. "Freely add models / datasets" entry points

- **Text models** — `src/model/__init__.py:get_model` reads `model_handler` (default `AutoModelForCausalLM`). Only `AutoModelForCausalLM` + `ProbedLlamaForCausalLM` are registered, so **any HF CausalLM = one YAML** (no code). ✅ matches the paper claim.
- **Multimodal models** — `src/model/multimodal.py:get_mm_model` loads via generic `AutoModelForImageTextToText` + auto-discovers LoRA targets (`find_all_linear_names`). So a new `AutoModelForImageTextToText`-compatible VLM is also ~1 YAML. ⚠️ Caveat: per-family **prompt/chat-template formatting** lives in the MM data loaders; BLIP-2 / InstructBLIP are flagged "limited (no chat template)". A genuinely new family may need a data-side branch.
- **Datasets** — `src/data/__init__.py`: implement a `Dataset` class → `_register_data(MyDataset)` → YAML with `handler:`. Registered today: QA/Pretraining/Completion + edit (Editing, ZSRE, CounterFact, ELKEN, UnKE, ConceptEdit, AKEW, LEME, EditEvery) + MM-edit (MMEditVQA, MMEditCaption, MMKEBench, VLKEB) + inject (Inject, Alpaca, ShareGPT). ⚠️ **RWKU** and **MQuAKE** are *not* in `DATASET_REGISTRY`; they are wired through their **evaluators** (`RWKUEvaluator`, `MQuAKEMultiHopEvaluator`) — by design, not a bug. **CLEAR / MLLMU-Bench / FIUBench** (MM unlearn) load through the separate MM path (`data/clear_dataset.py`, `data/multimodal.py`, `data/fiubench_dataset.py`), not `DATASET_REGISTRY`.

---

## 5. Paper ↔ code discrepancies (drives the `paper/` edits)

**Safe, verified — applied:**
- **MMUnlearner citation** (appendix `tab:all-methods`, currently `--`) → add `mmunlearner2025` = *MMUnlearner: Reformulating Multimodal Machine Unlearning…*, ACL 2025 Findings, arXiv 2502.11051. Code matches (saliency-map gradient ascent).
- **Abstract "28+ methods"** → inconsistent with the 48 used everywhere else (features §, appendix caption). Update to the real number.
- **Abstract "Gradio-based web interface"** → the shipped UI is **Flask + Vue 3** (`new_ui/`); the old Gradio `webui/` is gone. Update.

**Needs your decision — flagged, not changed:**
- **Inject count 5 vs 6** — `tab:methods` header says "Inj. (6)" but lists 5; the 48 total depends on 6. Either add the 6th (full FT / `InjectTrainer`) explicitly or correct the count to 5 (→ 47 total).
- **AnyEdit** — collides with the published ICML-2025 AnyEdit (arXiv 2502.05628), a different method. Options: (i) rename your aggregator (e.g., "UniStrategyEdit"), (ii) cite the real AnyEdit only if you intend to represent it, or (iii) keep the `†` footnote as-is. Your call.
- **`tab:knowledge` rule-based row (RuleTaker, FOLIO)** — no registered dataset class found; only `scripts/prepare_logic_reasoning_public_data.py` exists. Confirm these are actually wired before claiming them.
- **`MMKLMin` / `MMRetainFT`** — `--` is fine (generic KL-min / retain-FT objectives with no single source), unless you want to attribute them.

---

## 6. Suggested next actions

1. Decide SFRON: register + config + paper row, or delete.
2. Decide AnyEdit naming/citation; decide the inject 5-vs-6 count.
3. (Optional) Implement `SERAC.train_serac()` or soften the paper to say SERAC runs in archive / on-the-fly modes.
4. Confirm RuleTaker/FOLIO wiring or drop them from `tab:knowledge`.
5. (Optional) Ask me to verify the remaining 43 `references.bib` entries against Google Scholar.
