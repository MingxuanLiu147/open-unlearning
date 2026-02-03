## 1. Foundation (v1)

- [ ] 1.1 Add model configs for Llama-3.2-1B Instruct and Qwen3-1.7B Base
- [ ] 1.2 Add two TOFU demo experiment configs with flexible split overrides
- [ ] 1.3 Implement Python pipeline wrapper to orchestrate train.py -> eval.py
- [ ] 1.4 Emit evaluation artifacts (manifest/metrics/traces) in a standard layout
- [ ] 1.5 Add compatibility metadata structure for models/methods/datasets
- [ ] 1.6 Add CLI validation for incompatible model-method-dataset combos

## 2. Knowledge Editing (v2)

- [ ] 2.1 Add editing pipeline entrypoint and registry hooks
- [ ] 2.2 Add dataset configs for ZSRE and CounterFact
- [ ] 2.3 Add editing metrics (reliability/locality/generalization/portability)

## 3. Injection / PEFT (v2.5)

- [ ] 3.1 Add PEFT integration for LoRA/DoRA/AdaLoRA/LoReFT/BREP-REFT
- [ ] 3.2 Add injection evaluation hooks and configs

## 4. Multimodal (v3)

- [ ] 4.1 Define multimodal data schema and dataset adapters
- [ ] 4.2 Add VLM model adapters (LLaVA/Qwen-VL)
- [ ] 4.3 Add multimodal benchmarks and metrics

## 5. GUI Orchestrator (v2 end)

- [ ] 5.1 Add Gradio GUI shell that loads/saves the same configs as CLI
- [ ] 5.2 Implement wizard-style selection flow (model/method/dataset/eval)
- [ ] 5.3 Add basic run status + metrics visualization
