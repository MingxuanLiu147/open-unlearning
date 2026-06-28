# Model Compatibility Matrix

This document lists the supported models and their compatibility with each method across the three tasks: **Edit**, **Unlearn**, and **Inject**.

## Supported Model Configurations

### Single-Modal Text Models (31 configs)

| Series | Size | Config Name | Architecture |
|--------|------|-------------|-------------|
| LLaMA 2 | 7B | `Llama-2-7b-hf` | model.layers |
| LLaMA 2 | 7B-Chat | `Llama-2-7b-chat-hf` | model.layers |
| LLaMA 2 | 13B | `Llama-2-13b-hf` | model.layers |
| LLaMA 3.2 | 1B | `Llama-3.2-1B-Instruct` | model.layers |
| LLaMA 3.2 | 3B | `Llama-3.2-3B-Instruct` | model.layers |
| LLaMA 3.1 | 8B | `Llama-3.1-8B-Instruct` | model.layers |
| Qwen 2.5 | 0.5B | `Qwen2.5-0.5B-Instruct` | model.layers |
| Qwen 2.5 | 1.5B | `Qwen2.5-1.5B-Instruct` | model.layers |
| Qwen 2.5 | 3B | `Qwen2.5-3B-Instruct` | model.layers |
| Qwen 2.5 | 7B | `Qwen2.5-7B-Instruct` | model.layers |
| Qwen 2.5 | 14B | `Qwen2.5-14B-Instruct` | model.layers |
| Mistral | 7B | `Mistral-7B-Instruct-v0.3` | model.layers |
| InternLM 2.5 | 7B | `internlm2_5-7b-chat` | model.layers |
| Baichuan 2 | 7B | `Baichuan2-7B-Chat` | model.layers |
| Baichuan 2 | 13B | `Baichuan2-13B-Chat` | model.layers |
| ChatGLM 4 | 9B | `glm-4-9b-chat` | encoder.layers |
| GPT-2 | 124M | `gpt2` | transformer.h |
| GPT-2 | 355M | `gpt2-medium` | transformer.h |
| Gemma | 7B | `gemma-7b-it` | model.layers |
| Gemma 2 | 2B | `gemma-2-2b-it` | model.layers |
| Gemma 2 | 9B | `gemma-2-9b-it` | model.layers |
| Yi 1.5 | 6B | `Yi-1.5-6B-Chat` | model.layers |
| Yi 1.5 | 9B | `Yi-1.5-9B-Chat` | model.layers |
| DeepSeek-R1 | 1.5B | `DeepSeek-R1-Distill-Qwen-1.5B` | model.layers |
| DeepSeek-R1 | 7B | `DeepSeek-R1-Distill-Qwen-7B` | model.layers |
| DeepSeek-R1 | 8B | `DeepSeek-R1-Distill-Llama-8B` | model.layers |
| Phi | 1.3B | `phi-1_5` | model.layers |
| Phi 3.5 | 3.8B | `Phi-3.5-mini-instruct` | model.layers |
| Zephyr | 7B | `zephyr-7b-beta` | model.layers |

### Multi-Modal Models (7 configs)

| Model | Config Name | Usage | Chat Template |
|-------|-------------|-------|---------------|
| Qwen2-VL-2B | `Qwen2VL-2B` | MM-Edit, MM-Unlearn | Yes |
| Qwen3-VL-4B | `Qwen3-VL-4B` | MM-Edit, MM-Unlearn | Yes |
| LLaVA-1.5-7B | `llava-1.5-7b` | MM-Edit, MM-Unlearn | Yes |
| LLaVA-OneVision-7B | `llava-onevision-7b` | MM-Edit, MM-Unlearn | Yes |
| InternVL2.5-2B | `InternVL2_5-2B` | MM-Edit, MM-Unlearn | Yes |
| InternVL2.5-8B | `InternVL2_5-8B` | MM-Edit, MM-Unlearn | Yes |
| BLIP-2-OPT-2.7B | `blip2-opt-2.7b` | MM-Unlearn (limited) | No (plain text fallback) |

---

## Edit Methods Compatibility

| Method | LLaMA / Qwen / Mistral / InternLM / Gemma / Yi / DeepSeek / Phi / Zephyr | Baichuan 2 | GPT-2 | ChatGLM 4 | Multi-Modal VL |
|--------|---------|------------|-------|-----------|---------------|
| ROME | OK | OK | OK | Not supported | - |
| MEMIT | OK | OK | OK | Not supported | - |
| MEND | OK | OK | OK | Not supported | - |
| MALMEN | OK | OK | OK | Not supported | - |
| AlphaEdit | OK | OK | Override templates | Not supported | - |
| AnyEdit | OK | OK | Override templates | Not supported | - |
| UNKE | OK | OK | Override templates | Not supported | - |
| InstructEdit | OK | OK | OK | Not supported | - |
| GRACE | OK | OK | Override inner_params | Not supported | - |
| WISE | OK | OK | Override inner_params | Not supported | - |
| IKE | OK | OK | OK | Not supported | - |
| SERAC | OK | OK | OK | Not supported | - |
| MM-IKE | - | - | - | - | OK |
| MM-GRACE | - | - | - | - | OK |
| MM-WISE | - | - | - | - | OK |
| MM-MEND | - | - | - | - | OK |
| MM-SERAC | - | - | - | - | OK |

**Notes:**
- "OK" = works out of the box with default config
- "Override templates" = need to set `rewrite_module_tmp` / `layer_module_tmp` / `inner_params` via trainer YAML
- "Not supported" = architecture not compatible, do not use

---

## Unlearn Methods Compatibility

| Method | All model.layers models | GPT-2 | ChatGLM 4 | Notes |
|--------|------------------------|-------|-----------|-------|
| GradAscent | OK | OK | OK | No architecture dependency |
| GradDiff | OK | OK | OK | |
| NPO | OK | OK | OK | |
| DPO | OK | OK | OK | |
| SimNPO | OK | OK | OK | |
| UNDIAL | OK | OK | OK | |
| CEU | OK | OK | OK | |
| SatImp | OK | OK | OK | |
| WGA | OK | OK | OK | |
| PDU | OK | OK | OK | |
| RMU | OK | Override regex | Override regex | Default regex targets `model.layers`; override via YAML |

---

## Inject Methods Compatibility

| Method | LLaMA / Qwen / Mistral / InternLM / Gemma / Yi / DeepSeek | Baichuan 2 | ChatGLM 4 | GPT-2 |
|--------|-----|------------|-----------|-------|
| LoRA | OK (`q/k/v/o_proj`) | Use `LoRA_baichuan` (`W_pack`) | Use `LoRA_chatglm` (`query_key_value`) | Use `LoRA_gpt2` (`c_attn`) |
| DoRA | OK | Use `LoRA_baichuan` targets | Use `LoRA_chatglm` targets | Use `LoRA_gpt2` targets |
| AdaLoRA | OK | Use `LoRA_baichuan` targets | Use `LoRA_chatglm` targets | Use `LoRA_gpt2` targets |
| LoReFT | OK | OK | OK | OK |
| BREP | OK | OK | Not supported | Not supported |
| InjectTrainer | OK | OK | OK | OK |

---

## Custom Data Formats

### Inject (Alpaca)
```json
{"instruction": "Translate to English", "input": "Hello", "output": "Hello"}
```

### Inject (ShareGPT)
```json
{"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello!"}]}
```

### Unlearn (QA)
```json
{"question": "What is X?", "answer": "Y"}
```

### Edit (Triple)
```json
{"prompt": "X is", "subject": "X", "target_new": "Y", "target_old": "Z"}
```

Validate data before training:
```bash
python scripts/validate_data.py --mode inject --data /path/to/data.jsonl
```

---

## Custom Models

Users can add any HuggingFace model by creating a YAML in `configs/model/`:

```yaml
model_args:
  pretrained_model_name_or_path: "my-org/my-model"
  torch_dtype: bfloat16
tokenizer_args:
  pretrained_model_name_or_path: "my-org/my-model"
template_args:
  apply_chat_template: true
  system_prompt: "You are a helpful assistant."
  user_start_tag: "<|user|>"
  user_end_tag: "<|end|>"
  asst_start_tag: "<|assistant|>"
  asst_end_tag: "<|end|>"
```

- Models using `model.layers` architecture: full Inject/Unlearn/Edit support
- Other architectures: Inject/Unlearn work; Edit methods may need template overrides
