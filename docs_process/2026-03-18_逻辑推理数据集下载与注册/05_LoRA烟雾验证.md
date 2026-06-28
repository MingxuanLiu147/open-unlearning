# LoRA 烟雾验证

## 目标

使用真实 `Llama-3.2-1B-Instruct`，对本次接入的逻辑/规则数据集逐个执行最小 1-step LoRA 训练，验证：

- 数据集配置能被 Hydra 正常加载
- `InjectDataset` / `AlpacaDataset` 能正确读取样本
- `src/train.py` 的 inject 训练链路能完成至少 1 个 step
- PEFT adapter 能成功落盘

## 验证设置

- 模型：`meta-llama/Llama-3.2-1B-Instruct`
- 模型快照：`/home/liumingxuan/.huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6`
- 训练器：`inject/LoRA`
- 统一参数：
  - `trainer.args.do_eval=false`
  - `per_device_train_batch_size=1`
  - `gradient_accumulation_steps=1`
  - `num_train_epochs=1`
  - `max_steps=1`
  - `save_steps=1`
  - `report_to=none`
  - `max_length=128`
- 运行环境：
  - 当前 `.venv` 中 `torch.cuda.is_available()` 为 `False`
  - 本轮验证实际在 CPU 上完成

## 统一命令模式

```bash
CUDA_VISIBLE_DEVICES='' HYDRA_FULL_ERROR=1 PYTHONPATH=src .venv/bin/python src/train.py \
  --config-name=inject.yaml \
  model=Llama-3.2-1B-Instruct \
  trainer=inject/LoRA \
  data/datasets@data.train=<DATASET_CONFIG> \
  model.model_args.pretrained_model_name_or_path=/home/liumingxuan/.huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  model.tokenizer_args.pretrained_model_name_or_path=/home/liumingxuan/.huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  model.model_args.attn_implementation=eager \
  model.model_args.torch_dtype=float32 \
  trainer.args.bf16=false \
  trainer.args.do_eval=false \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.gradient_accumulation_steps=1 \
  trainer.args.num_train_epochs=1 \
  +trainer.args.max_steps=1 \
  trainer.args.logging_steps=1 \
  trainer.args.save_steps=1 \
  +trainer.args.overwrite_output_dir=true \
  +trainer.args.report_to=none \
  data.train.<DATASET_KEY>.args.max_length=128 \
  paths.output_dir=<OUTPUT_DIR> \
  task_name=<TASK_NAME>
```

## 结果

- 原始批量输出根目录：`/tmp/logic_lora_smoke_0rHKvn`
- 结论：`6 / 6` 全部跑通

| 数据集 | 配置名 | 数据键 | 样本数 | 退出码 | `train_runtime` | 输出目录 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| CLUTRR | `CLUTRR_train_inject` | `CLUTRR_train` | 53,518 | 0 | 3.7723s | `/tmp/logic_lora_smoke_0rHKvn/clutrr` |
| RuleTaker | `RuleTaker_train_inject` | `RuleTaker_train` | 480,152 | 0 | 3.0199s | `/tmp/logic_lora_smoke_0rHKvn/ruletaker` |
| ProofWriter | `ProofWriter_train_inject` | `ProofWriter_train` | 585,552 | 0 | 3.1400s | `/tmp/logic_lora_smoke_0rHKvn/proofwriter` |
| FOLIO | `FOLIO_train_inject` | `FOLIO_train` | 1,001 | 0 | 2.0857s | `/tmp/logic_lora_smoke_0rHKvn/folio` |
| LogicBench | `LogicBench_train_inject` | `LogicBench_train` | 12,908 | 0 | 1.8867s | `/tmp/logic_lora_smoke_0rHKvn/logicbench` |
| RuleArena | `RuleArena_eval_inject` | `RuleArena_eval` | 816 | 0 | 3.0734s | `/tmp/logic_lora_smoke_0rHKvn/rulearena` |

## 关键观察

- `CLUTRR`、`RuleTaker`、`ProofWriter`、`FOLIO`、`RuleArena` 的 1-step 日志中 `train_loss` 为 `0.0`
- `LogicBench` 的 1-step 日志中 `train_loss` 为 `10.254220962524414`
- 所有数据集都出现了 `InjectDataset loaded with ... samples` 或同等成功加载日志
- 所有数据集都成功保存了 `checkpoint-1/adapter_model.safetensors`

## 特殊说明

- `RuleArena` 在正式注册时是评测集配置，不是训练集配置
- 这次为了验证数据文件本身能否被 inject 训练链路消费，临时用 `data/datasets@data.train=RuleArena_eval_inject` 做了 1-step 烟雾训练
- 这不改变它在项目中的正式语义；正式使用时仍应把它当作 eval 数据集
