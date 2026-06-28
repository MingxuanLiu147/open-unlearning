# Know-Surgery Extension Guide

How to add new models, methods, and datasets to the toolkit.

## 1. Add a New Model

Create a YAML config file in `configs/model/`:

```yaml
# configs/model/My-Model-7B.yaml
model_args:
  pretrained_model_name_or_path: "your-org/your-model-7b"
  torch_dtype: bfloat16
tokenizer_args:
  pretrained_model_name_or_path: "your-org/your-model-7b"
template_args:
  apply_chat_template: True
  system_prompt: "You are a helpful assistant."
  user_start_tag: "<|user|>"
  user_end_tag: ""
  asst_start_tag: "<|assistant|>"
  asst_end_tag: ""
```

Run with: `python src/train.py model=My-Model-7B ...`

For models requiring custom loading logic, register a handler in `src/model/__init__.py`:

```python
from model.my_model import MySpecialModel
_register_model(MySpecialModel)
```

Then set `model_handler: MySpecialModel` in the YAML.

## 2. Add a New Editing Method

### Step 1: Implement the editor class

```python
# src/trainer/edit/my_method.py
from trainer.edit.base import EditTrainer, EditRequest

class MyMethodEditor(EditTrainer):
    def __init__(self, model, tokenizer, args, **method_args):
        super().__init__(model, tokenizer, args, **method_args)

    def edit(self, requests: list[EditRequest], **kwargs):
        weights_copy = {}
        for request in requests:
            # Your editing logic here
            pass
        return self.model, weights_copy
```

### Step 2: Register

```python
# src/trainer/edit/__init__.py
from trainer.edit.my_method import MyMethodEditor

# src/trainer/__init__.py
_register_trainer(MyMethodEditor)
```

### Step 3: Create config

```yaml
# configs/trainer/edit/MyMethod.yaml
defaults:
  - edit/base_editor@_here_
  - _self_

handler: MyMethodEditor
method_args:
  layers: [5]
  learning_rate: 0.5
```

### Step 4: Run

```bash
python src/train.py experiment=edit/zsre/default trainer=edit/MyMethod
```

## 3. Add a New Unlearning Method

Same pattern as editing, but inherit from `FinetuneTrainer` and override `compute_loss`:

```python
# src/trainer/unlearn/my_unlearn.py
from trainer.base import FinetuneTrainer

class MyUnlearnMethod(FinetuneTrainer):
    def compute_loss(self, model, inputs, **kwargs):
        forget_inputs = inputs["forget"]
        retain_inputs = inputs["retain"]
        # Your unlearning loss
        return loss
```

Register in `src/trainer/__init__.py` and create `configs/trainer/unlearn/MyUnlearn.yaml`.

## 4. Add a New Injection Method

Inherit from `InjectTrainer`:

```python
# src/trainer/inject/my_inject.py
from trainer.inject.base import InjectTrainer

class MyInjectTrainer(InjectTrainer):
    def setup_peft(self, model):
        # Configure your PEFT method
        return model
```

## 5. Add a New Dataset

### For editing datasets:

```python
# In src/data/editing.py
class MyEditDataset(EditingDataset):
    def normalize_record(self, record):
        return EditingSample(
            prompt=record["question"],
            subject=record.get("subject", ""),
            target_new=record["answer"],
            target_old=record.get("old_answer"),
        )
```

Register in `src/data/__init__.py`:
```python
_register_data(MyEditDataset)
```

Create config:
```yaml
# configs/data/datasets/MyData_edit.yaml
handler: MyEditDataset
args:
  data_path: "data/edit/mydata/test.json"
  split: "test"
  max_length: 512
```

### For unlearning datasets:

Use the existing `QADataset` handler with a new config pointing to your data files.

## 6. Create an Experiment Config

Combine model + trainer + data + eval:

```yaml
# configs/experiment/edit/mydata/default.yaml
defaults:
  - override /model: Qwen2.5-7B-Instruct
  - override /trainer: edit/ROME
  - override /data: edit
  - override /eval: edit

data:
  edit:
    MyData_edit:
      handler: MyEditDataset
      args:
        split: "test"
        max_length: 512
```

## Architecture Overview

```
train.py (unified entry)
  |
  +-- model/__init__.py    MODEL_REGISTRY     <- configs/model/*.yaml
  +-- data/__init__.py     DATASET_REGISTRY   <- configs/data/datasets/*.yaml
  +-- trainer/__init__.py  TRAINER_REGISTRY   <- configs/trainer/{edit,unlearn,inject}/*.yaml
  +-- evals/__init__.py    EVALUATOR_REGISTRY <- configs/eval/*.yaml
  |
  +-- Hydra combines them via configs/experiment/*/*.yaml
```

Each component is independent. Methods don't depend on specific datasets or models.
