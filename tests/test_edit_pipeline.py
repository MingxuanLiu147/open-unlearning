from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import BatchEncoding, TrainingArguments

from data.editing import (
    AKEWDataset,
    ConceptEditDataset,
    CounterFactDataset,
    EditEveryDataset,
    EditingDataset,
    EditingSample,
    ELKENDataset,
    LEMEDataset,
    UnKEDataset,
    ZSREDataset,
)
from trainer.edit.memit import MEMITEditor
from trainer.edit.rome import ROMEEditor
from trainer.edit.base import EditRequest
from trainer.edit.pipeline import (
    build_edit_requests,
    execute_edit_requests,
    save_edit_artifacts,
)
from trainer.edit import (
    AlphaEditEditor,
    UNKEEditor,
    GRACEEditor,
    WISEEditor,
    IKEEditor,
    SERACEditor,
    MALMENEditor,
    InstructEditEditor,
    AnyEditEditor,
)


class DummyDataset:
    def __init__(self, requests):
        self._requests = list(requests)

    def to_edit_requests(self, limit=None):
        requests = list(self._requests)
        return requests[:limit] if limit is not None else requests


class DummyTrainer:
    def __init__(self):
        self.calls = []
        self.preserve_memory = True
        self.edit_history = []

    def edit(self, requests):
        if isinstance(requests, list):
            payload = requests
            edited_count = len(requests)
        else:
            payload = [requests]
            edited_count = 1
        self.calls.append(payload)
        return {"success": True, "edited_count": edited_count}


class ToyTokenizer:
    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def _encode(self, text: str):
        tokens = text.split()
        if not tokens:
            return [self.eos_token_id]
        return [abs(hash(token)) % (self.vocab_size - 2) + 2 for token in tokens]

    def __call__(
        self,
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=None,
        truncation=False,
        padding=None,
    ):
        ids = self._encode(text)
        if add_special_tokens:
            ids = ids + [self.eos_token_id]
        if max_length is not None and truncation:
            ids = ids[:max_length]
        attention_mask = [1] * len(ids)
        if padding == "max_length" and max_length is not None and len(ids) < max_length:
            pad_len = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        if return_tensors == "pt":
            return BatchEncoding(
                {
                    "input_ids": torch.tensor([ids], dtype=torch.long),
                    "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
                }
            )
        return {"input_ids": ids, "attention_mask": attention_mask}


class ToyMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.c_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.c_proj(hidden_states)


class ToyBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = ToyMLP(hidden_size)

    def forward(self, hidden_states):
        return self.mlp(hidden_states)


class ToyTransformer(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.h = nn.ModuleList([ToyBlock(hidden_size) for _ in range(num_layers)])

    def forward(self, hidden_states):
        for layer in self.h:
            hidden_states = layer(hidden_states)
        return hidden_states


class ToyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 4096, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.transformer = ToyTransformer(hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(is_encoder_decoder=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.embed(input_ids)
        hidden_states = self.transformer(hidden_states)
        logits = self.lm_head(hidden_states)
        return SimpleNamespace(logits=logits, loss=None)


def _request(idx: int) -> EditRequest:
    return EditRequest(
        prompt=f"prompt-{idx}",
        subject=f"subject-{idx}",
        target_new=f"target-{idx}",
    )


def test_build_edit_requests_respects_max_edits_across_datasets():
    edit_data = {
        "a": DummyDataset([_request(1), _request(2)]),
        "b": DummyDataset([_request(3)]),
    }

    requests = build_edit_requests(edit_data, max_edits=2)

    assert [request.prompt for request in requests] == ["prompt-1", "prompt-2"]


def test_execute_edit_requests_batch_and_history():
    trainer = DummyTrainer()
    requests = [_request(1), _request(2)]

    summary = execute_edit_requests(trainer, requests, edit_type="batch")

    assert summary["requested_count"] == 2
    assert summary["result"]["edited_count"] == 2
    assert len(trainer.calls) == 1
    assert len(trainer.edit_history) == 2


def test_execute_edit_requests_sequential_runs_each_request():
    trainer = DummyTrainer()
    requests = [_request(1), _request(2), _request(3)]

    summary = execute_edit_requests(trainer, requests, edit_type="sequential")

    assert summary["success"] is True
    assert len(summary["step_results"]) == 3
    assert len(trainer.calls) == 3
    assert len(trainer.edit_history) == 3


def test_save_edit_artifacts_writes_preview_and_summary(tmp_path: Path):
    requests = [_request(1)]
    summary = {"method": "DummyTrainer", "result": {"success": True}}

    save_edit_artifacts(tmp_path, requests, summary)

    preview_path = tmp_path / "edit_requests_preview.json"
    summary_path = tmp_path / "edit_results.json"

    assert preview_path.exists()
    assert summary_path.exists()
    assert "prompt-1" in preview_path.read_text(encoding="utf-8")
    assert "DummyTrainer" in summary_path.read_text(encoding="utf-8")


def test_editing_dataset_to_edit_requests_preserves_fields():
    dataset = EditingDataset.__new__(EditingDataset)
    dataset.data = [
        EditingSample(
            prompt="Where was Ada born?",
            subject="Ada",
            target_new="London",
            target_old="Paris",
            locality_inputs={
                "Relation_Specificity": [
                    {"prompt": "Where is Rome?", "ground_truth": "Italy"}
                ]
            },
            portability_inputs={
                "Reasoning": [{"prompt": "Ada was born in", "ground_truth": "London"}]
            },
        )
    ]

    requests = dataset.to_edit_requests()

    assert len(requests) == 1
    assert requests[0].prompt == "Where was Ada born?"
    assert requests[0].target_old == "Paris"
    assert (
        requests[0].locality_inputs["Relation_Specificity"][0]["ground_truth"]
        == "Italy"
    )


def test_local_default_paths_for_zsre_and_counterfact():
    zsre = ZSREDataset()
    zsre_train = ZSREDataset(split="train")
    counterfact = CounterFactDataset()

    assert zsre.data_path.name == "ZsRE-test-all.json"
    assert zsre_train.data_path.name == "zsre_mend_train.json"
    assert counterfact.data_path.name == "counterfact_train.json"
    assert len(zsre) > 0
    assert len(zsre_train) > 0
    assert len(counterfact) > 0


def test_new_benchmark_adapters_build_valid_edit_requests():
    datasets = [
        ELKENDataset(split="train"),
        UnKEDataset(version="v3"),
        ConceptEditDataset(variant="inter"),
        AKEWDataset(subset="mquake-cf"),
        LEMEDataset(subset="zsre"),
    ]

    for dataset in datasets:
        requests = dataset.to_edit_requests(limit=1)
        assert len(requests) == 1
        request = requests[0]
        assert request.prompt
        assert request.subject
        assert request.target_new
        assert request.subject in request.prompt


def test_rome_and_memit_fallback_to_available_layers(tmp_path: Path):
    tokenizer = ToyTokenizer()

    def _model():
        return ToyCausalLM(vocab_size=tokenizer.vocab_size, hidden_size=64, num_layers=2)

    rome_args = TrainingArguments(
        output_dir=str(tmp_path / "rome"),
        report_to=[],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_strategy="no",
        save_strategy="no",
        eval_strategy="no",
        disable_tqdm=True,
        use_cpu=True,
    )
    memit_args = TrainingArguments(
        output_dir=str(tmp_path / "memit"),
        report_to=[],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_strategy="no",
        save_strategy="no",
        eval_strategy="no",
        disable_tqdm=True,
        use_cpu=True,
    )

    rome = ROMEEditor(
        model=_model(),
        tokenizer=tokenizer,
        args=rome_args,
        v_num_grad_steps=1,
    )
    memit = MEMITEditor(
        model=_model(),
        tokenizer=tokenizer,
        args=memit_args,
        v_num_grad_steps=1,
    )

    rome_result = rome.edit(ZSREDataset().to_edit_requests(limit=1)[0])
    memit_result = memit.edit(CounterFactDataset().to_edit_requests(limit=1)[0])

    assert rome_result["success"] is True
    assert rome_result["edited_count"] == 1
    assert memit_result["success"] is True
    assert memit_result["edited_count"] == 1


def test_editevery_dataset_loads_and_produces_requests():
    ds = EditEveryDataset()
    assert len(ds) > 0
    requests = ds.to_edit_requests(limit=2)
    assert len(requests) == 2
    for r in requests:
        assert r.prompt
        assert r.target_new


def test_new_editor_classes_importable():
    """Smoke-test that all 9 new editor classes are importable and registered."""
    from trainer import TRAINER_REGISTRY

    expected = [
        "AlphaEditEditor",
        "UNKEEditor",
        "GRACEEditor",
        "WISEEditor",
        "IKEEditor",
        "SERACEditor",
        "MALMENEditor",
        "InstructEditEditor",
        "AnyEditEditor",
    ]
    for name in expected:
        assert name in TRAINER_REGISTRY, f"{name} not found in TRAINER_REGISTRY"
