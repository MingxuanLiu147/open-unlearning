import json
import sys
from pathlib import Path

from data.inject import InjectDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from webui.utils.config_loader import ConfigLoader


def test_inject_dataset_supports_dotted_field_paths(tmp_path):
    dataset_path = tmp_path / "prm800k_raw.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "question": {
                    "problem": "What is 2 + 2?",
                    "ground_truth_solution": "4",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = InjectDataset(
        data_path=str(dataset_path),
        format_type="alpaca",
        instruction_key="question.problem",
        output_key="question.ground_truth_solution",
    )

    sample = dataset[0]

    assert sample["prompt"] == "### Instruction:\nWhat is 2 + 2?\n\n### Response:\n"
    assert sample["text"].endswith("4")


def test_config_loader_detects_inject_datasets_by_yaml_content(tmp_path):
    datasets_dir = tmp_path / "data" / "datasets"
    datasets_dir.mkdir(parents=True)

    (datasets_dir / "brep_prm800k.yaml").write_text(
        """
train:
  BREP_prm800k_train:
    handler: AlpacaDataset
    args:
      data_path: /tmp/prm800k.json
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (datasets_dir / "brep_gsm8k_eval.yaml").write_text(
        """
eval:
  BREP_gsm8k_eval:
    handler: InjectDataset
    args:
      data_path: /tmp/gsm8k.json
      format_type: alpaca
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (datasets_dir / "counterfact_edit.yaml").write_text(
        """
edit:
  CounterFact_edit:
    handler: CounterFactDataset
    args: {}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    loader = ConfigLoader(str(tmp_path))

    datasets = loader.get_datasets("inject")

    assert datasets["train"] == ["brep_prm800k"]
    assert datasets["eval"] == ["brep_gsm8k_eval"]
    assert datasets["analysis"] == []
