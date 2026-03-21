"""
评估结果解析服务 - 适配自 webui/utils/result_parser.py，返回 JSON 友好结构。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


METRIC_DESCRIPTIONS = {
    "forget_quality": "Forget Quality",
    "model_utility": "Model Utility",
    "forget_Q_A_Prob": "Forget QA Prob",
    "forget_Q_A_ROUGE": "Forget QA ROUGE",
    "forget_Truth_Ratio": "Forget Truth Ratio",
    "privleak": "Privacy Leak",
    "extraction_strength": "Extraction Strength",
    "exact_memorization": "Exact Memorization",
    "forget_knowmem_ROUGE": "Forget KnowMem",
    "forget_verbmem_ROUGE": "Forget VerbMem",
    "retain_knowmem_ROUGE": "Retain KnowMem",
    "reliability": "Reliability",
    "generalization": "Generalization",
    "locality": "Locality",
    "portability": "Portability",
    "task_accuracy": "Task Accuracy",
    "knowledge_retention": "Knowledge Retention",
}


def format_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4e}" if abs(v) < 0.01 or abs(v) > 1000 else f"{v:.4f}"
    return str(v)


def parse_summary(filepath: str) -> Optional[Dict]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        name = Path(filepath).stem.replace("_SUMMARY", "")
        return {"name": name, "metrics": data, "file": filepath}
    except Exception:
        return None


def parse_run_results(output_dir: str) -> List[Dict]:
    p = Path(output_dir)
    if not p.exists():
        return []
    files = sorted(str(f) for f in p.rglob("*_SUMMARY.json"))
    return [r for f in files if (r := parse_summary(f)) is not None]


def compare_runs(run_results: Dict[str, List[Dict]]) -> Dict:
    """
    run_results: { label: [ {name, metrics, file}, ... ] }
    返回: { eval_name: { metric: { label: value, ... } } }
    """
    out: Dict[str, Dict] = {}
    for label, results in run_results.items():
        for r in results:
            en = r["name"]
            if en not in out:
                out[en] = {}
            for mk, mv in r["metrics"].items():
                if mk not in out[en]:
                    out[en][mk] = {}
                out[en][mk][label] = mv
    return out
