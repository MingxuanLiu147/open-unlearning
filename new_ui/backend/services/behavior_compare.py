"""Lightweight before/after text for the Results UI (metrics + optional eval samples)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config_loader import ConfigLoader
from .result_parser import METRIC_DESCRIPTIONS, format_value, parse_summary


MAX_SAMPLES = 5


def _run_dir_from_label(loader: ConfigLoader, label: str) -> Optional[Path]:
    for r in loader.get_eval_runs():
        if r["label"] != label:
            continue
        mode = r.get("mode", "")
        task = r.get("task_name", "")
        ckpt = r.get("checkpoint", "")
        if not task or not ckpt:
            continue
        p = loader.saves_dir / mode / task / ckpt
        if p.is_dir():
            return p
    return None


def _collect_samples(
    eval_dir: Path, max_n: int = MAX_SAMPLES
) -> List[Dict[str, str]]:
    """从 EVAL JSON 中收集最多 max_n 个 QA 样本。"""
    samples: List[Dict[str, str]] = []
    evals = eval_dir / "evals"
    if not evals.is_dir():
        return samples
    for jf in sorted(evals.rglob("*_EVAL.json")):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        _extract_qa_samples(data, samples, max_n)
        if len(samples) >= max_n:
            break
    return samples[:max_n]


def _extract_qa_samples(
    obj: Any, out: List[Dict[str, str]], max_n: int
) -> None:
    """递归从 EVAL JSON 中提取 QA 样本。"""
    if len(out) >= max_n:
        return
    if isinstance(obj, dict):
        for metric_val in obj.values():
            if not isinstance(metric_val, dict):
                continue
            vbi = metric_val.get("value_by_index")
            if isinstance(vbi, dict) and vbi:
                for entry in vbi.values():
                    if len(out) >= max_n:
                        return
                    if isinstance(entry, dict):
                        q = str(
                            entry.get("question")
                            or entry.get("prompt")
                            or ""
                        )
                        pred = str(
                            entry.get("prediction")
                            or entry.get("generated")
                            or entry.get("answer")
                            or ""
                        )
                        if q or pred:
                            out.append({"question": q or "(prompt)", "answer": pred})
            # 递归搜索嵌套结构
            for v in metric_val.values() if isinstance(metric_val, dict) else []:
                if isinstance(v, (dict, list)):
                    _extract_qa_samples(v, out, max_n)


def _compute_metric_deltas(
    before_metrics: Dict[str, float], after_metrics: Dict[str, float]
) -> List[Dict[str, Any]]:
    """计算 before/after 指标差值，返回 [{name, before, after, delta, direction}]。"""
    deltas: List[Dict[str, Any]] = []
    all_keys = sorted(set(before_metrics) | set(after_metrics))
    for k in all_keys:
        bv = before_metrics.get(k)
        av = after_metrics.get(k)
        if bv is None or av is None:
            continue
        try:
            bf = float(bv)
            af = float(av)
        except (TypeError, ValueError):
            continue
        d = af - bf
        deltas.append({
            "name": METRIC_DESCRIPTIONS.get(k, k),
            "key": k,
            "before": format_value(bf),
            "after": format_value(af),
            "delta": format_value(d),
            "direction": "up" if d > 0 else ("down" if d < 0 else "same"),
        })
    return deltas


def _parse_metrics_from_run(loader: ConfigLoader, lbl: str) -> Dict[str, float]:
    """从 run 的 summary 文件中提取所有指标的数值。"""
    metrics: Dict[str, float] = {}
    all_runs = {r["label"]: r for r in loader.get_eval_runs()}
    r = all_runs.get(lbl)
    if not r:
        return metrics
    for sf in r.get("summary_files", []):
        p = parse_summary(sf)
        if not p:
            continue
        raw = p.get("metrics") or {}
        for mk, mv in raw.items():
            try:
                metrics[mk] = float(mv)
            except (TypeError, ValueError):
                pass
    return metrics


def build_behavior_compare(loader: ConfigLoader, labels: List[str], question: str) -> Dict[str, Any]:
    if len(labels) < 2:
        return {"error": "need_at_least_two_runs"}

    def run_card(lbl: str) -> Dict[str, Any]:
        summary_lines: List[str] = []
        all_runs = {r["label"]: r for r in loader.get_eval_runs()}
        r = all_runs.get(lbl)
        if r:
            for sf in r.get("summary_files", []):
                p = parse_summary(sf)
                if not p:
                    continue
                metrics = p.get("metrics") or {}
                bits = []
                for mk, mv in sorted(metrics.items()):
                    label_m = METRIC_DESCRIPTIONS.get(mk, mk)
                    bits.append(f"{label_m}: {format_value(mv)}")
                summary_lines.append(f"[{p.get('name', 'eval')}] " + "; ".join(bits))

        rd = _run_dir_from_label(loader, lbl)
        samples: List[Dict[str, str]] = []
        if rd:
            samples = _collect_samples(rd)

        return {
            "label": lbl,
            "metrics_text": "\n".join(summary_lines) if summary_lines else "",
            "samples": samples,
        }

    before = run_card(labels[0])
    after = run_card(labels[1])

    # 合并样本：按 question 配对 before/after 答案
    paired_samples: List[Dict[str, str]] = []
    before_map = {s["question"]: s["answer"] for s in before.get("samples", [])}
    after_map = {s["question"]: s["answer"] for s in after.get("samples", [])}
    all_questions = list(dict.fromkeys(
        [s["question"] for s in before.get("samples", [])]
        + [s["question"] for s in after.get("samples", [])]
    ))
    for q in all_questions[:MAX_SAMPLES]:
        paired_samples.append({
            "question": q,
            "before_answer": before_map.get(q, "—"),
            "after_answer": after_map.get(q, "—"),
        })

    q_use = question.strip() or (paired_samples[0]["question"] if paired_samples else "")

    # 计算指标 delta
    before_metrics = _parse_metrics_from_run(loader, labels[0])
    after_metrics = _parse_metrics_from_run(loader, labels[1])
    metric_deltas = _compute_metric_deltas(before_metrics, after_metrics)

    return {
        "question": q_use,
        "before": before,
        "after": after,
        "samples": paired_samples,
        "metric_deltas": metric_deltas,
        "note": "展示最多 5 个评估样本的 before/after 对比。",
    }
