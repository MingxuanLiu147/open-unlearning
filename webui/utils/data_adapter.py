# -*- coding: utf-8 -*-
"""
数据适配层
==========

负责解析用户输入的 JSONL / 单条 JSON 数据，验证字段合法性，
生成各任务模式所需的字段映射和 Hydra dataset config。

支持的任务模式及推荐字段：
  inject  : instruction / input / output
  edit    : prompt / subject / target_new / target_old
  unlearn : question / answer / split  或  分别指定 forget/retain
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# 每个 mode 的推荐字段定义
FIELD_SCHEMA: Dict[str, Dict] = {
    "inject": {
        "required": ["output"],
        "optional": ["instruction", "input"],
        "description": {
            "zh": "指令微调格式（Alpaca 风格）：instruction=指令, input=补充输入, output=期望输出",
            "en": "Instruction-tuning format (Alpaca-style): instruction, input(optional), output",
        },
        "example": {
            "instruction": "介绍量子纠缠现象",
            "input": "",
            "output": "量子纠缠是一种量子力学现象，两个粒子的量子态相互关联...",
        },
    },
    "edit": {
        "required": ["prompt", "target_new"],
        "optional": ["subject", "target_old"],
        "description": {
            "zh": "知识编辑格式：prompt=原始问题, target_new=新答案, target_old=旧答案（可选）",
            "en": "Knowledge editing format: prompt, target_new, subject(optional), target_old(optional)",
        },
        "example": {
            "prompt": "The capital of France is",
            "subject": "France",
            "target_new": "Paris",
            "target_old": "Lyon",
        },
    },
    "unlearn": {
        "required": ["question", "answer"],
        "optional": ["split"],
        "description": {
            "zh": "遗忘格式：question=问题, answer=答案, split=forget/retain（决定数据归属）",
            "en": "Unlearn format: question, answer, split=forget/retain",
        },
        "example": {
            "question": "What is the birthplace of Harry Potter?",
            "answer": "Godric's Hollow",
            "split": "forget",
        },
    },
}

# 上传文件保存目录
UPLOADS_DIR = Path(__file__).parent.parent / "uploads"


def _ensure_uploads_dir() -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOADS_DIR


def parse_jsonl_text(text: str) -> Tuple[List[Dict], Optional[str]]:
    """解析 JSONL 文本（每行一个 JSON 对象），也兼容单个 JSON 数组。

    Returns:
        (records, error_msg) — error_msg 为 None 表示成功
    """
    if not text or not text.strip():
        return [], "内容为空"

    text = text.strip()

    # 尝试按行解析 JSONL
    records = []
    errors = []
    lines = [l for l in text.splitlines() if l.strip()]

    # 如果只有一行且是数组，当作 JSON 数组处理
    if len(lines) == 1 and lines[0].strip().startswith("["):
        try:
            records = json.loads(lines[0])
            if isinstance(records, list):
                return records, None
        except json.JSONDecodeError:
            pass

    # 按行解析
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
            else:
                errors.append(f"第 {i+1} 行不是 JSON 对象")
        except json.JSONDecodeError as e:
            errors.append(f"第 {i+1} 行解析失败: {e.msg}")

    if errors and not records:
        return [], "\n".join(errors[:5])
    return records, "\n".join(errors[:3]) if errors else None


def parse_jsonl_file(file_path: str) -> Tuple[List[Dict], Optional[str]]:
    """从文件路径读取 JSONL。"""
    try:
        content = Path(file_path).read_text(encoding="utf-8")
        return parse_jsonl_text(content)
    except Exception as e:
        return [], f"读取文件失败: {e}"


def validate_records(records: List[Dict], mode: str) -> Tuple[bool, List[str]]:
    """检查记录是否符合指定 mode 的字段要求。

    Returns:
        (all_valid, warning_list)
    """
    schema = FIELD_SCHEMA.get(mode, {})
    required = schema.get("required", [])
    warnings = []

    if not records:
        return False, ["没有有效记录"]

    missing_counts: Dict[str, int] = {f: 0 for f in required}
    for rec in records:
        for f in required:
            if f not in rec or rec[f] is None or str(rec[f]).strip() == "":
                missing_counts[f] += 1

    for field, cnt in missing_counts.items():
        if cnt > 0:
            warnings.append(f"字段 `{field}` 在 {cnt}/{len(records)} 条记录中缺失")

    all_valid = all(v == 0 for v in missing_counts.values())
    return all_valid, warnings


def get_example_jsonl(mode: str) -> str:
    """返回指定 mode 的示例 JSONL 文本（3 条）。"""
    schema = FIELD_SCHEMA.get(mode, {})
    example = schema.get("example", {})
    if not example:
        return ""
    lines = [json.dumps(example, ensure_ascii=False) for _ in range(3)]
    return "\n".join(lines)


def preview_records_html(
    records: List[Dict],
    mode: str,
    max_rows: int = 8,
) -> str:
    """生成字段预览 HTML 表格（最多 max_rows 行）。"""
    if not records:
        return "<p style='color:#9CA3AF;font-size:0.82rem;'>暂无数据，请先输入或上传</p>"

    schema = FIELD_SCHEMA.get(mode, {})
    required = schema.get("required", [])
    optional = schema.get("optional", [])
    show_fields = required + optional

    # 取实际出现的字段（按优先顺序）
    actual_fields = [f for f in show_fields if any(f in r for r in records)]
    extra_fields = [
        f for r in records for f in r if f not in show_fields and f not in actual_fields
    ]
    # 去重
    seen = set()
    all_fields = []
    for f in actual_fields + extra_fields:
        if f not in seen:
            all_fields.append(f)
            seen.add(f)

    # 构造表头
    header_cells = "".join(
        f"<th style='padding:5px 10px;text-align:left;font-size:0.78rem;"
        f"color:#1D4ED8;background:#EFF6FF;white-space:nowrap;'>"
        f"{'★ ' if f in required else ''}{f}</th>"
        for f in all_fields
    )

    # 构造数据行
    display_records = records[:max_rows]
    row_html = ""
    for i, rec in enumerate(display_records):
        bg = "white" if i % 2 == 0 else "#F9FAFB"
        cells = "".join(
            f"<td style='padding:5px 10px;font-size:0.78rem;color:#374151;"
            f"max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>"
            f"{_truncate(str(rec.get(f, '—')), 60)}</td>"
            for f in all_fields
        )
        row_html += f"<tr style='background:{bg};'>{cells}</tr>"

    total = len(records)
    more_hint = (
        f"<p style='font-size:0.75rem;color:#6B7280;margin-top:4px;'>"
        f"共 {total} 条，显示前 {max_rows} 条</p>"
        if total > max_rows
        else f"<p style='font-size:0.75rem;color:#6B7280;margin-top:4px;'>共 {total} 条</p>"
    )

    return f"""
<div style="overflow-x:auto;">
  <table style="width:100%;border-collapse:collapse;border:1px solid #DBEAFE;border-radius:6px;overflow:hidden;">
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{row_html}</tbody>
  </table>
  {more_hint}
</div>"""


def _truncate(s: str, n: int) -> str:
    return s[:n] + "…" if len(s) > n else s


def save_records_to_file(
    records: List[Dict],
    mode: str,
    filename: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """将记录保存为 JSONL 文件到 uploads 目录。

    Returns:
        (saved_path, error_msg)
    """
    try:
        uploads = _ensure_uploads_dir()
        if not filename:
            import time
            filename = f"custom_{mode}_{int(time.time())}.jsonl"
        out_path = uploads / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return str(out_path), None
    except Exception as e:
        return "", f"保存失败: {e}"


def build_dataset_override(mode: str, file_path: str) -> Dict[str, str]:
    """根据 mode 和文件路径生成 Hydra override dict。

    Returns 示例:
      inject -> {"data/datasets@data.train": "Custom_inject"}
      unlearn -> {"data/datasets@data.forget": "Custom_unlearn_forget",
                  "data/datasets@data.retain": "Custom_unlearn_retain"}
      edit   -> {"data/datasets@data.edit": "Custom_edit"}
    """
    overrides: Dict[str, str] = {}
    if mode == "inject":
        overrides["data/datasets@data.train"] = "Custom_inject"
        overrides["__custom_data_path_train__"] = file_path
    elif mode == "edit":
        overrides["data/datasets@data.edit"] = "Custom_edit"
        overrides["__custom_data_path_edit__"] = file_path
    elif mode == "unlearn":
        overrides["data/datasets@data.forget"] = "Custom_unlearn_forget"
        overrides["data/datasets@data.retain"] = "Custom_unlearn_retain"
        overrides["__custom_data_path_forget__"] = file_path
    return overrides


def single_record_to_jsonl(mode: str, **kwargs) -> Tuple[List[Dict], Optional[str]]:
    """将用户单条输入的字段值构造为一条记录，并做基础校验。"""
    schema = FIELD_SCHEMA.get(mode, {})
    required = schema.get("required", [])
    optional = schema.get("optional", [])

    rec: Dict[str, Any] = {}
    for f in required + optional:
        val = kwargs.get(f, "")
        if val and str(val).strip():
            rec[f] = str(val).strip()

    missing = [f for f in required if f not in rec]
    if missing:
        return [], f"必填字段缺失：{', '.join(missing)}"

    return [rec], None
