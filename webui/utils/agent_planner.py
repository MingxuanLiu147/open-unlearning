# -*- coding: utf-8 -*-
"""
Agent Planner — 结构化建议生成与校验
========================================

定义 Assistant 输出合同 (Output Contract)，
基于 skills 上下文构建 system prompt，
从 Agent 回复中提取 JSON 建议并校验字段完整性。
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

from utils.skills_context import (
    build_skills_context,
    get_valid_skill_ids,
    get_skill_config_by_id,
)
from utils.i18n import get_language


# ────────────────────────────────────────────────
# Output Contract
# ────────────────────────────────────────────────

REQUIRED_FIELDS = [
    "mode",
    "recommended_skill",
    "recommended_method",
    "recommended_model",
    "data_plan",
    "eval_plan",
    "core_overrides",
    "reasoning_summary",
    "risk_notes",
]

OPTIONAL_FIELDS = [
    "ui_actions",
    "dataset_overrides",
    "parameter_overrides",
    "warnings",
]

VALID_MODES = ["unlearn", "inject", "edit"]


@dataclass
class Recommendation:
    """Agent 生成的结构化建议对象。"""
    mode: str = ""
    recommended_skill: str = ""
    recommended_method: str = ""
    recommended_model: str = ""
    data_plan: str = ""
    eval_plan: str = ""
    core_overrides: Dict[str, Any] = field(default_factory=dict)
    reasoning_summary: str = ""
    risk_notes: str = ""
    # optional
    ui_actions: List[Dict[str, Any]] = field(default_factory=list)
    dataset_overrides: Dict[str, Any] = field(default_factory=dict)
    parameter_overrides: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    # parse metadata
    _raw_json: Optional[str] = field(default=None, repr=False)
    _parse_errors: List[str] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_raw_json", None)
        d.pop("_parse_errors", None)
        return d

    @property
    def is_valid(self) -> bool:
        return bool(self.mode and self.mode in VALID_MODES and not self._parse_errors)

    def get_apply_config(self) -> Dict[str, Any]:
        """将建议转换为可直接传给 apply 流程的配置字典。"""
        cfg: Dict[str, Any] = {}
        if self.mode:
            cfg["mode"] = self.mode

        skill_cfg = get_skill_config_by_id(self.recommended_skill)
        if skill_cfg:
            cfg.update(skill_cfg)

        if self.recommended_model:
            cfg["model"] = self.recommended_model
        if self.recommended_method:
            cfg["trainer"] = self.recommended_method

        if self.core_overrides:
            for k, v in self.core_overrides.items():
                if k in ("learning_rate", "num_epochs", "batch_size",
                         "gradient_accumulation", "max_length", "warmup_ratio"):
                    cfg[k] = v

        if self.parameter_overrides:
            cfg.update(self.parameter_overrides)

        return cfg


# ────────────────────────────────────────────────
# System prompt 构建
# ────────────────────────────────────────────────

_CONTRACT_SPEC = """
You MUST output a JSON object with exactly these fields:
{
  "mode": "unlearn | inject | edit",
  "recommended_skill": "<skill id from the library above>",
  "recommended_method": "<trainer name, e.g. SimNPO, inject/LoRA, edit/ROME>",
  "recommended_model": "<model name>",
  "data_plan": "<brief data preparation guidance>",
  "eval_plan": "<recommended evaluation suite>",
  "core_overrides": { "<hydra override key>": "<value>", ... },
  "reasoning_summary": "<why this recommendation>",
  "risk_notes": "<potential risks and mitigations>"
}
Wrap the JSON in ```json ... ``` fences.
""".strip()


def build_system_prompt(
    lang: str = "zh",
    goal_hint: Optional[str] = None,
) -> str:
    """构建包含 skills 知识库的 system prompt。"""
    skills_text = build_skills_context(lang=lang, goal_filter=goal_hint)

    if lang == "zh":
        role = (
            "你是 Know-Surgery 智能助手，专门帮助用户配置大模型知识更新实验。\n"
            "Know-Surgery 支持三种操作：Unlearn（知识遗忘）、Inject（知识注入）、Edit（知识编辑）。\n"
            "你要根据用户的目标描述，从下方的 Skill 模板库中选择最合适的模板，\n"
            "给出完整的结构化配置建议，并解释你的推荐理由。\n"
            "请用中文回复。\n"
        )
    else:
        role = (
            "You are the Know-Surgery assistant that helps users configure LLM knowledge surgery experiments.\n"
            "Know-Surgery supports three operations: Unlearn, Inject, and Edit.\n"
            "Based on the user's goal, select the best Skill template from the library below\n"
            "and provide a structured configuration recommendation with reasoning.\n"
        )

    return f"{role}\n---\n## Skill Template Library\n\n{skills_text}\n\n---\n## Output Format\n\n{_CONTRACT_SPEC}"


# ────────────────────────────────────────────────
# JSON 提取与校验
# ────────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def extract_json_from_text(text: str) -> Optional[str]:
    """从 Agent 回复中提取第一个 JSON 块。"""
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    for m in _BARE_JSON_RE.finditer(text):
        candidate = m.group(0)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    return None


def parse_recommendation(text: str) -> Recommendation:
    """从 Agent 原始回复文本解析出 Recommendation。"""
    raw_json = extract_json_from_text(text)
    if raw_json is None:
        rec = Recommendation()
        rec._raw_json = None
        rec._parse_errors = ["No JSON block found in agent response"]
        return rec

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        rec = Recommendation()
        rec._raw_json = raw_json
        rec._parse_errors = [f"JSON decode error: {e}"]
        return rec

    if not isinstance(data, dict):
        rec = Recommendation()
        rec._raw_json = raw_json
        rec._parse_errors = ["Parsed value is not a JSON object"]
        return rec

    errors: List[str] = []

    mode = data.get("mode", "")
    if mode not in VALID_MODES:
        errors.append(f"Invalid mode: {mode!r}")

    rec = Recommendation(
        mode=data.get("mode", ""),
        recommended_skill=data.get("recommended_skill", ""),
        recommended_method=data.get("recommended_method", ""),
        recommended_model=data.get("recommended_model", ""),
        data_plan=data.get("data_plan", ""),
        eval_plan=data.get("eval_plan", ""),
        core_overrides=data.get("core_overrides", {}),
        reasoning_summary=data.get("reasoning_summary", ""),
        risk_notes=data.get("risk_notes", ""),
        ui_actions=data.get("ui_actions", []),
        dataset_overrides=data.get("dataset_overrides", {}),
        parameter_overrides=data.get("parameter_overrides", {}),
        warnings=data.get("warnings", []),
    )
    rec._raw_json = raw_json
    rec._parse_errors = errors
    return rec


# ────────────────────────────────────────────────
# 建议卡片渲染
# ────────────────────────────────────────────────

def render_recommendation_card(rec: Recommendation, lang: str = "zh") -> str:
    """将 Recommendation 渲染为 HTML 建议卡片。"""
    if not rec.is_valid:
        err_detail = "; ".join(rec._parse_errors) if rec._parse_errors else "unknown"
        return f"<p style='color:#EF4444;font-size:0.85rem;'>Parse error: {err_detail}</p>"

    from utils.i18n import t

    def _row(label_key: str, value: str) -> str:
        label = t(label_key)
        return (
            f"<tr>"
            f"<td style='padding:5px 10px;color:#475569;font-size:0.82rem;white-space:nowrap;'>{label}</td>"
            f"<td style='padding:5px 10px;font-weight:600;color:#0F172A;font-size:0.82rem;'>{value}</td>"
            f"</tr>"
        )

    rows = "".join([
        _row("assistant_card_mode", rec.mode),
        _row("assistant_card_skill", rec.recommended_skill),
        _row("assistant_card_method", rec.recommended_method),
        _row("assistant_card_model", rec.recommended_model),
        _row("assistant_card_eval", rec.eval_plan),
        _row("assistant_card_data", rec.data_plan),
    ])

    overrides_html = ""
    if rec.core_overrides:
        ov_rows = "".join(
            f"<span style='display:inline-block;padding:2px 8px;border-radius:4px;"
            f"background:#EEF3FB;color:#1D3FDB;font-size:0.75rem;margin:2px;font-family:monospace;'>"
            f"{k}={v}</span>"
            for k, v in rec.core_overrides.items()
        )
        overrides_label = t("assistant_card_overrides")
        overrides_html = f"<div style='margin-top:8px;'><strong style='font-size:0.78rem;color:#355CFF;'>{overrides_label}</strong><br/>{ov_rows}</div>"

    reasoning_label = t("assistant_card_reasoning")
    risk_label = t("assistant_card_risk")

    return f"""
<div style="border:1px solid #D7E0F0;border-radius:10px;overflow:hidden;background:white;">
  <div style="background:#355CFF;color:white;padding:8px 14px;font-weight:700;font-size:0.875rem;">
    {t("assistant_recommendation_title")}
  </div>
  <div style="padding:12px;">
    <table style="width:100%;border-collapse:collapse;">{rows}</table>
    {overrides_html}
    <div style="margin-top:10px;padding:8px;background:#F8FAFC;border-radius:6px;">
      <strong style="font-size:0.78rem;color:#334155;">{reasoning_label}</strong>
      <p style="font-size:0.82rem;color:#475569;margin:4px 0 0 0;">{rec.reasoning_summary}</p>
    </div>
    <div style="margin-top:8px;padding:8px;background:#FFF7ED;border-radius:6px;border-left:3px solid #F59E0B;">
      <strong style="font-size:0.78rem;color:#92400E;">{risk_label}</strong>
      <p style="font-size:0.82rem;color:#78350F;margin:4px 0 0 0;">{rec.risk_notes}</p>
    </div>
  </div>
</div>"""


def render_diff_html(
    current: Dict[str, Any],
    recommended: Dict[str, Any],
    lang: str = "zh",
) -> str:
    """渲染当前配置 vs 建议配置的 diff 表格。"""
    from utils.i18n import t

    all_keys = sorted(set(list(current.keys()) + list(recommended.keys())))
    field_label = t("assistant_diff_field")
    current_label = t("assistant_diff_current")
    rec_label = t("assistant_diff_recommended")

    rows = ""
    for k in all_keys:
        cur_val = current.get(k, "—")
        rec_val = recommended.get(k, "—")
        if str(cur_val) == str(rec_val):
            continue
        rows += (
            f"<tr>"
            f"<td style='padding:4px 8px;font-family:monospace;font-size:0.78rem;color:#334155;'>{k}</td>"
            f"<td style='padding:4px 8px;font-size:0.82rem;color:#94A3B8;'>{cur_val}</td>"
            f"<td style='padding:4px 8px;font-size:0.82rem;color:#355CFF;font-weight:600;'>{rec_val}</td>"
            f"</tr>"
        )

    if not rows:
        no_diff = "无差异" if lang == "zh" else "No differences"
        return f"<p style='color:#94A3B8;font-size:0.82rem;'>{no_diff}</p>"

    return f"""
<table style="width:100%;border-collapse:collapse;border:1px solid #D7E0F0;border-radius:8px;overflow:hidden;">
  <thead>
    <tr style="background:#EEF3FB;">
      <th style="padding:6px 8px;text-align:left;font-size:0.75rem;color:#355CFF;">{field_label}</th>
      <th style="padding:6px 8px;text-align:left;font-size:0.75rem;color:#94A3B8;">{current_label}</th>
      <th style="padding:6px 8px;text-align:left;font-size:0.75rem;color:#355CFF;">{rec_label}</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>"""


# ────────────────────────────────────────────────
# Phase 7B: 建议分组视图与风险提醒
# ────────────────────────────────────────────────

def render_grouped_recommendation(rec: Recommendation, lang: str = "zh") -> str:
    """渲染分组视图的建议卡片（基础配置 / 参数 / 风险 三组）。"""
    if not rec.is_valid:
        return render_recommendation_card(rec, lang)

    from utils.i18n import t

    # 基础配置组
    basic_rows = "".join([
        f"<tr><td style='padding:4px 8px;color:#475569;font-size:0.82rem;'>{t('assistant_card_mode')}</td>"
        f"<td style='padding:4px 8px;font-weight:600;color:#0F172A;'>{rec.mode}</td></tr>",
        f"<tr><td style='padding:4px 8px;color:#475569;font-size:0.82rem;'>{t('assistant_card_skill')}</td>"
        f"<td style='padding:4px 8px;font-weight:600;color:#0F172A;'>{rec.recommended_skill}</td></tr>",
        f"<tr><td style='padding:4px 8px;color:#475569;font-size:0.82rem;'>{t('assistant_card_method')}</td>"
        f"<td style='padding:4px 8px;font-weight:600;color:#0F172A;'>{rec.recommended_method}</td></tr>",
        f"<tr><td style='padding:4px 8px;color:#475569;font-size:0.82rem;'>{t('assistant_card_model')}</td>"
        f"<td style='padding:4px 8px;font-weight:600;color:#0F172A;'>{rec.recommended_model}</td></tr>",
    ])

    # 参数组
    param_items = ""
    if rec.core_overrides:
        for k, v in rec.core_overrides.items():
            param_items += (
                f"<span style='display:inline-block;padding:2px 8px;border-radius:4px;"
                f"background:#EEF3FB;color:#1D3FDB;font-size:0.75rem;margin:2px;font-family:monospace;'>"
                f"{k}={v}</span>"
            )

    # 风险组
    risk_html = ""
    if rec.risk_notes:
        risk_html = f"<p style='font-size:0.82rem;color:#78350F;'>{rec.risk_notes}</p>"
    if rec.warnings:
        for w in rec.warnings:
            risk_html += f"<p style='font-size:0.78rem;color:#92400E;'>⚠ {w}</p>"

    return f"""
<div style="border:1px solid #D7E0F0;border-radius:10px;overflow:hidden;background:white;">
  <div style="background:#355CFF;color:white;padding:8px 14px;font-weight:700;font-size:0.875rem;">
    {t("assistant_recommendation_title")}
  </div>
  <div style="padding:12px;">
    <div style="margin-bottom:10px;">
      <div style="font-size:0.75rem;font-weight:600;color:#355CFF;text-transform:uppercase;margin-bottom:4px;">基础配置</div>
      <table style="width:100%;border-collapse:collapse;">{basic_rows}</table>
    </div>
    <div style="margin-bottom:10px;">
      <div style="font-size:0.75rem;font-weight:600;color:#355CFF;text-transform:uppercase;margin-bottom:4px;">核心参数</div>
      <div>{param_items or '<span style="color:#94A3B8;font-size:0.78rem;">使用默认参数</span>'}</div>
    </div>
    <div style="margin-bottom:10px;padding:8px;background:#F8FAFC;border-radius:6px;">
      <div style="font-size:0.75rem;font-weight:600;color:#334155;margin-bottom:4px;">推荐理由</div>
      <p style="font-size:0.82rem;color:#475569;margin:0;">{rec.reasoning_summary}</p>
    </div>
    <div style="padding:8px;background:#FFF7ED;border-radius:6px;border-left:3px solid #F59E0B;">
      <div style="font-size:0.75rem;font-weight:600;color:#92400E;margin-bottom:4px;">⚠ 风险提醒</div>
      {risk_html or '<p style="font-size:0.78rem;color:#92400E;">无特殊风险</p>'}
    </div>
  </div>
</div>"""
