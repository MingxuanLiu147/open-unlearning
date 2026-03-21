# -*- coding: utf-8 -*-
"""
Skills 注入与上下文装配层
==========================

将已有的 skill 模板序列化为大模型可理解的上下文片段，
让 Agent 基于 skills 知识库自主完成智能匹配和建议生成。

供 agent_planner 的 system prompt 构建使用。
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from utils.skill_schema import SkillLoader, Skill

_loader = SkillLoader()


def load_skills(force_reload: bool = False) -> List[Skill]:
    return _loader.load_all(force_reload=force_reload)


def skill_to_prompt_block(skill: Skill, lang: str = "zh") -> str:
    """将单个 Skill 序列化为 Agent 可读的文本块。"""
    name = skill.get_display_name(lang)
    desc = skill.get_description(lang)
    tags = ", ".join(skill.get_tags(lang))

    hints = skill.prompt_hints_en if lang == "en" and skill.prompt_hints_en else skill.prompt_hints
    constraints = skill.constraints_en if lang == "en" and skill.constraints_en else skill.constraints
    risks = skill.risk_notes_en if lang == "en" and skill.risk_notes_en else skill.risk_notes
    resource = skill.resource_estimate_en if lang == "en" and skill.resource_estimate_en else skill.resource_estimate

    lines = [
        f"### Skill: {name}  (id={skill.id}, goal={skill.goal})",
        f"Description: {desc}",
        f"Tags: {tags}",
        f"Recommended eval: {skill.recommended_eval}",
        f"Resource estimate: {resource}",
        f"Recommended models: {', '.join(skill.recommended_models)}",
        f"Config: {json.dumps(skill.config, ensure_ascii=False)}",
        f"Core overrides: {json.dumps(skill.core_overrides, ensure_ascii=False)}",
    ]
    if hints:
        lines.append(f"When to recommend: {hints}")
    if constraints:
        lines.append(f"Constraints: {'; '.join(constraints)}")
    if risks:
        lines.append(f"Risks: {'; '.join(risks)}")

    return "\n".join(lines)


def build_skills_context(lang: str = "zh", goal_filter: Optional[str] = None) -> str:
    """构建完整的 skills 知识库上下文文本。

    Args:
        lang: 语言 zh/en
        goal_filter: 若提供，只返回指定 goal 的 skills

    Returns:
        供注入 system prompt 的多段 Skill 描述文本
    """
    skills = load_skills()
    if goal_filter:
        skills = [s for s in skills if s.goal == goal_filter]

    if not skills:
        return "(No skill templates available)"

    header = (
        "Below is the complete Know-Surgery skill template library. "
        "Use these skills as the basis for your recommendations.\n"
    )
    blocks = [skill_to_prompt_block(s, lang) for s in skills]
    return header + "\n\n".join(blocks)


def build_skills_summary_json(lang: str = "zh") -> List[Dict[str, Any]]:
    """返回 skills 摘要的结构化列表，用于前端或校验。"""
    skills = load_skills()
    result = []
    for s in skills:
        result.append({
            "id": s.id,
            "name": s.get_display_name(lang),
            "goal": s.goal,
            "trainer": s.config.get("trainer", ""),
            "model": s.config.get("model", ""),
            "eval": s.recommended_eval,
            "tags": s.get_tags(lang),
        })
    return result


def get_valid_skill_ids() -> List[str]:
    """返回所有合法的 skill id，用于校验 Agent 输出。"""
    return [s.id for s in load_skills()]


def get_skill_config_by_id(skill_id: str) -> Optional[Dict[str, Any]]:
    """根据 skill id 获取其完整 config 字段。"""
    skill = _loader.get_by_id(skill_id)
    if skill is None:
        return None
    return skill.config.copy()
