# -*- coding: utf-8 -*-
"""skills_context 单测"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.skills_context import (
    load_skills,
    skill_to_prompt_block,
    build_skills_context,
    build_skills_summary_json,
    get_valid_skill_ids,
    get_skill_config_by_id,
)


def test_load_skills():
    skills = load_skills(force_reload=True)
    assert isinstance(skills, list)
    assert len(skills) >= 1, "至少应有一个 skill 模板"


def test_skill_to_prompt_block():
    skills = load_skills(force_reload=True)
    if skills:
        block = skill_to_prompt_block(skills[0], lang="zh")
        assert "### Skill:" in block
        assert skills[0].id in block


def test_build_skills_context():
    ctx = build_skills_context(lang="zh")
    assert "skill template library" in ctx.lower() or "Skill" in ctx
    assert len(ctx) > 100


def test_build_skills_context_with_filter():
    ctx = build_skills_context(lang="en", goal_filter="unlearn")
    if "No skill" not in ctx:
        assert "unlearn" in ctx.lower()


def test_build_skills_summary_json():
    summary = build_skills_summary_json("zh")
    assert isinstance(summary, list)
    if summary:
        assert "id" in summary[0]
        assert "goal" in summary[0]


def test_get_valid_skill_ids():
    ids = get_valid_skill_ids()
    assert isinstance(ids, list)
    for sid in ids:
        assert isinstance(sid, str) and len(sid) > 0


def test_get_skill_config_by_id():
    ids = get_valid_skill_ids()
    if ids:
        cfg = get_skill_config_by_id(ids[0])
        assert cfg is not None
        assert "mode" in cfg

    assert get_skill_config_by_id("nonexistent_skill_xyz") is None


if __name__ == "__main__":
    test_load_skills()
    test_skill_to_prompt_block()
    test_build_skills_context()
    test_build_skills_context_with_filter()
    test_build_skills_summary_json()
    test_get_valid_skill_ids()
    test_get_skill_config_by_id()
    print("All skills_context tests passed!")
