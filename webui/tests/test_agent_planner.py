# -*- coding: utf-8 -*-
"""agent_planner 单测"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.agent_planner import (
    Recommendation,
    build_system_prompt,
    extract_json_from_text,
    parse_recommendation,
    render_recommendation_card,
    render_diff_html,
    render_grouped_recommendation,
    REQUIRED_FIELDS,
    VALID_MODES,
)


def test_build_system_prompt_zh():
    prompt = build_system_prompt(lang="zh")
    assert "Know-Surgery" in prompt
    assert "Skill Template Library" in prompt
    assert "JSON" in prompt


def test_build_system_prompt_en():
    prompt = build_system_prompt(lang="en")
    assert "Know-Surgery" in prompt
    assert "mode" in prompt


def test_extract_json_fenced():
    text = 'Some text\n```json\n{"mode": "unlearn"}\n```\nMore text'
    result = extract_json_from_text(text)
    assert result is not None
    data = json.loads(result)
    assert data["mode"] == "unlearn"


def test_extract_json_bare():
    text = 'Here is my recommendation: {"mode": "inject", "recommended_skill": "inject_alpaca"}'
    result = extract_json_from_text(text)
    assert result is not None
    data = json.loads(result)
    assert data["mode"] == "inject"


def test_extract_json_none():
    text = "No JSON here at all"
    assert extract_json_from_text(text) is None


def test_parse_recommendation_valid():
    raw = json.dumps({
        "mode": "unlearn",
        "recommended_skill": "unlearn_tofu",
        "recommended_method": "SimNPO",
        "recommended_model": "Qwen2.5-7B-Instruct",
        "data_plan": "Download TOFU",
        "eval_plan": "tofu",
        "core_overrides": {"trainer.args.learning_rate": "1e-5"},
        "reasoning_summary": "Good balance",
        "risk_notes": "May affect retention",
    })
    text = f"```json\n{raw}\n```"
    rec = parse_recommendation(text)
    assert rec.is_valid
    assert rec.mode == "unlearn"
    assert rec.recommended_skill == "unlearn_tofu"


def test_parse_recommendation_invalid_mode():
    text = '```json\n{"mode": "invalid_mode"}\n```'
    rec = parse_recommendation(text)
    assert not rec.is_valid


def test_parse_recommendation_no_json():
    rec = parse_recommendation("Just plain text with no JSON")
    assert not rec.is_valid
    assert len(rec._parse_errors) > 0


def test_recommendation_to_dict():
    rec = Recommendation(mode="edit", recommended_skill="edit_zsre")
    d = rec.to_dict()
    assert d["mode"] == "edit"
    assert "_raw_json" not in d
    assert "_parse_errors" not in d


def test_recommendation_get_apply_config():
    rec = Recommendation(
        mode="unlearn",
        recommended_skill="unlearn_tofu",
        recommended_method="SimNPO",
        recommended_model="Qwen2.5-7B-Instruct",
        core_overrides={"learning_rate": "2e-5", "num_epochs": 5},
    )
    cfg = rec.get_apply_config()
    assert cfg["mode"] == "unlearn"
    assert cfg.get("learning_rate") == "2e-5"
    assert cfg.get("num_epochs") == 5


def test_render_recommendation_card():
    rec = Recommendation(
        mode="inject",
        recommended_skill="inject_alpaca",
        recommended_method="inject/LoRA",
        recommended_model="Qwen2.5-7B-Instruct",
        reasoning_summary="LoRA is efficient",
        risk_notes="May override knowledge",
    )
    html = render_recommendation_card(rec, "zh")
    assert "inject" in html.lower()
    assert "LoRA" in html


def test_render_grouped_recommendation():
    rec = Recommendation(
        mode="unlearn",
        recommended_skill="unlearn_tofu",
        recommended_method="SimNPO",
        recommended_model="Qwen2.5-7B-Instruct",
        core_overrides={"trainer.args.learning_rate": "1e-5"},
        reasoning_summary="Good balance",
        risk_notes="Check retain set",
    )
    html = render_grouped_recommendation(rec, "zh")
    assert "基础配置" in html
    assert "核心参数" in html
    assert "风险提醒" in html


def test_render_diff_html():
    current = {"mode": "unlearn", "model": "A"}
    recommended = {"mode": "unlearn", "model": "B", "trainer": "SimNPO"}
    html = render_diff_html(current, recommended, "zh")
    assert "model" in html
    assert "B" in html


def test_render_diff_no_changes():
    same = {"mode": "unlearn", "model": "A"}
    html = render_diff_html(same, same, "zh")
    assert "无差异" in html


if __name__ == "__main__":
    test_build_system_prompt_zh()
    test_build_system_prompt_en()
    test_extract_json_fenced()
    test_extract_json_bare()
    test_extract_json_none()
    test_parse_recommendation_valid()
    test_parse_recommendation_invalid_mode()
    test_parse_recommendation_no_json()
    test_recommendation_to_dict()
    test_recommendation_get_apply_config()
    test_render_recommendation_card()
    test_render_grouped_recommendation()
    test_render_diff_html()
    test_render_diff_no_changes()
    print("All agent_planner tests passed!")
