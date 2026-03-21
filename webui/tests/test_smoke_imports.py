# -*- coding: utf-8 -*-
"""Smoke test: 确保所有关键模块可以成功 import。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_import_skills_context():
    from utils.skills_context import build_skills_context, get_valid_skill_ids
    assert callable(build_skills_context)
    assert callable(get_valid_skill_ids)


def test_import_agent_planner():
    from utils.agent_planner import (
        Recommendation, build_system_prompt, parse_recommendation,
        render_recommendation_card, render_diff_html,
        render_grouped_recommendation,
    )
    assert callable(build_system_prompt)
    assert callable(parse_recommendation)


def test_import_agent_provider():
    from utils.agent_provider import (
        AgentConfig, OpenAIProvider, DeepSeekProvider,
        AgentSession, create_provider, create_session,
    )
    assert callable(create_provider)


def test_import_agent_settings():
    from utils.agent_settings import (
        load_settings, save_settings, get_provider_default, test_connection,
    )
    assert callable(load_settings)


def test_import_model_adapter():
    from utils.model_adapter import (
        get_model_meta, supports_task, get_defaults_for_model,
        validate_model_name, get_registered_models,
    )
    assert callable(get_model_meta)


def test_import_result_parser():
    from utils.result_parser import ResultParser, EvalResult
    assert hasattr(ResultParser, "METRIC_DIRECTION")
    assert hasattr(ResultParser, "get_best_value")


def test_import_skill_schema():
    from utils.skill_schema import Skill, SkillLoader, SkillValidator
    assert callable(SkillValidator.validate)


def test_import_i18n():
    from utils.i18n import t, set_language, get_language, TRANSLATIONS
    assert callable(t)
    assert "assistant_sidebar_title" in TRANSLATIONS
    assert "custom_model_label" in TRANSLATIONS


def test_import_data_adapter():
    from utils.data_adapter import (
        parse_jsonl_text, validate_records, get_example_jsonl,
        preview_records_html, save_records_to_file,
    )
    assert callable(parse_jsonl_text)


def test_import_ui_state():
    from utils.ui_state import build_config_dict, apply_config_dict
    assert callable(build_config_dict)


if __name__ == "__main__":
    test_import_skills_context()
    test_import_agent_planner()
    test_import_agent_provider()
    test_import_agent_settings()
    test_import_model_adapter()
    test_import_result_parser()
    test_import_skill_schema()
    test_import_i18n()
    test_import_data_adapter()
    test_import_ui_state()
    print("All smoke import tests passed!")
