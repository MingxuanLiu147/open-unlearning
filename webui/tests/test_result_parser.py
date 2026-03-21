# -*- coding: utf-8 -*-
"""result_parser 单测（含 metric direction 校验）"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.result_parser import ResultParser, EvalResult


def test_metric_direction_defined():
    """所有 METRIC_DESCRIPTIONS 中的指标都应有 direction 定义。"""
    for metric in ResultParser.METRIC_DESCRIPTIONS:
        assert metric in ResultParser.METRIC_DIRECTION, (
            f"Metric {metric!r} is in DESCRIPTIONS but missing from DIRECTION"
        )


def test_get_best_value_higher():
    best = ResultParser.get_best_value([0.5, 0.8, 0.3], "model_utility")
    assert best == 0.8


def test_get_best_value_lower():
    best = ResultParser.get_best_value([0.5, 0.2, 0.9], "privleak")
    assert best == 0.2


def test_get_best_value_with_none():
    best = ResultParser.get_best_value([None, 0.5, None, 0.3], "forget_quality")
    assert best == 0.5


def test_get_best_value_all_none():
    best = ResultParser.get_best_value([None, None], "reliability")
    assert best is None


def test_format_metric_value_float():
    assert "0.1235" in ResultParser.format_metric_value(0.12345)


def test_format_metric_value_small():
    formatted = ResultParser.format_metric_value(0.001)
    assert "e" in formatted.lower()


def test_format_metric_value_string():
    assert ResultParser.format_metric_value("hello") == "hello"


def test_render_compare_html_empty():
    html = ResultParser.render_compare_html({})
    assert "请先选择" in html


def test_render_compare_html_single_run():
    runs = {
        "run1": [
            EvalResult(name="TOFU", metrics={"forget_quality": 0.85, "model_utility": 0.72}, file_path="/tmp/f")
        ]
    }
    html = ResultParser.render_compare_html(runs)
    assert "TOFU" in html
    assert "0.85" in html
    assert "↑" in html or "↓" in html


def test_render_compare_html_multi_run():
    runs = {
        "run1": [EvalResult(name="TOFU", metrics={"forget_quality": 0.85}, file_path="/tmp/a")],
        "run2": [EvalResult(name="TOFU", metrics={"forget_quality": 0.90}, file_path="/tmp/b")],
    }
    html = ResultParser.render_compare_html(runs)
    assert "run1" in html or "a" in html
    assert "#DCFCE7" in html


if __name__ == "__main__":
    test_metric_direction_defined()
    test_get_best_value_higher()
    test_get_best_value_lower()
    test_get_best_value_with_none()
    test_get_best_value_all_none()
    test_format_metric_value_float()
    test_format_metric_value_small()
    test_format_metric_value_string()
    test_render_compare_html_empty()
    test_render_compare_html_single_run()
    test_render_compare_html_multi_run()
    print("All result_parser tests passed!")
