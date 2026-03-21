"""
结果对比页面（Tab 2） — Phase 8 重构版
========================================

新增筛选区（模式/评测类型）、对比表格、指标方向标注。
"""

import gradio as gr
from typing import Dict, Any, List
import sys
from pathlib import Path

webui_dir = Path(__file__).parent.parent
if str(webui_dir) not in sys.path:
    sys.path.insert(0, str(webui_dir))

from utils.config_loader import ConfigLoader
from utils.result_parser import ResultParser
from utils.i18n import t


def create_results_compare_tab(config_loader: ConfigLoader) -> Dict[str, Any]:
    components: Dict[str, Any] = {}

    with gr.Row(equal_height=False):
        # ── 左栏：筛选区 ──
        with gr.Column(scale=1, min_width=240):
            gr.HTML(f'<div class="ks-col-title">{t("compare_select_title")}</div>')

            components["filter_mode"] = gr.Dropdown(
                choices=["all", "unlearn", "inject", "edit"],
                value="all",
                label=t("compare_filter_mode") if "compare_filter_mode" in {} else "按模式筛选",
            )

            components["refresh_btn"] = gr.Button(
                t("refresh_runs"), size="sm", variant="secondary"
            )

            initial_runs = list(config_loader.get_eval_runs().keys())
            components["run_selector"] = gr.CheckboxGroup(
                choices=initial_runs,
                value=[],
                label=t("run_selector_label"),
                info=t("run_selector_info"),
            )

            components["compare_btn"] = gr.Button(
                t("generate_compare"), variant="primary"
            )

            gr.HTML(
                '<div style="margin-top:12px;padding:8px;background:#EEF3FB;border-radius:6px;">'
                '<p style="font-size:0.72rem;color:#355CFF;margin:0;">↑ higher is better &nbsp;·&nbsp; ↓ lower is better</p>'
                '<p style="font-size:0.72rem;color:#355CFF;margin:2px 0 0 0;">最佳值用 <span style="background:#DCFCE7;padding:1px 4px;border-radius:3px;">绿色</span> 标记</p>'
                '</div>'
            )

        # ── 右栏：对比结果 ──
        with gr.Column(scale=3):
            gr.HTML(f'<div class="ks-col-title">{t("compare_output_title")}</div>')

            components["summary_html"] = gr.HTML(value="")

            components["compare_output"] = gr.HTML(
                value=f"<p style='color:#94A3B8;padding:20px;'>{t('compare_placeholder')}</p>"
            )

    return components


def bind_results_compare_events(
    components: Dict[str, Any],
    config_loader: ConfigLoader,
) -> None:

    def refresh_runs(mode_filter):
        runs = config_loader.get_eval_runs()
        if mode_filter and mode_filter != "all":
            runs = {
                k: v for k, v in runs.items()
                if mode_filter in k.lower()
            }
        labels = list(runs.keys())
        return gr.update(choices=labels, value=[])

    def do_compare(selected_labels: List[str]):
        if not selected_labels:
            return "", "<p style='color:#94A3B8;padding:20px;'>请先勾选至少一个 Run</p>"

        all_runs = config_loader.get_eval_runs()
        run_results: Dict[str, List] = {}

        for label in selected_labels:
            info = all_runs.get(label)
            if not info:
                continue
            results = []
            for sf in info["summary_files"]:
                r = ResultParser.parse_summary_file(sf)
                if r:
                    results.append(r)
            if results:
                run_results[label] = results

        if not run_results:
            return "", "<p style='color:#94A3B8;padding:20px;'>所选 Run 中没有找到评估结果</p>"

        total_runs = len(run_results)
        total_metrics = sum(
            len(r.metrics) for results in run_results.values() for r in results
        )
        summary = (
            f"<div style='padding:8px 12px;background:#EEF3FB;border-radius:8px;margin-bottom:12px;'>"
            f"<span style='font-size:0.82rem;color:#355CFF;font-weight:600;'>"
            f"对比 {total_runs} 个实验，共 {total_metrics} 个指标</span></div>"
        )

        compare_html = ResultParser.render_compare_html(run_results)
        return summary, compare_html

    components["refresh_btn"].click(
        fn=refresh_runs,
        inputs=[components["filter_mode"]],
        outputs=[components["run_selector"]],
    )

    components["compare_btn"].click(
        fn=do_compare,
        inputs=[components["run_selector"]],
        outputs=[components["summary_html"], components["compare_output"]],
    )
