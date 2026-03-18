"""
结果对比页面（Tab 2）
=====================

扫描 saves/ 目录下所有含评估结果的 checkpoint run，
支持用户选择多个 run 进行横向指标对比。
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
    """创建结果对比 Tab 的所有组件。

    Returns:
        组件字典
    """
    components = {}

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(f'<div class="ks-col-title">{t("compare_select_title")}</div>')

            # 刷新按钮
            components["refresh_btn"] = gr.Button(t("refresh_runs"), size="sm", variant="secondary")

            # 多选 Run
            initial_runs = list(config_loader.get_eval_runs().keys())
            components["run_selector"] = gr.CheckboxGroup(
                choices=initial_runs,
                value=[],
                label=t("run_selector_label"),
                info=t("run_selector_info"),
            )

            components["compare_btn"] = gr.Button(t("generate_compare"), variant="primary")

        with gr.Column(scale=3):
            gr.HTML(f'<div class="ks-col-title">{t("compare_output_title")}</div>')
            components["compare_output"] = gr.HTML(
                value=f"<p style='color:#888;padding:20px;'>{t('compare_placeholder')}</p>"
            )

    return components


def bind_results_compare_events(
    components: Dict[str, Any],
    config_loader: ConfigLoader,
) -> None:
    """绑定 Tab 2 的事件处理。

    Args:
        components: create_results_compare_tab 返回的组件字典
        config_loader: 配置加载器
    """

    def refresh_runs():
        runs = list(config_loader.get_eval_runs().keys())
        return gr.update(choices=runs, value=[])

    def do_compare(selected_labels: List[str]):
        if not selected_labels:
            return "<p style='color:#888;padding:20px;'>请先勾选至少一个 Run</p>"

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

        return ResultParser.render_compare_html(run_results)

    components["refresh_btn"].click(
        fn=refresh_runs,
        inputs=[],
        outputs=[components["run_selector"]],
    )

    components["compare_btn"].click(
        fn=do_compare,
        inputs=[components["run_selector"]],
        outputs=[components["compare_output"]],
    )
