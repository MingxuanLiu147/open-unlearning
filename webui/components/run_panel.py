"""
运行面板组件
============

右侧运行区，包含命令预览、运行控制、日志输出、结果展示。
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# 添加 webui 目录到路径
webui_dir = Path(__file__).parent.parent
if str(webui_dir) not in sys.path:
    sys.path.insert(0, str(webui_dir))

import gradio as gr
from utils.result_parser import ResultParser
from utils.i18n import t, get_language


def create_run_panel() -> Dict[str, Any]:
    """创建运行面板
    
    Returns:
        组件字典
    """
    components = {}
    
    with gr.Column(scale=1):
        gr.Markdown(f"## {t('run_title')}")
        
        # 运行控制
        with gr.Group():
            with gr.Row():
                components["run_btn"] = gr.Button(
                    t("run_btn"),
                    variant="primary",
                    scale=2
                )
                components["stop_btn"] = gr.Button(
                    t("stop_btn"),
                    variant="stop",
                    scale=1,
                    interactive=False
                )
            
            # 运行状态
            components["status"] = gr.Markdown(
                value=t("status_ready"),
                visible=True
            )
        
        # 命令预览
        with gr.Group():
            gr.Markdown(f"### {t('command_preview_title')}")
            components["command_preview"] = gr.Textbox(
                value="# 选择配置后生成命令",
                label=t("command_preview_label"),
                lines=8,
                interactive=False
            )
            components["copy_btn"] = gr.Button(t("copy_command"), size="sm")
        
        # GPU 选择
        with gr.Group():
            gr.Markdown(f"### {t('env_config_title')}")
            components["cuda_devices"] = gr.Textbox(
                value="0",
                label=t("cuda_devices_label"),
                info=t("cuda_devices_info")
            )
        
        # 实时日志
        with gr.Group():
            gr.Markdown(f"### {t('log_title')}")
            components["log_output"] = gr.Textbox(
                value="",
                label=t("log_label"),
                lines=20,
                max_lines=30,
                interactive=False,
                autoscroll=True
            )
            with gr.Row():
                components["clear_log_btn"] = gr.Button(t("clear_log"), size="sm")
                components["scroll_btn"] = gr.Button(t("scroll_bottom"), size="sm")
        
        # 结果展示区
        with gr.Group():
            gr.Markdown(f"### {t('result_title')}")
            components["result_summary"] = gr.HTML(
                value=f"<p style='color: #888;'>{t('result_placeholder')}</p>"
            )
            with gr.Row():
                components["output_dir"] = gr.Textbox(
                    value="",
                    label=t("output_dir_label"),
                    interactive=False,
                    scale=3
                )
                components["load_results_btn"] = gr.Button(t("load_results"), size="sm", scale=1)
    
    return components


def generate_command_preview(
    mode: str,
    model: str = None,
    trainer: str = None,
    experiment: str = None,
    task_name: str = None,
    seed: int = None,
    eval_suite: str = None,
    saved_model_path: str = None,
    overrides: dict = None,
) -> str:
    """生成命令预览
    
    根据当前配置生成等价的 CLI 命令
    """
    # eval 模式使用 eval.py
    if mode == "eval":
        cmd_parts = [
            "python src/eval.py",
            "--config-name=eval.yaml",
        ]
        
        # 添加评测套件
        if eval_suite:
            cmd_parts.append(f"eval={eval_suite}")
        
        # 添加模型路径或模型配置
        if saved_model_path:
            cmd_parts.append(f"model.model_args.pretrained_model_name_or_path={saved_model_path}")
        elif model:
            cmd_parts.append(f"model={model}")
        
        # 添加任务名称
        if task_name:
            cmd_parts.append(f"task_name={task_name}")
        
        return " \\\n    ".join(cmd_parts)
    
    # 训练模式使用 train.py
    cmd_parts = [
        "python src/train.py",
        f"--config-name={mode}.yaml",
    ]
    
    # 添加实验模板（如果不是"无模板"）
    if experiment and not experiment.startswith("("):
        cmd_parts.append(f"experiment={experiment}")
    
    # 添加模型
    if model:
        cmd_parts.append(f"model={model}")
    
    # 添加方法
    if trainer:
        cmd_parts.append(f"trainer={trainer}")
    
    # 添加评测套件（训练模式也需要指定 eval）
    if eval_suite:
        cmd_parts.append(f"eval={eval_suite}")
    
    # 添加任务名称和种子
    if task_name:
        cmd_parts.append(f"task_name={task_name}")
    if seed is not None:
        cmd_parts.append(f"trainer.args.seed={int(seed)}")
    
    # 添加额外 overrides（高级参数和数据集配置）
    if overrides:
        for key, value in overrides.items():
            if value is not None and value != "":
                cmd_parts.append(f"{key}={value}")
    
    # 格式化为多行
    return " \\\n    ".join(cmd_parts)


def update_status(running: bool, exit_code: int = None) -> Tuple[gr.update, gr.update, gr.update]:
    """更新运行状态（支持 i18n）
    
    Returns:
        (status_update, run_btn_update, stop_btn_update)
    """
    if running:
        status = t("status_running")
        run_interactive = False
        stop_interactive = True
    elif exit_code is not None:
        if exit_code == 0:
            status = t("status_success")
        else:
            status = t("status_failed", exit_code=exit_code)
        run_interactive = True
        stop_interactive = False
    else:
        status = t("status_ready")
        run_interactive = True
        stop_interactive = False
    
    return (
        gr.update(value=status),
        gr.update(interactive=run_interactive),
        gr.update(interactive=stop_interactive),
    )


def load_eval_results(output_dir: str) -> str:
    """加载评估结果并渲染为 HTML
    
    Args:
        output_dir: 输出目录路径
        
    Returns:
        HTML 格式的结果展示
    """
    if not output_dir:
        return "<p style='color: #888;'>请先指定输出目录</p>"
    
    results = ResultParser.parse_results(output_dir)
    
    if not results:
        return f"<p style='color: #888;'>在 {output_dir} 中未找到评估结果</p>"
    
    return ResultParser.render_metrics_html(results)
