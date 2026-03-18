"""
配置面板组件
============

左侧配置区，包含 Mode 选择、模型选择、方法选择、数据集选择、评测套件选择。
"""

import gradio as gr
from typing import Dict, Any, List, Tuple
import sys
from pathlib import Path

# 添加 webui 目录到路径
webui_dir = Path(__file__).parent.parent
if str(webui_dir) not in sys.path:
    sys.path.insert(0, str(webui_dir))

from utils.config_loader import ConfigLoader
from utils.ui_state import apply_mode_config
from utils.i18n import t


def create_config_panel(config_loader: ConfigLoader) -> Dict[str, Any]:
    """创建任务配置面板（左栏）。

    不包含外层 gr.Column，由调用方（app.py）的三栏布局管理。

    Args:
        config_loader: 配置加载器

    Returns:
        组件字典
    """
    components = {}

    gr.HTML(f'<div class="ks-col-title">{t("config_title")}</div>')

    # Mode 选择 —— elem_id 供 CSS 卡片样式命中
    with gr.Group():
        components["mode"] = gr.Radio(
            choices=["unlearn", "inject", "edit", "eval"],
            value="unlearn",
            label=t("mode_label"),
            info=t("mode_info"),
            elem_id="mode-radio",
        )
        
        # 实验模板
        with gr.Group():
            initial_experiments = config_loader.get_experiments("unlearn")
            components["experiment"] = gr.Dropdown(
                choices=initial_experiments,
                value=initial_experiments[0] if initial_experiments else None,
                label=t("experiment_label"),
                info=t("experiment_info")
            )

        # 模型选择
        with gr.Group():
            models = config_loader.get_models()
            components["model"] = gr.Dropdown(
                choices=models,
                value="Qwen2.5-7B-Instruct" if "Qwen2.5-7B-Instruct" in models else (models[0] if models else None),
                label=t("model_label"),
                info=t("model_info")
            )

        # 方法选择
        with gr.Group():
            initial_trainers = config_loader.get_trainers("unlearn")
            components["trainer"] = gr.Dropdown(
                choices=initial_trainers,
                value="SimNPO" if "SimNPO" in initial_trainers else (initial_trainers[0] if initial_trainers else None),
                label=t("trainer_label"),
                info=t("trainer_info")
            )

        # 数据集选择（动态根据 mode 变化）
        with gr.Group():
            
            # Unlearn 数据集（默认显示）
            initial_datasets = config_loader.get_datasets("unlearn")
            with gr.Column(visible=True) as unlearn_datasets:
                components["forget_dataset"] = gr.Dropdown(
                    choices=initial_datasets.get("forget", []),
                    value=initial_datasets.get("forget", [""])[0] if initial_datasets.get("forget") else None,
                    label=t("forget_dataset_label"),
                    info=t("forget_dataset_info"),
                    allow_custom_value=True,
                )
                components["retain_dataset"] = gr.Dropdown(
                    choices=initial_datasets.get("retain", []),
                    value=initial_datasets.get("retain", [""])[0] if initial_datasets.get("retain") else None,
                    label=t("retain_dataset_label"),
                    info=t("retain_dataset_info"),
                    allow_custom_value=True,
                )
            components["unlearn_datasets_group"] = unlearn_datasets
            
            # Inject 数据集（默认隐藏）
            with gr.Column(visible=False) as inject_datasets:
                inject_data = config_loader.get_datasets("inject")
                components["train_dataset"] = gr.Dropdown(
                    choices=inject_data.get("train", []),
                    value=inject_data.get("train", [""])[0] if inject_data.get("train") else None,
                    label=t("train_dataset_label"),
                    info=t("train_dataset_info"),
                    allow_custom_value=True,
                )
            components["inject_datasets_group"] = inject_datasets
            
            # Edit 数据集（默认隐藏）
            with gr.Column(visible=False) as edit_datasets:
                edit_data = config_loader.get_datasets("edit")
                components["edit_dataset"] = gr.Dropdown(
                    choices=edit_data.get("edit", []),
                    value=edit_data.get("edit", [""])[0] if edit_data.get("edit") else None,
                    label=t("edit_dataset_label"),
                    info=t("edit_dataset_info"),
                    allow_custom_value=True,
                )
            components["edit_datasets_group"] = edit_datasets
        
        # 评测套件（训练模式）
        with gr.Group(visible=True) as train_eval_group:
            initial_evals = config_loader.get_evals("unlearn")
            components["eval_suite"] = gr.Dropdown(
                choices=initial_evals,
                value=initial_evals[0] if initial_evals else None,
                label=t("eval_suite_label"),
                info=t("eval_suite_info")
            )
        components["train_eval_group"] = train_eval_group
        
        # Eval 模式专用配置（默认隐藏）
        with gr.Group(visible=False) as eval_mode_group:
            # 评测套件选择
            eval_suites = config_loader.get_eval_suites()
            components["eval_suite_select"] = gr.Dropdown(
                choices=eval_suites,
                value="tofu" if "tofu" in eval_suites else (eval_suites[0] if eval_suites else None),
                label=t("eval_suite_label"),
                info=t("eval_suite_info")
            )
            
            # 已保存模型选择
            saved_models = config_loader.get_saved_models()
            components["saved_model_path"] = gr.Dropdown(
                choices=saved_models,
                value=saved_models[0] if saved_models else None,
                label=t("saved_model_label"),
                info=t("saved_model_info")
            )
            
            # 刷新按钮
            components["refresh_saved_models"] = gr.Button(t("refresh_models"), size="sm")
        components["eval_mode_group"] = eval_mode_group
        
        # 运行参数
        with gr.Group():
            with gr.Row():
                components["task_name"] = gr.Textbox(
                    value="my_experiment",
                    label=t("task_name_label"),
                    info=t("task_name_info")
                )
            with gr.Row():
                components["seed"] = gr.Number(
                    value=42,
                    label=t("seed_label"),
                    precision=0
                )
        
        # 导入/导出
        with gr.Row():
            components["import_btn"] = gr.Button(t("import_config"), size="sm")
            components["export_btn"] = gr.Button(t("export_config"), size="sm")
        
        # 隐藏的文件上传/下载组件
        components["config_upload"] = gr.File(
            label=t("import_config"),
            file_types=[".yaml", ".yml"],
            visible=False
        )
        components["config_download"] = gr.File(
            label=t("export_config"),
            visible=False
        )
    
    return components


def update_config_on_mode_change(
    mode: str,
    config_loader: ConfigLoader
) -> Tuple[gr.update, ...]:
    """当 mode 改变时更新配置面板
    
    使用统一的 apply_mode_config 函数确保联动顺序正确：
    先更新 choices，再设置 value，避免 dropdown warning
    
    Args:
        mode: 新的模式
        config_loader: 配置加载器
        
    Returns:
        12 个组件的更新值元组
    """
    return apply_mode_config(mode, config_loader)
