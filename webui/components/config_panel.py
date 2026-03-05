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


def create_config_panel(config_loader: ConfigLoader) -> Dict[str, Any]:
    """创建配置面板
    
    Args:
        config_loader: 配置加载器
        
    Returns:
        组件字典
    """
    components = {}
    
    with gr.Column(scale=1):
        gr.Markdown("## 配置选择")
        
        # Step 0: Mode 选择
        with gr.Group():
            gr.Markdown("### Step 1: 选择任务模式")
            components["mode"] = gr.Radio(
                choices=["unlearn", "inject", "edit", "eval"],
                value="unlearn",
                label="任务模式",
                info="Unlearn=知识删除, Inject=知识注入, Edit=知识编辑, Eval=评估"
            )
        
        # Step 1: 实验模板
        with gr.Group():
            gr.Markdown("### Step 2: 选择实验模板 (可选)")
            initial_experiments = config_loader.get_experiments("unlearn")
            components["experiment"] = gr.Dropdown(
                choices=initial_experiments,
                value=initial_experiments[0] if initial_experiments else None,
                label="实验模板",
                info="选择预设配置模板，或自定义配置"
            )
        
        # Step 2: 模型选择
        with gr.Group():
            gr.Markdown("### Step 3: 选择模型")
            models = config_loader.get_models()
            components["model"] = gr.Dropdown(
                choices=models,
                value="Qwen2.5-7B-Instruct" if "Qwen2.5-7B-Instruct" in models else (models[0] if models else None),
                label="模型",
                info="选择预训练模型"
            )
        
        # Step 3: 方法选择
        with gr.Group():
            gr.Markdown("### Step 4: 选择方法")
            initial_trainers = config_loader.get_trainers("unlearn")
            components["trainer"] = gr.Dropdown(
                choices=initial_trainers,
                value="SimNPO" if "SimNPO" in initial_trainers else (initial_trainers[0] if initial_trainers else None),
                label="训练方法",
                info="选择具体的算法/方法"
            )
        
        # Step 4: 数据集选择（动态根据 mode 变化）
        with gr.Group():
            gr.Markdown("### Step 5: 选择数据集")
            
            # Unlearn 数据集（默认显示）
            initial_datasets = config_loader.get_datasets("unlearn")
            with gr.Column(visible=True) as unlearn_datasets:
                components["forget_dataset"] = gr.Dropdown(
                    choices=initial_datasets.get("forget", []),
                    value=initial_datasets.get("forget", [""])[0] if initial_datasets.get("forget") else None,
                    label="遗忘数据集 (Forget)",
                    info="需要遗忘的数据"
                )
                components["retain_dataset"] = gr.Dropdown(
                    choices=initial_datasets.get("retain", []),
                    value=initial_datasets.get("retain", [""])[0] if initial_datasets.get("retain") else None,
                    label="保留数据集 (Retain)",
                    info="需要保持的数据"
                )
            components["unlearn_datasets_group"] = unlearn_datasets
            
            # Inject 数据集（默认隐藏）
            with gr.Column(visible=False) as inject_datasets:
                inject_data = config_loader.get_datasets("inject")
                components["train_dataset"] = gr.Dropdown(
                    choices=inject_data.get("train", []),
                    value=inject_data.get("train", [""])[0] if inject_data.get("train") else None,
                    label="训练数据集",
                    info="微调训练数据"
                )
            components["inject_datasets_group"] = inject_datasets
            
            # Edit 数据集（默认隐藏）
            with gr.Column(visible=False) as edit_datasets:
                edit_data = config_loader.get_datasets("edit")
                components["edit_dataset"] = gr.Dropdown(
                    choices=edit_data.get("edit", []),
                    value=edit_data.get("edit", [""])[0] if edit_data.get("edit") else None,
                    label="编辑数据集",
                    info="知识编辑数据"
                )
            components["edit_datasets_group"] = edit_datasets
        
        # Step 5: 评测套件（训练模式）
        with gr.Group(visible=True) as train_eval_group:
            gr.Markdown("### Step 6: 选择评测套件")
            initial_evals = config_loader.get_evals("unlearn")
            components["eval_suite"] = gr.Dropdown(
                choices=initial_evals,
                value=initial_evals[0] if initial_evals else None,
                label="评测套件",
                info="选择评估指标集合"
            )
        components["train_eval_group"] = train_eval_group
        
        # Eval 模式专用配置（默认隐藏）
        with gr.Group(visible=False) as eval_mode_group:
            gr.Markdown("### Step 3: 评估配置")
            
            # 评测套件选择
            eval_suites = config_loader.get_eval_suites()
            components["eval_suite_select"] = gr.Dropdown(
                choices=eval_suites,
                value="tofu" if "tofu" in eval_suites else (eval_suites[0] if eval_suites else None),
                label="评测套件",
                info="选择要运行的评测套件"
            )
            
            # 已保存模型选择
            saved_models = config_loader.get_saved_models()
            components["saved_model_path"] = gr.Dropdown(
                choices=saved_models,
                value=saved_models[0] if saved_models else None,
                label="已保存模型",
                info="选择 saves/ 目录下的已训练模型，或留空使用 HuggingFace 模型"
            )
            
            # 刷新按钮
            components["refresh_saved_models"] = gr.Button("🔄 刷新模型列表", size="sm")
        components["eval_mode_group"] = eval_mode_group
        
        # Step 6: 运行参数
        with gr.Group():
            gr.Markdown("### Step 7: 运行参数")
            with gr.Row():
                components["task_name"] = gr.Textbox(
                    value="my_experiment",
                    label="任务名称 (task_name)",
                    info="用于标识本次实验"
                )
            with gr.Row():
                components["seed"] = gr.Number(
                    value=42,
                    label="随机种子 (seed)",
                    precision=0
                )
        
        # 导入/导出
        with gr.Row():
            components["import_btn"] = gr.Button("📥 导入配置", size="sm")
            components["export_btn"] = gr.Button("📤 导出配置", size="sm")
        
        # 隐藏的文件上传/下载组件
        components["config_upload"] = gr.File(
            label="上传配置文件",
            file_types=[".yaml", ".yml"],
            visible=False
        )
        components["config_download"] = gr.File(
            label="下载配置",
            visible=False
        )
    
    return components


def update_config_on_mode_change(
    mode: str,
    config_loader: ConfigLoader
) -> Tuple[gr.update, ...]:
    """当 mode 改变时更新配置面板
    
    Args:
        mode: 新的模式
        config_loader: 配置加载器
        
    Returns:
        各组件的更新值
    """
    # eval 模式特殊处理
    is_eval_mode = mode == "eval"
    is_train_mode = mode in ["unlearn", "inject", "edit"]
    
    # 获取新模式对应的配置
    experiments = config_loader.get_experiments(mode) if is_train_mode else ["(无模板 - 自定义配置)"]
    trainers = config_loader.get_trainers(mode) if is_train_mode else []
    datasets = config_loader.get_datasets(mode) if is_train_mode else {}
    evals = config_loader.get_evals(mode) if is_train_mode else []
    
    # 确定数据集面板可见性
    unlearn_visible = mode == "unlearn"
    inject_visible = mode == "inject"
    edit_visible = mode == "edit"
    
    # 获取默认值
    default_trainer = trainers[0] if trainers else None
    if mode == "unlearn" and "SimNPO" in trainers:
        default_trainer = "SimNPO"
    elif mode == "inject" and "inject/LoRA" in trainers:
        default_trainer = "inject/LoRA"
    elif mode == "edit" and "edit/ROME" in trainers:
        default_trainer = "edit/ROME"
    
    return (
        # experiment 更新
        gr.update(choices=experiments, value=experiments[0] if experiments else None, visible=is_train_mode),
        # trainer 更新
        gr.update(choices=trainers, value=default_trainer, visible=is_train_mode),
        # unlearn 数据集组可见性
        gr.update(visible=unlearn_visible),
        # inject 数据集组可见性
        gr.update(visible=inject_visible),
        # edit 数据集组可见性
        gr.update(visible=edit_visible),
        # forget 数据集更新
        gr.update(
            choices=datasets.get("forget", []),
            value=datasets.get("forget", [""])[0] if datasets.get("forget") else None
        ),
        # retain 数据集更新
        gr.update(
            choices=datasets.get("retain", []),
            value=datasets.get("retain", [""])[0] if datasets.get("retain") else None
        ),
        # train 数据集更新
        gr.update(
            choices=datasets.get("train", []),
            value=datasets.get("train", [""])[0] if datasets.get("train") else None
        ),
        # edit 数据集更新
        gr.update(
            choices=datasets.get("edit", []),
            value=datasets.get("edit", [""])[0] if datasets.get("edit") else None
        ),
        # eval 更新（训练模式评测套件）
        gr.update(choices=evals, value=evals[0] if evals else None),
        # train_eval_group 可见性
        gr.update(visible=is_train_mode),
        # eval_mode_group 可见性
        gr.update(visible=is_eval_mode),
    )
