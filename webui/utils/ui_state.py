# -*- coding: utf-8 -*-
"""
UI 状态管理工具
================

提供统一的 UI 状态更新逻辑，避免状态分散在多个回调中。
解决 mode 切换时的联动顺序问题。
"""

from typing import Dict, Any, List, Optional, Tuple
import gradio as gr


class UIState:
    """UI 状态管理类，提供统一的状态更新接口"""

    @staticmethod
    def build_dropdown_update(
        choices: List[str],
        value: Optional[str] = None,
        visible: bool = True,
        default_if_missing: bool = True,
    ) -> gr.update:
        """构建 Dropdown 的更新，确保 value 在 choices 中
        
        Args:
            choices: 可选项列表
            value: 期望的值
            visible: 是否可见
            default_if_missing: 如果 value 不在 choices 中，是否使用第一个选项
            
        Returns:
            gr.update 对象
        """
        if not choices:
            return gr.update(choices=[], value=None, visible=visible)
        
        if value and value in choices:
            final_value = value
        elif default_if_missing and choices:
            final_value = choices[0]
        else:
            final_value = None
            
        return gr.update(choices=choices, value=final_value, visible=visible)

    @staticmethod
    def get_mode_defaults(mode: str) -> Dict[str, str]:
        """获取各模式的推荐默认值
        
        Args:
            mode: 训练模式 (unlearn/inject/edit/eval)
            
        Returns:
            包含推荐默认值的字典
        """
        defaults = {
            "unlearn": {
                "trainer": "SimNPO",
                "model": "Qwen2.5-7B-Instruct",
            },
            "inject": {
                "trainer": "inject/LoRA",
                "model": "Qwen2.5-7B-Instruct",
            },
            "edit": {
                "trainer": "edit/ROME",
                "model": "Qwen2.5-7B-Instruct",
            },
            "eval": {
                "trainer": None,
                "model": "Qwen2.5-7B-Instruct",
            },
        }
        return defaults.get(mode, defaults["unlearn"])


def apply_mode_config(
    mode: str,
    config_loader,
    current_trainer: Optional[str] = None,
) -> Tuple:
    """统一的 mode 切换配置应用函数
    
    解决联动顺序问题：先更新 choices，再设置 value
    
    Args:
        mode: 目标模式
        config_loader: 配置加载器
        current_trainer: 当前选中的 trainer（用于保持选择）
        
    Returns:
        12 个组件的 gr.update 元组
    """
    is_eval_mode = mode == "eval"
    is_train_mode = mode in ["unlearn", "inject", "edit"]
    
    defaults = UIState.get_mode_defaults(mode)
    
    # 获取可用配置
    if is_train_mode:
        experiments = config_loader.get_experiments(mode)
        trainers = config_loader.get_trainers(mode)
        datasets = config_loader.get_datasets(mode)
        evals = config_loader.get_evals(mode)
    else:
        experiments = ["(无模板 - 自定义配置)"]
        trainers = []
        datasets = {}
        evals = []
    
    # 确定 trainer 默认值
    default_trainer = defaults.get("trainer")
    if current_trainer and current_trainer in trainers:
        default_trainer = current_trainer
    elif default_trainer and default_trainer not in trainers and trainers:
        default_trainer = trainers[0]
    
    # 数据集面板可见性
    unlearn_visible = mode == "unlearn"
    inject_visible = mode == "inject"
    edit_visible = mode == "edit"
    
    return (
        # experiment
        UIState.build_dropdown_update(
            choices=experiments,
            value=experiments[0] if experiments else None,
            visible=is_train_mode,
        ),
        # trainer
        UIState.build_dropdown_update(
            choices=trainers,
            value=default_trainer,
            visible=is_train_mode,
        ),
        # unlearn_datasets_group
        gr.update(visible=unlearn_visible),
        # inject_datasets_group
        gr.update(visible=inject_visible),
        # edit_datasets_group
        gr.update(visible=edit_visible),
        # forget_dataset
        UIState.build_dropdown_update(
            choices=datasets.get("forget", []),
            value=datasets.get("forget", [""])[0] if datasets.get("forget") else None,
        ),
        # retain_dataset
        UIState.build_dropdown_update(
            choices=datasets.get("retain", []),
            value=datasets.get("retain", [""])[0] if datasets.get("retain") else None,
        ),
        # train_dataset
        UIState.build_dropdown_update(
            choices=datasets.get("train", []),
            value=datasets.get("train", [""])[0] if datasets.get("train") else None,
        ),
        # edit_dataset
        UIState.build_dropdown_update(
            choices=datasets.get("edit", []),
            value=datasets.get("edit", [""])[0] if datasets.get("edit") else None,
        ),
        # eval_suite (训练模式评测套件)
        UIState.build_dropdown_update(
            choices=evals,
            value=evals[0] if evals else None,
        ),
        # train_eval_group
        gr.update(visible=is_train_mode),
        # eval_mode_group
        gr.update(visible=is_eval_mode),
    )


def build_config_dict(
    mode: str,
    model: str,
    trainer: str,
    experiment: str,
    task_name: str,
    seed: int,
    forget_dataset: str,
    retain_dataset: str,
    train_dataset: str,
    edit_dataset: str,
    eval_suite: str,
    learning_rate: str,
    num_epochs: int,
    batch_size: int,
    gradient_accumulation: int,
    max_length: int,
    warmup_ratio: float,
    method_args_json: str,
    overrides_text: str,
) -> Dict[str, Any]:
    """构建完整的配置字典（用于导出）
    
    Args:
        各 UI 组件的值
        
    Returns:
        完整的配置字典
    """
    import json
    
    config = {
        "mode": mode,
        "model": model,
        "trainer": trainer,
        "task_name": task_name,
        "seed": int(seed) if seed else 42,
    }
    
    # 实验模板
    if experiment and not experiment.startswith("("):
        config["experiment"] = experiment
    
    # 数据集
    if mode == "unlearn":
        config["data"] = {"forget": forget_dataset, "retain": retain_dataset}
    elif mode == "inject":
        config["data"] = {"train": train_dataset}
    elif mode == "edit":
        config["data"] = {"edit": edit_dataset}
    
    # 评测套件
    if eval_suite:
        config["eval_suite"] = eval_suite
    
    # 训练参数
    config["trainer_args"] = {
        "learning_rate": learning_rate,
        "num_train_epochs": int(num_epochs) if num_epochs else 3,
        "per_device_train_batch_size": int(batch_size) if batch_size else 4,
        "gradient_accumulation_steps": int(gradient_accumulation) if gradient_accumulation else 4,
        "max_length": int(max_length) if max_length else 512,
        "warmup_ratio": float(warmup_ratio) if warmup_ratio is not None else 0.1,
    }
    
    # 方法参数
    if method_args_json and method_args_json != "{}":
        try:
            config["method_args"] = json.loads(method_args_json)
        except json.JSONDecodeError:
            pass
    
    # 额外 overrides
    if overrides_text:
        config["overrides"] = overrides_text
    
    return config


def apply_config_dict(config: Dict[str, Any]) -> Tuple:
    """从配置字典恢复 UI 状态
    
    Args:
        config: 配置字典
        
    Returns:
        UI 组件更新元组（共 18 个组件）
    """
    import json
    
    data = config.get("data", {})
    trainer_args = config.get("trainer_args", {})
    method_args = config.get("method_args", {})
    
    return (
        # 基础配置 (10 个)
        gr.update(value=config.get("mode", "unlearn")),
        gr.update(value=config.get("model")),
        gr.update(value=config.get("trainer")),
        gr.update(value=config.get("experiment")),
        gr.update(value=config.get("task_name", "my_experiment")),
        gr.update(value=config.get("seed", 42)),
        gr.update(value=data.get("forget")),
        gr.update(value=data.get("retain")),
        gr.update(value=data.get("train")),
        gr.update(value=data.get("edit")),
        # 训练参数 (6 个)
        gr.update(value=trainer_args.get("learning_rate", "1e-5")),
        gr.update(value=trainer_args.get("num_train_epochs", 3)),
        gr.update(value=trainer_args.get("per_device_train_batch_size", 4)),
        gr.update(value=trainer_args.get("gradient_accumulation_steps", 4)),
        gr.update(value=trainer_args.get("max_length", 512)),
        gr.update(value=trainer_args.get("warmup_ratio", 0.1)),
        # 方法参数和 overrides (2 个)
        gr.update(value=json.dumps(method_args, indent=2, ensure_ascii=False) if method_args else "{}"),
        gr.update(value=config.get("overrides", "")),
    )
