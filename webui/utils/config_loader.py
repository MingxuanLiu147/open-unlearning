"""
配置加载工具
=============

扫描 configs/ 目录，提取可用的模型、方法、数据集、评测配置。
为 Gradio 下拉框提供数据源。
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import yaml


class ConfigLoader:
    """Hydra 配置加载器，为 GUI 提供配置选项"""
    
    def __init__(self, configs_dir: str = None):
        """初始化配置加载器
        
        Args:
            configs_dir: configs 目录路径，默认为项目根目录下的 configs/
        """
        if configs_dir is None:
            # 默认从 webui 目录向上找 configs
            base_dir = Path(__file__).parent.parent.parent
            configs_dir = base_dir / "configs"
        self.configs_dir = Path(configs_dir)
        
    def _scan_yaml_files(self, subdir: str, recursive: bool = False) -> List[str]:
        """扫描子目录下的 yaml 文件
        
        Args:
            subdir: 子目录名（相对于 configs/）
            recursive: 是否递归扫描子目录
            
        Returns:
            配置名列表（不含 .yaml 后缀）
        """
        target_dir = self.configs_dir / subdir
        if not target_dir.exists():
            return []
        
        configs = []
        if recursive:
            for yaml_file in target_dir.rglob("*.yaml"):
                # 生成相对路径作为配置名
                rel_path = yaml_file.relative_to(target_dir)
                config_name = str(rel_path).replace(".yaml", "").replace(os.sep, "/")
                configs.append(config_name)
        else:
            for yaml_file in target_dir.glob("*.yaml"):
                configs.append(yaml_file.stem)
        
        return sorted(configs)
    
    def _load_yaml(self, filepath: Path) -> Optional[Dict]:
        """加载单个 yaml 文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception:
            return None
    
    def get_models(self) -> List[str]:
        """获取可用模型列表
        
        扫描 configs/model/*.yaml
        """
        return self._scan_yaml_files("model")
    
    def get_trainers(self, mode: str = None) -> List[str]:
        """获取可用训练方法列表
        
        Args:
            mode: 训练模式 (unlearn/inject/edit)，用于过滤方法
            
        Returns:
            方法名列表
        """
        trainer_dir = self.configs_dir / "trainer"
        if not trainer_dir.exists():
            return []
        
        trainers = []
        
        # 根据 mode 过滤
        if mode == "unlearn":
            # Unlearn 方法在 trainer/ 顶层
            for yaml_file in trainer_dir.glob("*.yaml"):
                if yaml_file.stem not in ["finetune"]:  # 排除非 unlearn 方法
                    trainers.append(yaml_file.stem)
        elif mode == "inject":
            # Inject 方法在 trainer/inject/
            inject_dir = trainer_dir / "inject"
            if inject_dir.exists():
                for yaml_file in inject_dir.glob("*.yaml"):
                    if yaml_file.stem != "base_inject":
                        trainers.append(f"inject/{yaml_file.stem}")
        elif mode == "edit":
            # Edit 方法在 trainer/edit/
            edit_dir = trainer_dir / "edit"
            if edit_dir.exists():
                for yaml_file in edit_dir.glob("*.yaml"):
                    if yaml_file.stem != "base_editor":
                        trainers.append(f"edit/{yaml_file.stem}")
        else:
            # 返回所有方法
            trainers = self._scan_yaml_files("trainer", recursive=True)
            # 过滤掉 base 配置
            trainers = [t for t in trainers if not t.endswith("base_inject") 
                       and not t.endswith("base_editor")]
        
        return sorted(trainers)
    
    def get_datasets(self, mode: str = None) -> Dict[str, List[str]]:
        """获取数据集配置
        
        Args:
            mode: 训练模式，决定返回的数据集类型
            
        Returns:
            数据集配置字典，包含 forget/retain/edit/train 等键
        """
        datasets_dir = self.configs_dir / "data" / "datasets"
        if not datasets_dir.exists():
            return {}
        
        all_datasets = []
        for yaml_file in datasets_dir.glob("*.yaml"):
            all_datasets.append(yaml_file.stem)
        
        result = {}
        
        if mode == "unlearn":
            # Unlearn 需要 forget 和 retain 数据集
            result["forget"] = [d for d in all_datasets if "forget" in d.lower()]
            result["retain"] = [d for d in all_datasets if "retain" in d.lower()]
        elif mode == "edit":
            # Edit 需要编辑数据集
            result["edit"] = [d for d in all_datasets if "edit" in d.lower()]
        elif mode == "inject":
            # Inject 需要训练数据集
            result["train"] = [d for d in all_datasets if "inject" in d.lower() 
                              or "alpaca" in d.lower() or "custom" in d.lower()]
        else:
            result["all"] = sorted(all_datasets)
        
        return result
    
    def get_evals(self, mode: str = None) -> List[str]:
        """获取评测套件列表
        
        Args:
            mode: 训练模式，用于过滤评测套件
        """
        eval_dir = self.configs_dir / "eval"
        if not eval_dir.exists():
            return []
        
        evals = []
        for yaml_file in eval_dir.glob("*.yaml"):
            eval_name = yaml_file.stem
            # 根据 mode 过滤
            if mode == "unlearn":
                if eval_name in ["tofu", "muse", "wmdp"]:
                    evals.append(eval_name)
            elif mode == "edit":
                if eval_name in ["edit"]:
                    evals.append(eval_name)
            elif mode == "inject":
                if eval_name in ["inject"]:
                    evals.append(eval_name)
            else:
                evals.append(eval_name)
        
        return sorted(evals)
    
    def get_experiments(self, mode: str = None) -> List[str]:
        """获取实验模板列表
        
        Args:
            mode: 训练模式，用于过滤实验模板
        """
        exp_dir = self.configs_dir / "experiment"
        if not exp_dir.exists():
            return ["(无模板 - 自定义配置)"]
        
        experiments = ["(无模板 - 自定义配置)"]
        
        if mode:
            mode_dir = exp_dir / mode
            if mode_dir.exists():
                for yaml_file in mode_dir.rglob("*.yaml"):
                    rel_path = yaml_file.relative_to(exp_dir)
                    exp_name = str(rel_path).replace(".yaml", "").replace(os.sep, "/")
                    experiments.append(exp_name)
        else:
            for yaml_file in exp_dir.rglob("*.yaml"):
                rel_path = yaml_file.relative_to(exp_dir)
                exp_name = str(rel_path).replace(".yaml", "").replace(os.sep, "/")
                experiments.append(exp_name)
        
        return experiments
    
    def get_trainer_config(self, trainer_name: str) -> Dict:
        """获取特定 trainer 的配置内容
        
        Args:
            trainer_name: trainer 名称（如 "SimNPO" 或 "inject/LoRA"）
            
        Returns:
            配置字典
        """
        trainer_path = self.configs_dir / "trainer" / f"{trainer_name}.yaml"
        return self._load_yaml(trainer_path) or {}
    
    def get_model_config(self, model_name: str) -> Dict:
        """获取特定模型的配置内容"""
        model_path = self.configs_dir / "model" / f"{model_name}.yaml"
        return self._load_yaml(model_path) or {}
    
    def get_default_config(self, mode: str) -> Dict:
        """获取某个模式的默认入口配置
        
        Args:
            mode: 模式名 (unlearn/inject/edit)
        """
        config_path = self.configs_dir / f"{mode}.yaml"
        return self._load_yaml(config_path) or {}
