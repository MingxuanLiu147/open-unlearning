"""
配置加载工具
=============

扫描 configs/ 目录，提取可用的模型、方法、数据集、评测配置。
为 Gradio 下拉框提供数据源。
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
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
        self.project_root = self.configs_dir.parent
        self.saves_dir = self.project_root / "saves"
        
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

    def _collect_handlers(self, node: Any) -> List[str]:
        """递归提取配置中的 handler 名称。"""
        if isinstance(node, dict):
            handlers = []
            handler = node.get("handler")
            if isinstance(handler, str):
                handlers.append(handler)
            for value in node.values():
                handlers.extend(self._collect_handlers(value))
            return handlers
        if isinstance(node, list):
            handlers = []
            for value in node:
                handlers.extend(self._collect_handlers(value))
            return handlers
        return []

    def _infer_inject_split(self, config_name: str) -> str:
        """根据 inject 配置文件名推断所属 split。"""
        lowered = config_name.lower()
        if "_analysis_" in lowered or lowered.endswith("_analysis_inject"):
            return "analysis"
        if "_eval_" in lowered or lowered.endswith("_eval_inject"):
            return "eval"
        return "train"

    def _scan_dataset_configs(
        self,
        split: str,
        handler_whitelist: Optional[Set[str]] = None,
    ) -> List[str]:
        """按 split 和 handler 扫描数据集配置。"""
        datasets_dir = self.configs_dir / "data" / "datasets"
        if not datasets_dir.exists():
            return []

        matches = []
        for yaml_file in datasets_dir.glob("*.yaml"):
            cfg = self._load_yaml(yaml_file)
            if not isinstance(cfg, dict):
                continue

            split_cfg = cfg.get(split)
            if isinstance(split_cfg, dict):
                if handler_whitelist is not None:
                    handlers = set(self._collect_handlers(split_cfg))
                    if not handlers.intersection(handler_whitelist):
                        continue
                matches.append(yaml_file.stem)
                continue

            # 兼容 unlearn 风格的单层数据集配置：
            #   DatasetName:
            #     handler: AlpacaDataset
            #     args: ...
            if self._infer_inject_split(yaml_file.stem) != split:
                continue

            if handler_whitelist is not None:
                handlers = set(self._collect_handlers(cfg))
                if not handlers.intersection(handler_whitelist):
                    continue

            matches.append(yaml_file.stem)

        return sorted(matches)
    
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
            # Inject 按配置内容识别，避免依赖文件名约定。
            inject_handlers = {"InjectDataset", "AlpacaDataset", "ShareGPTDataset"}
            result["train"] = self._scan_dataset_configs("train", inject_handlers)
            result["eval"] = self._scan_dataset_configs("eval", inject_handlers)
            result["analysis"] = self._scan_dataset_configs(
                "analysis", inject_handlers
            )
        else:
            result["all"] = sorted(all_datasets)
        
        return result
    
    def get_evals(self, mode: str = None) -> List[str]:
        """获取评测套件列表
        
        Args:
            mode: 训练模式，用于智能排序（推荐的套件排前面）
        """
        eval_dir = self.configs_dir / "eval"
        if not eval_dir.exists():
            return []
        
        # 扫描所有评估配置
        all_evals = []
        for yaml_file in eval_dir.glob("*.yaml"):
            all_evals.append(yaml_file.stem)
        
        # 根据模式智能排序：推荐的评估套件排在前面
        recommended = {
            "unlearn": ["tofu", "muse"],
            "edit": ["edit"],
            "inject": ["inject"],
        }
        
        if mode and mode in recommended:
            # 推荐套件在前，其他在后
            rec_list = [e for e in recommended[mode] if e in all_evals]
            other_list = sorted([e for e in all_evals if e not in rec_list])
            return rec_list + other_list
        
        return sorted(all_evals)
    
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
            mode: 模式名 (unlearn/inject/edit/eval)
        """
        config_path = self.configs_dir / f"{mode}.yaml"
        return self._load_yaml(config_path) or {}
    
    def get_saved_models(self) -> List[str]:
        """获取 saves/ 目录下已保存的模型路径
        
        扫描 saves/unlearn/, saves/finetune/, saves/edit/ 等目录，
        查找包含 model.safetensors 或 pytorch_model.bin 的目录
        
        Returns:
            模型路径列表（相对于项目根目录）
        """
        saved_models = []
        
        if not self.saves_dir.exists():
            return saved_models
        
        # 扫描的子目录
        subdirs = ["unlearn", "finetune", "edit", "inject"]
        
        for subdir in subdirs:
            subdir_path = self.saves_dir / subdir
            if not subdir_path.exists():
                continue
            
            # 遍历所有子目录
            for model_dir in subdir_path.iterdir():
                if not model_dir.is_dir():
                    continue
                
                # 检查是否包含模型文件
                has_model = (
                    (model_dir / "model.safetensors").exists() or
                    (model_dir / "pytorch_model.bin").exists() or
                    (model_dir / "adapter_model.safetensors").exists()
                )
                
                if has_model:
                    # 返回相对路径
                    rel_path = model_dir.relative_to(self.project_root)
                    saved_models.append(str(rel_path))
        
        return sorted(saved_models)
    
    def get_eval_suites(self) -> List[str]:
        """获取可用的评测套件列表（用于 eval 模式）
        
        Returns:
            评测套件名列表
        """
        eval_dir = self.configs_dir / "eval"
        if not eval_dir.exists():
            return []
        
        suites = []
        for yaml_file in eval_dir.glob("*.yaml"):
            suites.append(yaml_file.stem)
        
        return sorted(suites)

    def get_eval_runs(self) -> Dict[str, Dict]:
        """扫描 saves/ 目录下所有含评估结果的 run，用于 Tab 2 结果对比。

        遍历 saves/{mode}/{task_name}/checkpoint-{N}/evals/*_SUMMARY.json，
        按 "{mode}/{task_name} @ checkpoint-{N}" 组织返回值。

        Returns:
            有序字典，key 为显示标签，value 为
            {"mode": str, "task_name": str, "checkpoint": str,
             "checkpoint_step": int, "summary_files": List[str]}
        """
        from collections import defaultdict

        runs: Dict[str, Dict] = {}

        if not self.saves_dir.exists():
            return runs

        mode_dirs = ["unlearn", "inject", "edit", "finetune"]

        for mode in mode_dirs:
            mode_path = self.saves_dir / mode
            if not mode_path.exists():
                continue

            for task_dir in sorted(mode_path.iterdir()):
                if not task_dir.is_dir():
                    continue
                task_name = task_dir.name

                for ckpt_dir in sorted(task_dir.iterdir()):
                    if not ckpt_dir.is_dir():
                        continue
                    ckpt_name = ckpt_dir.name

                    # 解析 checkpoint 步数，用于排序
                    step = 0
                    if ckpt_name.startswith("checkpoint-"):
                        try:
                            step = int(ckpt_name.split("-", 1)[1])
                        except ValueError:
                            pass

                    evals_dir = ckpt_dir / "evals"
                    if not evals_dir.exists():
                        continue

                    summary_files = sorted(str(f) for f in evals_dir.glob("*_SUMMARY.json"))
                    if not summary_files:
                        continue

                    label = f"{mode}/{task_name} @ {ckpt_name}"
                    runs[label] = {
                        "mode": mode,
                        "task_name": task_name,
                        "checkpoint": ckpt_name,
                        "checkpoint_step": step,
                        "summary_files": summary_files,
                    }

        # 按 task_name + step 排序
        return dict(
            sorted(runs.items(), key=lambda x: (x[1]["mode"], x[1]["task_name"], x[1]["checkpoint_step"]))
        )
