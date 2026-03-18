# -*- coding: utf-8 -*-
"""
Skill Schema 定义与校验
========================

定义 skill 模板的结构化字段，用于规则推荐和 Agent 推荐的共同知识源。
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class SkillConfig:
    """Skill 的核心配置"""
    mode: str                          # unlearn / inject / edit
    model: Optional[str] = None        # 推荐模型
    trainer: Optional[str] = None      # 训练方法
    experiment: Optional[str] = None   # 实验模板
    task_name: str = "my_experiment"
    seed: int = 42
    learning_rate: str = "1e-5"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4
    max_length: int = 512
    warmup_ratio: float = 0.1


@dataclass
class Skill:
    """完整的 Skill 模板定义
    
    包含算法核心配置、适用场景、推荐参数等信息，
    供规则引擎和 Agent 共同使用。
    """
    # 基础信息
    id: str                            # 唯一标识
    name: str                          # 显示名称
    name_en: str = ""                  # 英文名称
    description: str = ""              # 中文描述
    description_en: str = ""           # 英文描述
    goal: str = "unlearn"              # 目标类型: unlearn / inject / edit
    
    # 核心配置
    config: Dict[str, Any] = field(default_factory=dict)
    
    # 适用场景
    supported_data_modes: List[str] = field(default_factory=lambda: ["single", "batch"])
    supported_input_formats: List[str] = field(default_factory=lambda: ["jsonl", "json"])
    
    # 推荐配置
    recommended_models: List[str] = field(default_factory=list)
    recommended_eval: str = ""
    resource_estimate: str = ""
    resource_estimate_en: str = ""
    
    # 核心 overrides（Agent 可直接应用）
    core_overrides: Dict[str, Any] = field(default_factory=dict)
    
    # 约束和风险
    constraints: List[str] = field(default_factory=list)
    constraints_en: List[str] = field(default_factory=list)
    risk_notes: List[str] = field(default_factory=list)
    risk_notes_en: List[str] = field(default_factory=list)
    
    # Agent 提示
    prompt_hints: str = ""
    prompt_hints_en: str = ""
    
    # 标签
    tags: List[str] = field(default_factory=list)
    tags_en: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def get_display_name(self, lang: str = "zh") -> str:
        """获取显示名称"""
        if lang == "en" and self.name_en:
            return self.name_en
        return self.name
    
    def get_description(self, lang: str = "zh") -> str:
        """获取描述"""
        if lang == "en" and self.description_en:
            return self.description_en
        return self.description
    
    def get_tags(self, lang: str = "zh") -> List[str]:
        """获取标签"""
        if lang == "en" and self.tags_en:
            return self.tags_en
        return self.tags


class SkillLoader:
    """Skill 加载器"""
    
    def __init__(self, skills_dir: Path = None):
        if skills_dir is None:
            skills_dir = Path(__file__).parent.parent / "skills"
        self.skills_dir = Path(skills_dir)
        self._cache: Dict[str, Skill] = {}
    
    def load_all(self, force_reload: bool = False) -> List[Skill]:
        """加载所有 skill
        
        Args:
            force_reload: 是否强制重新加载
            
        Returns:
            Skill 列表
        """
        if self._cache and not force_reload:
            return list(self._cache.values())
        
        self._cache.clear()
        
        if not self.skills_dir.exists():
            return []
        
        for f in sorted(self.skills_dir.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    skill = Skill.from_dict(data)
                    self._cache[skill.id] = skill
            except Exception as e:
                print(f"[WARN] Failed to load skill {f}: {e}")
        
        return list(self._cache.values())
    
    def get_by_id(self, skill_id: str) -> Optional[Skill]:
        """根据 ID 获取 skill"""
        if not self._cache:
            self.load_all()
        return self._cache.get(skill_id)
    
    def get_by_goal(self, goal: str) -> List[Skill]:
        """根据目标类型筛选 skill"""
        if not self._cache:
            self.load_all()
        return [s for s in self._cache.values() if s.goal == goal]
    
    def get_by_name(self, name: str) -> Optional[Skill]:
        """根据名称获取 skill"""
        if not self._cache:
            self.load_all()
        for skill in self._cache.values():
            if skill.name == name or skill.name_en == name:
                return skill
        return None


class SkillValidator:
    """Skill schema 校验器"""
    
    REQUIRED_FIELDS = ["id", "name", "goal", "config"]
    VALID_GOALS = ["unlearn", "inject", "edit"]
    VALID_DATA_MODES = ["single", "batch"]
    VALID_INPUT_FORMATS = ["jsonl", "json", "csv", "text"]
    
    @classmethod
    def validate(cls, skill: Skill) -> List[str]:
        """校验 skill，返回错误列表
        
        Args:
            skill: 要校验的 Skill
            
        Returns:
            错误信息列表，空列表表示校验通过
        """
        errors = []
        
        # 必填字段
        if not skill.id:
            errors.append("Missing required field: id")
        if not skill.name:
            errors.append("Missing required field: name")
        if not skill.goal:
            errors.append("Missing required field: goal")
        if not skill.config:
            errors.append("Missing required field: config")
        
        # 目标类型
        if skill.goal and skill.goal not in cls.VALID_GOALS:
            errors.append(f"Invalid goal: {skill.goal}, must be one of {cls.VALID_GOALS}")
        
        # config 中的 mode 必须与 goal 一致
        if skill.config and skill.config.get("mode") != skill.goal:
            errors.append(f"config.mode ({skill.config.get('mode')}) must match goal ({skill.goal})")
        
        # 数据模式校验
        for mode in skill.supported_data_modes:
            if mode not in cls.VALID_DATA_MODES:
                errors.append(f"Invalid data mode: {mode}")
        
        # 输入格式校验
        for fmt in skill.supported_input_formats:
            if fmt not in cls.VALID_INPUT_FORMATS:
                errors.append(f"Invalid input format: {fmt}")
        
        return errors
    
    @classmethod
    def is_valid(cls, skill: Skill) -> bool:
        """检查 skill 是否有效"""
        return len(cls.validate(skill)) == 0


def create_skill_template(
    skill_id: str,
    name: str,
    goal: str,
    description: str = "",
    model: str = "Qwen2.5-7B-Instruct",
    trainer: str = None,
    experiment: str = None,
    **kwargs
) -> Skill:
    """创建 skill 模板的便捷函数
    
    Args:
        skill_id: 唯一标识
        name: 显示名称
        goal: 目标类型
        description: 描述
        model: 推荐模型
        trainer: 训练方法
        experiment: 实验模板
        **kwargs: 其他参数
        
    Returns:
        Skill 对象
    """
    # 根据 goal 设置默认 trainer
    default_trainers = {
        "unlearn": "SimNPO",
        "inject": "inject/LoRA",
        "edit": "edit/ROME",
    }
    if trainer is None:
        trainer = default_trainers.get(goal, "SimNPO")
    
    config = {
        "mode": goal,
        "model": model,
        "trainer": trainer,
        "task_name": f"{skill_id}_run",
        "seed": 42,
        "learning_rate": kwargs.get("learning_rate", "1e-5"),
        "num_epochs": kwargs.get("num_epochs", 3),
        "batch_size": kwargs.get("batch_size", 4),
        "gradient_accumulation": kwargs.get("gradient_accumulation", 4),
    }
    
    if experiment:
        config["experiment"] = experiment
    
    return Skill(
        id=skill_id,
        name=name,
        goal=goal,
        description=description,
        config=config,
        recommended_models=[model],
        recommended_eval=kwargs.get("recommended_eval", goal),
        resource_estimate=kwargs.get("resource_estimate", ""),
        tags=kwargs.get("tags", []),
    )
