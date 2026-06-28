"""
Agent LLM 抽象层 - 支持 Skill-based prompt routing 和结构化 action 输出。
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List


SETTINGS_PATH = Path(__file__).resolve().parent.parent / ".agent_settings.json"

# ─── Base prompt ───

BASE_SYSTEM_PROMPT = """你是 Know-Surgery 智能助手，专注于大模型知识更新实验。
你帮助用户完成 Unlearn（知识遗忘）、Inject（知识注入）、Edit（知识编辑）三类操作的配置、分析和优化。

重要规则：
- 如果用户使用中文提问，你必须全程使用中文回复，不要切换到英文。
- 如果用户使用英文提问，你使用英文回复。
- 回复要简洁实用，直接给出配置建议和操作步骤。
- 不要输出冗长的思考过程或 disclaimer。"""

ACTION_FORMAT_INSTRUCTION = """
当你推荐配置时，请在回复末尾输出一个 JSON 代码块，格式如下：
```json
{"actions": [{"type": "apply_config", "preview": "简短描述", "payload": {...}}]}
```
action type 可以是:
- "apply_config": payload 字段说明：
  - mode: 字符串，如 "unlearn"、"inject"、"edit"
  - model: 字符串，模型配置名（如 "Llama-3.2-1B-Instruct"），不要用嵌套对象
  - trainer: 字符串，训练方法名（如 "GradAscent"），不要用嵌套对象
  - datasets: 对象，键为 forget/retain/edit/train，值为数据集配置名字符串（如 {"forget": "TOFU_QA_forget", "retain": "TOFU_QA_retain"}）
  - eval: 字符串，评测套件名
  - params: 对象，训练参数（如 {"learning_rate": 1e-5, "num_epochs": 5}）
- "navigate_view": payload 包含 {"view": "workshop|monitor|results|skills"}
如果不需要执行操作，可以不输出 JSON 代码块，只输出文本回答。
"""

# ─── Skill prompts ───

SKILL_PROMPTS: Dict[str, str] = {
    "config_wizard": """你正在帮助用户配置实验。根据用户描述的目标：
- 推荐合适的 mode（unlearn/inject/edit）
- 推荐模型：必须从上下文中"系统中可用的模型"列表里选择，payload.model 使用字符串名称
- 推荐训练方法：必须从上下文中"系统中可用的训练方法"列表里选择，payload.trainer 使用字符串名称
- 推荐数据集：必须从上下文中"系统中可用的数据集"列表里选择，payload.datasets 使用确切名称如 {"forget": "TOFU_QA_forget", "retain": "TOFU_QA_retain"}
- 推荐评测套件：必须从上下文中"系统中可用的评测套件"列表里选择
每次推荐都输出 apply_config action。""",

    "param_tuner": """你正在帮助用户调优训练参数。基于已选的模型和方法：
- 推荐 learning_rate（通常 1e-5 到 5e-5）
- 推荐 num_epochs（通常 3-10）
- 推荐 batch_size（根据 GPU 显存）
- 解释各参数对遗忘/保留效果的影响
输出 apply_config action，payload.params 包含推荐参数。""",

    "skill_recommender": """你正在帮助用户选择实验模板。当前可用的 Skill 模板列表会在上下文中提供。
根据用户目标匹配最合适的模板，解释为什么推荐它，并输出 apply_config action 加载该模板的配置。""",

    "edit_config_guide": """你正在帮助用户配置 Edit（知识编辑）实验。
Edit 模式使用 ROME、MEMIT 等方法，与 Unlearn 不同：
- 需要指定 target_new（新知识）和 subject（编辑对象）
- ROME 适合单条编辑，MEMIT 适合批量编辑
- 学习率通常设置较低（5e-5 到 1e-4）
- 编辑数据集格式包含 prompt/subject/target_new 三元组
根据用户需求推荐 Edit 专用配置，数据集必须从"系统中可用的数据集"列表选择。""",

    "dataset_guide": """你正在帮助用户选择数据集。
重要规则：你必须只推荐上下文中"系统中可用的数据集"列表里存在的名称，绝不要编造不存在的数据集路径或名称。
常见数据集分组：
- TOFU 系列(unlearn): TOFU_QA_forget, TOFU_QA_retain, TOFU_QA_full 等，基于虚构作者的 QA 数据
- MUSE 系列(unlearn): MUSE_forget, MUSE_retain, MUSE_train 等，新闻文本遗忘基准
- RWKU 系列(unlearn): RWKU_forget_level1, RWKU_neighbor_level1 等，真实世界知识遗忘
- WMDP 系列(unlearn): WMDP_forget, WMDP_retain，危险知识遗忘
- Edit 系列: CounterFact_edit, ZSRE_edit, UnKE_edit, ELKEN_edit 等
- Inject 系列: Alpaca_inject, SmokeAlpaca_inject, Custom_inject 等
在 apply_config action 的 payload.datasets 中使用确切的数据集配置名（字符串），例如 {"forget": "TOFU_QA_forget", "retain": "TOFU_QA_retain"}。""",

    "forget_retain_advisor": """你正在帮助用户理解 Forget/Retain 数据划分。核心原则：
- Forget set：需要模型"忘记"的目标知识
- Retain set：模型应保留的知识（防止灾难性遗忘）
- 比例建议：retain 通常大于等于 forget
- 评估时需要在两个集上分别测试
推荐数据集时必须使用上下文中"系统中可用的数据集"列表里的确切名称。""",

    "result_analyzer": """你正在帮助用户分析实验结果。关键指标包括：
- Forget Accuracy / Forget Quality：遗忘效果（越低越好表示遗忘越彻底）
- Retain Accuracy：保留效果（越高越好表示没有灾难性遗忘）
- Model Utility：模型通用能力保持度
- Truth Ratio / Probability：更细粒度的遗忘质量指标
解释结果含义，指出 trade-off，如果在 Results 页面可建议 navigate_view 到 workshop 调整。""",

    "next_step_advisor": """你正在帮助用户规划下一步实验。基于当前结果：
- 如果遗忘不充分：建议增加 epochs、提高 learning_rate、或换更强的遗忘方法
- 如果保留下降严重：建议降低 learning_rate、增加 retain 数据比例、或用 GradDiff/NPO
- 如果整体不理想：建议换模型或换 benchmark
输出 apply_config action 和 navigate_view action，所有推荐项必须来自系统可用资源列表。""",

    "concept_explainer": """你正在回答关于知识更新的概念问题。核心概念：
- Unlearn（机器遗忘）：让模型忘记特定训练数据，应对隐私/版权/安全需求
- Inject（知识注入）：向模型注入新知识，让它学会新的事实
- Edit（知识编辑）：精准修改模型中的特定知识，不影响其他知识
用清晰的类比和例子解释，不需要输出 action。""",

    "method_comparator": """你正在帮助用户对比不同方法。以下是框架中已集成的完整方法清单：

【Unlearn 遗忘类】
- GradAscent: 梯度上升，简单直接的基线，可能过度遗忘
- GradDiff: 梯度差异，同时在 forget 上梯度上升 + retain 上梯度下降，平衡遗忘与保持
- NPO: 负偏好优化 (383引用)，基于 DPO 思想避免灾难性崩溃
- SimNPO: 简化版 NPO (NeurIPS 2025)，无需参考模型，TOFU+MUSE+WMDP 三 bench SOTA
- RMU: 表示误导法 (ICML 2024)，将危险知识表示重定向，安全遗忘 SOTA
- DPO: 偏好优化用于遗忘
- UNDIAL: 自蒸馏 + 调整 Logits (NAACL 2025)，鲁棒性好
- CEU: Cross-Entropy 遗忘
- SatImp / WGA / PDU: 其他变体方法
- MM-GradAscent / MM-NPO: 多模态遗忘方法，适用于 LLaVA 等 MLLM

【Inject 注入类】
- InjectTrainer: 全参数微调，效果最好但计算成本高
- LoRA: 低秩适配器，仅训练 <1% 参数，显存友好
- DoRA: 权重分解低秩适配，LoRA 改进版
- AdaLoRA: 自适应秩分配的 LoRA
- LoReFT: 低秩表示微调

【Edit 编辑类】
- ROME: 因果追踪 + 秩一修改 (NeurIPS 2022)，单条精准编辑
- MEMIT: ROME 批量扩展，多层同时修改
- AlphaEdit: 零空间投影 (ICLR 2025 Outstanding)，+36.7% 提升
- AnyEdit: 知识分块迭代编辑 (ICML 2025)，支持长文本/非结构化知识
- MEND: 元学习编辑器
- IKE: 上下文示例注入，无需修改权重
- GRACE: 激活空间 codebook，终身编辑
- WISE: 双记忆系统 (NeurIPS 2024)，终身编辑 SOTA
- SERAC: 外部记忆 + 范围分类器
- NMKE: 神经元级归因 (NeurIPS 2025)，终身编辑数千次 SOTA
- MEMIT-Merge: 修复 MEMIT 同主语批量编辑冲突 (ACL 2025)
- MM-IKE / UniKE: 多模态编辑方法

根据用户场景推荐方法，推荐的方法名必须来自系统可用资源列表，可输出 apply_config action。""",

    "scenario_advisor": """你正在帮助用户根据具体场景选择最佳方法。

场景匹配指南：
1. 长文本/非结构化知识编辑 → AnyEdit (ICML 2025)，支持诗歌/代码/数学等多格式
2. 单条精准事实编辑 → ROME 或 AlphaEdit
3. 批量事实编辑 → MEMIT 或 MEMIT-Merge（同主语时用 Merge）
4. 终身/序列编辑（数千次） → NMKE > WISE > GRACE
5. 隐私数据遗忘 → SimNPO（最新 SOTA）或 NPO
6. 安全/危险知识删除 → RMU
7. 多模态知识编辑 → UniKE（NeurIPS 2024 Spotlight）或 MM-IKE
8. 多模态遗忘 → MM-NPO 或 MM-GradAscent
9. 知识注入（显存有限） → LoRA 或 DoRA
10. 知识注入（追求效果） → InjectTrainer (Full FT)
11. 快速验证/无需训练 → IKE（上下文编辑）
12. 平衡遗忘与保持 → GradDiff 或 NPO

请根据用户描述的场景，推荐 1-3 个最合适的方法，说明理由，并给出配置建议。
推荐的方法名必须来自系统可用资源列表，可输出 apply_config action。""",
}

# ─── Agent skill metadata (returned to frontend) ───

AGENT_SKILLS_META = [
    {"id": "config_wizard", "name": "实验配置向导", "name_en": "Config Wizard", "category": "config", "route": "workshop"},
    {"id": "param_tuner", "name": "参数调优", "name_en": "Param Tuner", "category": "config", "route": "workshop"},
    {"id": "skill_recommender", "name": "模板推荐", "name_en": "Skill Recommender", "category": "config", "route": "workshop"},
    {"id": "edit_config_guide", "name": "Edit 配置引导", "name_en": "Edit Config Guide", "category": "config", "route": "workshop"},
    {"id": "dataset_guide", "name": "数据集指南", "name_en": "Dataset Guide", "category": "data", "route": "workshop"},
    {"id": "forget_retain_advisor", "name": "数据划分建议", "name_en": "Forget/Retain Advisor", "category": "data", "route": "workshop"},
    {"id": "result_analyzer", "name": "结果分析", "name_en": "Result Analyzer", "category": "results", "route": "results"},
    {"id": "next_step_advisor", "name": "下一步建议", "name_en": "Next Step Advisor", "category": "results", "route": "results"},
    {"id": "concept_explainer", "name": "概念解释", "name_en": "Concept Explainer", "category": "knowledge", "route": "*"},
    {"id": "method_comparator", "name": "方法对比", "name_en": "Method Comparator", "category": "knowledge", "route": "*"},
    {"id": "scenario_advisor", "name": "场景推荐", "name_en": "Scenario Advisor", "category": "knowledge", "route": "*"},
]


# ─── Prompt construction ───

def build_system_prompt(context: dict) -> str:
    """Build the full system prompt based on route, active skill, and experiment context."""
    parts = [BASE_SYSTEM_PROMPT]

    active_skill = context.get("activeSkill")
    route = context.get("route", "workshop")

    if not active_skill:
        if route == "results":
            active_skill = "result_analyzer"
        elif route == "workshop":
            mode = context.get("mode", "")
            if mode == "edit":
                active_skill = "edit_config_guide"
            elif context.get("model") and context.get("trainer"):
                active_skill = "param_tuner"
            else:
                active_skill = "config_wizard"

    if active_skill and active_skill in SKILL_PROMPTS:
        parts.append(f"\n--- 当前能力模式: {active_skill} ---")
        parts.append(SKILL_PROMPTS[active_skill])

    parts.append(ACTION_FORMAT_INSTRUCTION)

    ctx_lines: list[str] = []
    for key, label in [
        ("mode", "当前实验模式"),
        ("model", "已选模型"),
        ("trainer", "已选方法"),
        ("eval", "评测套件"),
    ]:
        if context.get(key):
            ctx_lines.append(f"{label}: {context[key]}")

    for key, label in [
        ("datasets", "已选数据集"),
        ("params", "当前参数"),
        ("skills", "可用 Skill 模板"),
        ("results", "结果摘要"),
    ]:
        if context.get(key):
            ctx_lines.append(f"{label}: {json.dumps(context[key], ensure_ascii=False)}")

    if ctx_lines:
        parts.append("\n--- 当前实验上下文 ---")
        parts.extend(ctx_lines)

    catalog_lines: list[str] = []
    if context.get("available_datasets"):
        catalog_lines.append(f"系统中可用的数据集: {json.dumps(context['available_datasets'], ensure_ascii=False)}")
    if context.get("available_models"):
        catalog_lines.append(f"系统中可用的模型: {json.dumps(context['available_models'], ensure_ascii=False)}")
    if context.get("available_trainers"):
        catalog_lines.append(f"系统中可用的训练方法: {json.dumps(context['available_trainers'], ensure_ascii=False)}")
    if context.get("available_evals"):
        catalog_lines.append(f"系统中可用的评测套件: {json.dumps(context['available_evals'], ensure_ascii=False)}")

    if catalog_lines:
        parts.append("\n--- 系统可用资源（只推荐以下列表中存在的项目） ---")
        parts.extend(catalog_lines)

    return "\n".join(parts)


def get_suggested_skills(context: dict) -> List[str]:
    """Return a list of recommended skill IDs based on current context."""
    route = context.get("route", "workshop")
    mode = context.get("mode", "")
    suggestions: list[str] = []

    if route == "workshop":
        suggestions = ["config_wizard", "param_tuner", "dataset_guide"]
        if mode == "edit":
            suggestions.insert(0, "edit_config_guide")
        if context.get("skills"):
            suggestions.append("skill_recommender")
    elif route == "results":
        suggestions = ["result_analyzer", "next_step_advisor"]
    elif route == "skills":
        suggestions = ["skill_recommender", "config_wizard"]

    suggestions.extend(["concept_explainer", "method_comparator", "scenario_advisor"])
    return suggestions[:6]


def parse_agent_response(raw_text: str) -> dict:
    """Extract structured actions from the LLM's raw text output."""
    result: Dict = {"message": raw_text, "actions": []}

    pattern = r"```json\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, raw_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "actions" in data:
                actions = data["actions"]
                for i, a in enumerate(actions):
                    a.setdefault("id", f"act_{i}")
                    a.setdefault("status", "pending")
                result["actions"] = actions
            result["message"] = raw_text[: match.start()].strip()
        except (json.JSONDecodeError, KeyError):
            pass

    return result


# ─── Config & LLM client ───

@dataclass
class AgentConfig:
    provider: str = "openai"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048


def load_settings() -> AgentConfig:
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text("utf-8"))
            return AgentConfig(
                **{k: v for k, v in data.items() if hasattr(AgentConfig, k)}
            )
        except Exception:
            pass
    return AgentConfig(
        provider="openai",
        base_url=os.getenv("OPENAI_API_BASE", "https://api.gpt.ge/v1/"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-5.2-high"),
    )


def save_settings(cfg: AgentConfig):
    SETTINGS_PATH.write_text(
        json.dumps(cfg.__dict__, ensure_ascii=False, indent=2), "utf-8"
    )


def _get_client(cfg: AgentConfig):
    from openai import OpenAI
    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=60)


def test_connection(cfg: AgentConfig) -> Dict:
    try:
        client = _get_client(cfg)
        client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        return {"ok": True, "model": cfg.model}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def stream_chat(
    messages: List[Dict[str, str]],
    cfg: AgentConfig = None,
    system_prompt: str = None,
) -> Generator[str, None, None]:
    if cfg is None:
        cfg = load_settings()
    client = _get_client(cfg)

    full_messages = [
        {"role": "system", "content": system_prompt or BASE_SYSTEM_PROMPT}
    ]
    full_messages.extend(messages)

    try:
        stream = client.chat.completions.create(
            model=cfg.model,
            messages=full_messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"\n[ERROR] {e}"
