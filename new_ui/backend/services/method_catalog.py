"""三大方法分类清单 + 场景推荐函数。

基于综述文件 (multimodal_knowledge_editing_survey.md, multimodal_unlearning_survey.md,
text_llm_unlearning_survey.md) 和仓库已有方法，提供结构化的方法推荐。
"""

from __future__ import annotations

from typing import Any, Dict, List

# 每个方法条目的字段：
#   name: 方法名
#   category: unlearn / inject / edit
#   subcategory: 细分类别
#   config_entry: 框架中的 trainer= 配置名，None 表示未集成
#   description: 一句话描述
#   scenarios: 适用场景关键词列表
#   data_format: 数据格式要求
#   recommended_models: 推荐模型列表
#   source: 综述来源
#   status: integrated / not_integrated

METHOD_CATALOG: List[Dict[str, Any]] = [
    # ═══════════════════════════════════════════════════
    # Unlearn 类方法
    # ═══════════════════════════════════════════════════
    {
        "name": "GradAscent",
        "category": "unlearn",
        "subcategory": "梯度方法",
        "config_entry": "GradAscent",
        "description": "梯度上升，在遗忘集上反向优化使模型忘记目标知识。简单直接的基线方法。",
        "scenarios": ["隐私遗忘", "简单基线", "快速实验", "QA知识遗忘"],
        "data_format": "QA 格式: {question, answer}",
        "recommended_models": ["Llama-2-7B", "Llama-3-8B", "GPT-2-XL"],
        "source": "通用基线方法",
        "status": "integrated",
    },
    {
        "name": "GradDiff",
        "category": "unlearn",
        "subcategory": "梯度方法",
        "config_entry": "GradDiff",
        "description": "梯度差异法，同时在遗忘集上梯度上升 + 保留集上梯度下降，平衡遗忘与保持。",
        "scenarios": ["隐私遗忘", "平衡遗忘保持", "有retain数据", "QA知识遗忘"],
        "data_format": "QA 格式: {question, answer}，需要 forget + retain 两个数据集",
        "recommended_models": ["Llama-2-7B", "Llama-3-8B"],
        "source": "通用基线方法",
        "status": "integrated",
    },
    {
        "name": "NPO",
        "category": "unlearn",
        "subcategory": "偏好优化",
        "config_entry": "NPO",
        "description": "负偏好优化，基于 DPO 思想避免灾难性崩溃。383 引用，首个大规模遗忘有效方法。",
        "scenarios": ["隐私遗忘", "大规模遗忘", "避免崩溃", "TOFU基准", "QA知识遗忘"],
        "data_format": "QA 格式: {question, answer}",
        "recommended_models": ["Llama-2-7B", "Phi-1.5"],
        "source": "arXiv 2024, 383 引用",
        "status": "integrated",
    },
    {
        "name": "SimNPO",
        "category": "unlearn",
        "subcategory": "偏好优化",
        "config_entry": "SimNPO",
        "description": "简化版 NPO，去除参考模型依赖。NeurIPS 2025，TOFU+MUSE+WMDP 三 bench SOTA。",
        "scenarios": ["隐私遗忘", "无参考模型", "高效遗忘", "多基准SOTA", "QA知识遗忘"],
        "data_format": "QA 格式: {question, answer}",
        "recommended_models": ["Llama-2-7B", "Zephyr-7B"],
        "source": "NeurIPS 2025, 87 引用",
        "status": "integrated",
    },
    {
        "name": "RMU",
        "category": "unlearn",
        "subcategory": "表示操作",
        "config_entry": "RMU",
        "description": "表示误导法，将危险知识的表示重定向到随机方向。安全遗忘 SOTA。",
        "scenarios": ["安全遗忘", "危险知识删除", "WMDP基准", "生物化学网络安全"],
        "data_format": "QA 格式: {question, answer}",
        "recommended_models": ["Zephyr-7B", "Llama-2-7B"],
        "source": "ICML 2024 (WMDP)",
        "status": "integrated",
    },
    {
        "name": "DPO",
        "category": "unlearn",
        "subcategory": "偏好优化",
        "config_entry": "DPO",
        "description": "偏好优化用于遗忘，成熟的对齐技术。",
        "scenarios": ["偏好对齐遗忘", "QA知识遗忘"],
        "data_format": "QA 格式: {question, answer}",
        "recommended_models": ["Llama-2-7B"],
        "source": "通用基线",
        "status": "integrated",
    },
    {
        "name": "UNDIAL",
        "category": "unlearn",
        "subcategory": "蒸馏方法",
        "config_entry": "UNDIAL",
        "description": "自蒸馏 + 调整 Logits，鲁棒性和可扩展性好。",
        "scenarios": ["鲁棒遗忘", "可扩展遗忘", "QA知识遗忘"],
        "data_format": "QA 格式: {question, answer}",
        "recommended_models": ["Llama-2-7B"],
        "source": "NAACL 2025",
        "status": "integrated",
    },
    {
        "name": "CEU",
        "category": "unlearn",
        "subcategory": "梯度方法",
        "config_entry": "CEU",
        "description": "Cross-Entropy 遗忘，简洁的遗忘方法。",
        "scenarios": ["简单遗忘", "QA知识遗忘"],
        "data_format": "QA 格式: {question, answer}",
        "recommended_models": ["Llama-2-7B"],
        "source": "—",
        "status": "integrated",
    },
    # 多模态遗忘方法
    {
        "name": "MM-GradAscent",
        "category": "unlearn",
        "subcategory": "多模态遗忘",
        "config_entry": "MM_GradAscent",
        "description": "多模态梯度上升遗忘，适用于 LLaVA 等 MLLM。",
        "scenarios": ["多模态遗忘", "MLLM隐私", "图文遗忘"],
        "data_format": "多模态 QA + 图像",
        "recommended_models": ["LLaVA-1.5-7B", "LLaVA-1.5-13B"],
        "source": "框架内置",
        "status": "integrated",
    },
    {
        "name": "MM-NPO",
        "category": "unlearn",
        "subcategory": "多模态遗忘",
        "config_entry": "MM_NPO",
        "description": "多模态 NPO，将负偏好优化扩展到多模态场景。",
        "scenarios": ["多模态遗忘", "MLLM隐私", "图文遗忘", "避免崩溃"],
        "data_format": "多模态 QA + 图像",
        "recommended_models": ["LLaVA-1.5-7B"],
        "source": "框架内置",
        "status": "integrated",
    },
    # ═══════════════════════════════════════════════════
    # Inject 类方法
    # ═══════════════════════════════════════════════════
    {
        "name": "InjectTrainer (Full FT)",
        "category": "inject",
        "subcategory": "全参数微调",
        "config_entry": "InjectTrainer",
        "description": "全参数微调注入新知识，效果最好但计算成本高。",
        "scenarios": ["知识注入", "全参数微调", "小数据集", "指令微调"],
        "data_format": "Alpaca: {instruction, output} 或 ShareGPT: {conversations}",
        "recommended_models": ["Llama-2-7B", "Llama-3-8B", "Qwen2.5-7B"],
        "source": "通用方法",
        "status": "integrated",
    },
    {
        "name": "LoRA",
        "category": "inject",
        "subcategory": "参数高效微调",
        "config_entry": "LoRA",
        "description": "低秩适配器，仅训练 <1% 参数即可注入新知识。显存友好。",
        "scenarios": ["知识注入", "参数高效", "显存有限", "大模型微调", "指令微调"],
        "data_format": "Alpaca: {instruction, output} 或 ShareGPT: {conversations}",
        "recommended_models": ["Llama-2-7B", "Llama-3-8B", "Qwen2.5-7B", "Mistral-7B"],
        "source": "通用方法",
        "status": "integrated",
    },
    {
        "name": "DoRA",
        "category": "inject",
        "subcategory": "参数高效微调",
        "config_entry": "DoRA",
        "description": "权重分解低秩适配，LoRA 的改进版，分解方向和幅度。",
        "scenarios": ["知识注入", "参数高效", "LoRA改进"],
        "data_format": "Alpaca: {instruction, output} 或 ShareGPT: {conversations}",
        "recommended_models": ["Llama-2-7B", "Llama-3-8B"],
        "source": "通用方法",
        "status": "integrated",
    },
    {
        "name": "AdaLoRA",
        "category": "inject",
        "subcategory": "参数高效微调",
        "config_entry": "AdaLoRA",
        "description": "自适应秩分配的 LoRA，自动为不同层分配不同秩。",
        "scenarios": ["知识注入", "参数高效", "自适应秩"],
        "data_format": "Alpaca: {instruction, output} 或 ShareGPT: {conversations}",
        "recommended_models": ["Llama-2-7B", "Llama-3-8B"],
        "source": "通用方法",
        "status": "integrated",
    },
    {
        "name": "LoReFT",
        "category": "inject",
        "subcategory": "表示微调",
        "config_entry": "LoReFT",
        "description": "低秩表示微调，在表示空间而非权重空间进行适配。",
        "scenarios": ["知识注入", "表示空间微调", "参数高效"],
        "data_format": "Alpaca: {instruction, output} 或 ShareGPT: {conversations}",
        "recommended_models": ["Llama-2-7B", "Llama-3-8B"],
        "source": "通用方法",
        "status": "integrated",
    },
    # ═══════════════════════════════════════════════════
    # Edit 类方法
    # ═══════════════════════════════════════════════════
    {
        "name": "ROME",
        "category": "edit",
        "subcategory": "定位编辑 (Locate-then-Edit)",
        "config_entry": "ROME",
        "description": "因果追踪定位 MLP 层 + 秩一矩阵修改 FFN 权重。单条精准编辑的奠基方法。",
        "scenarios": ["单条事实编辑", "结构化三元组", "精准编辑", "因果追踪"],
        "data_format": "Edit 三元组: {prompt, subject, target_new}",
        "recommended_models": ["GPT-J-6B", "GPT-2-XL", "Llama-2-7B"],
        "source": "NeurIPS 2022, 1000+ 引用",
        "status": "integrated",
    },
    {
        "name": "MEMIT",
        "category": "edit",
        "subcategory": "定位编辑 (Locate-then-Edit)",
        "config_entry": "MEMIT",
        "description": "ROME 的批量扩展，多层 MLP 同时修改。适合批量编辑。",
        "scenarios": ["批量事实编辑", "结构化三元组", "多条同时编辑"],
        "data_format": "Edit 三元组: {prompt, subject, target_new}",
        "recommended_models": ["GPT-J-6B", "GPT-NeoX-20B", "Llama-2-7B"],
        "source": "NeurIPS 2022, 800+ 引用",
        "status": "integrated",
    },
    {
        "name": "AlphaEdit",
        "category": "edit",
        "subcategory": "定位编辑 (Locate-then-Edit)",
        "config_entry": "AlphaEdit",
        "description": "零空间投影，将编辑扰动投射到保留知识的零空间。ICLR 2025 Outstanding Paper，平均提升 36.7%。",
        "scenarios": ["精准编辑", "序列编辑", "保持原有知识", "结构化三元组"],
        "data_format": "Edit 三元组: {prompt, subject, target_new}",
        "recommended_models": ["GPT-J-6B", "GPT-2-XL", "Llama-3-8B"],
        "source": "ICLR 2025 Outstanding Paper",
        "status": "integrated",
    },
    {
        "name": "AnyEdit",
        "category": "edit",
        "subcategory": "自回归编辑",
        "config_entry": "AnyEdit",
        "description": "知识分块 + 迭代编辑关键 token。支持长文本、非结构化知识编辑。ICML 2025。",
        "scenarios": ["长文本编辑", "非结构化知识", "多格式编辑", "诗歌代码数学"],
        "data_format": "支持多种格式，包括长文本",
        "recommended_models": ["GPT-J-6B", "Llama-2-7B", "Llama-3-8B"],
        "source": "ICML 2025",
        "status": "integrated",
    },
    {
        "name": "MEND",
        "category": "edit",
        "subcategory": "元学习编辑",
        "config_entry": "MEND",
        "description": "超网络学习编辑方向，小梯度高效修正。需要预训练编辑器。",
        "scenarios": ["快速编辑", "元学习", "结构化三元组"],
        "data_format": "Edit 三元组: {prompt, subject, target_new}",
        "recommended_models": ["GPT-2-XL", "GPT-J-6B"],
        "source": "ICLR 2022, 500+ 引用",
        "status": "integrated",
    },
    {
        "name": "IKE",
        "category": "edit",
        "subcategory": "上下文编辑 (In-Context)",
        "config_entry": "IKE",
        "description": "上下文示例注入，无需修改权重。最简单的编辑方式。",
        "scenarios": ["无需训练", "上下文编辑", "快速验证", "结构化三元组"],
        "data_format": "Edit 三元组: {prompt, subject, target_new}",
        "recommended_models": ["GPT-J-6B", "Llama-2-7B", "任意模型"],
        "source": "ACL 2023",
        "status": "integrated",
    },
    {
        "name": "GRACE",
        "category": "edit",
        "subcategory": "记忆编辑 (Memory-based)",
        "config_entry": "GRACE",
        "description": "激活空间离散 codebook 编辑，无需改权重，支持终身编辑。",
        "scenarios": ["终身编辑", "序列编辑", "不修改权重", "结构化三元组"],
        "data_format": "Edit 三元组: {prompt, subject, target_new}",
        "recommended_models": ["GPT-2-XL", "GPT-J-6B"],
        "source": "NeurIPS 2023, 200+ 引用",
        "status": "integrated",
    },
    {
        "name": "WISE",
        "category": "edit",
        "subcategory": "记忆编辑 (Memory-based)",
        "config_entry": "WISE",
        "description": "双记忆系统（主记忆+侧记忆）+ 知识分片 + 无冲突合并。终身编辑 SOTA。",
        "scenarios": ["终身编辑", "大规模序列编辑", "知识分片", "结构化三元组"],
        "data_format": "Edit 三元组: {prompt, subject, target_new}",
        "recommended_models": ["GPT-J-6B", "Llama-2-7B", "Mistral-7B"],
        "source": "NeurIPS 2024",
        "status": "integrated",
    },
    {
        "name": "SERAC",
        "category": "edit",
        "subcategory": "记忆编辑 (Memory-based)",
        "config_entry": "SERAC",
        "description": "外部记忆存储 + 范围分类器路由。不修改原始模型权重。",
        "scenarios": ["外部记忆", "不修改权重", "结构化三元组"],
        "data_format": "Edit 三元组: {prompt, subject, target_new}",
        "recommended_models": ["GPT-J-6B"],
        "source": "ICML 2022, 400+ 引用",
        "status": "integrated",
    },
    {
        "name": "NMKE",
        "category": "edit",
        "subcategory": "神经元级编辑",
        "config_entry": "NMKE",
        "description": "神经元级归因 + 熵引导动态稀疏掩码。终身编辑（数千次）SOTA。",
        "scenarios": ["终身编辑", "大规模序列编辑", "神经元级精准", "结构化三元组"],
        "data_format": "Edit 三元组: {prompt, subject, target_new}",
        "recommended_models": ["Llama-2-7B", "Llama-3-8B"],
        "source": "NeurIPS 2025",
        "status": "integrated",
    },
    {
        "name": "MEMIT-Merge",
        "category": "edit",
        "subcategory": "定位编辑 (Locate-then-Edit)",
        "config_entry": "MEMITMerge",
        "description": "修复 MEMIT 同主语批量编辑时的键值冲突问题 (46%→98%)。",
        "scenarios": ["批量编辑", "同主语编辑", "MEMIT改进", "结构化三元组"],
        "data_format": "Edit 三元组: {prompt, subject, target_new}",
        "recommended_models": ["GPT-J-6B", "Llama-2-7B"],
        "source": "ACL 2025 Findings",
        "status": "integrated",
    },
    # 多模态编辑方法
    {
        "name": "MM-IKE",
        "category": "edit",
        "subcategory": "多模态编辑",
        "config_entry": "MM_IKE",
        "description": "多模态上下文编辑，将 IKE 扩展到视觉-语言场景。",
        "scenarios": ["多模态编辑", "上下文编辑", "图文知识更新"],
        "data_format": "多模态 Edit 三元组 + 图像",
        "recommended_models": ["BLIP2-OPT-2.7B", "LLaVA-1.5-7B"],
        "source": "MMEdit (EMNLP 2023)",
        "status": "integrated",
    },
    {
        "name": "UniKE",
        "category": "edit",
        "subcategory": "多模态编辑",
        "config_entry": "UniKE",
        "description": "统一内在编辑(ROME) + 外部知识(IKE)为向量化 KV 记忆。NeurIPS 2024 Spotlight。",
        "scenarios": ["多模态编辑", "统一编辑", "图文知识更新"],
        "data_format": "多模态 Edit 三元组 + 图像",
        "recommended_models": ["BLIP2-OPT-2.7B", "LLaVA-1.5-7B"],
        "source": "NeurIPS 2024 Spotlight",
        "status": "integrated",
    },
]


# ─── 场景关键词到方法的映射 ───

_SCENARIO_KEYWORDS = {
    "长文本": ["AnyEdit", "WISE", "GRACE"],
    "非结构化": ["AnyEdit", "InjectTrainer (Full FT)", "LoRA"],
    "非结构化知识编辑": ["AnyEdit"],
    "结构化三元组": ["ROME", "MEMIT", "AlphaEdit", "MEND"],
    "单条编辑": ["ROME", "AlphaEdit"],
    "批量编辑": ["MEMIT", "MEMIT-Merge", "AlphaEdit"],
    "终身编辑": ["WISE", "GRACE", "NMKE"],
    "序列编辑": ["WISE", "NMKE", "AlphaEdit"],
    "隐私遗忘": ["NPO", "SimNPO", "GradDiff", "GradAscent"],
    "安全遗忘": ["RMU", "SimNPO"],
    "危险知识": ["RMU"],
    "知识注入": ["LoRA", "InjectTrainer (Full FT)", "DoRA", "AdaLoRA"],
    "指令微调": ["LoRA", "InjectTrainer (Full FT)"],
    "参数高效": ["LoRA", "DoRA", "AdaLoRA", "LoReFT"],
    "显存有限": ["LoRA", "DoRA"],
    "多模态": ["MM-IKE", "UniKE", "MM-GradAscent", "MM-NPO"],
    "图文": ["MM-IKE", "UniKE", "MM-GradAscent", "MM-NPO"],
    "MLLM": ["MM-IKE", "UniKE", "MM-GradAscent", "MM-NPO"],
    "上下文编辑": ["IKE", "MM-IKE"],
    "无需训练": ["IKE", "MM-IKE"],
    "快速": ["IKE", "GradAscent", "ROME"],
    "精准编辑": ["ROME", "AlphaEdit", "NMKE"],
}

_CATALOG_BY_NAME = {m["name"]: m for m in METHOD_CATALOG}


def recommend_methods(
    scenario: str, top_k: int = 5
) -> List[Dict[str, Any]]:
    """根据用户描述的场景关键词匹配推荐方法。

    返回按匹配度排序的方法列表（最多 top_k 个）。
    """
    scenario_lower = scenario.lower()
    scores: Dict[str, int] = {}

    for keyword, methods in _SCENARIO_KEYWORDS.items():
        if keyword in scenario_lower or keyword in scenario:
            for m in methods:
                scores[m] = scores.get(m, 0) + 1

    # 也检查方法自身的 scenarios 字段
    for m in METHOD_CATALOG:
        for s in m["scenarios"]:
            if s in scenario_lower or s in scenario:
                scores[m["name"]] = scores.get(m["name"], 0) + 1

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    results: List[Dict[str, Any]] = []
    for name, score in ranked[:top_k]:
        if name in _CATALOG_BY_NAME:
            entry = dict(_CATALOG_BY_NAME[name])
            entry["match_score"] = score
            results.append(entry)
    return results


def get_catalog_summary() -> Dict[str, List[Dict[str, str]]]:
    """返回按 category 分组的方法摘要，供 API 和 Agent prompt 使用。"""
    groups: Dict[str, List[Dict[str, str]]] = {
        "unlearn": [],
        "inject": [],
        "edit": [],
    }
    for m in METHOD_CATALOG:
        cat = m["category"]
        if cat not in groups:
            groups[cat] = []
        groups[cat].append({
            "name": m["name"],
            "subcategory": m["subcategory"],
            "config_entry": m.get("config_entry") or "未集成",
            "description": m["description"],
            "scenarios": ", ".join(m["scenarios"][:4]),
            "status": m["status"],
        })
    return groups
