# -*- coding: utf-8 -*-
"""
国际化（i18n）支持
==================

提供中英文界面切换功能。
所有界面文案统一管理，支持一键切换语言。
"""

from typing import Dict, Any
import gradio as gr


# 支持的语言
SUPPORTED_LANGUAGES = ["zh", "en"]
DEFAULT_LANGUAGE = "zh"


# 文案字典
#
# Key naming rule:
# - `<scope>_<name>` for shared UI nouns, e.g. `tab_config`, `run_title`
# - suffix `_label` for component labels
# - suffix `_info` for helper text / tooltip text
# - suffix `_title` for section headings
# - suffix `_placeholder` for empty-state copy
# - suffix `_failed` / `_success` / `_required` for runtime states
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # ==================== 通用 ====================
    "app_title": {
        "zh": "🔬 Know-Surgery WebUI",
        "en": "🔬 Know-Surgery WebUI",
    },
    "app_subtitle": {
        "zh": "大模型知识可控更新工具包",
        "en": "LLM Knowledge Surgery Toolkit",
    },
    "app_description": {
        "zh": "Unlearn · Inject · Edit · Eval",
        "en": "Unlearn · Inject · Edit · Eval",
    },
    
    # ==================== Tab 名称 ====================
    "tab_config": {
        "zh": "⚙️ 配置与运行",
        "en": "⚙️ Config & Run",
    },
    "tab_compare": {
        "zh": "📊 结果对比",
        "en": "📊 Results Compare",
    },
    "tab_demo": {
        "zh": "🎬 交互演示",
        "en": "🎬 Interactive Demo",
    },
    "tab_wizard": {
        "zh": "🧙 智能向导",
        "en": "🧙 Smart Wizard",
    },
    
    # ==================== 配置面板 ====================
    "config_title": {
        "zh": "任务配置",
        "en": "Task Config",
    },
    "mode_label": {
        "zh": "任务模式",
        "en": "Task Mode",
    },
    "mode_info": {
        "zh": "Unlearn=知识删除 · Inject=知识注入 · Edit=知识编辑 · Eval=评估",
        "en": "Unlearn=Delete · Inject=Add · Edit=Modify · Eval=Evaluate",
    },
    "experiment_label": {
        "zh": "实验模板（可选）",
        "en": "Experiment Template (Optional)",
    },
    "experiment_info": {
        "zh": "选择预设配置模板，留空自定义",
        "en": "Select preset template or leave empty for custom",
    },
    "model_label": {
        "zh": "模型",
        "en": "Model",
    },
    "model_info": {
        "zh": "选择预训练模型",
        "en": "Select pretrained model",
    },
    "trainer_label": {
        "zh": "训练方法",
        "en": "Training Method",
    },
    "trainer_info": {
        "zh": "选择算法/方法",
        "en": "Select algorithm/method",
    },
    "forget_dataset_label": {
        "zh": "遗忘数据集 (Forget)",
        "en": "Forget Dataset",
    },
    "forget_dataset_info": {
        "zh": "需要遗忘的数据",
        "en": "Data to forget",
    },
    "retain_dataset_label": {
        "zh": "保留数据集 (Retain)",
        "en": "Retain Dataset",
    },
    "retain_dataset_info": {
        "zh": "需要保持的数据",
        "en": "Data to retain",
    },
    "train_dataset_label": {
        "zh": "训练数据集",
        "en": "Training Dataset",
    },
    "train_dataset_info": {
        "zh": "微调训练数据",
        "en": "Fine-tuning data",
    },
    "edit_dataset_label": {
        "zh": "编辑数据集",
        "en": "Edit Dataset",
    },
    "edit_dataset_info": {
        "zh": "知识编辑数据",
        "en": "Knowledge editing data",
    },
    "eval_suite_label": {
        "zh": "评测套件",
        "en": "Evaluation Suite",
    },
    "eval_suite_info": {
        "zh": "选择评估指标集合",
        "en": "Select evaluation metrics",
    },
    "eval_suite_select_info": {
        "zh": "选择要运行的评测套件",
        "en": "Select the evaluation suite to run",
    },
    "saved_model_label": {
        "zh": "已保存模型",
        "en": "Saved Model",
    },
    "saved_model_info": {
        "zh": "选择 saves/ 目录下的已训练模型，或留空使用 HuggingFace 模型",
        "en": "Select trained model from saves/, or leave empty for HuggingFace model",
    },
    "task_name_label": {
        "zh": "任务名称 (task_name)",
        "en": "Task Name",
    },
    "task_name_info": {
        "zh": "用于标识本次实验",
        "en": "Identifier for this experiment",
    },
    "seed_label": {
        "zh": "随机种子 (seed)",
        "en": "Random Seed",
    },
    "config_upload_label": {
        "zh": "上传配置文件",
        "en": "Upload Config File",
    },
    "config_download_label": {
        "zh": "下载配置",
        "en": "Download Config",
    },
    "import_config": {
        "zh": "📥 导入配置",
        "en": "📥 Import Config",
    },
    "export_config": {
        "zh": "📤 导出配置",
        "en": "📤 Export Config",
    },
    "refresh_models": {
        "zh": "🔄 刷新模型列表",
        "en": "🔄 Refresh Models",
    },
    
    # ==================== 参数面板 ====================
    "params_title": {
        "zh": "参数调节",
        "en": "Parameters",
    },
    "advanced_params_title": {
        "zh": "高级参数",
        "en": "Advanced Parameters",
    },
    "trainer_args_title": {
        "zh": "训练参数 (trainer.args)",
        "en": "Training Args (trainer.args)",
    },
    "learning_rate_label": {
        "zh": "学习率 (learning_rate)",
        "en": "Learning Rate",
    },
    "num_epochs_label": {
        "zh": "训练轮数 (num_train_epochs)",
        "en": "Training Epochs",
    },
    "batch_size_label": {
        "zh": "批次大小 (per_device_train_batch_size)",
        "en": "Batch Size",
    },
    "gradient_accumulation_label": {
        "zh": "梯度累积步数",
        "en": "Gradient Accumulation Steps",
    },
    "max_length_label": {
        "zh": "最大序列长度",
        "en": "Max Sequence Length",
    },
    "warmup_ratio_label": {
        "zh": "预热比例 (warmup_ratio)",
        "en": "Warmup Ratio",
    },
    "method_args_title": {
        "zh": "方法参数 (method_args)",
        "en": "Method Args",
    },
    "method_args_label": {
        "zh": "方法特定参数 (JSON 格式)",
        "en": "Method-specific Args (JSON)",
    },
    "overrides_title": {
        "zh": "Override 编辑器",
        "en": "Override Editor",
    },
    "overrides_label": {
        "zh": "额外 Hydra Overrides",
        "en": "Extra Hydra Overrides",
    },
    "overrides_info": {
        "zh": "每行一个 override，如: trainer.args.weight_decay=0.01",
        "en": "One override per line, e.g.: trainer.args.weight_decay=0.01",
    },
    "overrides_placeholder": {
        "zh": "trainer.args.weight_decay=0.01\nmodel.model_args.torch_dtype=float16",
        "en": "trainer.args.weight_decay=0.01\nmodel.model_args.torch_dtype=float16",
    },
    
    # ==================== 运行面板 ====================
    "run_title": {
        "zh": "运行控制",
        "en": "Run Control",
    },
    "run_btn": {
        "zh": "▶️ 开始运行",
        "en": "▶️ Start Run",
    },
    "stop_btn": {
        "zh": "⏹️ 停止",
        "en": "⏹️ Stop",
    },
    "status_ready": {
        "zh": "**状态**: 就绪",
        "en": "**Status**: Ready",
    },
    "status_running": {
        "zh": "**状态**: 🏃 运行中...",
        "en": "**Status**: 🏃 Running...",
    },
    "status_success": {
        "zh": "**状态**: ✅ 运行成功",
        "en": "**Status**: ✅ Success",
    },
    "status_failed": {
        "zh": "**状态**: ❌ 运行失败 (退出码: {exit_code})",
        "en": "**Status**: ❌ Failed (exit code: {exit_code})",
    },
    "command_preview_title": {
        "zh": "命令预览",
        "en": "Command Preview",
    },
    "command_preview_label": {
        "zh": "CLI 命令",
        "en": "CLI Command",
    },
    "command_preview_placeholder": {
        "zh": "# 选择配置后生成命令",
        "en": "# Command will be generated after selecting config",
    },
    "copy_command": {
        "zh": "📋 复制命令",
        "en": "📋 Copy Command",
    },
    "env_config_title": {
        "zh": "环境配置",
        "en": "Environment Config",
    },
    "cuda_devices_label": {
        "zh": "CUDA_VISIBLE_DEVICES",
        "en": "CUDA_VISIBLE_DEVICES",
    },
    "cuda_devices_info": {
        "zh": "GPU 设备 ID，多卡用逗号分隔（如: 0,1）",
        "en": "GPU device IDs, comma-separated (e.g.: 0,1)",
    },
    "log_title": {
        "zh": "运行日志",
        "en": "Run Logs",
    },
    "log_label": {
        "zh": "日志输出",
        "en": "Log Output",
    },
    "clear_log": {
        "zh": "🗑️ 清空日志",
        "en": "🗑️ Clear Log",
    },
    "scroll_bottom": {
        "zh": "⬇️ 滚动到底部",
        "en": "⬇️ Scroll to Bottom",
    },
    "result_title": {
        "zh": "运行结果",
        "en": "Results",
    },
    "result_placeholder": {
        "zh": "运行完成后显示结果摘要",
        "en": "Results summary will appear after run",
    },
    "output_dir_required": {
        "zh": "请先指定输出目录",
        "en": "Please specify an output directory first",
    },
    "results_not_found": {
        "zh": "在 {output_dir} 中未找到评估结果",
        "en": "No evaluation results found in {output_dir}",
    },
    "output_dir_label": {
        "zh": "输出目录",
        "en": "Output Directory",
    },
    "load_results": {
        "zh": "📊 加载结果",
        "en": "📊 Load Results",
    },
    
    # ==================== 结果对比页 ====================
    "compare_select_title": {
        "zh": "选择实验 Run",
        "en": "Select Experiment Runs",
    },
    "refresh_runs": {
        "zh": "🔄 刷新 Run 列表",
        "en": "🔄 Refresh Run List",
    },
    "run_selector_label": {
        "zh": "可用的 Run（勾选后对比）",
        "en": "Available Runs (check to compare)",
    },
    "run_selector_info": {
        "zh": "格式：mode/task_name @ checkpoint-N",
        "en": "Format: mode/task_name @ checkpoint-N",
    },
    "generate_compare": {
        "zh": "📊 生成对比",
        "en": "📊 Generate Comparison",
    },
    "compare_output_title": {
        "zh": "指标对比",
        "en": "Metrics Comparison",
    },
    "compare_placeholder": {
        "zh": "勾选左侧 Run 后点击「生成对比」",
        "en": "Select runs on the left and click 'Generate Comparison'",
    },
    "compare_empty_selection": {
        "zh": "请先勾选至少一个 Run",
        "en": "Please select at least one run first",
    },
    
    # ==================== 交互演示页 ====================
    "demo_examples_title": {
        "zh": "预置示例",
        "en": "Preset Examples",
    },
    "demo_warning": {
        "zh": "⚠️ 以下为预计算演示数据，非实时推理",
        "en": "⚠️ Pre-computed demo data, not real-time inference",
    },
    "category_filter_label": {
        "zh": "类别筛选",
        "en": "Category Filter",
    },
    "category_all": {
        "zh": "全部",
        "en": "All",
    },
    "category_unlearn": {
        "zh": "🗑️ 知识遗忘",
        "en": "🗑️ Unlearn",
    },
    "category_inject": {
        "zh": "💉 知识注入",
        "en": "💉 Inject",
    },
    "category_edit": {
        "zh": "✏️ 知识编辑",
        "en": "✏️ Edit",
    },
    "select_example_label": {
        "zh": "选择示例",
        "en": "Select Example",
    },
    "before_title": {
        "zh": "原始模型（Before）",
        "en": "Original Model (Before)",
    },
    "after_title": {
        "zh": "处理后模型（After）",
        "en": "Modified Model (After)",
    },
    "before_answer_label": {
        "zh": "原始模型回答",
        "en": "Original Model Answer",
    },
    "after_answer_label": {
        "zh": "处理后模型回答",
        "en": "Modified Model Answer",
    },
    "effect_explanation": {
        "zh": "💡 效果说明",
        "en": "💡 Effect Explanation",
    },
    "demo_placeholder": {
        "zh": "请点击左侧示例查看",
        "en": "Select an example on the left to view it",
    },
    "demo_not_found": {
        "zh": "未找到对应示例",
        "en": "Matching example not found",
    },
    "demo_question_label": {
        "zh": "问题：",
        "en": "Question:",
    },
    "demo_answer_label": {
        "zh": "回答：",
        "en": "Answer:",
    },
    
    # ==================== 智能向导页 ====================
    "wizard_goal_title": {
        "zh": "选择目标",
        "en": "Select Goal",
    },
    "wizard_goal_label": {
        "zh": "我的目标是...",
        "en": "My goal is...",
    },
    "goal_unlearn": {
        "zh": "🗑️ 我想遗忘某类知识（Unlearn）",
        "en": "🗑️ I want to forget knowledge (Unlearn)",
    },
    "goal_inject": {
        "zh": "💉 我想注入新知识（Inject）",
        "en": "💉 I want to inject new knowledge (Inject)",
    },
    "goal_edit": {
        "zh": "✏️ 我想编辑一个事实（Edit）",
        "en": "✏️ I want to edit a fact (Edit)",
    },
    "wizard_skill_title": {
        "zh": "推荐 Skill 模板",
        "en": "Recommended Skill Templates",
    },
    "select_template_label": {
        "zh": "选择模板",
        "en": "Select Template",
    },
    "config_preview_title": {
        "zh": "推荐配置预览",
        "en": "Recommended Config Preview",
    },
    "wizard_no_skill": {
        "zh": "请选择一个 Skill 模板",
        "en": "Please select a skill template",
    },
    "wizard_table_param": {
        "zh": "参数",
        "en": "Parameter",
    },
    "wizard_table_value": {
        "zh": "推荐值",
        "en": "Recommended",
    },
    "wizard_resource_estimate": {
        "zh": "资源估算：",
        "en": "Resources:",
    },
    "wizard_recommended_eval": {
        "zh": "推荐评测套件：",
        "en": "Eval Suite:",
    },
    "apply_config": {
        "zh": "⚡ 一键应用到配置页",
        "en": "⚡ Apply to Config Page",
    },
    "apply_success": {
        "zh": "✅ 已应用到配置页",
        "en": "✅ Applied to config page",
    },
    "apply_failed": {
        "zh": "❌ 未找到对应模板",
        "en": "❌ Template not found",
    },
    
    # ==================== 智能助手 ====================
    "assistant_title": {
        "zh": "智能助手",
        "en": "Smart Assistant",
    },
    "assistant_input_placeholder": {
        "zh": "描述你想要完成的任务，例如：我想让模型遗忘某个人的信息...",
        "en": "Describe your task, e.g.: I want the model to forget someone's information...",
    },
    "assistant_send": {
        "zh": "发送",
        "en": "Send",
    },
    "assistant_clear": {
        "zh": "清空对话",
        "en": "Clear Chat",
    },
    
    # ==================== 语言切换 ====================
    "lang_zh": {
        "zh": "中文",
        "en": "Chinese",
    },
    "lang_en": {
        "zh": "English",
        "en": "English",
    },
    "import_failed": {
        "zh": "导入配置失败: {error}",
        "en": "Failed to import config: {error}",
    },

    # ==================== Phase 4: 数据上传 ====================
    "data_upload_title": {
        "zh": "📂 自定义数据输入",
        "en": "📂 Custom Data Input",
    },
    "data_input_mode_label": {
        "zh": "输入方式",
        "en": "Input Mode",
    },
    "data_input_single": {
        "zh": "单条输入",
        "en": "Single Record",
    },
    "data_input_batch": {
        "zh": "批量 JSONL",
        "en": "Batch JSONL",
    },
    "data_mode_hint_inject": {
        "zh": "inject 格式：instruction（指令）/ input（补充，可空）/ output（期望输出）",
        "en": "inject format: instruction / input (optional) / output",
    },
    "data_mode_hint_edit": {
        "zh": "edit 格式：prompt（原问题）/ target_new（新答案）/ subject（可空）/ target_old（可空）",
        "en": "edit format: prompt / target_new / subject (optional) / target_old (optional)",
    },
    "data_mode_hint_unlearn": {
        "zh": "unlearn 格式：question（问题）/ answer（答案）/ split（forget 或 retain）",
        "en": "unlearn format: question / answer / split (forget or retain)",
    },
    "field_instruction_label": {
        "zh": "instruction（指令）",
        "en": "instruction",
    },
    "field_input_label": {
        "zh": "input（补充输入，可空）",
        "en": "input (optional)",
    },
    "field_output_label": {
        "zh": "output（期望输出）",
        "en": "output",
    },
    "field_prompt_label": {
        "zh": "prompt（原始问题）",
        "en": "prompt",
    },
    "field_subject_label": {
        "zh": "subject（主体，可空）",
        "en": "subject (optional)",
    },
    "field_target_new_label": {
        "zh": "target_new（新答案）",
        "en": "target_new",
    },
    "field_target_old_label": {
        "zh": "target_old（旧答案，可空）",
        "en": "target_old (optional)",
    },
    "field_question_label": {
        "zh": "question（问题）",
        "en": "question",
    },
    "field_answer_label": {
        "zh": "answer（答案）",
        "en": "answer",
    },
    "field_split_label": {
        "zh": "split（forget / retain）",
        "en": "split (forget / retain)",
    },
    "jsonl_input_label": {
        "zh": "粘贴 JSONL 内容（每行一条 JSON）",
        "en": "Paste JSONL content (one JSON per line)",
    },
    "jsonl_upload_label": {
        "zh": "或上传 .jsonl 文件",
        "en": "Or upload .jsonl file",
    },
    "jsonl_example_btn": {
        "zh": "📋 填入示例数据",
        "en": "📋 Fill Example",
    },
    "data_parse_btn": {
        "zh": "🔍 解析预览",
        "en": "🔍 Parse & Preview",
    },
    "data_add_single_btn": {
        "zh": "➕ 添加本条",
        "en": "➕ Add Record",
    },
    "data_clear_btn": {
        "zh": "🗑 清空",
        "en": "🗑 Clear",
    },
    "data_save_btn": {
        "zh": "💾 保存并应用到 Tab 1",
        "en": "💾 Save & Apply to Tab 1",
    },
    "data_preview_title": {
        "zh": "字段预览",
        "en": "Field Preview",
    },
    "data_save_path_label": {
        "zh": "保存路径",
        "en": "Saved Path",
    },
    "data_parse_success": {
        "zh": "✅ 解析成功：{count} 条记录",
        "en": "✅ Parsed: {count} records",
    },
    "data_parse_warning": {
        "zh": "⚠️ 解析完成，存在警告：{warnings}",
        "en": "⚠️ Parsed with warnings: {warnings}",
    },
    "data_parse_error": {
        "zh": "❌ 解析失败：{error}",
        "en": "❌ Parse failed: {error}",
    },
    "data_save_success": {
        "zh": "✅ 已保存到 {path}，并应用到 Tab 1 数据集字段",
        "en": "✅ Saved to {path}, applied to Tab 1 dataset field",
    },
    "data_save_error": {
        "zh": "❌ 保存失败：{error}",
        "en": "❌ Save failed: {error}",
    },
    "data_no_records": {
        "zh": "⚠️ 请先输入或上传数据，再保存",
        "en": "⚠️ Please input or upload data first",
    },

    # ==================== Phase 6: Agent 配置 ====================
    "agent_settings_title": {
        "zh": "🤖 Agent 配置",
        "en": "🤖 Agent Settings",
    },
    "agent_provider_label": {
        "zh": "Provider",
        "en": "Provider",
    },
    "agent_api_key_label": {
        "zh": "API Key",
        "en": "API Key",
    },
    "agent_api_key_placeholder": {
        "zh": "sk-... 或 ds-...",
        "en": "sk-... or ds-...",
    },
    "agent_base_url_label": {
        "zh": "Base URL（可选，留空使用默认）",
        "en": "Base URL (optional, leave blank for default)",
    },
    "agent_model_label": {
        "zh": "模型名称",
        "en": "Model Name",
    },
    "agent_temperature_label": {
        "zh": "Temperature",
        "en": "Temperature",
    },
    "agent_max_tokens_label": {
        "zh": "Max Tokens",
        "en": "Max Tokens",
    },
    "agent_timeout_label": {
        "zh": "超时（秒）",
        "en": "Timeout (s)",
    },
    "agent_test_btn": {
        "zh": "🔗 测试连接",
        "en": "🔗 Test Connection",
    },
    "agent_save_btn": {
        "zh": "💾 保存配置",
        "en": "💾 Save Settings",
    },
    "agent_test_success": {
        "zh": "✅ {message}",
        "en": "✅ {message}",
    },
    "agent_test_failed": {
        "zh": "❌ {message}",
        "en": "❌ {message}",
    },
    "agent_save_success": {
        "zh": "✅ 配置已保存",
        "en": "✅ Settings saved",
    },
    "agent_save_failed": {
        "zh": "❌ 保存失败：{error}",
        "en": "❌ Save failed: {error}",
    },
    "agent_testing": {
        "zh": "⏳ 测试中...",
        "en": "⏳ Testing...",
    },
    "agent_key_hint": {
        "zh": "API Key 仅保存在本地，不会上传",
        "en": "API Key is saved locally only, never uploaded",
    },
    "agent_chat_title": {
        "zh": "💬 快速对话",
        "en": "💬 Quick Chat",
    },
    "agent_chat_placeholder": {
        "zh": "描述你的需求，Agent 将推荐合适的配置...",
        "en": "Describe your goal, Agent will recommend configurations...",
    },
    "agent_chat_send_btn": {
        "zh": "发送",
        "en": "Send",
    },
    "agent_chat_clear_btn": {
        "zh": "🗑 清空对话",
        "en": "🗑 Clear Chat",
    },
    "agent_not_configured": {
        "zh": "⚠️ 请先在「Agent 配置」中填写 API Key 并保存",
        "en": "⚠️ Please fill in the API Key in Agent Settings first",
    },
    "agent_provider_openai_hint": {
        "zh": "OpenAI GPT 系列（gpt-4o, gpt-4-turbo...）",
        "en": "OpenAI GPT series (gpt-4o, gpt-4-turbo...)",
    },
    "agent_provider_deepseek_hint": {
        "zh": "DeepSeek 系列（deepseek-chat, deepseek-reasoner...）",
        "en": "DeepSeek series (deepseek-chat, deepseek-reasoner...)",
    },
}


class I18n:
    """国际化管理类"""
    
    def __init__(self, language: str = DEFAULT_LANGUAGE):
        self.language = language if language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
    
    def t(self, key: str, **kwargs) -> str:
        """获取翻译文案
        
        Args:
            key: 文案键名
            **kwargs: 格式化参数
            
        Returns:
            翻译后的文案
        """
        if key not in TRANSLATIONS:
            return key
        
        text = TRANSLATIONS[key].get(self.language, TRANSLATIONS[key].get(DEFAULT_LANGUAGE, key))
        
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError:
                pass
        
        return text
    
    def set_language(self, language: str):
        """设置语言"""
        if language in SUPPORTED_LANGUAGES:
            self.language = language
    
    def get_language(self) -> str:
        """获取当前语言"""
        return self.language
    
    def get_all_translations(self) -> Dict[str, str]:
        """获取当前语言的所有翻译"""
        return {key: self.t(key) for key in TRANSLATIONS}


# 全局 i18n 实例
i18n = I18n()


def t(key: str, **kwargs) -> str:
    """便捷翻译函数"""
    return i18n.t(key, **kwargs)


def set_language(language: str):
    """设置全局语言"""
    i18n.set_language(language)


def get_language() -> str:
    """获取当前语言"""
    return i18n.get_language()
