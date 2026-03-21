"""
智能向导页面（Tab 4） — Phase 7A 重构版
============================================

布局分为三个逻辑区块：
  A. 智能助手区  — 目标输入 + Agent 对话 + 建议卡片 + diff + 应用
  B. Skill 模板区 — 保留原有模板选择 + 一键应用
  C. 数据工作区  — 自定义数据输入、预览、保存（与 Assistant 解耦）

Agent 配置（provider / API Key）保留在 Accordion 中。
"""

import json
import gradio as gr
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

webui_dir = Path(__file__).parent.parent
if str(webui_dir) not in sys.path:
    sys.path.insert(0, str(webui_dir))

from utils.i18n import t, get_language
from utils.data_adapter import (
    FIELD_SCHEMA,
    parse_jsonl_text,
    parse_jsonl_file,
    validate_records,
    get_example_jsonl,
    preview_records_html,
    save_records_to_file,
    single_record_to_jsonl,
)
from utils.agent_settings import (
    load_settings,
    save_settings,
    get_provider_default,
    test_connection,
)

SKILLS_DIR = Path(__file__).parent.parent / "skills"


# ──────────────────────────────────────────────
# Skill 辅助
# ──────────────────────────────────────────────

def _get_goal_options() -> Dict[str, str]:
    return {
        t("goal_unlearn"): "unlearn",
        t("goal_inject"): "inject",
        t("goal_edit"): "edit",
    }


def _load_all_skills() -> List[Dict]:
    skills = []
    if not SKILLS_DIR.exists():
        return skills
    for f in sorted(SKILLS_DIR.glob("*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                skills.append(json.load(fp))
        except Exception:
            pass
    return skills


def _render_config_preview_html(skill: Optional[Dict]) -> str:
    if not skill:
        return "<p style='color:#888;'>请选择一个 Skill 模板</p>"

    lang = get_language()
    cfg = skill.get("config", {})
    rows = "".join(
        f"<tr><td style='padding:5px 10px;color:#6B7280;font-size:0.82rem;'>{k}</td>"
        f"<td style='padding:5px 10px;font-weight:600;color:#0F172A;font-size:0.82rem;'>{v}</td></tr>"
        for k, v in cfg.items()
    )

    name = skill.get("name_en", skill["name"]) if lang == "en" and skill.get("name_en") else skill["name"]
    desc = skill.get("description_en", skill.get("description", "")) if lang == "en" and skill.get("description_en") else skill.get("description", "")
    tags = skill.get("tags_en", skill.get("tags", [])) if lang == "en" and skill.get("tags_en") else skill.get("tags", [])
    resource = skill.get("resource_estimate_en", skill.get("resource_estimate", "—")) if lang == "en" and skill.get("resource_estimate_en") else skill.get("resource_estimate", "—")

    tags_html = "".join(
        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
        f"background:#EEF3FB;color:#1D3FDB;font-size:0.72rem;margin:2px;'>{tag}</span>"
        for tag in tags
    )

    param_label = "Parameter" if lang == "en" else "参数"
    value_label = "Recommended" if lang == "en" else "推荐值"
    resource_label = "Resources:" if lang == "en" else "资源估算："
    eval_label = "Eval Suite:" if lang == "en" else "推荐评测套件："

    return f"""
<div style="border:1px solid #D7E0F0;border-radius:10px;overflow:hidden;background:white;">
  <div style="background:#355CFF;color:white;padding:8px 14px;font-weight:700;font-size:0.875rem;">
    {name}
  </div>
  <div style="padding:10px;">
    <p style="font-size:0.82rem;color:#6B7280;margin-bottom:10px;">{desc}</p>
    <div style="margin-bottom:8px;">{tags_html}</div>
    <table style="width:100%;border-collapse:collapse;">
      <thead>
        <tr style="background:#F5F7FB;">
          <th style="padding:5px 10px;text-align:left;font-size:0.78rem;color:#355CFF;">{param_label}</th>
          <th style="padding:5px 10px;text-align:left;font-size:0.78rem;color:#355CFF;">{value_label}</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    <div style="margin-top:10px;font-size:0.8rem;color:#355CFF;">
      ⚡ <strong>{resource_label}</strong>{resource} &nbsp;|&nbsp;
      📊 <strong>{eval_label}</strong>{skill.get('recommended_eval','—')}
    </div>
  </div>
</div>"""


def _render_mode_hint(mode: str) -> str:
    schema = FIELD_SCHEMA.get(mode, {})
    lang = get_language()
    desc = schema.get("description", {}).get(lang, schema.get("description", {}).get("zh", ""))
    required = schema.get("required", [])
    optional = schema.get("optional", [])

    req_tags = "".join(
        f"<span style='display:inline-block;padding:2px 8px;border-radius:4px;"
        f"background:#DBEAFE;color:#1D4ED8;font-size:0.75rem;margin:2px;font-family:monospace;'>"
        f"★ {f}</span>"
        for f in required
    )
    opt_tags = "".join(
        f"<span style='display:inline-block;padding:2px 8px;border-radius:4px;"
        f"background:#F3F4F6;color:#6B7280;font-size:0.75rem;margin:2px;font-family:monospace;'>"
        f"{f}</span>"
        for f in optional
    )

    req_label = "Required" if lang == "en" else "必填"
    opt_label = "Optional" if lang == "en" else "可选"

    return f"""
<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;padding:10px 14px;margin-bottom:6px;">
  <p style="font-size:0.8rem;color:#1E40AF;margin:0 0 6px 0;">{desc}</p>
  <div><strong style="font-size:0.75rem;color:#2563EB;">{req_label}:</strong> {req_tags}</div>
  <div style="margin-top:4px;"><strong style="font-size:0.75rem;color:#6B7280;">{opt_label}:</strong> {opt_tags}</div>
</div>"""


# ──────────────────────────────────────────────
# 主 Tab 创建
# ──────────────────────────────────────────────

def create_skill_wizard_tab() -> Dict[str, Any]:
    components: Dict[str, Any] = {}
    all_skills = _load_all_skills()
    goal_options = _get_goal_options()

    # ══════════════════════════════════════════
    # Section A: 智能助手区
    # ══════════════════════════════════════════
    gr.HTML(f'<div class="ks-col-title">{t("assistant_sidebar_title")}</div>')

    with gr.Row(equal_height=False):
        # ── 左列：目标 + 对话 ──
        with gr.Column(scale=1):
            components["goal_radio"] = gr.Radio(
                choices=list(goal_options.keys()),
                value=list(goal_options.keys())[0],
                label=t("wizard_goal_label"),
                container=False,
            )
            components["assistant_goal_input"] = gr.Textbox(
                label=t("assistant_goal_input_label"),
                placeholder=t("assistant_goal_input_placeholder"),
                lines=3,
            )
            with gr.Row():
                components["assistant_get_rec_btn"] = gr.Button(
                    t("assistant_get_recommendation"), variant="primary", size="sm"
                )

            components["agent_chatbot"] = gr.Chatbot(
                value=[],
                height=300,
                show_label=False,
            )
            with gr.Row():
                components["agent_input"] = gr.Textbox(
                    placeholder=t("agent_chat_placeholder"),
                    label="",
                    lines=2,
                    scale=4,
                    container=False,
                )
                with gr.Column(scale=1, min_width=80):
                    components["agent_send_btn"] = gr.Button(
                        t("agent_chat_send_btn"), variant="primary", size="sm"
                    )
                    components["agent_clear_chat_btn"] = gr.Button(
                        t("agent_chat_clear_btn"), variant="secondary", size="sm"
                    )

        # ── 右列：建议卡片 + diff + 应用 ──
        with gr.Column(scale=1):
            gr.HTML(f'<div class="ks-col-title">{t("assistant_recommendation_title")}</div>')
            components["recommendation_html"] = gr.HTML(
                value=f"<p style='color:#94A3B8;font-size:0.82rem;'>{t('assistant_recommendation_empty')}</p>"
            )
            gr.HTML(f'<div class="ks-col-title" style="margin-top:12px;">{t("assistant_diff_title")}</div>')
            components["diff_html"] = gr.HTML(
                value=f"<p style='color:#94A3B8;font-size:0.82rem;'>{t('assistant_diff_empty')}</p>"
            )
            components["operation_level"] = gr.Radio(
                choices=["仅生成建议", "生成并预览", "直接应用"],
                value="生成并预览",
                label="操作层级",
                container=False,
            )
            components["assistant_apply_btn"] = gr.Button(
                t("assistant_apply_recommendation"), variant="primary"
            )
            components["assistant_apply_status"] = gr.HTML(value="")

    # hidden states for assistant
    components["_agent_history"] = gr.State(value=[])
    components["_recommendation_state"] = gr.State(value=None)
    components["_goal_mode_state"] = gr.State(value="unlearn")
    components["_skills_state"] = gr.State(value=all_skills)

    gr.HTML('<div style="height:14px;"></div>')

    # ══════════════════════════════════════════
    # Section B: Skill 模板选择
    # ══════════════════════════════════════════
    with gr.Accordion(t("wizard_skill_title"), open=False):
        with gr.Row():
            with gr.Column(scale=1):
                initial_skills = [s for s in all_skills if s.get("goal") == "unlearn"]
                skill_choices = [s["name"] for s in initial_skills]
                components["skill_selector"] = gr.Radio(
                    choices=skill_choices,
                    value=skill_choices[0] if skill_choices else None,
                    label=t("select_template_label"),
                    container=False,
                    elem_id="skill-selector-radio",
                )
            with gr.Column(scale=1):
                gr.HTML(f'<div class="ks-col-title">{t("config_preview_title")}</div>')
                first_skill = initial_skills[0] if initial_skills else None
                components["config_preview_html"] = gr.HTML(
                    value=_render_config_preview_html(first_skill)
                )
                components["apply_btn"] = gr.Button(
                    t("apply_config"), variant="primary"
                )
                components["apply_status"] = gr.HTML(value="")

    gr.HTML('<div style="height:10px;"></div>')

    # ══════════════════════════════════════════
    # Section C: 数据工作区（从 Assistant 解耦）
    # ══════════════════════════════════════════
    with gr.Accordion(t("assistant_data_section_title"), open=False):
        components["data_mode_hint"] = gr.HTML(
            value=_render_mode_hint("unlearn")
        )

        components["data_input_mode"] = gr.Radio(
            choices=[t("data_input_single"), t("data_input_batch")],
            value=t("data_input_single"),
            label=t("data_input_mode_label"),
            container=False,
        )

        gr.HTML('<div style="height:6px;"></div>')

        # ── 单条输入 ──
        with gr.Group(visible=True, elem_id="single-input-group") as components["single_group"]:
            with gr.Group(visible=True) as components["single_unlearn_group"]:
                components["f_question"] = gr.Textbox(
                    label=t("field_question_label"), lines=2, placeholder="What is the birthplace of Harry Potter?"
                )
                components["f_answer"] = gr.Textbox(
                    label=t("field_answer_label"), lines=2, placeholder="Godric's Hollow"
                )
                components["f_split"] = gr.Dropdown(
                    choices=["forget", "retain"], value="forget",
                    label=t("field_split_label"),
                )
            with gr.Group(visible=False) as components["single_inject_group"]:
                components["f_instruction"] = gr.Textbox(
                    label=t("field_instruction_label"), lines=2, placeholder="介绍量子纠缠现象"
                )
                components["f_input"] = gr.Textbox(
                    label=t("field_input_label"), lines=1, placeholder="（可留空）"
                )
                components["f_output"] = gr.Textbox(
                    label=t("field_output_label"), lines=3, placeholder="量子纠缠是..."
                )
            with gr.Group(visible=False) as components["single_edit_group"]:
                components["f_prompt"] = gr.Textbox(
                    label=t("field_prompt_label"), lines=2, placeholder="The capital of France is"
                )
                components["f_subject"] = gr.Textbox(
                    label=t("field_subject_label"), lines=1, placeholder="France"
                )
                components["f_target_new"] = gr.Textbox(
                    label=t("field_target_new_label"), lines=1, placeholder="Paris"
                )
                components["f_target_old"] = gr.Textbox(
                    label=t("field_target_old_label"), lines=1, placeholder="Lyon"
                )
            components["add_single_btn"] = gr.Button(
                t("data_add_single_btn"), variant="secondary", size="sm"
            )

        # ── 批量 JSONL ──
        with gr.Group(visible=False, elem_id="batch-input-group") as components["batch_group"]:
            with gr.Row():
                with gr.Column(scale=3):
                    components["jsonl_text"] = gr.Textbox(
                        label=t("jsonl_input_label"), lines=8,
                        placeholder='{"question": "...", "answer": "...", "split": "forget"}',
                    )
                with gr.Column(scale=1):
                    components["jsonl_upload"] = gr.File(
                        label=t("jsonl_upload_label"),
                        file_types=[".jsonl", ".json", ".txt"],
                        file_count="single",
                    )
                    components["jsonl_example_btn"] = gr.Button(
                        t("jsonl_example_btn"), variant="secondary", size="sm"
                    )
            components["parse_btn"] = gr.Button(
                t("data_parse_btn"), variant="secondary", size="sm"
            )

        components["parse_status"] = gr.HTML(value="")
        gr.HTML('<div style="height:6px;"></div>')
        gr.HTML(
            f'<div style="font-size:0.82rem;font-weight:600;color:#374151;margin-bottom:4px;">'
            f'{t("data_preview_title")}</div>'
        )

        components["_records_state"] = gr.State(value=[])
        components["preview_html"] = gr.HTML(value=preview_records_html([], "unlearn"))

        with gr.Row():
            components["clear_btn"] = gr.Button(
                t("data_clear_btn"), variant="secondary", size="sm"
            )
            components["save_apply_btn"] = gr.Button(
                t("data_save_btn"), variant="primary", size="sm"
            )
        components["save_status"] = gr.HTML(value="")
        components["save_path_display"] = gr.Textbox(
            label=t("data_save_path_label"), interactive=False, visible=False,
        )

    gr.HTML('<div style="height:10px;"></div>')

    # ══════════════════════════════════════════
    # Agent 配置（Provider / API Key）
    # ══════════════════════════════════════════
    _init = load_settings()
    with gr.Accordion(t("agent_settings_title"), open=False):
        with gr.Row():
            with gr.Column(scale=1):
                components["agent_provider"] = gr.Radio(
                    choices=["openai", "deepseek"],
                    value=_init.get("provider", "openai"),
                    label=t("agent_provider_label"), container=False,
                )
                gr.HTML(
                    f"<p style='font-size:0.75rem;color:#6B7280;margin-top:2px;' "
                    f"id='agent-provider-hint'>{t('agent_provider_openai_hint')}</p>"
                )
                components["agent_api_key"] = gr.Textbox(
                    label=t("agent_api_key_label"),
                    value=_init.get("api_key", ""),
                    placeholder=t("agent_api_key_placeholder"),
                    type="password",
                )
                gr.HTML(
                    f"<p style='font-size:0.72rem;color:#9CA3AF;margin-top:-6px;'>"
                    f"🔒 {t('agent_key_hint')}</p>"
                )
            with gr.Column(scale=1):
                components["agent_base_url"] = gr.Textbox(
                    label=t("agent_base_url_label"),
                    value=_init.get("base_url", ""),
                    placeholder="https://api.openai.com/v1",
                )
                components["agent_model"] = gr.Textbox(
                    label=t("agent_model_label"),
                    value=_init.get("model", "gpt-4o"),
                    placeholder="gpt-4o",
                )
                with gr.Row():
                    components["agent_temperature"] = gr.Slider(
                        minimum=0.0, maximum=2.0, step=0.1,
                        value=_init.get("temperature", 0.7),
                        label=t("agent_temperature_label"),
                    )
                    components["agent_max_tokens"] = gr.Number(
                        value=_init.get("max_tokens", 2048),
                        label=t("agent_max_tokens_label"),
                        minimum=64, maximum=32768, step=256,
                    )
        with gr.Row():
            components["agent_test_btn"] = gr.Button(
                t("agent_test_btn"), variant="secondary", size="sm"
            )
            components["agent_save_btn"] = gr.Button(
                t("agent_save_btn"), variant="primary", size="sm"
            )
        components["agent_conn_status"] = gr.HTML(value="")

    return components


# ──────────────────────────────────────────────
# 事件绑定
# ──────────────────────────────────────────────

def bind_skill_wizard_events(
    components: Dict[str, Any],
    config_components: Dict[str, Any],
    params_components: Dict[str, Any],
) -> None:

    # ════════════════════════════════════════
    # A. 智能助手事件
    # ════════════════════════════════════════

    def _build_agent_system_prompt(goal_hint: Optional[str] = None) -> str:
        from utils.agent_planner import build_system_prompt
        return build_system_prompt(lang=get_language(), goal_hint=goal_hint)

    def _get_current_config_snapshot(
        mode, model, trainer, experiment, task_name, seed,
        lr, epochs, bs, ga,
    ) -> Dict[str, Any]:
        return {
            "mode": mode, "model": model, "trainer": trainer,
            "experiment": experiment, "task_name": task_name, "seed": seed,
            "learning_rate": lr, "num_epochs": epochs,
            "batch_size": bs, "gradient_accumulation": ga,
        }

    # ── 获取建议按钮 ──
    def get_recommendation(
        goal_label, user_goal, history,
        provider, api_key, base_url, model_name, temperature, max_tokens,
        cur_mode, cur_model, cur_trainer, cur_experiment, cur_task, cur_seed,
        cur_lr, cur_epochs, cur_bs, cur_ga,
        skills, goal_mode, operation_level,
    ):
        from utils.agent_planner import (
            build_system_prompt, parse_recommendation,
            render_recommendation_card, render_diff_html,
            render_grouped_recommendation,
        )

        goal_options = _get_goal_options()
        goal_key = goal_options.get(goal_label, goal_mode or "unlearn")

        current_cfg = _get_current_config_snapshot(
            cur_mode, cur_model, cur_trainer, cur_experiment, cur_task, cur_seed,
            cur_lr, cur_epochs, cur_bs, cur_ga,
        )

        user_text = (user_goal or "").strip()
        if not user_text:
            user_text = f"I want to perform {goal_key} operation. Please recommend the best skill template and configuration."

        lang = get_language()

        if not api_key or not api_key.strip():
            from utils.skills_context import load_skills
            matched = [s for s in load_skills() if s.goal == goal_key]
            if matched:
                sk = matched[0]
                rec_data = {
                    "mode": goal_key,
                    "recommended_skill": sk.id,
                    "recommended_method": sk.config.get("trainer", ""),
                    "recommended_model": sk.config.get("model", ""),
                    "data_plan": sk.get_description(lang),
                    "eval_plan": sk.recommended_eval,
                    "core_overrides": sk.core_overrides,
                    "reasoning_summary": sk.prompt_hints if lang == "zh" else sk.prompt_hints_en,
                    "risk_notes": "; ".join(sk.risk_notes if lang == "zh" else sk.risk_notes_en),
                }
                fake_json = json.dumps(rec_data, ensure_ascii=False)
                rec = parse_recommendation(f"```json\n{fake_json}\n```")
                card_html = render_grouped_recommendation(rec, lang)
                show_diff = operation_level != "仅生成建议"
                diff_html = render_diff_html(current_cfg, rec.get_apply_config(), lang) if show_diff else ""
                bot_msg = f"基于规则匹配，推荐使用 **{sk.get_display_name(lang)}** 模板。\n\n" if lang == "zh" else f"Based on rule matching, recommending **{sk.get_display_name(lang)}**.\n\n"
                new_history = history + [[user_text, bot_msg + f"```json\n{fake_json}\n```"]]
                return new_history, new_history, "", card_html, diff_html, rec.to_dict()
            empty_card = f"<p style='color:#F59E0B;'>{t('agent_not_configured')}</p>"
            return history, history, "", empty_card, "", None

        try:
            from utils.agent_provider import AgentConfig, OpenAIProvider, DeepSeekProvider, Message

            system_prompt = build_system_prompt(lang=lang, goal_hint=goal_key)

            cfg = AgentConfig(
                provider=provider,
                api_key=api_key.strip(),
                base_url=base_url.strip() or "",
                model=model_name.strip(),
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
            prov = DeepSeekProvider(cfg) if provider == "deepseek" else OpenAIProvider(cfg)

            messages = [Message(role="system", content=system_prompt)]
            for u, a in history:
                messages.append(Message(role="user", content=u))
                if a:
                    messages.append(Message(role="assistant", content=a))
            messages.append(Message(role="user", content=user_text))

            response = prov.chat(messages)
            if response.error:
                err_html = f"<p style='color:#EF4444;'>Agent error: {response.error}</p>"
                new_history = history + [[user_text, f"❌ {response.error}"]]
                return new_history, new_history, "", err_html, "", None

            agent_reply = response.content
            rec = parse_recommendation(agent_reply)

            if rec.is_valid:
                card_html = render_grouped_recommendation(rec, lang)
                show_diff = operation_level != "仅生成建议"
                diff_html = render_diff_html(current_cfg, rec.get_apply_config(), lang) if show_diff else ""
            else:
                card_html = f"<p style='color:#F59E0B;'>{t('assistant_parse_failed')}</p>"
                diff_html = ""

            new_history = history + [[user_text, agent_reply]]
            return new_history, new_history, "", card_html, diff_html, rec.to_dict() if rec.is_valid else None

        except Exception as e:
            err_html = f"<p style='color:#EF4444;'>Error: {e}</p>"
            new_history = history + [[user_text, f"❌ {e}"]]
            return new_history, new_history, "", err_html, "", None

    _config_snapshot_inputs = [
        "mode", "model", "trainer", "experiment", "task_name", "seed",
    ]
    _params_snapshot_inputs = [
        "learning_rate", "num_epochs", "batch_size", "gradient_accumulation",
    ]

    def _bind_get_recommendation(components, config_components, params_components):
        components["assistant_get_rec_btn"].click(
            fn=get_recommendation,
            inputs=[
                components["goal_radio"],
                components["assistant_goal_input"],
                components["_agent_history"],
                components["agent_provider"],
                components["agent_api_key"],
                components["agent_base_url"],
                components["agent_model"],
                components["agent_temperature"],
                components["agent_max_tokens"],
                config_components["mode"],
                config_components["model"],
                config_components["trainer"],
                config_components["experiment"],
                config_components["task_name"],
                config_components["seed"],
                params_components["learning_rate"],
                params_components["num_epochs"],
                params_components["batch_size"],
                params_components["gradient_accumulation"],
                components["_skills_state"],
                components["_goal_mode_state"],
                components["operation_level"],
            ],
            outputs=[
                components["agent_chatbot"],
                components["_agent_history"],
                components["assistant_goal_input"],
                components["recommendation_html"],
                components["diff_html"],
                components["_recommendation_state"],
            ],
        )

    _bind_get_recommendation(components, config_components, params_components)

    # ── 应用建议到 Tab 1 ──
    def apply_recommendation(rec_dict):
        if not rec_dict:
            return [gr.update()] * 10 + [f"<span style='color:#F59E0B;'>{t('assistant_no_recommendation')}</span>"]

        cfg = rec_dict
        skill_id = cfg.get("recommended_skill", "")
        from utils.skills_context import get_skill_config_by_id
        skill_cfg = get_skill_config_by_id(skill_id) or {}

        mode = cfg.get("mode", skill_cfg.get("mode", "unlearn"))
        model_val = cfg.get("recommended_model", skill_cfg.get("model"))
        trainer_val = cfg.get("recommended_method", skill_cfg.get("trainer"))
        experiment_val = skill_cfg.get("experiment")
        task_name = skill_cfg.get("task_name", "my_experiment")
        seed = skill_cfg.get("seed", 42)
        lr = skill_cfg.get("learning_rate", "1e-5")
        epochs = skill_cfg.get("num_epochs", 3)
        bs = skill_cfg.get("batch_size", 4)
        ga = skill_cfg.get("gradient_accumulation", 4)

        overrides = cfg.get("core_overrides", {})
        lr = overrides.get("trainer.args.learning_rate", lr)
        epochs_ov = overrides.get("trainer.args.num_train_epochs")
        if epochs_ov is not None:
            epochs = epochs_ov

        status = f"<span style='color:#10B981;'>{t('assistant_apply_success')}</span>"
        return (
            gr.update(value=mode),
            gr.update(value=model_val),
            gr.update(value=trainer_val),
            gr.update(value=experiment_val),
            gr.update(value=task_name),
            gr.update(value=seed),
            gr.update(value=str(lr)),
            gr.update(value=epochs),
            gr.update(value=bs),
            gr.update(value=ga),
            status,
        )

    components["assistant_apply_btn"].click(
        fn=apply_recommendation,
        inputs=[components["_recommendation_state"]],
        outputs=[
            config_components["mode"],
            config_components["model"],
            config_components["trainer"],
            config_components["experiment"],
            config_components["task_name"],
            config_components["seed"],
            params_components["learning_rate"],
            params_components["num_epochs"],
            params_components["batch_size"],
            params_components["gradient_accumulation"],
            components["assistant_apply_status"],
        ],
    )

    # ── Agent 自由对话（流式） ──
    def do_chat(user_msg, history, provider, api_key, base_url, model_name, temperature, max_tokens, goal_mode):
        if not user_msg or not user_msg.strip():
            yield history, history, ""
            return

        if not api_key or not api_key.strip():
            warn = t("agent_not_configured")
            new_history = history + [[user_msg, warn]]
            yield new_history, new_history, ""
            return

        new_history = history + [[user_msg, ""]]
        yield new_history, new_history, ""

        try:
            from utils.agent_provider import AgentConfig, OpenAIProvider, DeepSeekProvider, Message

            system_prompt = _build_agent_system_prompt(goal_hint=goal_mode)

            cfg = AgentConfig(
                provider=provider,
                api_key=api_key.strip(),
                base_url=base_url.strip() or "",
                model=model_name.strip(),
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
            prov = DeepSeekProvider(cfg) if provider == "deepseek" else OpenAIProvider(cfg)

            messages = [Message(role="system", content=system_prompt)]
            for u, a in history:
                messages.append(Message(role="user", content=u))
                if a:
                    messages.append(Message(role="assistant", content=a))
            messages.append(Message(role="user", content=user_msg))

            accumulated = ""
            for chunk in prov.stream_chat(messages):
                accumulated += chunk
                new_history[-1][1] = accumulated
                yield new_history, new_history, ""

        except Exception as e:
            new_history[-1][1] = f"❌ {e}"
            yield new_history, new_history, ""

    _chat_inputs = [
        components["agent_input"],
        components["_agent_history"],
        components["agent_provider"],
        components["agent_api_key"],
        components["agent_base_url"],
        components["agent_model"],
        components["agent_temperature"],
        components["agent_max_tokens"],
        components["_goal_mode_state"],
    ]
    _chat_outputs = [
        components["agent_chatbot"],
        components["_agent_history"],
        components["agent_input"],
    ]

    components["agent_send_btn"].click(fn=do_chat, inputs=_chat_inputs, outputs=_chat_outputs)
    components["agent_input"].submit(fn=do_chat, inputs=_chat_inputs, outputs=_chat_outputs)

    def clear_chat():
        return [], [], ""

    components["agent_clear_chat_btn"].click(
        fn=clear_chat, inputs=[],
        outputs=[components["agent_chatbot"], components["_agent_history"], components["agent_input"]],
    )

    # ════════════════════════════════════════
    # B. Skill 模板事件
    # ════════════════════════════════════════

    def on_goal_change(goal_label, skills):
        goal_options = _get_goal_options()
        goal_key = goal_options.get(goal_label, "unlearn")
        filtered = [s for s in skills if s.get("goal") == goal_key]
        first = filtered[0] if filtered else None
        choices = [s["name"] for s in filtered]
        return (
            gr.update(choices=choices, value=choices[0] if choices else None),
            _render_config_preview_html(first),
            _render_mode_hint(goal_key),
            goal_key,
            gr.update(visible=(goal_key == "unlearn")),
            gr.update(visible=(goal_key == "inject")),
            gr.update(visible=(goal_key == "edit")),
        )

    def on_skill_select(skill_name, skills):
        sk = next((s for s in skills if s["name"] == skill_name), None)
        return _render_config_preview_html(sk)

    def apply_skill(skill_name, skills):
        sk = next((s for s in skills if s["name"] == skill_name), None)
        if not sk:
            status = f"<span style='color:#EF4444;'>{t('apply_failed')}</span>"
            return [gr.update()] * 10 + [status]
        cfg = sk.get("config", {})
        status = f"<span style='color:#10B981;'>{t('apply_success')}</span>"
        return (
            gr.update(value=cfg.get("mode", "unlearn")),
            gr.update(value=cfg.get("model")),
            gr.update(value=cfg.get("trainer")),
            gr.update(value=cfg.get("experiment")),
            gr.update(value=cfg.get("task_name", "my_experiment")),
            gr.update(value=cfg.get("seed", 42)),
            gr.update(value=cfg.get("learning_rate", "1e-5")),
            gr.update(value=cfg.get("num_epochs", 3)),
            gr.update(value=cfg.get("batch_size", 4)),
            gr.update(value=cfg.get("gradient_accumulation", 4)),
            status,
        )

    components["goal_radio"].change(
        fn=on_goal_change,
        inputs=[components["goal_radio"], components["_skills_state"]],
        outputs=[
            components["skill_selector"],
            components["config_preview_html"],
            components["data_mode_hint"],
            components["_goal_mode_state"],
            components["single_unlearn_group"],
            components["single_inject_group"],
            components["single_edit_group"],
        ],
    )

    components["skill_selector"].change(
        fn=on_skill_select,
        inputs=[components["skill_selector"], components["_skills_state"]],
        outputs=[components["config_preview_html"]],
    )

    components["apply_btn"].click(
        fn=apply_skill,
        inputs=[components["skill_selector"], components["_skills_state"]],
        outputs=[
            config_components["mode"],
            config_components["model"],
            config_components["trainer"],
            config_components["experiment"],
            config_components["task_name"],
            config_components["seed"],
            params_components["learning_rate"],
            params_components["num_epochs"],
            params_components["batch_size"],
            params_components["gradient_accumulation"],
            components["apply_status"],
        ],
    )

    # ════════════════════════════════════════
    # C. 数据工作区事件（保持与原逻辑一致）
    # ════════════════════════════════════════

    def on_input_mode_change(mode_label):
        is_single = mode_label == t("data_input_single")
        return gr.update(visible=is_single), gr.update(visible=not is_single)

    components["data_input_mode"].change(
        fn=on_input_mode_change,
        inputs=[components["data_input_mode"]],
        outputs=[components["single_group"], components["batch_group"]],
    )

    def fill_example(goal_mode):
        return gr.update(value=get_example_jsonl(goal_mode))

    components["jsonl_example_btn"].click(
        fn=fill_example,
        inputs=[components["_goal_mode_state"]],
        outputs=[components["jsonl_text"]],
    )

    def on_file_upload(file):
        if file is None:
            return gr.update()
        records, err = parse_jsonl_file(file.name)
        if err and not records:
            return gr.update(value=f"// 读取失败: {err}")
        lines = [json.dumps(r, ensure_ascii=False) for r in records]
        return gr.update(value="\n".join(lines))

    components["jsonl_upload"].change(
        fn=on_file_upload,
        inputs=[components["jsonl_upload"]],
        outputs=[components["jsonl_text"]],
    )

    def add_single_record(
        goal_mode, records,
        question, answer, split,
        instruction, inp, output,
        prompt, subject, target_new, target_old,
    ):
        kwargs = {}
        if goal_mode == "unlearn":
            kwargs = {"question": question, "answer": answer, "split": split}
        elif goal_mode == "inject":
            kwargs = {"instruction": instruction, "input": inp, "output": output}
        elif goal_mode == "edit":
            kwargs = {"prompt": prompt, "subject": subject, "target_new": target_new, "target_old": target_old}

        new_recs, err = single_record_to_jsonl(goal_mode, **kwargs)
        if err:
            status = f"<span style='color:#EF4444;'>❌ {err}</span>"
            return records, preview_records_html(records, goal_mode), status

        combined = records + new_recs
        status = f"<span style='color:#10B981;'>{t('data_parse_success', count=len(combined))}</span>"
        return combined, preview_records_html(combined, goal_mode), status

    components["add_single_btn"].click(
        fn=add_single_record,
        inputs=[
            components["_goal_mode_state"], components["_records_state"],
            components["f_question"], components["f_answer"], components["f_split"],
            components["f_instruction"], components["f_input"], components["f_output"],
            components["f_prompt"], components["f_subject"], components["f_target_new"], components["f_target_old"],
        ],
        outputs=[components["_records_state"], components["preview_html"], components["parse_status"]],
    )

    def parse_and_preview(jsonl_text, goal_mode):
        records, parse_err = parse_jsonl_text(jsonl_text)
        if parse_err and not records:
            status = f"<span style='color:#EF4444;'>{t('data_parse_error', error=parse_err)}</span>"
            return [], preview_records_html([], goal_mode), status
        valid, warnings = validate_records(records, goal_mode)
        if warnings:
            warn_text = "；".join(warnings)
            status = f"<span style='color:#F59E0B;'>{t('data_parse_warning', warnings=warn_text)}</span>"
        else:
            status = f"<span style='color:#10B981;'>{t('data_parse_success', count=len(records))}</span>"
        return records, preview_records_html(records, goal_mode), status

    components["parse_btn"].click(
        fn=parse_and_preview,
        inputs=[components["jsonl_text"], components["_goal_mode_state"]],
        outputs=[components["_records_state"], components["preview_html"], components["parse_status"]],
    )

    def clear_data(goal_mode):
        return [], preview_records_html([], goal_mode), "", ""

    components["clear_btn"].click(
        fn=clear_data,
        inputs=[components["_goal_mode_state"]],
        outputs=[components["_records_state"], components["preview_html"], components["parse_status"], components["save_status"]],
    )

    def save_and_apply(records, goal_mode):
        if not records:
            status = f"<span style='color:#F59E0B;'>{t('data_no_records')}</span>"
            return status, gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update()

        saved_path, err = save_records_to_file(records, goal_mode)
        if err:
            status = f"<span style='color:#EF4444;'>{t('data_save_error', error=err)}</span>"
            return status, gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update()

        status = f"<span style='color:#10B981;'>{t('data_save_success', path=saved_path)}</span>"

        forget_update = gr.update()
        retain_update = gr.update()
        train_update = gr.update()
        edit_update = gr.update()

        if goal_mode == "unlearn":
            forget_update = gr.update(value=saved_path)
        elif goal_mode == "inject":
            train_update = gr.update(value=saved_path)
        elif goal_mode == "edit":
            edit_update = gr.update(value=saved_path)

        path_display = gr.update(value=saved_path, visible=True)
        return status, path_display, forget_update, retain_update, train_update, edit_update

    components["save_apply_btn"].click(
        fn=save_and_apply,
        inputs=[components["_records_state"], components["_goal_mode_state"]],
        outputs=[
            components["save_status"], components["save_path_display"],
            config_components["forget_dataset"], config_components["retain_dataset"],
            config_components["train_dataset"], config_components["edit_dataset"],
        ],
    )

    # ════════════════════════════════════════
    # D. Agent 配置事件
    # ════════════════════════════════════════

    def on_provider_change(provider):
        defaults = get_provider_default(provider)
        lang = get_language()
        hint_key = f"agent_provider_{provider}_hint"
        from utils.i18n import TRANSLATIONS
        hint = TRANSLATIONS.get(hint_key, {}).get(lang, "")
        return (
            gr.update(value=defaults["base_url"]),
            gr.update(value=defaults["model"]),
            f"<p style='font-size:0.75rem;color:#6B7280;margin-top:2px;'>{hint}</p>",
        )

    components["agent_provider"].change(
        fn=on_provider_change,
        inputs=[components["agent_provider"]],
        outputs=[components["agent_base_url"], components["agent_model"], components["agent_conn_status"]],
    )

    def do_test_connection(provider, api_key, base_url, model):
        status_html = f"<span style='color:#6B7280;'>{t('agent_testing')}</span>"
        yield status_html
        ok, msg = test_connection(provider=provider, api_key=api_key, base_url=base_url, model=model, timeout=15)
        if ok:
            yield f"<span style='color:#10B981;'>{t('agent_test_success', message=msg)}</span>"
        else:
            yield f"<span style='color:#EF4444;'>{t('agent_test_failed', message=msg)}</span>"

    components["agent_test_btn"].click(
        fn=do_test_connection,
        inputs=[components["agent_provider"], components["agent_api_key"], components["agent_base_url"], components["agent_model"]],
        outputs=[components["agent_conn_status"]],
    )

    def do_save_settings(provider, api_key, base_url, model, temperature, max_tokens):
        err = save_settings({
            "provider": provider, "api_key": api_key, "base_url": base_url,
            "model": model, "temperature": float(temperature), "max_tokens": int(max_tokens),
        })
        if err:
            return f"<span style='color:#EF4444;'>{t('agent_save_failed', error=err)}</span>"
        return f"<span style='color:#10B981;'>{t('agent_save_success')}</span>"

    components["agent_save_btn"].click(
        fn=do_save_settings,
        inputs=[
            components["agent_provider"], components["agent_api_key"],
            components["agent_base_url"], components["agent_model"],
            components["agent_temperature"], components["agent_max_tokens"],
        ],
        outputs=[components["agent_conn_status"]],
    )
