"""
交互演示页面（Tab 3）
======================

提供 Before/After 对比展示。
选中逻辑由原生 gr.Radio 承担（CSS 样式化为卡片），预计算 JSON 提供示例数据。
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

EXAMPLES_FILE = Path(__file__).parent.parent / "examples" / "unlearn_examples.json"


def _get_category_labels() -> Dict[str, str]:
    """动态获取类别标签（支持 i18n）"""
    return {
        "unlearn": t("category_unlearn"),
        "inject": t("category_inject"),
        "edit": t("category_edit"),
    }


def _load_examples() -> List[Dict]:
    if not EXAMPLES_FILE.exists():
        return []
    try:
        with open(EXAMPLES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _render_qa_html(question: str, answer: str, side: str = "before") -> str:
    border_color = "#94A3B8" if side == "before" else "#10B981"
    label = t("before_answer_label") if side == "before" else t("after_answer_label")
    label_color = "#64748B" if side == "before" else "#10B981"
    answer_escaped = answer.replace("\n", "<br>")
    lang = get_language()
    q_label = "Question:" if lang == "en" else "问题："
    a_label = "Answer:" if lang == "en" else "回答："
    return f"""
<div style="border-left:3px solid {border_color};padding-left:12px;">
  <div style="font-size:0.75rem;font-weight:700;color:{label_color};
              text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px;">{label}</div>
  <div style="background:#F8FAFC;border-radius:8px;padding:10px 14px;
              font-size:0.875rem;color:#374151;border:1px solid #E2E8F0;margin-bottom:10px;">
    <span style="color:#94A3B8;font-size:0.8rem;">{q_label}</span><br>
    <strong>{question}</strong>
  </div>
  <div style="background:{"#F1F5F9" if side == "before" else "#ECFDF5"};border-radius:8px;
              padding:10px 14px;font-size:0.875rem;color:#374151;
              border:1px solid {"#E2E8F0" if side == "before" else "#A7F3D0"};">
    <span style="color:{"#94A3B8" if side == "before" else "#059669"};font-size:0.8rem;">{a_label}</span><br>
    {answer_escaped}
  </div>
</div>"""


def _example_title(ex: Dict) -> str:
    """根据当前语言返回示例标题"""
    lang = get_language()
    return ex.get("title_en", ex["title"]) if lang == "en" else ex["title"]


def _example_highlight(ex: Dict) -> str:
    """根据当前语言返回效果说明"""
    lang = get_language()
    return ex.get("highlight_en", ex.get("highlight", "")) if lang == "en" else ex.get("highlight", "")


def _find_example(title: str, examples: List[Dict]) -> Optional[Dict]:
    """按显示标题查找示例，同时兼容中英文标题"""
    return next(
        (e for e in examples if e["title"] == title or e.get("title_en") == title),
        None,
    )


def create_interactive_demo_tab() -> Dict[str, Any]:
    """创建交互演示 Tab 的所有组件。"""
    components = {}
    examples = _load_examples()
    category_labels = _get_category_labels()

    with gr.Row():
        # ====== 左列：示例选择 ======
        with gr.Column(scale=1, min_width=220):
            gr.HTML(f'<div class="ks-col-title">{t("demo_examples_title")}</div>')
            gr.HTML(
                f"<p style='font-size:0.8rem;color:#6B7280;margin-bottom:8px;'>"
                f"{t('demo_warning')}</p>"
            )

            # 类别筛选
            components["category_filter"] = gr.Radio(
                choices=[t("category_all")] + list(category_labels.values()),
                value=t("category_all"),
                label=t("category_filter_label"),
                container=False,
            )

            # 示例选择器 —— 原生 gr.Radio，CSS 样式化为卡片
            all_choices = [_example_title(ex) for ex in examples]
            components["example_selector"] = gr.Radio(
                choices=all_choices,
                value=all_choices[0] if all_choices else None,
                label=t("select_example_label"),
                container=False,
                elem_id="example-selector-radio",
            )

            # 缓存全量数据
            components["_examples_state"] = gr.State(value=examples)

        # ====== 中列：Before ======
        with gr.Column(scale=2):
            gr.HTML(f'<div class="ks-col-title">{t("before_title")}</div>')
            components["before_output"] = gr.HTML(
                value="<p style='color:#888;padding:20px;'>请点击左侧示例查看</p>"
            )

        # ====== 右列：After ======
        with gr.Column(scale=2):
            gr.HTML(f'<div class="ks-col-title">{t("after_title")}</div>')
            components["after_output"] = gr.HTML(
                value="<p style='color:#888;padding:20px;'>请点击左侧示例查看</p>"
            )

    components["highlight_bar"] = gr.HTML(value="")

    return components


def bind_interactive_demo_events(components: Dict[str, Any]) -> None:
    """绑定 Tab 3 的事件处理。"""
    def on_example_select(selected_title: str, examples: List[Dict]):
        ex = _find_example(selected_title, examples)
        if not ex:
            empty = "<p style='color:#888;padding:20px;'>未找到对应示例</p>"
            return empty, empty, ""
        before_html = _render_qa_html(ex["question"], ex["before_answer"], side="before")
        after_html  = _render_qa_html(ex["question"], ex["after_answer"],  side="after")
        highlight_text = _example_highlight(ex)
        highlight_html = f"""
<div style="background:#F0FDFA;border:1px solid #99F6E4;border-radius:8px;
            padding:10px 14px;margin-top:8px;font-size:0.85rem;color:#0F766E;">
  {t("effect_explanation")} {highlight_text}
</div>"""
        return before_html, after_html, highlight_html

    def on_category_filter(category: str, examples: List[Dict]):
        category_labels = _get_category_labels()
        reverse_map = {v: k for k, v in category_labels.items()}
        if category == t("category_all"):
            filtered = examples
        else:
            cat_key = reverse_map.get(category, category)
            filtered = [e for e in examples if e["category"] == cat_key]
        titles = [_example_title(e) for e in filtered]
        first = titles[0] if titles else None
        return gr.update(choices=titles, value=first)

    components["example_selector"].change(
        fn=on_example_select,
        inputs=[components["example_selector"], components["_examples_state"]],
        outputs=[
            components["before_output"],
            components["after_output"],
            components["highlight_bar"],
        ],
    )

    components["category_filter"].change(
        fn=on_category_filter,
        inputs=[components["category_filter"], components["_examples_state"]],
        outputs=[components["example_selector"]],
    )
