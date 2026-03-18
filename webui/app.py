# -*- coding: utf-8 -*-
"""
Know-Surgery WebUI
==================

基于 Gradio 的大模型知识可控更新工具包图形界面。
支持 Unlearn(删)、Inject(增)、Edit(改) 三大操作的配置、运行和结果可视化。

启动方式:
    python webui/app.py

或指定端口:
    python webui/app.py --port 7860 --share
"""

import sys
import os
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Generator

import gradio as gr

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import ConfigLoader
from utils.runner import CommandRunner
from components.config_panel import create_config_panel, update_config_on_mode_change
from components.run_panel import (
    create_run_panel,
    generate_command_preview,
    update_status,
    load_eval_results,
)
from components.params import (
    create_params_panel,
    parse_overrides,
    update_params_from_trainer,
)
from components.results_compare import (
    create_results_compare_tab,
    bind_results_compare_events,
)
from components.interactive_demo import (
    create_interactive_demo_tab,
    bind_interactive_demo_events,
)
from components.skill_wizard import create_skill_wizard_tab, bind_skill_wizard_events
from utils.ui_state import build_config_dict, apply_config_dict
from utils.i18n import t, set_language, get_language, SUPPORTED_LANGUAGES, TRANSLATIONS

# 语言偏好持久化文件
LANG_PREF_FILE = Path(__file__).parent / ".lang_preference"


def _load_lang_preference() -> str:
    """从文件加载语言偏好"""
    if LANG_PREF_FILE.exists():
        try:
            return LANG_PREF_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return "zh"


def _save_lang_preference(lang: str):
    """保存语言偏好到文件"""
    try:
        LANG_PREF_FILE.write_text(lang, encoding="utf-8")
    except Exception:
        pass


# 在模块加载时恢复语言偏好
_saved_lang = _load_lang_preference()
if _saved_lang in SUPPORTED_LANGUAGES:
    set_language(_saved_lang)


def _build_header_html(lang: str) -> str:
    """根据语言生成 header HTML，消除重复代码"""
    if lang == "en":
        subtitle = "LLM Knowledge Surgery Toolkit &nbsp;·&nbsp; Unlearn · Inject · Edit · Eval"
    else:
        subtitle = "大模型知识可控更新工具包 &nbsp;·&nbsp; Unlearn · Inject · Edit · Eval"
    return f"""
    <div id="ks-header">
      <h1>🔬 Know-Surgery WebUI</h1>
      <p>{subtitle}</p>
    </div>
    """


def _build_zh_to_en_js() -> str:
    """从 i18n 翻译字典生成 JS 格式的中→英映射，用于客户端文字替换"""
    pairs = []
    for trans in TRANSLATIONS.values():
        zh = trans.get("zh", "")
        en = trans.get("en", "")
        if zh and en and zh != en:
            zh_esc = zh.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            en_esc = en.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            pairs.append(f'  "{zh_esc}": "{en_esc}"')
    return "{\n" + ",\n".join(pairs) + "\n}"

config_loader = ConfigLoader(PROJECT_ROOT / "configs")
runner = CommandRunner(PROJECT_ROOT)

CSS_FILE = Path(__file__).parent / "assets" / "custom.css"


def _load_custom_css() -> str:
    if CSS_FILE.exists():
        return CSS_FILE.read_text(encoding="utf-8")
    return ""


def build_ui() -> gr.Blocks:
    """构建完整的 WebUI 界面（4-Tab 布局，支持中英文切换）"""
    
    # 获取当前语言设置（服务启动时读取一次，之后通过 app.load 动态更新）
    current_lang = get_language()

    with gr.Blocks(title="Know-Surgery WebUI") as app:
        # 全局语言状态
        lang_state = gr.State(value=current_lang)
        
        # ===== 标题栏（含语言切换） =====
        with gr.Row(elem_id="ks-header-row"):
            with gr.Column(scale=10):
                header_html = gr.HTML(_build_header_html(current_lang))
            with gr.Column(scale=1, min_width=120):
                lang_btn = gr.Radio(
                    choices=["中文", "English"],
                    value="English" if current_lang == "en" else "中文",
                    label="🌐",
                    container=False,
                    elem_id="lang-switcher",
                )

        with gr.Tabs():
            # ==================== Tab 1: 配置与运行 ====================
            with gr.Tab(t("tab_config")) as tab_config:
                with gr.Row(equal_height=False):
                    # ---- 左栏：任务配置 ----
                    with gr.Column(scale=1, min_width=260):
                        config_components = create_config_panel(config_loader)

                    # ---- 中栏：参数调节 ----
                    with gr.Column(scale=1, min_width=240):
                        params_components = create_params_panel(
                            config_loader, use_accordion=False
                        )

                    # ---- 右栏：运行控制 ----
                    with gr.Column(scale=1, min_width=260):
                        run_components = create_run_panel()

            # ==================== Tab 2: 结果对比 ====================
            with gr.Tab(t("tab_compare")) as tab_compare:
                compare_components = create_results_compare_tab(config_loader)

            # ==================== Tab 3: 交互演示 ====================
            with gr.Tab(t("tab_demo")) as tab_demo:
                demo_components = create_interactive_demo_tab()

            # ==================== Tab 4: 智能向导 ====================
            with gr.Tab(t("tab_wizard")) as tab_wizard:
                wizard_components = create_skill_wizard_tab()

        # ==================== 事件绑定 ====================

        # --- 语言切换事件 ---
        def on_language_change(lang_choice):
            lang = "en" if lang_choice == "English" else "zh"
            set_language(lang)
            _save_lang_preference(lang)
            # t() 在 set_language 后调用，返回新语言的翻译
            return (
                lang,
                _build_header_html(lang),
                gr.update(label=t("tab_config")),
                gr.update(label=t("tab_compare")),
                gr.update(label=t("tab_demo")),
                gr.update(label=t("tab_wizard")),
            )

        # 点击即切换：去掉 location.reload()，改用 JS TreeWalker 替换文字节点
        _ZH_TO_EN_JS = _build_zh_to_en_js()
        lang_btn.change(
            fn=on_language_change,
            inputs=[lang_btn],
            outputs=[lang_state, header_html, tab_config, tab_compare, tab_demo, tab_wizard],
            js="""(lang) => {
                const newLang = lang === 'English' ? 'en' : 'zh';
                localStorage.setItem('ks_language', newLang);
                window._ksApplyI18n && window._ksApplyI18n(newLang);
                return lang;
            }"""
        )

        # --- Tab 2 事件 ---
        bind_results_compare_events(compare_components, config_loader)

        # --- Tab 3 事件 ---
        bind_interactive_demo_events(demo_components)

        # --- Tab 4 事件（含跨 Tab 一键应用） ---
        bind_skill_wizard_events(
            wizard_components, config_components, params_components
        )

        # --- Tab 1: Mode 改变时更新配置 ---
        config_components["mode"].change(
            fn=lambda mode: update_config_on_mode_change(mode, config_loader),
            inputs=[config_components["mode"]],
            outputs=[
                config_components["experiment"],
                config_components["trainer"],
                config_components["unlearn_datasets_group"],
                config_components["inject_datasets_group"],
                config_components["edit_datasets_group"],
                config_components["forget_dataset"],
                config_components["retain_dataset"],
                config_components["train_dataset"],
                config_components["edit_dataset"],
                config_components["eval_suite"],
                config_components["train_eval_group"],
                config_components["eval_mode_group"],
            ],
        )

        # 刷新已保存模型列表
        def refresh_saved_models():
            saved_models = config_loader.get_saved_models()
            return gr.update(
                choices=saved_models, value=saved_models[0] if saved_models else None
            )

        config_components["refresh_saved_models"].click(
            fn=refresh_saved_models,
            inputs=[],
            outputs=[config_components["saved_model_path"]],
        )

        # --- Tab 1: Trainer 改变时同步关键默认参数 ---
        config_components["trainer"].change(
            fn=lambda trainer: update_params_from_trainer(trainer, config_loader),
            inputs=[config_components["trainer"]],
            outputs=[
                params_components["learning_rate"],
                params_components["num_epochs"],
                params_components["batch_size"],
                params_components["gradient_accumulation"],
                params_components["max_length"],
                params_components["warmup_ratio"],
                params_components["method_args_json"],
            ],
        )

        # --- Tab 1: 配置改变时更新命令预览 ---
        config_inputs = [
            config_components["mode"],
            config_components["model"],
            config_components["trainer"],
            config_components["experiment"],
            config_components["task_name"],
            config_components["seed"],
            config_components["forget_dataset"],
            config_components["retain_dataset"],
            config_components["train_dataset"],
            config_components["edit_dataset"],
            config_components["eval_suite"],  # 训练模式的评测套件
            config_components["eval_suite_select"],  # eval 模式的评测套件
            config_components["saved_model_path"],
        ]

        params_inputs = [
            params_components["learning_rate"],
            params_components["num_epochs"],
            params_components["batch_size"],
            params_components["gradient_accumulation"],
            params_components["max_length"],
            params_components["warmup_ratio"],
            params_components["method_args_json"],
            params_components["overrides"],
        ]

        all_preview_inputs = config_inputs + params_inputs

        def update_command(*args):
            (
                mode,
                model,
                trainer,
                experiment,
                task_name,
                seed,
                forget,
                retain,
                train,
                edit,
                train_eval_suite,  # 训练模式的评测套件
                eval_mode_suite,  # eval 模式的评测套件
                saved_model,
            ) = args[:13]
            (
                learning_rate,
                num_epochs,
                batch_size,
                gradient_accumulation,
                max_length,
                warmup_ratio,
                method_args_json,
                overrides_text,
            ) = args[13:]

            overrides = parse_overrides(
                learning_rate,
                num_epochs,
                batch_size,
                gradient_accumulation,
                max_length,
                warmup_ratio,
                method_args_json,
                overrides_text,
            )

            use_experiment = experiment and not experiment.startswith("(")
            if not use_experiment:
                if mode == "unlearn":
                    if forget:
                        overrides["data/datasets@data.forget"] = forget
                    if retain:
                        overrides["data/datasets@data.retain"] = retain
                elif mode == "inject":
                    if train:
                        overrides["data/datasets@data.train"] = train
                elif mode == "edit":
                    if edit:
                        overrides["data/datasets@data.edit"] = edit

            # 根据模式选择正确的 eval_suite
            eval_suite = eval_mode_suite if mode == "eval" else train_eval_suite

            return generate_command_preview(
                mode=mode,
                model=model,
                trainer=trainer,
                experiment=experiment,
                task_name=task_name,
                seed=int(seed) if seed else 42,
                eval_suite=eval_suite,
                saved_model_path=saved_model,
                overrides=overrides,
            )

        for component in all_preview_inputs:
            component.change(
                fn=update_command,
                inputs=all_preview_inputs,
                outputs=[run_components["command_preview"]],
            )

        # --- Tab 1: 运行按钮 ---
        def run_training(
            mode,
            model,
            trainer,
            experiment,
            task_name,
            seed,
            forget_dataset,
            retain_dataset,
            train_dataset,
            edit_dataset,
            train_eval_suite,  # 训练模式的评测套件
            eval_mode_suite,  # eval 模式的评测套件
            saved_model_path,
            cuda_devices,
            learning_rate,
            num_epochs,
            batch_size,
            gradient_accumulation,
            max_length,
            warmup_ratio,
            method_args_json,
            overrides_text,
        ):
            if mode == "eval":
                command = runner.build_command(
                    mode="eval",
                    model=model,
                    task_name=task_name,
                    eval_suite=eval_mode_suite,
                    model_path=saved_model_path,
                    overrides={},
                )
            else:
                overrides = parse_overrides(
                    learning_rate,
                    num_epochs,
                    batch_size,
                    gradient_accumulation,
                    max_length,
                    warmup_ratio,
                    method_args_json,
                    overrides_text,
                )

                use_experiment = experiment and not experiment.startswith("(")
                if not use_experiment:
                    if mode == "unlearn":
                        if forget_dataset:
                            overrides["data/datasets@data.forget"] = forget_dataset
                        if retain_dataset:
                            overrides["data/datasets@data.retain"] = retain_dataset
                    elif mode == "inject":
                        if train_dataset:
                            overrides["data/datasets@data.train"] = train_dataset
                    elif mode == "edit":
                        if edit_dataset:
                            overrides["data/datasets@data.edit"] = edit_dataset

                # 训练模式添加 eval 配置
                if train_eval_suite:
                    overrides["eval"] = train_eval_suite

                command = runner.build_command(
                    mode=mode,
                    model=model,
                    trainer=trainer,
                    experiment=use_experiment and experiment or None,
                    task_name=task_name,
                    overrides=overrides,
                )

            log_lines = []
            output_dir = ""

            def log_callback(line: str):
                nonlocal output_dir
                log_lines.append(line)
                # 从日志中解析输出目录（Hydra 会打印类似 "Working directory: ..." 的信息）
                if (
                    "Working directory:" in line
                    or "output_dir:" in line
                    or "保存到:" in line
                ):
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        output_dir = parts[1].strip()
                # 也尝试解析 "Saving model to ..." 格式
                if "Saving model to" in line or "Model saved to" in line:
                    parts = line.split("to", 1)
                    if len(parts) > 1:
                        output_dir = parts[1].strip().rstrip("/")

            env = {"CUDA_VISIBLE_DEVICES": cuda_devices} if cuda_devices else {}
            runner.run(command, log_callback=log_callback, env=env)
            time.sleep(0.5)

            last_log_count = 0
            while runner.is_running():
                if len(log_lines) > last_log_count:
                    last_log_count = len(log_lines)
                    yield (
                        "\n".join(log_lines),
                        *update_status(running=True),
                        gr.update(),  # output_dir 保持不变
                        gr.update(),  # result_summary 保持不变
                    )
                time.sleep(0.3)

            # 如果未从日志解析到 output_dir，尝试根据配置推断
            if not output_dir and mode != "eval":
                # 格式: saves/{mode}/{date}_{task_name}_{model}_{trainer}/
                from datetime import datetime

                date_str = datetime.now().strftime("%Y%m%d")
                model_short = model.split("/")[-1] if model else "model"
                trainer_short = trainer.split("/")[-1] if trainer else "method"
                output_dir = (
                    f"saves/{mode}/{date_str}_{task_name}_{model_short}_{trainer_short}"
                )

            # 运行结束，返回最终结果
            final_status = update_status(
                running=False, exit_code=runner.status.exit_code
            )

            # 尝试自动加载结果
            result_html = ""
            if output_dir and runner.status.exit_code == 0:
                result_html = load_eval_results(output_dir)

            yield (
                "\n".join(log_lines),
                *final_status,
                gr.update(value=output_dir),
                gr.update(value=result_html) if result_html else gr.update(),
            )

        run_components["run_btn"].click(
            fn=run_training,
            inputs=[
                config_components["mode"],
                config_components["model"],
                config_components["trainer"],
                config_components["experiment"],
                config_components["task_name"],
                config_components["seed"],
                config_components["forget_dataset"],
                config_components["retain_dataset"],
                config_components["train_dataset"],
                config_components["edit_dataset"],
                config_components["eval_suite"],  # 训练模式的评测套件
                config_components["eval_suite_select"],  # eval 模式的评测套件
                config_components["saved_model_path"],
                run_components["cuda_devices"],
                params_components["learning_rate"],
                params_components["num_epochs"],
                params_components["batch_size"],
                params_components["gradient_accumulation"],
                params_components["max_length"],
                params_components["warmup_ratio"],
                params_components["method_args_json"],
                params_components["overrides"],
            ],
            outputs=[
                run_components["log_output"],
                run_components["status"],
                run_components["run_btn"],
                run_components["stop_btn"],
                run_components["output_dir"],  # 输出目录回填
                run_components["result_summary"],  # 结果摘要自动加载
            ],
        )

        run_components["load_results_btn"].click(
            fn=load_eval_results,
            inputs=[run_components["output_dir"]],
            outputs=[run_components["result_summary"]],
        )

        def stop_training():
            runner.stop()
            return update_status(running=False, exit_code=-1)

        run_components["stop_btn"].click(
            fn=stop_training,
            inputs=[],
            outputs=[
                run_components["status"],
                run_components["run_btn"],
                run_components["stop_btn"],
            ],
        )

        run_components["clear_log_btn"].click(
            fn=lambda: "", inputs=[], outputs=[run_components["log_output"]]
        )

        run_components["copy_btn"].click(
            fn=None,
            inputs=[run_components["command_preview"]],
            outputs=[],
            js="(text) => { navigator.clipboard.writeText(text); }",
        )

        # 导出配置（使用统一的 build_config_dict）
        def export_config(
            mode,
            model,
            trainer,
            experiment,
            task_name,
            seed,
            forget_dataset,
            retain_dataset,
            train_dataset,
            edit_dataset,
            eval_suite,
            learning_rate,
            num_epochs,
            batch_size,
            gradient_accumulation,
            max_length,
            warmup_ratio,
            method_args_json,
            overrides_text,
        ):
            import yaml

            config = build_config_dict(
                mode=mode,
                model=model,
                trainer=trainer,
                experiment=experiment,
                task_name=task_name,
                seed=seed,
                forget_dataset=forget_dataset,
                retain_dataset=retain_dataset,
                train_dataset=train_dataset,
                edit_dataset=edit_dataset,
                eval_suite=eval_suite,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
                gradient_accumulation=gradient_accumulation,
                max_length=max_length,
                warmup_ratio=warmup_ratio,
                method_args_json=method_args_json,
                overrides_text=overrides_text,
            )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
                return f.name

        config_components["export_btn"].click(
            fn=export_config,
            inputs=[
                config_components["mode"],
                config_components["model"],
                config_components["trainer"],
                config_components["experiment"],
                config_components["task_name"],
                config_components["seed"],
                config_components["forget_dataset"],
                config_components["retain_dataset"],
                config_components["train_dataset"],
                config_components["edit_dataset"],
                config_components["eval_suite"],
                params_components["learning_rate"],
                params_components["num_epochs"],
                params_components["batch_size"],
                params_components["gradient_accumulation"],
                params_components["max_length"],
                params_components["warmup_ratio"],
                params_components["method_args_json"],
                params_components["overrides"],
            ],
            outputs=[config_components["config_download"]],
        )

        # 导入配置（使用统一的 apply_config_dict，完整恢复所有字段）
        def import_config(file):
            if file is None:
                return [gr.update()] * 18

            import yaml

            try:
                with open(file.name, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                return apply_config_dict(config)
            except Exception as e:
                gr.Warning(f"导入配置失败: {e}")
                return [gr.update()] * 18

        config_components["import_btn"].click(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[config_components["config_upload"]],
        )

        config_components["config_upload"].change(
            fn=import_config,
            inputs=[config_components["config_upload"]],
            outputs=[
                # 基础配置 (10 个)
                config_components["mode"],
                config_components["model"],
                config_components["trainer"],
                config_components["experiment"],
                config_components["task_name"],
                config_components["seed"],
                config_components["forget_dataset"],
                config_components["retain_dataset"],
                config_components["train_dataset"],
                config_components["edit_dataset"],
                # 训练参数 (6 个)
                params_components["learning_rate"],
                params_components["num_epochs"],
                params_components["batch_size"],
                params_components["gradient_accumulation"],
                params_components["max_length"],
                params_components["warmup_ratio"],
                # 方法参数和 overrides (2 个)
                params_components["method_args_json"],
                params_components["overrides"],
            ],
        )

        # 初始化命令预览
        app.load(
            fn=lambda: generate_command_preview(
                mode="unlearn",
                model="Qwen2.5-7B-Instruct",
                trainer="SimNPO",
                experiment=None,
                task_name="my_experiment",
                seed=42,
                overrides={},
            ),
            inputs=[],
            outputs=[run_components["command_preview"]],
        )
        
        # 页面加载时：定义全局翻译函数，并从 localStorage 恢复语言（客户端）
        app.load(
            fn=None,
            inputs=[],
            outputs=[],
            js=f"""() => {{
                const ZH_TO_EN = {_build_zh_to_en_js()};
                const EN_TO_ZH = Object.fromEntries(Object.entries(ZH_TO_EN).map(([k, v]) => [v, k]));

                // 全局文字替换函数：通过 TreeWalker 遍历所有文本节点
                window._ksApplyI18n = function(lang) {{
                    const dict = lang === 'en' ? ZH_TO_EN : EN_TO_ZH;
                    const walker = document.createTreeWalker(
                        document.body,
                        NodeFilter.SHOW_TEXT,
                        {{
                            acceptNode: function(node) {{
                                const tag = node.parentElement && node.parentElement.tagName.toLowerCase();
                                if (!tag || ['script', 'style', 'textarea', 'input', 'code', 'pre'].includes(tag)) {{
                                    return NodeFilter.FILTER_REJECT;
                                }}
                                return NodeFilter.FILTER_ACCEPT;
                            }}
                        }}
                    );
                    const nodes = [];
                    let n;
                    while (n = walker.nextNode()) nodes.push(n);
                    nodes.forEach(function(n) {{
                        const trimmed = n.textContent.trim();
                        if (trimmed && dict[trimmed] !== undefined) {{
                            const pre = n.textContent.match(/^\\s*/)[0];
                            const suf = n.textContent.match(/\\s*$/)[0];
                            n.textContent = pre + dict[trimmed] + suf;
                        }}
                    }});
                    // 替换 placeholder 属性
                    document.querySelectorAll('[placeholder]').forEach(function(el) {{
                        const ph = el.getAttribute('placeholder');
                        if (ph && dict[ph] !== undefined) el.setAttribute('placeholder', dict[ph]);
                    }});
                }};

                // 从 localStorage 读取并应用语言（延迟等待 Gradio 渲染完成）
                const savedLang = localStorage.getItem('ks_language');
                if (savedLang && savedLang !== 'zh') {{
                    setTimeout(() => window._ksApplyI18n(savedLang), 800);
                }}
            }}"""
        )

        # 页面加载时：服务端返回正确的 lang_btn 值和 Tab label（修复刷新后状态）
        def _init_lang_state():
            lang = get_language()
            return (
                gr.update(value="English" if lang == "en" else "中文"),
                _build_header_html(lang),
                gr.update(label=t("tab_config")),
                gr.update(label=t("tab_compare")),
                gr.update(label=t("tab_demo")),
                gr.update(label=t("tab_wizard")),
            )

        app.load(
            fn=_init_lang_state,
            inputs=[],
            outputs=[lang_btn, header_html, tab_config, tab_compare, tab_demo, tab_wizard],
        )

    return app


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Know-Surgery WebUI")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务地址")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    args = parser.parse_args()

    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║           Know-Surgery WebUI                              ║
    ║   大模型知识可控更新工具包 - 图形化界面                         ║ 
    ╚═══════════════════════════════════════════════════════════╝

    启动参数:
      - Host: {args.host}
      - Port: {args.port}
      - Share: {args.share}

    项目根目录: {PROJECT_ROOT}
    配置目录: {PROJECT_ROOT / "configs"}
    """)

    app = build_ui()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=_load_custom_css(),
    )


if __name__ == "__main__":
    main()
