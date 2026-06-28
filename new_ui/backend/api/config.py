import re
from pathlib import Path

import yaml
from flask import jsonify, request
from . import config_bp
from ..services.config_loader import ConfigLoader

loader = ConfigLoader()
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODEL_CFG_DIR = PROJECT_ROOT / "configs" / "model"


def _safe_model_name(name: str) -> str:
    n = re.sub(r"[^a-zA-Z0-9_\-.]", "_", name.strip())
    return n or "custom_model"


@config_bp.route("/api/config/models")
def models():
    return jsonify(loader.get_models())


@config_bp.route("/api/config/trainers")
def trainers():
    mode = request.args.get("mode")
    return jsonify(loader.get_trainers(mode))


@config_bp.route("/api/config/datasets")
def datasets():
    mode = request.args.get("mode")
    return jsonify(loader.get_datasets(mode))


@config_bp.route("/api/config/evals")
def evals():
    mode = request.args.get("mode")
    return jsonify(loader.get_evals(mode))


@config_bp.route("/api/config/experiments")
def experiments():
    mode = request.args.get("mode")
    return jsonify(loader.get_experiments(mode))


@config_bp.route("/api/config/trainer-params")
def trainer_params():
    name = request.args.get("name", "")
    return jsonify(loader.get_trainer_params(name))


@config_bp.route("/api/config/import", methods=["POST"])
def import_config():
    data = request.get_json(silent=True) or {}
    return jsonify({"ok": True, "config": data})


@config_bp.route("/api/config/export", methods=["POST"])
def export_config():
    data = request.get_json(silent=True) or {}
    return jsonify({"ok": True, "config": data})


@config_bp.route("/api/config/create-model", methods=["POST"])
def create_model():
    body = request.get_json(silent=True) or {}
    raw_name = (body.get("name") or "").strip()
    if not raw_name:
        return jsonify({"ok": False, "error": "name required"}), 400
    name = _safe_model_name(raw_name)
    path = (body.get("path") or "").strip()
    dtype = (body.get("dtype") or "bfloat16").strip()
    apply_chat = bool(body.get("apply_chat_template", True))
    if not path:
        return jsonify({"ok": False, "error": "path required"}), 400

    MODEL_CFG_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_CFG_DIR / f"{name}.yaml"
    if out.exists():
        return jsonify({"ok": False, "error": "config name already exists"}), 409

    template_args = {
        "apply_chat_template": apply_chat,
        "system_prompt": "You are a helpful assistant.",
        "user_start_tag": "<|user|>",
        "user_end_tag": "<|end|>",
        "asst_start_tag": "<|assistant|>",
        "asst_end_tag": "<|end|>",
    }
    cfg = {
        "model_args": {
            "pretrained_model_name_or_path": path,
            "torch_dtype": dtype,
        },
        "tokenizer_args": {
            "pretrained_model_name_or_path": path,
        },
        "template_args": template_args,
    }
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return jsonify({"ok": True, "model": name})


@config_bp.route("/api/config/method-catalog")
def method_catalog():
    """返回三大方法分类清单 + 可选场景推荐。"""
    from ..services.method_catalog import get_catalog_summary, recommend_methods

    scenario = request.args.get("scenario", "")
    result: dict = {"catalog": get_catalog_summary()}
    if scenario:
        result["recommendations"] = recommend_methods(scenario)
    return jsonify(result)
