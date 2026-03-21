from flask import jsonify, request
from . import config_bp
from ..services.config_loader import ConfigLoader

loader = ConfigLoader()


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
