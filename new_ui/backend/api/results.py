from flask import jsonify, request
from . import results_bp
from ..services.behavior_compare import build_behavior_compare
from ..services.config_loader import ConfigLoader
from ..services.result_parser import parse_run_results, compare_runs

loader = ConfigLoader()


@results_bp.route("/api/results/list")
def list_runs():
    return jsonify(loader.get_eval_runs())


@results_bp.route("/api/results/<path:run_id>")
def get_run(run_id: str):
    runs = loader.get_eval_runs()
    for r in runs:
        if r["label"] == run_id:
            results = []
            for sf in r.get("summary_files", []):
                from ..services.result_parser import parse_summary
                parsed = parse_summary(sf)
                if parsed:
                    results.append(parsed)
            return jsonify({"run": r, "results": results})
    return jsonify({"error": "not found"}), 404


@results_bp.route("/api/results/compare", methods=["POST"])
def compare():
    body = request.get_json(silent=True) or {}
    labels = body.get("labels", [])
    all_runs = {r["label"]: r for r in loader.get_eval_runs()}

    run_results = {}
    for lbl in labels:
        r = all_runs.get(lbl)
        if not r:
            continue
        parsed = []
        for sf in r.get("summary_files", []):
            from ..services.result_parser import parse_summary
            p = parse_summary(sf)
            if p:
                parsed.append(p)
        run_results[lbl] = parsed

    return jsonify(compare_runs(run_results))


@results_bp.route("/api/results/behavior-compare", methods=["POST"])
def behavior_compare():
    body = request.get_json(silent=True) or {}
    labels = body.get("labels", [])
    question = body.get("question", "")
    if len(labels) < 2:
        return jsonify({"error": "at least 2 runs required"}), 400
    payload = build_behavior_compare(loader, labels, question)
    if payload.get("error"):
        return jsonify(payload), 400
    return jsonify(payload)
