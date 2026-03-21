from flask import jsonify, request
from . import results_bp
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
