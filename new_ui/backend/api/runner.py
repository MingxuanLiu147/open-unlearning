import json
import time
from flask import Response, jsonify, request
from . import runner_bp
from ..services.command_runner import CommandRunner

_runner = CommandRunner()


@runner_bp.route("/api/run/start", methods=["POST"])
def start():
    if _runner.status.running:
        return jsonify({"ok": False, "error": "already running"}), 409

    body = request.get_json(silent=True) or {}
    env = {}
    gpu = body.get("gpu", "")
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = _runner.build_command(
        mode=body.get("mode", "unlearn"),
        model=body.get("model"),
        trainer=body.get("trainer"),
        experiment=body.get("experiment"),
        task_name=body.get("task_name", "experiment"),
        overrides=body.get("overrides"),
        eval_suite=body.get("eval_suite"),
        model_path=body.get("model_path"),
    )

    ok = _runner.run(cmd, env=env if env else None)
    return jsonify({"ok": ok, "command": cmd})


@runner_bp.route("/api/run/stop", methods=["POST"])
def stop():
    return jsonify({"ok": _runner.stop()})


@runner_bp.route("/api/run/status")
def status():
    return jsonify({
        "running": _runner.status.running,
        "exit_code": _runner.status.exit_code,
        "error": _runner.status.error,
    })


@runner_bp.route("/api/run/log")
def log_stream():
    """SSE endpoint - streams log lines in real-time."""
    sent = 0

    def generate():
        nonlocal sent
        while True:
            lines = list(_runner.log_lines)
            if sent < len(lines):
                for line in lines[sent:]:
                    yield f"data: {json.dumps({'line': line})}\n\n"
                sent = len(lines)
            if not _runner.status.running and sent >= len(_runner.log_lines):
                yield f"data: {json.dumps({'done': True, 'exit_code': _runner.status.exit_code})}\n\n"
                break
            _runner.wait_for_log(timeout=0.5)

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
