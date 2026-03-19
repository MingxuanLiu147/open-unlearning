import json
from flask import Response, jsonify, request
from . import agent_bp
from ..services.agent_provider import (
    AgentConfig, load_settings, save_settings, test_connection, stream_chat
)


@agent_bp.route("/api/agent/config")
def get_config():
    cfg = load_settings()
    safe = cfg.__dict__.copy()
    if safe.get("api_key"):
        safe["api_key"] = safe["api_key"][:8] + "..." if len(safe["api_key"]) > 8 else "***"
    return jsonify(safe)


@agent_bp.route("/api/agent/config", methods=["PUT"])
def put_config():
    body = request.get_json(silent=True) or {}
    cfg = AgentConfig(**{k: v for k, v in body.items() if hasattr(AgentConfig, k)})
    save_settings(cfg)
    return jsonify({"ok": True})


@agent_bp.route("/api/agent/test", methods=["POST"])
def test():
    body = request.get_json(silent=True) or {}
    cfg = AgentConfig(**{k: v for k, v in body.items() if hasattr(AgentConfig, k)})
    return jsonify(test_connection(cfg))


@agent_bp.route("/api/agent/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    messages = body.get("messages", [])
    context = body.get("context", {})

    system_prompt = None
    if context and any(context.get(k) for k in ('mode', 'model', 'trainer')):
        ctx_parts = []
        if context.get('mode'):
            ctx_parts.append(f"当前实验模式: {context['mode']}")
        if context.get('model'):
            ctx_parts.append(f"已选模型: {context['model']}")
        if context.get('trainer'):
            ctx_parts.append(f"已选方法: {context['trainer']}")
        if context.get('datasets'):
            ctx_parts.append(f"数据集: {json.dumps(context['datasets'], ensure_ascii=False)}")
        if context.get('skills'):
            ctx_parts.append(f"可用 Skills: {json.dumps(context['skills'], ensure_ascii=False)}")
        from ..services.agent_provider import DEFAULT_SYSTEM_PROMPT
        system_prompt = DEFAULT_SYSTEM_PROMPT + "\n\n--- 当前实验上下文 ---\n" + "\n".join(ctx_parts)

    def generate():
        for chunk in stream_chat(messages, system_prompt=system_prompt):
            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
