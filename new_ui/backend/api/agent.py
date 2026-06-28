import json
from flask import Response, jsonify, request
from . import agent_bp
from ..services.agent_provider import (
    AgentConfig,
    AGENT_SKILLS_META,
    load_settings,
    save_settings,
    test_connection,
    stream_chat,
    build_system_prompt,
    get_suggested_skills,
    parse_agent_response,
)
from ..services.config_loader import ConfigLoader

_config_loader = ConfigLoader()


def _enrich_context_with_catalog(context: dict) -> dict:
    """Inject real dataset/model/trainer catalogs so LLM recommends existing items."""
    mode = context.get("mode", "unlearn")

    if "available_datasets" not in context:
        ds = _config_loader.get_datasets(mode)
        context["available_datasets"] = ds

    if "available_models" not in context:
        models = _config_loader.get_models()
        context["available_models"] = [m["name"] for m in models]

    if "available_trainers" not in context:
        trainers = _config_loader.get_trainers(mode)
        context["available_trainers"] = [t["name"] for t in trainers]

    if "available_evals" not in context:
        context["available_evals"] = _config_loader.get_evals(mode)

    return context


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

    context = _enrich_context_with_catalog(context)
    system_prompt = build_system_prompt(context)
    suggested_skills = get_suggested_skills(context)

    def generate():
        yield f"data: {json.dumps({'type': 'skill_suggestions', 'skills': suggested_skills}, ensure_ascii=False)}\n\n"

        full_text = ""
        for chunk in stream_chat(messages, system_prompt=system_prompt):
            full_text += chunk
            yield f"data: {json.dumps({'type': 'message_delta', 'content': chunk}, ensure_ascii=False)}\n\n"

        parsed = parse_agent_response(full_text)
        if parsed["actions"]:
            yield f"data: {json.dumps({'type': 'actions', 'actions': parsed['actions']}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@agent_bp.route("/api/agent/skills")
def list_agent_skills():
    """List assistant skills (not experiment template skills)."""
    return jsonify(AGENT_SKILLS_META)
