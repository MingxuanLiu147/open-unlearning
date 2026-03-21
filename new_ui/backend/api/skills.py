from flask import jsonify, request
from . import skills_bp
from ..services import skill_engine


@skills_bp.route("/api/skills")
def list_skills():
    return jsonify(skill_engine.list_skills())


@skills_bp.route("/api/skills", methods=["POST"])
def create_skill():
    body = request.get_json(silent=True) or {}
    sid = skill_engine.save_skill("", body)
    return jsonify({"ok": True, "id": sid}), 201


@skills_bp.route("/api/skills/<skill_id>")
def get_skill(skill_id: str):
    s = skill_engine.get_skill(skill_id)
    if not s:
        return jsonify({"error": "not found"}), 404
    return jsonify(s)


@skills_bp.route("/api/skills/<skill_id>", methods=["PUT"])
def update_skill(skill_id: str):
    body = request.get_json(silent=True) or {}
    skill_engine.save_skill(skill_id, body)
    return jsonify({"ok": True, "id": skill_id})


@skills_bp.route("/api/skills/<skill_id>", methods=["DELETE"])
def delete_skill(skill_id: str):
    ok = skill_engine.delete_skill(skill_id)
    return jsonify({"ok": ok})


@skills_bp.route("/api/skills/<skill_id>/run", methods=["POST"])
def run_skill(skill_id: str):
    s = skill_engine.get_skill(skill_id)
    if not s:
        return jsonify({"error": "not found"}), 404
    return jsonify({"ok": True, "skill": s, "message": "Skill queued for execution"})
