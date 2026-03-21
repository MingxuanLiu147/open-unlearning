import json
import os
from pathlib import Path
from flask import jsonify, request
from . import data_bp

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"


@data_bp.route("/api/data/parse", methods=["POST"])
def parse():
    body = request.get_json(silent=True) or {}
    text = body.get("text", "")
    records = []
    errors = []
    for i, line in enumerate(text.strip().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: {e}")
    return jsonify({"records": records, "errors": errors, "count": len(records)})


@data_bp.route("/api/data/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    dest = DATA_DIR / "custom" / f.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    f.save(str(dest))
    return jsonify({"ok": True, "path": str(dest)})


@data_bp.route("/api/data/validate", methods=["POST"])
def validate():
    body = request.get_json(silent=True) or {}
    records = body.get("records", [])
    errors = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            errors.append(f"Record {i}: not an object")
    return jsonify({"valid": len(errors) == 0, "errors": errors, "count": len(records)})
