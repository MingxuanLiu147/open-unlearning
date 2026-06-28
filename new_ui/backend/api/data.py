import importlib.util
import re
import uuid
from pathlib import Path

import yaml
from flask import jsonify, request
from werkzeug.utils import secure_filename

from . import data_bp

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_CFG_DIR = PROJECT_ROOT / "configs" / "data" / "datasets"


def _load_validate_module():
    script = PROJECT_ROOT / "scripts" / "validate_data.py"
    spec = importlib.util.spec_from_file_location("validate_data", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_validation(data_path: str, mode: str):
    mod = _load_validate_module()
    return mod.validate(data_path, mode)


def _convert_txt_to_jsonl(txt_path: str, mode: str, jsonl_path: str):
    """将 .txt 自由文本转换为 .jsonl，返回 (记录数, 错误列表)。"""
    mod = _load_validate_module()
    return mod.convert_freetext_to_jsonl(txt_path, mode, jsonl_path)


def _safe_stem(name: str) -> str:
    base = secure_filename(name)
    base = re.sub(r"[^a-zA-Z0-9_]", "_", Path(base).stem)
    return base or "upload"


def _rel_data_path(abs_path: Path) -> str:
    try:
        return str(abs_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(abs_path)


def _write_dataset_yaml(stem: str, spec: dict) -> None:
    DATASET_CFG_DIR.mkdir(parents=True, exist_ok=True)
    path = DATASET_CFG_DIR / f"{stem}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True)


@data_bp.route("/api/data/parse", methods=["POST"])
def parse():
    import json

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
    mode = request.form.get("mode", "unlearn")
    purpose = request.form.get("purpose", "forget")
    if mode not in ("inject", "unlearn", "edit"):
        return jsonify({"error": "invalid mode"}), 400

    tag = uuid.uuid4().hex[:10]
    orig = secure_filename(f.filename or "data.jsonl")
    stem_file = _safe_stem(Path(orig).stem or "data")
    sfx = Path(orig).suffix.lower()
    is_txt = sfx == ".txt"
    if sfx not in (".json", ".jsonl", ".txt"):
        sfx = ".jsonl"
    dest = DATA_DIR / "custom" / f"{purpose}_{tag}_{stem_file}{sfx}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    f.save(str(dest))

    # .txt 文件先转换为 .jsonl，再走正常校验流程
    if is_txt:
        jsonl_dest = dest.with_suffix(".jsonl")
        count, conv_errors = _convert_txt_to_jsonl(str(dest), mode, str(jsonl_dest))
        if conv_errors:
            return jsonify({
                "ok": False,
                "path": str(dest),
                "total": count + len(conv_errors),
                "passed": count,
                "errors": conv_errors[:50],
            }), 400
        if count == 0:
            return jsonify({
                "ok": False,
                "path": str(dest),
                "total": 0,
                "passed": 0,
                "errors": ["文件不包含任何有效段落"],
            }), 400
        # 后续使用转换后的 jsonl 文件
        dest = jsonl_dest

    total, passed, errors = _run_validation(str(dest), mode)
    if errors:
        return jsonify({
            "ok": False,
            "path": str(dest),
            "total": total,
            "passed": passed,
            "errors": errors[:50],
        }), 400

    rel = _rel_data_path(dest)
    ds_stem = f"Custom_{purpose}_{tag}"

    if mode == "unlearn" and purpose in ("forget", "retain"):
        _write_dataset_yaml(ds_stem, {
            ds_stem: {
                "handler": "QADataset",
                "args": {
                    "data_path": rel,
                    "question_key": "question",
                    "answer_key": "answer",
                    "max_length": 512,
                },
            }
        })
    elif mode == "inject" and purpose == "train":
        _write_dataset_yaml(ds_stem, {
            ds_stem: {
                "handler": "InjectDataset",
                "args": {
                    "data_path": rel,
                    "format_type": "alpaca",
                    "instruction_key": "instruction",
                    "input_key": "input",
                    "output_key": "output",
                    "max_length": 2048,
                },
            }
        })
    elif mode == "edit" and purpose == "edit":
        _write_dataset_yaml(ds_stem, {
            ds_stem: {
                "handler": "EditingDataset",
                "args": {
                    "data_path": rel,
                    "max_length": 512,
                    "target_old_key": "target_old",
                },
            }
        })
    else:
        return jsonify({"error": "invalid purpose for mode"}), 400

    return jsonify({
        "ok": True,
        "path": str(dest),
        "dataset_stem": ds_stem,
        "total": total,
        "passed": passed,
        "errors": [],
    })


@data_bp.route("/api/data/validate", methods=["POST"])
def validate():
    body = request.get_json(silent=True) or {}
    path = body.get("path")
    mode = body.get("mode", "inject")
    if not path:
        return jsonify({"ok": False, "error": "invalid path"}), 400
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.is_file():
        return jsonify({"ok": False, "error": "file not found"}), 400
    total, passed, errors = _run_validation(str(p), mode)
    return jsonify({
        "ok": len(errors) == 0,
        "total": total,
        "passed": passed,
        "errors": errors[:100],
    })
