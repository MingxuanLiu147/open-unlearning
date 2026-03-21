"""
Skill 模板引擎 - 加载/保存/执行 YAML Skill 定义。
"""

import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import yaml

SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"


def _ensure_dir():
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)


def list_skills() -> List[Dict]:
    _ensure_dir()
    skills = []
    for yf in sorted(SKILLS_DIR.glob("*.yaml")):
        try:
            with open(yf, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            data["id"] = yf.stem
            data["file"] = str(yf)
            skills.append(data)
        except Exception:
            continue
    return skills


def get_skill(skill_id: str) -> Optional[Dict]:
    path = SKILLS_DIR / f"{skill_id}.yaml"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data["id"] = skill_id
        return data
    except Exception:
        return None


def save_skill(skill_id: str, data: Dict) -> str:
    _ensure_dir()
    if not skill_id:
        skill_id = str(uuid.uuid4())[:8]
    path = SKILLS_DIR / f"{skill_id}.yaml"
    clean = {k: v for k, v in data.items() if k not in ("id", "file")}
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(clean, f, allow_unicode=True, default_flow_style=False)
    return skill_id


def delete_skill(skill_id: str) -> bool:
    path = SKILLS_DIR / f"{skill_id}.yaml"
    if path.exists():
        path.unlink()
        return True
    return False
