"""
Hydra 配置扫描服务 - 适配自 webui/utils/config_loader.py，为 Flask API 提供 JSON 友好的输出。
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class ConfigLoader:

    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent.parent.parent
        self.project_root = Path(project_root)
        self.configs_dir = self.project_root / "configs"
        self.saves_dir = self.project_root / "saves"

    def _load_yaml(self, filepath: Path) -> Optional[Dict]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            return None

    # ── models ──────────────────────────────────────────────────────
    def get_models(self) -> List[Dict]:
        model_dir = self.configs_dir / "model"
        if not model_dir.exists():
            return []
        results = []
        for yf in sorted(model_dir.glob("*.yaml")):
            cfg = self._load_yaml(yf) or {}
            model_args = cfg.get("model_args", {})
            name = yf.stem
            results.append({
                "name": name,
                "path": model_args.get("pretrained_model_name_or_path", name),
                "modality": cfg.get("modality", "text"),
                "dtype": model_args.get("torch_dtype", ""),
                "attn": model_args.get("attn_implementation", ""),
            })
        return results

    # ── trainers ────────────────────────────────────────────────────
    def get_trainers(self, mode: str = None) -> List[Dict]:
        trainer_dir = self.configs_dir / "trainer"
        if not trainer_dir.exists():
            return []

        items: List[Dict] = []

        if mode == "unlearn":
            for yf in sorted(trainer_dir.glob("*.yaml")):
                if yf.stem in ("finetune",):
                    continue
                cfg = self._load_yaml(yf) or {}
                items.append({"name": yf.stem, "mode": "unlearn", "config": cfg})
        elif mode == "inject":
            sub = trainer_dir / "inject"
            if sub.exists():
                for yf in sorted(sub.glob("*.yaml")):
                    if yf.stem.startswith("base"):
                        continue
                    cfg = self._load_yaml(yf) or {}
                    items.append({"name": f"inject/{yf.stem}", "mode": "inject", "config": cfg})
        elif mode == "edit":
            sub = trainer_dir / "edit"
            if sub.exists():
                for yf in sorted(sub.glob("*.yaml")):
                    if yf.stem.startswith("base"):
                        continue
                    cfg = self._load_yaml(yf) or {}
                    items.append({"name": f"edit/{yf.stem}", "mode": "edit", "config": cfg})
        else:
            for yf in sorted(trainer_dir.rglob("*.yaml")):
                rel = yf.relative_to(trainer_dir)
                cfg_name = str(rel).replace(".yaml", "").replace(os.sep, "/")
                if "base" in cfg_name:
                    continue
                cfg = self._load_yaml(yf) or {}
                items.append({"name": cfg_name, "mode": "all", "config": cfg})
        return items

    # ── datasets ────────────────────────────────────────────────────
    def get_datasets(self, mode: str = None) -> Dict[str, List[str]]:
        ds_dir = self.configs_dir / "data" / "datasets"
        if not ds_dir.exists():
            return {}
        all_ds = sorted(yf.stem for yf in ds_dir.glob("*.yaml"))

        if mode == "unlearn":
            return {
                "forget": [d for d in all_ds if "forget" in d.lower()],
                "retain": [d for d in all_ds if "retain" in d.lower()],
            }
        if mode == "edit":
            return {"edit": [d for d in all_ds if "edit" in d.lower()]}
        if mode == "inject":
            return {"train": [d for d in all_ds if "inject" in d.lower() or "alpaca" in d.lower() or "custom" in d.lower()]}
        return {"all": all_ds}

    # ── evals ───────────────────────────────────────────────────────
    def get_evals(self, mode: str = None) -> List[str]:
        eval_dir = self.configs_dir / "eval"
        if not eval_dir.exists():
            return []
        all_evals = sorted(yf.stem for yf in eval_dir.glob("*.yaml"))
        recommended = {"unlearn": ["tofu", "muse"], "edit": ["edit"], "inject": ["inject"]}
        if mode and mode in recommended:
            rec = [e for e in recommended[mode] if e in all_evals]
            return rec + sorted(set(all_evals) - set(rec))
        return all_evals

    # ── experiments ─────────────────────────────────────────────────
    def get_experiments(self, mode: str = None) -> List[str]:
        exp_dir = self.configs_dir / "experiment"
        if not exp_dir.exists():
            return []
        base = exp_dir / mode if mode else exp_dir
        if not base.exists():
            base = exp_dir
        return sorted(
            str(yf.relative_to(exp_dir)).replace(".yaml", "").replace(os.sep, "/")
            for yf in base.rglob("*.yaml")
        )

    # ── trainer params ──────────────────────────────────────────────
    def get_trainer_params(self, name: str) -> Dict:
        path = self.configs_dir / "trainer" / f"{name}.yaml"
        return self._load_yaml(path) or {}

    # ── saved models ────────────────────────────────────────────────
    def get_saved_models(self) -> List[str]:
        if not self.saves_dir.exists():
            return []
        saved = []
        for sub in ("unlearn", "finetune", "edit", "inject"):
            sd = self.saves_dir / sub
            if not sd.exists():
                continue
            for md in sd.iterdir():
                if not md.is_dir():
                    continue
                has = any((md / f).exists() for f in ("model.safetensors", "pytorch_model.bin", "adapter_model.safetensors"))
                if has:
                    saved.append(str(md.relative_to(self.project_root)))
        return sorted(saved)

    # ── eval runs ───────────────────────────────────────────────────
    def get_eval_runs(self) -> List[Dict]:
        if not self.saves_dir.exists():
            return []
        runs = []
        for mode in ("unlearn", "inject", "edit", "finetune"):
            mp = self.saves_dir / mode
            if not mp.exists():
                continue
            for td in sorted(mp.iterdir()):
                if not td.is_dir():
                    continue
                for cd in sorted(td.iterdir()):
                    if not cd.is_dir():
                        continue
                    evals_dir = cd / "evals"
                    if not evals_dir.exists():
                        continue
                    sfiles = sorted(str(f) for f in evals_dir.glob("*_SUMMARY.json"))
                    if not sfiles:
                        continue
                    step = 0
                    if cd.name.startswith("checkpoint-"):
                        try:
                            step = int(cd.name.split("-", 1)[1])
                        except ValueError:
                            pass
                    runs.append({
                        "label": f"{mode}/{td.name} @ {cd.name}",
                        "mode": mode,
                        "task_name": td.name,
                        "checkpoint": cd.name,
                        "step": step,
                        "summary_files": sfiles,
                    })
        return runs
