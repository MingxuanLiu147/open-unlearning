"""Helpers for executing knowledge editing runs from the train entrypoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
import json
import logging
from pathlib import Path
from typing import Any

from trainer.edit.base import EditRequest

logger = logging.getLogger(__name__)


def build_edit_requests(edit_data: Any, max_edits: int | None = None) -> list[EditRequest]:
    """Build a bounded list of edit requests from one or more edit datasets."""
    if edit_data is None:
        raise ValueError("Edit mode requires `data.edit` to be configured.")

    if isinstance(edit_data, Sequence) and not isinstance(edit_data, (str, bytes)):
        if all(isinstance(item, EditRequest) for item in edit_data):
            requests = list(edit_data)
            return requests[:max_edits] if max_edits is not None else requests

    datasets = edit_data.values() if isinstance(edit_data, Mapping) else [edit_data]
    requests: list[EditRequest] = []
    for dataset in datasets:
        if not hasattr(dataset, "to_edit_requests"):
            raise TypeError(
                "Edit datasets must implement `to_edit_requests(limit=...)`."
            )

        remaining = None if max_edits is None else max_edits - len(requests)
        if remaining is not None and remaining <= 0:
            break

        dataset_requests = dataset.to_edit_requests(limit=remaining)
        requests.extend(dataset_requests)

    if not requests:
        raise ValueError("No edit requests were built from `data.edit`.")
    return requests


def execute_edit_requests(
    trainer: Any, requests: Sequence[EditRequest], edit_type: str
) -> dict[str, Any]:
    """Execute edit requests using single, batch, or sequential semantics."""
    if not requests:
        raise ValueError("At least one edit request is required.")

    edit_type = edit_type.lower()
    method_name = trainer.__class__.__name__

    if edit_type == "single":
        request = requests[0]
        result = trainer.edit(request)
        _record_history(trainer, [request])
        return {
            "method": method_name,
            "edit_type": edit_type,
            "requested_count": 1,
            "history_size": len(getattr(trainer, "edit_history", [])),
            "result": result,
        }

    if edit_type == "batch":
        batch = list(requests)
        result = trainer.edit(batch)
        _record_history(trainer, batch)
        return {
            "method": method_name,
            "edit_type": edit_type,
            "requested_count": len(batch),
            "history_size": len(getattr(trainer, "edit_history", [])),
            "result": result,
        }

    if edit_type == "sequential":
        step_results = []
        for request in requests:
            step_result = trainer.edit(request)
            step_results.append(step_result)
            _record_history(trainer, [request])

        return {
            "method": method_name,
            "edit_type": edit_type,
            "requested_count": len(requests),
            "history_size": len(getattr(trainer, "edit_history", [])),
            "step_results": step_results,
            "success": all(_is_success(result) for result in step_results),
        }

    raise ValueError(f"Unsupported edit_type: {edit_type}")


def save_edit_artifacts(
    output_dir: str | Path,
    requests: Sequence[EditRequest],
    summary: dict[str, Any],
) -> None:
    """Persist the executed requests and summary for later inspection."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    preview_path = output_path / "edit_requests_preview.json"
    summary_path = output_path / "edit_results.json"

    preview_payload = [asdict(request) for request in requests]
    summary_payload = _to_jsonable(summary)

    with preview_path.open("w", encoding="utf-8") as f:
        json.dump(preview_payload, f, ensure_ascii=False, indent=2)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    logger.info("Saved edit artifacts to %s", output_path)


def _record_history(trainer: Any, requests: Sequence[EditRequest]) -> None:
    if getattr(trainer, "preserve_memory", False) and hasattr(trainer, "edit_history"):
        trainer.edit_history.extend(requests)


def _is_success(result: Any) -> bool:
    if isinstance(result, Mapping):
        return bool(result.get("success", True))
    return True


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        d = asdict(value)
        return {k: _to_jsonable(v) for k, v in d.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    try:
        from PIL.Image import Image as PILImage
        if isinstance(value, PILImage):
            return f"<PIL.Image mode={value.mode} size={value.size}>"
    except ImportError:
        pass
    return value
