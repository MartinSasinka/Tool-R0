import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


DEFAULT_TEXT_LIMIT = 4000
DEFAULT_TRACE_SAMPLES_PER_STEP = 2


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def logging_enabled() -> bool:
    return bool(os.environ.get("TOOL_R0_RUN_DIR"))


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def get_run_dir() -> Optional[str]:
    run_dir = os.environ.get("TOOL_R0_RUN_DIR")
    if not run_dir:
        return None
    return _ensure_dir(run_dir)


def get_step_dir(step_name: Optional[str] = None) -> Optional[str]:
    step_dir = os.environ.get("TOOL_R0_STEP_DIR")
    if step_dir:
        return _ensure_dir(step_dir)

    run_dir = get_run_dir()
    if run_dir is None:
        return None

    iteration = os.environ.get("TOOL_R0_ITERATION", "unknown")
    name = step_name or os.environ.get("TOOL_R0_STEP_NAME", "unknown_step")
    return _ensure_dir(os.path.join(run_dir, f"iter{iteration}", name))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate_text(text: Any, limit: Optional[int] = None) -> Any:
    if not isinstance(text, str):
        return text

    max_len = int(os.environ.get("TOOL_R0_TRACE_TEXT_LIMIT", limit or DEFAULT_TEXT_LIMIT))
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [TRUNCATED]"


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [to_serializable(v) for v in value]
    if isinstance(value, set):
        return [to_serializable(v) for v in sorted(value)]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return _truncate_text(value)


def trace_sample_limit() -> int:
    return int(os.environ.get("TOOL_R0_TRACE_SAMPLES_PER_STEP", DEFAULT_TRACE_SAMPLES_PER_STEP))


def append_jsonl(filename: str, payload: Dict[str, Any], step_name: Optional[str] = None) -> None:
    step_dir = get_step_dir(step_name=step_name)
    if step_dir is None:
        return

    record = {
        "timestamp": _now_iso(),
        "run_id": os.environ.get("TOOL_R0_RUN_ID"),
        "iteration": os.environ.get("TOOL_R0_ITERATION"),
        "step_name": step_name or os.environ.get("TOOL_R0_STEP_NAME"),
        **to_serializable(payload),
    }
    out_path = os.path.join(step_dir, filename)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(filename: str, payload: Dict[str, Any], step_name: Optional[str] = None) -> None:
    step_dir = get_step_dir(step_name=step_name)
    if step_dir is None:
        return

    out_path = os.path.join(step_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, ensure_ascii=False, indent=2)

