"""Shared I/O, config, and path helpers for trajectory analysis."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
VIZ_ROOT = REPO_ROOT / "vizualisation"

FEATURE_COLUMNS = [
    "format_score",
    "call_count_score",
    "tool_name_score",
    "label_score",
    "argument_key_score",
    "argument_value_score",
    "reference_score",
    "dependency_depth_score",
    "final_answer_score",
    "valid_json",
    "exact_trajectory_match",
    "final_answer_exact_match",
    "dependency_depth_pred",
    "dependency_depth_gold",
    "num_calls_pred",
    "num_calls_gold",
    "invalid_reference_count",
    "trajectory_edit_distance",
]

COMPONENT_LABELS = {
    "format_score": "JSON format",
    "call_count_score": "call count",
    "tool_name_score": "tools",
    "label_score": "labels",
    "argument_key_score": "argument keys",
    "argument_value_score": "argument values",
    "reference_score": "references",
    "dependency_depth_score": "dependency depth",
    "final_answer_score": "final answer",
}


def log(prefix: str, msg: str) -> None:
    print(f"[{prefix}] {msg}", flush=True)


def resolve_path(path: str | Path, *, base: Optional[Path] = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    root = base or REPO_ROOT
    return (root / p).resolve()


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = resolve_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["_config_path"] = str(path)
    return cfg


def run_dir_from_config(cfg: Dict[str, Any]) -> Path:
    out = cfg.get("output_dir") or f"vizualisation/runs/{cfg.get('run_name', 'run')}"
    return resolve_path(out)


def ensure_run_dir(cfg: Dict[str, Any]) -> Path:
    run_dir = run_dir_from_config(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "reports").mkdir(exist_ok=True)
    return run_dir


def save_config_copy(cfg: Dict[str, Any], run_dir: Path) -> None:
    copy = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(run_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(copy, fh, indent=2, ensure_ascii=False)


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                log("io", f"WARNING: skip malformed JSONL line {line_no} in {path}: {exc}")
                continue
            if isinstance(row, dict):
                yield row


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def load_jsonl_list(path: Path) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def checkpoints_order(cfg: Dict[str, Any]) -> List[str]:
    order = cfg.get("checkpoints_order")
    if order:
        return list(order)
    preds = cfg.get("input_predictions") or {}
    return list(preds.keys())


def all_prediction_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for key, rel in (cfg.get("input_predictions") or {}).items():
        if not rel:
            continue
        p = resolve_path(rel)
        if not p.is_file():
            raise FileNotFoundError(f"Required prediction file missing for '{key}': {p}")
        paths[key] = p
    optional = cfg.get("optional_predictions") or {}
    for key, rel in optional.items():
        if rel is None:
            continue
        p = resolve_path(rel)
        if p.is_file():
            paths[key] = p
        else:
            log("io", f"WARNING: optional prediction missing for '{key}': {p}")
    return paths


def add_repo_to_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
