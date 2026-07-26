"""Shared IO, hashing, config and resume helpers (stdlib + yaml only)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

MODULE_ROOT = Path(__file__).resolve().parents[2]   # targeted_tool_data_factory/
EXPERIMENTS_ROOT = MODULE_ROOT.parent               # experiments/


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def short_hash(obj: Any, n: int = 12) -> str:
    return sha256_obj(obj)[:n]


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML config with a single-level `extends` chain."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ext = cfg.pop("extends", None)
    if ext:
        base = load_config((path.parent / ext).resolve())
        cfg = _deep_merge(base, cfg)
    return cfg


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_target_path(target_cfg_value: str) -> Path:
    """Target config data paths are relative to experiments/ sibling layout."""
    p = Path(target_cfg_value)
    if p.is_absolute():
        return p
    return (MODULE_ROOT / target_cfg_value).resolve()


def normalize_query(q: str) -> str:
    q = q.lower()
    q = re.sub(r"-?\d+(?:\.\d+)?", "#", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def char_ngrams(s: str, n: int = 3) -> set:
    s = re.sub(r"\s+", " ", s.lower())
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


REF_RE = re.compile(r"^\$var_?\d+(\.[A-Za-z0-9_]+)?\$$")


def is_reference(value: Any) -> bool:
    return isinstance(value, str) and bool(REF_RE.match(value.strip()))


def is_numeric_string(value: Any) -> bool:
    return (isinstance(value, str) and not is_reference(value)
            and bool(re.fullmatch(r"-?\d+(\.\d+)?", value.strip())))


def arg_type_of(value: Any) -> str:
    if is_reference(value):
        return "reference"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "numeric_string" if is_numeric_string(value) else "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


class StepGuard:
    """Resume/overwrite protection for a pipeline step output directory."""

    def __init__(self, out_dir: Path, step: str, *, resume: bool, overwrite: bool):
        self.marker = out_dir / f"_{step}.DONE.json"
        self.out_dir = out_dir
        self.step = step
        self.resume = resume
        self.overwrite = overwrite

    def done(self) -> bool:
        return self.marker.is_file()

    def should_skip(self) -> bool:
        if self.done() and self.resume and not self.overwrite:
            print(f"[{self.step}] resume: already done -> {self.marker}")
            return True
        if self.done() and not self.overwrite and not self.resume:
            raise SystemExit(
                f"[{self.step}] output exists ({self.marker}). Use --resume or --overwrite.")
        return False

    def mark(self, payload: Optional[Dict[str, Any]] = None) -> None:
        write_json(self.marker, {"step": self.step, **(payload or {})})


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)
