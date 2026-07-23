"""Discover and classify invalid annotation rows for retry."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from weak_audit.agreement import parse_json_text, validate_raw_rows
from weak_audit.io_utils import read_jsonl, write_json
from weak_audit.schema import validate_annotation


def _classify_error(raw: dict) -> str:
    if raw.get("error"):
        return "provider_or_api_error"
    text = raw.get("response") or ""
    if not text.strip():
        return "empty_response"
    errors = raw.get("errors") or []
    err_txt = " ".join(errors).lower()
    if "invalid responsible_reward_component" in err_txt or "invalid root_cause" in err_txt:
        return "invalid_enum"
    if "evidence must be short" in err_txt:
        return "evidence_too_long"
    if "task_id mismatch" in err_txt or "missing task_id" in err_txt:
        return "missing_or_wrong_field"
    try:
        parse_json_text(text)
    except json.JSONDecodeError:
        if not text.rstrip().endswith("}"):
            return "truncation"
        return "invalid_json"
    except Exception:
        return "invalid_json"
    if errors:
        return "schema_validation"
    return "other"


def _raw_index(out_dir: Path) -> Dict[Tuple[str, str], dict]:
    idx: Dict[Tuple[str, str], dict] = {}
    for label in ("A", "B"):
        p = out_dir / f"pass_{label.lower()}_annotations_raw.jsonl"
        if not p.is_file():
            continue
        for row in read_jsonl(p):
            if row.get("task_id"):
                idx[(row["task_id"], label)] = row
    return idx


def _valid_pairs(out_dir: Path) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for label in ("A", "B"):
        p = out_dir / f"pass_{label.lower()}_annotations.jsonl"
        for row in read_jsonl(p):
            if row.get("task_id"):
                pairs.add((row["task_id"], label))
    return pairs


def _input_exists(out_dir: Path, task_id: str, pass_label: str) -> bool:
    p = out_dir / f"pass_{pass_label.lower()}_inputs.jsonl"
    for row in read_jsonl(p):
        if row.get("task_id") == task_id:
            return True
    return False


def discover_invalid(out_dir: Path) -> Tuple[List[dict], dict]:
    invalid_rows = read_jsonl(out_dir / "invalid_annotations.jsonl")
    valid_pairs = _valid_pairs(out_dir)
    raw_idx = _raw_index(out_dir)

    entries: List[dict] = []
    for raw in invalid_rows:
        tid = raw.get("task_id")
        pl = raw.get("pass")
        if not tid or pl not in ("A", "B"):
            continue
        key = (tid, pl)
        if key in valid_pairs:
            raise RuntimeError(f"valid annotation also in invalid list: {key}")
        src = raw_idx.get(key, raw)
        category = _classify_error(raw)
        entries.append({
            "task_id": tid,
            "pass_label": pl,
            "request_hash": src.get("request_hash"),
            "provider": src.get("provider"),
            "response_model": src.get("response_model"),
            "prompt_tokens": src.get("prompt_tokens"),
            "completion_tokens": src.get("completion_tokens"),
            "reasoning_tokens": src.get("reasoning_tokens"),
            "max_output_tokens_estimated": None,
            "validation_status": raw.get("validation_status"),
            "validation_errors": raw.get("errors"),
            "failure_category": category,
            "input_available": _input_exists(out_dir, tid, pl),
            "response_preview": (src.get("response") or "")[:240],
        })

    pairs = sorted({(e["task_id"], e["pass_label"]) for e in entries})
    manifest = {
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "n_invalid_pairs": len(pairs),
        "n_pass_a": sum(1 for _, p in pairs if p == "A"),
        "n_pass_b": sum(1 for _, p in pairs if p == "B"),
        "pairs": [{"task_id": t, "pass_label": p} for t, p in pairs],
        "failure_categories": dict(Counter(e["failure_category"] for e in entries)),
        "no_valid_in_retry": True,
        "pass_b_mapping_unchanged": (out_dir / "pass_b_mapping.json").is_file(),
    }
    return entries, manifest


def write_invalid_discovery(out_dir: Path) -> Tuple[List[dict], dict]:
    entries, manifest = discover_invalid(out_dir)
    write_json(out_dir / "invalid_retry_manifest.json", manifest)
    lines = [
        "# Invalid annotation retry discovery",
        "",
        f"**Invalid pairs:** {manifest['n_invalid_pairs']} "
        f"(A={manifest['n_pass_a']}, B={manifest['n_pass_b']})",
        "",
        "## Failure categories",
        "",
    ]
    for cat, n in sorted((manifest.get("failure_categories") or {}).items()):
        lines.append(f"- {cat}: {n}")
    lines += ["", "## Pairs", ""]
    for p in manifest.get("pairs") or []:
        lines.append(f"- {p['pass_label']} `{p['task_id']}`")
    lines += [
        "",
        "## Checks",
        "",
        f"- Valid annotations excluded from retry: {manifest.get('no_valid_in_retry')}",
        f"- Pass B mapping file present: {manifest.get('pass_b_mapping_unchanged')}",
        "- Case packets / pass inputs unchanged (retry reads existing inputs only)",
    ]
    (out_dir / "INVALID_RETRY_DISCOVERY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return entries, manifest
