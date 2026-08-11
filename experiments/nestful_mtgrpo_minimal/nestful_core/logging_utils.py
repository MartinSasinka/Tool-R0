"""CSV hygiene + reward-component logging helpers (shared by all v2 outputs).

The audit found several ``experiments/comparison/*.csv`` files were not loadable
by ``pandas.read_csv`` because free-text fields contained unescaped commas /
newlines. ``write_csv`` here uses ``csv.DictWriter`` with full quoting, forbids
commas in header names, and serialises list/dict cell values to compact JSON so
every emitted CSV round-trips through pandas.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Sequence


def _sanitize_header(name: str) -> str:
    # Header names must not contain commas/newlines (they break naive readers).
    return str(name).replace(",", "_").replace("\n", " ").replace("\r", " ").strip()


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return int(value)
    return value


def write_csv(path: str, rows: Sequence[Dict[str, Any]], *, fieldnames: List[str] | None = None) -> str:
    """Write ``rows`` to ``path`` with safe quoting. Returns the path.

    * ``fieldnames`` defaults to the union of keys (first-seen order preserved).
    * Complex cell values become compact JSON strings.
    * Always uses ``QUOTE_MINIMAL`` with proper escaping so commas/newlines in
      values never corrupt the column layout.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fieldnames is None:
        seen: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.append(k)
        fieldnames = seen
    safe_fields = [_sanitize_header(f) for f in fieldnames]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=safe_fields, quoting=csv.QUOTE_MINIMAL,
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({_sanitize_header(k): _cell(v) for k, v in r.items()})
    return path


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def write_json(path: str, obj: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
    return path


# Canonical list of reward-component / rate columns logged per training step and
# per checkpoint (request §15). Kept here so the trainer and the report share one
# schema.
REWARD_COMPONENT_FIELDS: List[str] = [
    "reward_total",
    "tool_final_answer_pass",
    "executable_trajectory",
    "tool_use_completeness",
    "valid_references",
    "small_gold_trace_progress",
    "parse_error_rate",
    "clipped_rate",
    "no_tool_call_rate",
    "too_few_calls_rate",
    "invalid_reference_rate",
    "executor_error_rate",
    "rollout_length_mean",
    "num_successful_calls_mean",
    "strict_parse_rate",
    "lenient_parse_rate",
    "parse_recovery_rate",
    "validation_react_win",
    "validation_direct_win",
    "validation_full_acc",
    "validation_partial",
    "kl",
    "entropy",
    "loss",
    "grad_norm",
    "learning_rate",
]


def aggregate_reward_components(diags: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Mean of execution_aware_v2 component + rate fields over a batch of diags."""
    if not diags:
        return {}
    n = len(diags)

    def _mean(key: str, default: float = 0.0) -> float:
        return sum(float(d.get(key, default) or 0.0) for d in diags) / n

    def _rate(key: str) -> float:
        return sum(1 for d in diags if d.get(key)) / n

    return {
        "reward_total": _mean("reward"),
        "tool_final_answer_pass": _mean("tool_final_answer_pass"),
        "executable_trajectory": _mean("executable_trajectory"),
        "tool_use_completeness": _mean("tool_use_completeness"),
        "valid_references": _mean("valid_references"),
        "small_gold_trace_progress": _mean("small_gold_trace_progress"),
        "parse_error_rate": _rate("parse_error"),
        "clipped_rate": _rate("clipped"),
        "no_tool_call_rate": _rate("no_tool_call"),
        "too_few_calls_rate": _rate("too_few_calls"),
        "invalid_reference_rate": _rate("invalid_reference"),
        "executor_error_rate": _rate("executor_error"),
        "num_successful_calls_mean": _mean("num_successful_calls"),
    }
