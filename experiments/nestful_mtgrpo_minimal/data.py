"""Standalone task loader for full-task JSONL datasets.

This file is a minimal standalone reimplementation inspired by the original
project loaders. It works directly on full-task JSONL rows (NOT the per-turn
expanded format) and is tolerant to small field-name differences across the
synthetic and official NESTFUL datasets.

A normalized task is a dict:
    {
      "task_id":    str,
      "question":   str,
      "tools":      list[dict],          # OpenAI-ish function schema
      "gold_calls": list[dict],          # [{"name","arguments","label"}, ...]
      "gold_answer": Any,
      "num_calls":  int,                 # len(gold_calls)
    }
"""
from __future__ import annotations

import ast
import json
import random
from typing import Any, Dict, List, Optional

_INPUT_FIELDS = ("input", "prompt", "query", "question")
_TOOLS_FIELDS = ("tools",)
_OUTPUT_FIELDS = ("gold_output", "gold_outputs", "output", "gold_calls")
_ANSWER_FIELDS = ("gold_answer", "answer", "final_answer")
_ID_FIELDS = ("sample_id", "task_id", "id")
_NCALLS_FIELDS = ("n_calls", "num_tool_calls", "num_calls", "stage")

_EXPECTED_FORMAT = (
    "Each JSONL row must provide:\n"
    "  - an input field   (one of: input / prompt / query / question)\n"
    "  - a tools field     (list or JSON-string of tool schemas)\n"
    "  - a gold output     (one of: output / gold_output / gold_outputs / gold_calls;\n"
    "                       a list or JSON-string of {name, arguments, label} calls)\n"
    "  - a gold answer     (one of: gold_answer / answer / final_answer)\n"
    "Example row:\n"
    '  {"sample_id": "ex-1", "input": "...", '
    '"tools": "[{...}]", "output": "[{\\"name\\":\\"add\\",'
    '\\"arguments\\":{\\"arg_0\\":1,\\"arg_1\\":2}}]", "gold_answer": 3}'
)


def _first(row: Dict[str, Any], fields) -> Any:
    for f in fields:
        if f in row and row[f] is not None:
            return row[f]
    return None


def _coerce_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return value
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return value
    return value


def _normalize_tool_schema(tools_raw: Any) -> List[Dict[str, Any]]:
    tools = _coerce_jsonish(tools_raw)
    if not isinstance(tools, list):
        return []
    out: List[Dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        params = t.get("parameters") or {}
        # Already OpenAI-style?
        if isinstance(params, dict) and "properties" in params:
            out.append(t)
            continue
        properties: Dict[str, Any] = {}
        for pname, pspec in params.items():
            if isinstance(pspec, dict):
                properties[pname] = {
                    "type": pspec.get("type", "string"),
                    "description": pspec.get("description", ""),
                }
        out.append({
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
            },
            # Preserve output schema (used by executor for var resolution hints).
            "output_parameters": t.get("output_parameters", {}),
        })
    return out


def _normalize_calls(output_raw: Any) -> List[Dict[str, Any]]:
    calls = _coerce_jsonish(output_raw)
    if not isinstance(calls, list):
        return []
    out: List[Dict[str, Any]] = []
    for step in calls:
        if not isinstance(step, dict):
            continue
        name = step.get("name") or step.get("tool_name") or step.get("tool")
        if not isinstance(name, str):
            continue
        args = step.get("arguments")
        if isinstance(args, str):
            args = _coerce_jsonish(args)
        if not isinstance(args, dict):
            args = {}
        out.append({
            "name": name.strip(),
            "arguments": args,
            "label": step.get("label", ""),
        })
    return out


def _coerce_gold_answer(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return value
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return value
    return value


def normalize_task(row: Dict[str, Any], idx: int = 0) -> Dict[str, Any]:
    question = _first(row, _INPUT_FIELDS)
    tools_raw = _first(row, _TOOLS_FIELDS)
    output_raw = _first(row, _OUTPUT_FIELDS)
    answer_raw = _first(row, _ANSWER_FIELDS)

    missing = []
    if question is None:
        missing.append("input/prompt/query")
    if tools_raw is None:
        missing.append("tools")
    if output_raw is None:
        missing.append("output/gold_output")
    if missing:
        raise ValueError(
            f"Unrecognized task row (index {idx}); missing fields: "
            f"{', '.join(missing)}.\n\n{_EXPECTED_FORMAT}"
        )

    gold_calls = _normalize_calls(output_raw)
    tools = _normalize_tool_schema(tools_raw)
    task_id = _first(row, _ID_FIELDS) or f"task_{idx}"

    return {
        "task_id": str(task_id),
        "question": str(question),
        "tools": tools,
        "gold_calls": gold_calls,
        "gold_answer": _coerce_gold_answer(answer_raw),
        "num_calls": len(gold_calls),
    }


def load_tasks(
    path: str,
    stage: Optional[int] = None,
    max_tasks: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load + normalize full-task JSONL.

    Args:
        path:      JSONL file path.
        stage:     if set, keep only tasks whose num_calls == stage.
        max_tasks: cap (applied after a deterministic shuffle by `seed`).
        seed:      shuffle seed for reproducible subsampling.
    """
    tasks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task = normalize_task(row, idx)
            if stage is not None and task["num_calls"] != int(stage):
                continue
            tasks.append(task)

    if not tasks:
        raise ValueError(
            f"No tasks loaded from {path}"
            + (f" with num_calls == {stage}" if stage is not None else "")
            + f".\n\n{_EXPECTED_FORMAT}"
        )

    rng = random.Random(seed)
    rng.shuffle(tasks)
    if max_tasks is not None:
        tasks = tasks[: int(max_tasks)]
    return tasks


def _load_all_from_file(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            out.append(normalize_task(json.loads(line), idx))
    return out


def load_tasks_mixed(
    stage_files: List[str],
    weights: Optional[List[float]] = None,
    max_tasks: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Mixed curriculum-replay loader.

    Builds a single training pool from several per-stage JSONL files, sampling
    each stage in proportion to `weights` (uniform if not given). The files are
    expected to already be per-stage (one call-depth each), so NO num_calls
    filtering is applied — every row in each file is used as that stage's pool.

    Returns a dict:
        {
          "tasks":  List[task],              # shuffled, weight-proportional mix
          "per_stage": [{"file","stage_index","available","sampled","weight"}, ...],
          "weights": List[float],            # normalized weights actually used
          "seed": int,
        }

    Sampling: each stage contributes `round(weight_i / sum_w * N_total)` samples
    where N_total = sum of available rows (so the mixed pool size ~ matches the
    union). Stages are sampled WITHOUT replacement when the target <= available,
    otherwise WITH replacement (oversampling a small stage).
    """
    if not stage_files:
        raise ValueError("load_tasks_mixed requires at least one stage file")

    rng = random.Random(seed)
    pools: List[List[Dict[str, Any]]] = []
    available: List[int] = []
    for path in stage_files:
        pool = _load_all_from_file(path)
        pools.append(pool)
        available.append(len(pool))

    n_stages = len(stage_files)
    if weights is None or len(weights) == 0:
        weights = [1.0] * n_stages
    if len(weights) != n_stages:
        # Truncate / pad with the last weight so a short list is tolerated.
        if len(weights) < n_stages:
            weights = list(weights) + [weights[-1]] * (n_stages - len(weights))
        else:
            weights = list(weights[:n_stages])
    weights = [max(0.0, float(w)) for w in weights]
    w_sum = sum(weights) or 1.0
    norm_w = [w / w_sum for w in weights]

    total_available = sum(available) or 0
    mixed: List[Dict[str, Any]] = []
    per_stage: List[Dict[str, Any]] = []
    for i, (path, pool, avail, w) in enumerate(zip(stage_files, pools, available, norm_w)):
        target = int(round(w * total_available))
        if avail == 0 or target <= 0:
            sampled_tasks: List[Dict[str, Any]] = []
        elif target <= avail:
            sampled_tasks = rng.sample(pool, target)
        else:
            sampled_tasks = [pool[rng.randrange(avail)] for _ in range(target)]
        for t in sampled_tasks:
            t["_stage"] = i + 1   # 1-based stage index (provenance for logging)
        mixed.extend(sampled_tasks)
        per_stage.append({
            "file": path,
            "stage_index": i + 1,
            "available": avail,
            "sampled": len(sampled_tasks),
            "weight": round(w, 4),
        })

    rng.shuffle(mixed)
    if max_tasks is not None:
        mixed = mixed[: int(max_tasks)]

    if not mixed:
        raise ValueError(
            "load_tasks_mixed produced an empty pool from: " + ", ".join(stage_files)
        )

    return {
        "tasks": mixed,
        "per_stage": per_stage,
        "weights": norm_w,
        "seed": seed,
    }
