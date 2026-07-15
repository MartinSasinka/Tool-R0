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

# Curriculum / provenance metadata preserved through normalization (audit Bug 7:
# stage-aware rewards need these; the old normalize_task silently dropped them).
_METADATA_FIELDS = (
    "stage",
    "terminal_stage",
    "motif_type",
    "prefix_of_motif",
    "target_full_motif",
    "source_failure_cluster",
    "trajectory_id",
    "tool_families",
    # v5 synthetic curriculum: gold per-call observations (used by graded
    # rewards under executor.mode=synthetic) + registry provenance so the
    # trainer can verify it executes the same registry the generator used.
    "observations",
    "registry_version",
    "registry_hash",
)

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

    task = {
        "task_id": str(task_id),
        "question": str(question),
        "tools": tools,
        "gold_calls": gold_calls,
        "gold_answer": _coerce_gold_answer(answer_raw),
        "num_calls": len(gold_calls),
    }
    # Preserve stage/motif metadata for stage-aware rewards (audit Bug 7).
    for key in _METADATA_FIELDS:
        if key in row and row[key] is not None:
            task[key] = row[key]
    return task


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
    replay_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """Mixed curriculum-replay loader.

    Builds a single training pool from several per-stage JSONL files. Two ways
    to specify the mix (mutually exclusive):

      * ``replay_ratio=r`` (RECOMMENDED for curriculum replay): the CURRENT
        stage (last file) gets weight ``1 - r``; the previous stages SHARE ``r``
        equally. E.g. ``replay_ratio=0.20`` with [stage1, stage2] gives
        stage1=0.20 / stage2=0.80 — NOT the 50/50 mix the old scalar-padding
        bug produced (audit Bug 6).
      * ``weights=[w1..wN]``: explicit per-stage weights, normalized. A SCALAR
        weight list for a multi-file mix is ambiguous and now REJECTED.

    Returns a dict:
        {
          "tasks":  List[task],              # shuffled, weight-proportional mix
          "per_stage": [{"file","stage_index","available","sampled","weight"}, ...],
          "weights": List[float],            # normalized weights actually used
          "intended_mix": List[float],
          "effective_mix": List[float],      # from actual sampled counts
          "seed": int,
        }

    Sampling: each stage contributes `round(weight_i / sum_w * N_total)` samples
    where N_total = sum of available rows (so the mixed pool size ~ matches the
    union). Stages are sampled WITHOUT replacement when the target <= available,
    otherwise WITH replacement (oversampling a small stage).

    Hard-fails when the effective (sampled) mix deviates from the intended mix
    by more than 1 percentage point on any stage.
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
    if replay_ratio is not None:
        if weights:
            raise ValueError(
                "load_tasks_mixed: pass EITHER weights OR replay_ratio, not both")
        r = float(replay_ratio)
        if not (0.0 <= r < 1.0):
            raise ValueError(f"replay_ratio must be in [0, 1); got {r}")
        if n_stages == 1:
            norm_w = [1.0]
        else:
            prev_share = r / (n_stages - 1)
            norm_w = [prev_share] * (n_stages - 1) + [1.0 - r]
    else:
        if weights is None or len(weights) == 0:
            weights = [1.0] * n_stages
        if len(weights) == 1 and n_stages > 1:
            # The old code padded a scalar to [w, w] and normalized to a 50/50
            # mix — silently misinterpreting "replay 20%" as "replay 50%".
            raise ValueError(
                f"load_tasks_mixed: got a SINGLE weight {weights[0]} for "
                f"{n_stages} stage files — ambiguous. Use replay_ratio="
                f"{weights[0]} for 'previous stages total={weights[0]:.0%}, "
                f"current stage={1 - float(weights[0]):.0%}', or pass one "
                f"weight per stage file.")
        if len(weights) != n_stages:
            raise ValueError(
                f"load_tasks_mixed: {len(weights)} weights for {n_stages} stage "
                f"files — pass exactly one weight per file (or replay_ratio).")
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

    # ── Verify effective mix ≈ intended mix (audit Bug 6 hard gate) ──────────
    total_sampled = sum(ps["sampled"] for ps in per_stage) or 1
    effective_mix = [ps["sampled"] / total_sampled for ps in per_stage]
    print("[data] effective_stage_mix:", flush=True)
    for ps, eff in zip(per_stage, effective_mix):
        print(f"[data]   stage{ps['stage_index']}: intended={ps['weight']:.4f} "
              f"effective={eff:.4f} ({ps['sampled']}/{total_sampled}) :: {ps['file']}",
              flush=True)
    for ps, eff, w in zip(per_stage, effective_mix, norm_w):
        if abs(eff - w) > 0.01 and ps["available"] > 0:
            raise ValueError(
                f"load_tasks_mixed: effective mix for stage{ps['stage_index']} "
                f"({eff:.4f}) deviates from intended ({w:.4f}) by more than 1% "
                f"— refusing to train on an unintended data mix.")

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
        "intended_mix": list(norm_w),
        "effective_mix": effective_mix,
        "seed": seed,
    }
