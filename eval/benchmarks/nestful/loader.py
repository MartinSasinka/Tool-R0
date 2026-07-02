"""
NESTFUL dataset loader.

Downloads the NESTFUL benchmark (ibm-research/nestful) from HuggingFace
and converts it into the standard eval task format.

Reference:
    Basu et al., "NESTFUL: A Benchmark for Evaluating LLMs on Nested
    Sequences of API Calls", EMNLP 2025. https://arxiv.org/abs/2409.03797
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def _convert_nestful_tools(tools_raw: Any) -> List[Dict[str, Any]]:
    """Convert NESTFUL tool specs to Tool-R0 style JSON tool descriptions."""
    if isinstance(tools_raw, str):
        tools_raw = json.loads(tools_raw)

    converted = []
    for t in tools_raw:
        properties = {}
        for param_name, param_spec in t.get("parameters", {}).items():
            properties[param_name] = {
                "type": param_spec.get("type", "string"),
                "description": param_spec.get("description", ""),
            }

        converted.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
            },
        })
    return converted


def _parse_gold_calls(output_raw: Any) -> List[Dict[str, Any]]:
    """Parse NESTFUL gold output into normalized call list."""
    if isinstance(output_raw, str):
        output_raw = json.loads(output_raw)

    calls = []
    for step in output_raw:
        calls.append({
            "name": step["name"],
            "arguments": step.get("arguments", {}),
            "label": step.get("label", ""),
        })
    return calls


def load_tasks(
    max_tasks: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load NESTFUL tasks from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "NESTFUL loader requires `datasets`. Install: pip install datasets"
        )

    print("[nestful] Downloading ibm-research/nestful from HuggingFace...")
    ds = load_dataset("ibm-research/nestful", split="train", cache_dir=cache_dir)

    tasks: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        if max_tasks is not None and idx >= max_tasks:
            break

        tools = _convert_nestful_tools(row["tools"])
        gold_calls = _parse_gold_calls(row["output"])

        gold_answer_raw = row.get("gold_answer")
        try:
            gold_answer = eval(str(gold_answer_raw)) if gold_answer_raw is not None else None
        except Exception:
            gold_answer = gold_answer_raw

        nesting_depth = 0
        for call in gold_calls:
            for v in call["arguments"].values():
                if isinstance(v, str) and v.startswith("$") and v.endswith("$"):
                    nesting_depth += 1
                    break

        tasks.append({
            "task_id": row.get("sample_id", f"nestful_{idx}"),
            "question": row["input"],
            "tools": tools,
            "gold_calls": gold_calls,
            "gold_answer": gold_answer,
            "num_gold_calls": len(gold_calls),
            "nesting_depth": nesting_depth,
        })

    print(f"[nestful] Loaded {len(tasks)} tasks "
          f"(avg {sum(t['num_gold_calls'] for t in tasks) / max(1, len(tasks)):.1f} calls/task)")
    return tasks


def build_user_content(task: Dict[str, Any]) -> str:
    """Build the user prompt for a NESTFUL task."""
    tools_json = json.dumps(task["tools"], indent=2, ensure_ascii=False)
    return (
        f"User request:\n{task['question']}\n\n"
        f"Available tools (JSON):\n{tools_json}"
    )
