"""
Training-context budget helpers for NESTFUL curriculum data.

Aligned with Tool-R0 GRPO on DGX (Qwen3-4B LoRA):
  prompt_length ~ 2048 tokens
  max_completion_length ~ 4096 tokens (fits ~28 GB/GPU with grad ckpt)

Uses char//3 token heuristic (same order of magnitude as nestful_evaluation/run.py).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

# Rough Tool-R0 / NESTFUL system prompt + chat template overhead
SYSTEM_PROMPT_OVERHEAD_CHARS = 1400

# Defaults matching DGX fast curriculum profile
DEFAULT_TARGET_PROMPT_TOKENS = 2048
DEFAULT_TARGET_MAX_COMPLETION_TOKENS = 4096
DEFAULT_MAX_INPUT_CHARS = 512
DEFAULT_TOOL_MENU_MIN = 4
DEFAULT_TOOL_MENU_MAX = 6
DEFAULT_MAX_TOOL_DESC_CHARS = 120
DEFAULT_MAX_PARAM_DESC_CHARS = 80


def compact_tool(
    tool: Dict[str, Any],
    max_desc: int = DEFAULT_MAX_TOOL_DESC_CHARS,
    max_param_desc: int = DEFAULT_MAX_PARAM_DESC_CHARS,
) -> Dict[str, Any]:
    name = tool.get("name", "")
    desc = str(tool.get("description") or "")[:max_desc]
    out: Dict[str, Any] = {"name": name, "description": desc}

    params = tool.get("parameters") or {}
    compact_params: Dict[str, Any] = {}
    if isinstance(params, dict):
        if "properties" in params:
            props = {}
            for k, v in (params.get("properties") or {}).items():
                if isinstance(v, dict):
                    props[k] = {
                        "type": v.get("type", "string"),
                        "description": str(v.get("description", ""))[:max_param_desc],
                    }
                else:
                    props[k] = v
            compact_params = props
        else:
            for k, v in params.items():
                if isinstance(v, dict):
                    compact_params[k] = {
                        "type": v.get("type", "string"),
                        "description": str(v.get("description", ""))[:max_param_desc],
                    }
                elif isinstance(v, str):
                    compact_params[k] = v
                else:
                    compact_params[k] = v
    out["parameters"] = compact_params

    out_params = tool.get("output_parameters") or tool.get("output_parameter")
    if isinstance(out_params, dict) and out_params:
        compact_out: Dict[str, Any] = {}
        for k, v in out_params.items():
            if isinstance(v, dict):
                compact_out[k] = {
                    "type": v.get("type", "string"),
                    "description": str(v.get("description", ""))[:max_param_desc],
                }
            else:
                compact_out[k] = v
        out["output_parameters"] = compact_out

    return out


def compact_tools_list(
    tools: List[Dict[str, Any]],
    max_desc: int = DEFAULT_MAX_TOOL_DESC_CHARS,
    max_param_desc: int = DEFAULT_MAX_PARAM_DESC_CHARS,
) -> List[Dict[str, Any]]:
    return [compact_tool(t, max_desc, max_param_desc) for t in tools if isinstance(t, dict)]


def trim_tool_menu(
    tools: List[Dict[str, Any]],
    calls: List[Dict[str, Any]],
    catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    rng=None,
    min_menu: int = DEFAULT_TOOL_MENU_MIN,
    max_menu: int = DEFAULT_TOOL_MENU_MAX,
    max_desc: int = DEFAULT_MAX_TOOL_DESC_CHARS,
) -> List[Dict[str, Any]]:
    """Keep tools used in calls; add a few distractors; cap menu size."""
    used_names: Set[str] = set()
    for c in calls:
        n = c.get("name")
        if isinstance(n, str):
            used_names.add(n)

    by_name = {t.get("name"): t for t in tools if isinstance(t, dict) and isinstance(t.get("name"), str)}
    if catalog:
        for n in used_names:
            if n not in by_name and n in catalog:
                by_name[n] = catalog[n]

    kept: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for n in sorted(used_names):
        if n in by_name and n not in seen:
            kept.append(by_name[n])
            seen.add(n)

    if catalog and rng is not None and len(kept) < min_menu:
        pool = [n for n in catalog.keys() if n not in seen]
        rng.shuffle(pool)
        for n in pool:
            if len(kept) >= min_menu:
                break
            kept.append(catalog[n])
            seen.add(n)

    if len(kept) > max_menu:
        required = [t for t in kept if t.get("name") in used_names]
        extras = [t for t in kept if t.get("name") not in used_names]
        kept = required + extras[: max(0, max_menu - len(required))]

    return compact_tools_list(kept[:max_menu], max_desc=max_desc)


def build_training_user_content(input_text: str, tools: List[Dict[str, Any]]) -> str:
    tools_json = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    return f"User request:\n{input_text}\n\nAvailable tools (JSON):\n{tools_json}"


def estimate_prompt_tokens(
    input_text: str,
    tools: List[Dict[str, Any]],
    system_overhead_chars: int = SYSTEM_PROMPT_OVERHEAD_CHARS,
) -> int:
    user = build_training_user_content(input_text, tools)
    return (len(user) + system_overhead_chars) // 3


def estimate_completion_tokens(
    calls: List[Dict[str, Any]],
    thinking_overhead_per_call: int = 120,
) -> int:
    body = json.dumps(calls, ensure_ascii=False, separators=(",", ":"))
    return len(body) // 3 + thinking_overhead_per_call * len(calls) + 80


def estimate_training_context(
    input_text: str,
    tools: List[Dict[str, Any]],
    calls: List[Dict[str, Any]],
) -> Dict[str, int]:
    pt = estimate_prompt_tokens(input_text, tools)
    ct = estimate_completion_tokens(calls)
    return {
        "prompt_tokens_est": pt,
        "completion_tokens_est": ct,
        "total_tokens_est": pt + ct,
        "input_chars": len(input_text),
        "tools_json_chars": len(json.dumps(tools, ensure_ascii=False, separators=(",", ":"))),
        "output_json_chars": len(json.dumps(calls, ensure_ascii=False, separators=(",", ":"))),
        "tool_count": len(tools),
    }


def check_context_budget(
    input_text: str,
    tools: List[Dict[str, Any]],
    calls: List[Dict[str, Any]],
    target_prompt_tokens: int = DEFAULT_TARGET_PROMPT_TOKENS,
    target_max_completion_tokens: int = DEFAULT_TARGET_MAX_COMPLETION_TOKENS,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    max_tool_menu: int = DEFAULT_TOOL_MENU_MAX,
) -> Tuple[bool, str, Dict[str, int]]:
    est = estimate_training_context(input_text, tools, calls)
    if len(input_text) > max_input_chars:
        return False, "input_too_long", est
    if est["tool_count"] > max_tool_menu:
        return False, "tool_menu_too_large", est
    if est["prompt_tokens_est"] > target_prompt_tokens:
        return False, "prompt_tokens_over_budget", est
    if est["completion_tokens_est"] > target_max_completion_tokens:
        return False, "completion_tokens_over_budget", est
    return True, "ok", est
