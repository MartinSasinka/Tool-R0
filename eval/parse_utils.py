"""
Robust tool-call parser for model outputs.

Handles many output formats that different models produce:
  - <tool_call_answer>...</tool_call_answer>  (Tool-R0 training format)
  - <tool_call>...</tool_call>                (Qwen native)
  - ```json ... ```                           (markdown fenced blocks)
  - Raw JSON arrays/objects
  - {"name": ..., "arguments": ...} pattern anywhere in text
  - Action/Action_Input style
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

# Tag patterns — order matters (most specific first)
_TAG_PATTERNS = [
    re.compile(r"<tool_call_answer>(.*?)</tool_call_answer>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<function_call>(.*?)</function_call>", re.DOTALL | re.IGNORECASE),
]

# Unclosed tags (model hit max_tokens before closing)
_UNCLOSED_TAG_PATTERNS = [
    re.compile(r"<tool_call_answer>\s*(.+)", re.DOTALL | re.IGNORECASE),
    re.compile(r"<tool_call>\s*(.+)", re.DOTALL | re.IGNORECASE),
]

_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

_TOOL_CALL_OBJ = re.compile(
    r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}',
    re.DOTALL,
)


def _loads_relaxed(text: str) -> Optional[Any]:
    """Parse JSON with relaxed fallbacks (single quotes, Python literals)."""
    if not text or not text.strip():
        return None

    s = text.strip()
    # Remove markdown fences if wrapping the whole string
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()

    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError):
        pass

    s2 = s.replace("None", "null").replace("True", "true").replace("False", "false")
    try:
        return json.loads(s2)
    except (json.JSONDecodeError, TypeError):
        pass

    return None


def _normalize_call(obj: Any) -> Optional[Dict[str, Any]]:
    """Normalize a single parsed object into {"name": str, "arguments": dict}."""
    if isinstance(obj, list):
        if not obj:
            return None
        obj = obj[0]

    if not isinstance(obj, dict):
        return None

    # OpenAI function wrapper: {"function": {"name": ..., "arguments": ...}}
    if "function" in obj and isinstance(obj["function"], dict):
        fn = obj["function"]
        name = fn.get("name")
        args = fn.get("arguments")
        if isinstance(args, str):
            args = _loads_relaxed(args)
        if isinstance(name, str) and isinstance(args, dict):
            return {"name": name, "arguments": args}

    # Standard: {"name": ..., "arguments": ...}
    name = obj.get("name") or obj.get("tool_name") or obj.get("tool")
    if not isinstance(name, str) or not name.strip():
        # Action/Action_Input style
        action = obj.get("Action") or obj.get("action")
        if isinstance(action, str) and action.strip():
            action_input = obj.get("Action_Input") or obj.get("action_input") or obj.get("parameters") or {}
            if isinstance(action_input, str):
                action_input = _loads_relaxed(action_input) or {}
            return {"name": action.strip(), "arguments": action_input if isinstance(action_input, dict) else {}}
        return None

    args = obj.get("arguments") or obj.get("parameters") or obj.get("args")
    if isinstance(args, str):
        args = _loads_relaxed(args)
    if isinstance(args, dict):
        return {"name": name.strip(), "arguments": args}

    # Treat remaining keys as arguments
    flat = {k: v for k, v in obj.items() if k not in ("name", "tool_name", "tool", "type", "id")}
    return {"name": name.strip(), "arguments": flat}


def _parse_call_list(obj: Any) -> Optional[List[Dict[str, Any]]]:
    """Parse a JSON value into a list of normalized tool calls."""
    if isinstance(obj, list):
        out = []
        for item in obj:
            norm = _normalize_call(item)
            if norm:
                out.append(norm)
        return out if out else None

    norm = _normalize_call(obj)
    return [norm] if norm else None


def parse_tool_calls(response: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """Extract tool calls from a model response.

    Returns:
        (calls, method) where calls is a list of normalized tool call dicts
        or None if parsing failed, and method describes which strategy worked.
    """
    if not response or not response.strip():
        return None, "empty_output"

    # Strategy 1: XML-style tags (closed)
    for pattern in _TAG_PATTERNS:
        m = pattern.search(response)
        if m:
            inner = m.group(1).strip()
            obj = _loads_relaxed(inner)
            if obj is not None:
                calls = _parse_call_list(obj)
                if calls:
                    return calls, "tag_closed"

    # Strategy 2: Unclosed tags (model hit token limit)
    for pattern in _UNCLOSED_TAG_PATTERNS:
        m = pattern.search(response)
        if m:
            inner = m.group(1).strip()
            # Try to fix truncated JSON
            inner = _try_close_json(inner)
            obj = _loads_relaxed(inner)
            if obj is not None:
                calls = _parse_call_list(obj)
                if calls:
                    return calls, "tag_unclosed"

    # Strategy 3: Markdown fenced JSON blocks
    for m in _FENCED_JSON.finditer(response):
        obj = _loads_relaxed(m.group(1))
        if obj is not None:
            calls = _parse_call_list(obj)
            if calls:
                return calls, "fenced_json"

    # Strategy 4: Find JSON array/object patterns in raw text
    # Look for the last JSON-like block (models often put tool calls at the end)
    json_candidates = _extract_json_candidates(response)
    for candidate in reversed(json_candidates):
        obj = _loads_relaxed(candidate)
        if obj is not None:
            calls = _parse_call_list(obj)
            if calls:
                return calls, "json_in_text"

    # Strategy 5: Regex for individual tool call objects
    matches = _TOOL_CALL_OBJ.findall(response)
    if matches:
        calls = []
        for match_str in matches:
            obj = _loads_relaxed(match_str)
            if obj:
                norm = _normalize_call(obj)
                if norm:
                    calls.append(norm)
        if calls:
            return calls, "regex_objects"

    # Strategy 6: Full response as JSON
    obj = _loads_relaxed(response)
    if obj is not None:
        calls = _parse_call_list(obj)
        if calls:
            return calls, "full_response_json"

    return None, "unparseable"


def _extract_json_candidates(text: str) -> List[str]:
    """Find substring candidates that look like JSON arrays or objects."""
    candidates = []
    depth = 0
    start = -1
    opener = ""

    for i, ch in enumerate(text):
        if ch in ("[", "{") and depth == 0:
            start = i
            opener = ch
            depth = 1
        elif depth > 0:
            if ch == opener:
                depth += 1
            elif (opener == "[" and ch == "]") or (opener == "{" and ch == "}"):
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    start = -1

    return candidates


def _try_close_json(text: str) -> str:
    """Try to close truncated JSON by balancing brackets."""
    open_sq = text.count("[") - text.count("]")
    open_cu = text.count("{") - text.count("}")
    suffix = "}" * max(0, open_cu) + "]" * max(0, open_sq)
    return text + suffix


def count_tool_calls_in_response(response: str) -> int:
    """Heuristic count of tool calls found in model output."""
    calls, _ = parse_tool_calls(response)
    if calls:
        return len(calls)
    return 0
