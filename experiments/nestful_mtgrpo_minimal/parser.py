"""Strict single-tool-call parser (gate, not a reward).

This file is a minimal standalone reimplementation inspired by the original
project parser (nestful_evaluation/run.py::parse_tool_calls). It deliberately
does NOT accept the many relaxed fallbacks of the original; the contract here is
a hard gate:

  - exactly one <tool_call_answer>...</tool_call_answer> tag
  - the tag contains valid JSON
  - the JSON resolves to exactly one call with a string `name` and dict `arguments`
  - an empty list `[]` is the explicit terminal/stop signal

Anything else fails. There is no format reward; parsing either passes or not.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_TAG_RE = re.compile(
    r"<tool_call_answer>(.*?)</tool_call_answer>", re.DOTALL | re.IGNORECASE
)


@dataclass
class ParseResult:
    ok: bool
    call: Optional[Dict[str, Any]] = None
    is_terminal: bool = False           # model emitted [] to end the episode
    reason: Optional[str] = None
    raw_inner: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def _loads_relaxed(text: str) -> Optional[Any]:
    if not text or not text.strip():
        return None
    s = text.strip()
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
        return None


def _normalize_one(obj: Any) -> Optional[Dict[str, Any]]:
    """Normalize a single call object to {name, arguments, label}."""
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool_name") or obj.get("tool")
    if not isinstance(name, str) or not name.strip():
        return None
    args = obj.get("arguments")
    if isinstance(args, str):
        args = _loads_relaxed(args)
    if not isinstance(args, dict):
        return None
    return {
        "name": name.strip(),
        "arguments": args,
        "label": obj.get("label", ""),
    }


_CLOSE_TAG_RE = re.compile(r"</tool_call[_a-z0-9]*>", re.IGNORECASE)
_OPEN_TAG = "<tool_call_answer>"


def _extract_balanced(s: str, open_ch: str, close_ch: str):
    """Yield every top-level balanced (open_ch ... close_ch) span in `s`,
    skipping brackets that appear inside JSON string literals."""
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield s[start : i + 1]
                start = None


def _find_call_payload(s: str):
    """Best-effort extraction of a tool-call JSON payload (list or object) from
    arbitrary model text. Prefers a balanced [...] / {...} that mentions "name"."""
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        spans = list(_extract_balanced(s, open_ch, close_ch))
        # Prefer a span that actually looks like a call (mentions "name").
        for span in spans:
            if '"name"' in span or "'name'" in span:
                obj = _loads_relaxed(span)
                if obj is not None:
                    return obj
        # Fallback: an empty list span is the terminal signal.
        for span in spans:
            obj = _loads_relaxed(span)
            if isinstance(obj, list) and len(obj) == 0:
                return obj
    return None


def _obj_to_result(obj, raw_inner: str) -> ParseResult | None:
    """Convert a parsed JSON object/list into a ParseResult, or None if it is
    not a usable tool call. Lenient: if a list has several calls, take the first
    valid one (multi-turn executes one call per turn)."""
    if isinstance(obj, list):
        if len(obj) == 0:
            return ParseResult(True, is_terminal=True, raw_inner=raw_inner)
        for item in obj:
            call = _normalize_one(item)
            if call is not None:
                return ParseResult(True, call=call, raw_inner=raw_inner)
        return None
    if isinstance(obj, dict):
        call = _normalize_one(obj)
        if call is not None:
            return ParseResult(True, call=call, raw_inner=raw_inner)
    return None


def parse_tool_call(text: str, lenient: bool = False) -> ParseResult:
    """Parse a single tool call from model text.

    Strict mode (lenient=False, the default — used for TRAINING reward gating):
      exactly one well-formed <tool_call_answer>[...]</tool_call_answer> tag.
      Failure reasons:
        empty_output, no_tag, multiple_tags, invalid_json, not_a_list_or_obj,
        multiple_tool_calls, missing_name_or_arguments, arguments_not_dict

    Lenient mode (lenient=True — used only for EVAL): recovers from mangled
    closing tags (e.g. </tool_call_call>), trailing junk after the JSON, repeated
    tags (takes the last), surrounding prose/LaTeX, and bare JSON arrays emitted
    without any tag. This mirrors NESTFUL's own model-specific lenient scoring and
    avoids penalising calls the model clearly intended to make.
    """
    if not text or not text.strip():
        return ParseResult(False, reason="empty_output")

    tags = _TAG_RE.findall(text)

    if not lenient:
        if not tags:
            return ParseResult(False, reason="no_tag")
        if len(tags) > 1:
            return ParseResult(False, reason="multiple_tags")

        inner = tags[0].strip()
        obj = _loads_relaxed(inner)
        if obj is None:
            return ParseResult(False, reason="invalid_json", raw_inner=inner)

        # Terminal signal: empty list.
        if isinstance(obj, list) and len(obj) == 0:
            return ParseResult(True, is_terminal=True, raw_inner=inner)

        if isinstance(obj, list):
            if len(obj) != 1:
                return ParseResult(
                    False, reason="multiple_tool_calls", raw_inner=inner
                )
            call = _normalize_one(obj[0])
        elif isinstance(obj, dict):
            call = _normalize_one(obj)
        else:
            return ParseResult(False, reason="not_a_list_or_obj", raw_inner=inner)

        if call is None:
            candidate = obj[0] if isinstance(obj, list) else obj
            if isinstance(candidate, dict) and (
                candidate.get("name") or candidate.get("tool_name") or candidate.get("tool")
            ):
                if not isinstance(candidate.get("arguments"), (dict, str)):
                    return ParseResult(
                        False, reason="arguments_not_dict", raw_inner=inner
                    )
            return ParseResult(
                False, reason="missing_name_or_arguments", raw_inner=inner
            )

        return ParseResult(True, call=call, raw_inner=inner)

    # ---- lenient (eval) recovery --------------------------------------
    candidates: List[str] = []

    # 1) Content of the LAST well-formed tag (the model's final decision).
    if tags:
        candidates.append(tags[-1].strip())

    # 2) Region after the last OPEN tag, even if the close tag is missing or
    #    misspelled (</tool_call_call>, </tool_call>, ...). Cut at any close tag.
    last_open = text.rfind(_OPEN_TAG)
    if last_open != -1:
        region = text[last_open + len(_OPEN_TAG) :]
        m = _CLOSE_TAG_RE.search(region)
        if m:
            region = region[: m.start()]
        candidates.append(region.strip())

    # 3) Whole text as a last resort (bare JSON array with no tag at all).
    candidates.append(text)

    for cand in candidates:
        if not cand:
            continue
        # Direct relaxed parse first (clean payloads).
        obj = _loads_relaxed(cand)
        if obj is None:
            # Extract a balanced JSON payload from surrounding prose/LaTeX/junk.
            obj = _find_call_payload(cand)
        if obj is None:
            continue
        res = _obj_to_result(obj, raw_inner=cand[:500])
        if res is not None:
            return res

    if not tags and last_open == -1:
        return ParseResult(False, reason="no_tag")
    return ParseResult(False, reason="invalid_json", raw_inner=text[:500])


def parse_tool_calls_all(text: str) -> list[dict]:
    """Extract the FULL sequence of tool calls from a single-shot (Direct) answer.

    Unlike parse_tool_call (which returns one call for multi-turn ReAct), the
    NESTFUL Direct-prompting paradigm asks the model to emit the entire plan in
    one response, e.g.
        <tool_call_answer>[{"name": ..., "arguments": {...}, "label": "$var1"},
                           {"name": ..., "arguments": {...}, "label": "$var2"}]</tool_call_answer>
    Returns a list of normalized call dicts (possibly empty). Uses the same
    lenient candidate extraction as parse_tool_call(lenient=True).
    """
    if not text or not text.strip():
        return []

    tags = _TAG_RE.findall(text)
    candidates: List[str] = []
    if tags:
        candidates.append(tags[-1].strip())
    last_open = text.rfind(_OPEN_TAG)
    if last_open != -1:
        region = text[last_open + len(_OPEN_TAG):]
        m = _CLOSE_TAG_RE.search(region)
        if m:
            region = region[: m.start()]
        candidates.append(region.strip())
    candidates.append(text)

    for cand in candidates:
        if not cand:
            continue
        obj = _loads_relaxed(cand)
        if obj is None:
            obj = _find_call_payload(cand)
        if obj is None:
            continue
        if isinstance(obj, dict):
            obj = [obj]
        if isinstance(obj, list):
            calls = [c for c in (_normalize_one(it) for it in obj) if c is not None]
            if calls:
                return calls
    return []
