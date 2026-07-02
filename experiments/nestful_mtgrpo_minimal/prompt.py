"""Prompt construction for the online multi-turn loop.

This file is a minimal standalone reimplementation inspired by the original
project prompt (nestful_evaluation/run.py). The system prompt text is vendored
locally so the folder has no external dependency.

Multi-turn history layout (NO gold prefix — the assistant turns are the model's
own generations, the user turns are real executor observations):

    system instruction
    user:   task input + tool schemas
    assistant: <tool_call_answer>[{...}]</tool_call_answer>
    user:   <tool_response>{"name": ..., "result": ...}</tool_response>
    assistant: ...
    user:   ...
    ...
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List

# MT-GRPO-aligned system prompt (vendored locally — no import from nestful_evaluation/).
#
# Key design decision vs. the original nestful_evaluation/run.py prompt:
#   ORIGINAL: "Plan the full call chain in <think> before emitting the first call."
#   HERE:     Per-turn reasoning — <think> is encouraged on EVERY turn, after seeing
#             the latest observation. This aligns with the multi-turn RL objective:
#             the model should react to real executor feedback, not pre-commit to a
#             rigid sequence it planned before seeing any results.
#
# Why it matters for MT-GRPO:
#   - Per-turn credit assignment (r_t) works best when the model actually re-reasons
#     based on the observation before choosing the next call.
#   - Full-chain upfront planning biases the model toward offline reasoning and makes
#     it hallucinate intermediate values it hasn't observed yet.
#   - <think> tokens are free w.r.t. reward — they enable reasoning without penalty.
SYSTEM_PROMPT = (
    "You are a tool-calling assistant. Solve problems step by step using the provided tools.\n"
    "Rules:\n"
    "- ALWAYS use the provided tools. NEVER solve any part of the problem mentally, and "
    "never answer from your own arithmetic — every value in the answer must come from a "
    "tool result you actually observed.\n"
    "- On every turn, use <think>...</think> to reason about what the latest observation "
    "tells you and which tool call to make next. Then emit exactly one tool call.\n"
    "- Every response must end with <tool_call_answer>[...]</tool_call_answer>. "
    "Never reply with only <think> or plain text, and never stop before the closing tag.\n"
    "- Emit EXACTLY ONE non-empty tool call per turn (never zero, never several). The only "
    "response with no call is the final empty finish below.\n"
    "- NEVER emit the empty finish <tool_call_answer>[]</tool_call_answer> on the first turn, "
    "and never before you have received at least one real <tool_response>. The first turn "
    "MUST contain a real tool call.\n"
    "- Finish with <tool_call_answer>[]</tool_call_answer> ONLY when the most recent "
    "observation already IS the final answer. If a tool returned an intermediate result, do "
    "NOT finish — continue with the next tool call until the task is fully solved.\n"
    "- Use the EXACT argument names from the chosen tool's schema (arg_0, arg_1, ... for "
    "tools that define them, or the named parameters the schema lists). Never invent, drop, "
    "or rename arguments.\n"
    "- Label each call in order: \"label\": \"$var1\" for the first call, \"$var2\" for the "
    "second, and so on.\n"
    "- Reference a previous call's output by its label and output field, NOT the concrete "
    "number you saw: use $var1.result$ (tools whose output is `result`) or $var1.output_0$ "
    "(tools whose output parameter is `output_0`).\n"
    "\n"
    "Example — What is (4 × 3) ÷ 2?\n"
    "[Turn 1 — no observations yet]\n"
    "<think>I need to first multiply 4 by 3.</think>\n"
    '<tool_call_answer>[{"name": "multiply", "arguments": {"arg_0": 4, "arg_1": 3}, "label": "$var1"}]</tool_call_answer>\n'
    "\n"
    "[Turn 2 — after receiving multiply result]\n"
    "<think>multiply returned 12. Now I divide 12 by 2.</think>\n"
    '<tool_call_answer>[{"name": "divide", "arguments": {"arg_0": "$var1.result$", "arg_1": 2}, "label": "$var2"}]</tool_call_answer>\n'
    "\n"
    "[Turn 3 — after receiving divide result (final answer)]\n"
    "<think>divide returned 6. That is the final answer.</think>\n"
    "<tool_call_answer>[]</tool_call_answer>"
)


def build_user_content(task: Dict[str, Any]) -> str:
    tools_json = json.dumps(task["tools"], indent=2, ensure_ascii=False)
    return (
        f"User request:\n{task['question']}\n\n"
        f"Available tools (JSON):\n{tools_json}"
    )


def _int_digit_len(n: int) -> int:
    if n == 0:
        return 1
    n = abs(n)
    d = 0
    while n:
        n //= 10
        d += 1
    return d


# ---------------------------------------------------------------------------
#  Observation truncation limits
# ---------------------------------------------------------------------------
# A tool can return an arbitrarily large object (e.g. a 50k-element list, a long
# string, a big numpy array). Serialised verbatim into <tool_response> and fed
# back on the next turn, this blows the prompt past the vLLM max_model_len and
# CRASHES the run ("decoder prompt longer than maximum model length"). These
# hard caps bound how much of an observation is ever placed into the prompt.
#
# Tunable from config via `set_observation_limits(config)` (generation.observation_limits).
_DEFAULT_OBS_LIMITS: Dict[str, int] = {
    "max_str_chars": 2000,     # any single string is truncated to this many chars
    "max_items": 200,          # any single list/dict is truncated to this many items
    "max_total_chars": 6000,   # whole <tool_response> payload hard-capped to this
}

_OBS_LIMITS: Dict[str, int] = dict(_DEFAULT_OBS_LIMITS)


def set_observation_limits(config: Dict[str, Any] | None) -> Dict[str, int]:
    """Load observation truncation limits from config['generation']['observation_limits'].

    Missing keys keep defaults. A value <= 0 disables that particular cap.
    Process-global (mirrors the reward-weight pattern); call once at startup.
    """
    global _OBS_LIMITS
    _OBS_LIMITS = dict(_DEFAULT_OBS_LIMITS)
    block = (((config or {}).get("generation", {}) or {}).get("observation_limits", {}) or {})
    for k in _DEFAULT_OBS_LIMITS:
        if k in block and block[k] is not None:
            try:
                _OBS_LIMITS[k] = int(block[k])
            except (TypeError, ValueError):
                pass
    return dict(_OBS_LIMITS)


def get_observation_limits() -> Dict[str, int]:
    return dict(_OBS_LIMITS)


def _truncate_str(s: str, limit: int) -> str:
    if limit and limit > 0 and len(s) > limit:
        return s[:limit] + f"…[truncated {len(s) - limit} chars]"
    return s


def _sanitize_for_json(value: Any) -> Any:
    """Make nested structures JSON-safe and length-bounded.

    Handles Python 3.11+ huge-int str limits, non-finite floats, exotic types,
    AND truncates oversized strings / lists / dicts so a runaway tool output can
    never explode the next-turn prompt.
    """
    max_str = _OBS_LIMITS.get("max_str_chars", 0)
    max_items = _OBS_LIMITS.get("max_items", 0)

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        try:
            str(value)
            return value
        except ValueError:
            return f"<int:{_int_digit_len(value)}_digits>"
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for i, (k, v) in enumerate(value.items()):
            if max_items and max_items > 0 and i >= max_items:
                out["__truncated__"] = f"[truncated {len(value) - max_items} more items]"
                break
            out[str(k)] = _sanitize_for_json(v)
        return out
    if isinstance(value, (list, tuple)):
        items = list(value)
        if max_items and max_items > 0 and len(items) > max_items:
            kept = [_sanitize_for_json(v) for v in items[:max_items]]
            kept.append(f"…[truncated {len(items) - max_items} more items]")
            return kept
        return [_sanitize_for_json(v) for v in items]
    if value is None:
        return value
    if isinstance(value, str):
        return _truncate_str(value, max_str)
    # Exotic type (numpy array, custom object, bytes, ...): stringify, then cap.
    return _truncate_str(str(value), max_str)


def format_tool_response(call: Dict[str, Any], result: Any) -> str:
    payload = {"name": call.get("name", ""), "result": _sanitize_for_json(result)}
    body = json.dumps(payload, ensure_ascii=False)
    # Final hard cap on the whole serialised payload — a defence-in-depth guard
    # for deeply nested structures whose per-element caps still sum to a lot.
    max_total = _OBS_LIMITS.get("max_total_chars", 0)
    if max_total and max_total > 0 and len(body) > max_total:
        safe_name = json.dumps(str(call.get("name", "")), ensure_ascii=False)
        marker = f"[truncated to {max_total} chars; was {len(body)}]"
        truncated_result = json.dumps(
            body[:max_total] + marker, ensure_ascii=False
        )
        body = f'{{"name": {safe_name}, "result": {truncated_result}}}'
    return f"<tool_response>{body}</tool_response>"


# Eval-only format reminder. NOT used during training (training keeps the exact
# SYSTEM_PROMPT the policy was optimised against). Appended only when
# eval_hardening=True so baseline and checkpoint are compared under the same,
# parser-friendly instructions. It only REINFORCES the existing contract:
#   - exactly one tool call per turn, correctly closed tag
#   - no prose/LaTeX answer (the $ sign collides with $varN$ references)
_EVAL_HARDENING = (
    "\n\n"
    "OUTPUT FORMAT (must follow exactly):\n"
    "- Reply with EXACTLY ONE tool call wrapped as "
    "<tool_call_answer>[{\"name\": ..., \"arguments\": {...}}]</tool_call_answer>.\n"
    "- The closing tag MUST be spelled exactly </tool_call_answer>.\n"
    "- Output ONLY that tag. Do NOT restate the answer in prose or bullet points, "
    "and do NOT stop before the tag is fully written and closed.\n"
    "- Do NOT use LaTeX or $$...$$ / \\( \\) math. The $ character is reserved "
    "for variable references such as $var1.result$.\n"
    "- Keep reasoning to at most one short sentence, then emit the tool call.\n"
    "- Use the EXACT argument names from the chosen tool's schema: arg_0, arg_1 "
    "for numeric tools that define them, or the named parameters (e.g. \"string\", "
    "\"values\", \"d\") when the schema uses names. Never invent or rename arguments.\n"
    "- Give every call a label: add \"label\": \"$var1\" to the first call, "
    "\"$var2\" to the second, and so on.\n"
    "- When an argument is the OUTPUT of a previous call, pass it as a variable "
    "reference (NOT the concrete number you observed): use the producing call's "
    "label and output field, e.g. $var1.result$ for numeric tools or "
    "$var1.output_0$ for tools whose output parameter is output_0.\n"
    "- If the latest tool result is already the final answer, reply with "
    "<tool_call_answer>[]</tool_call_answer> and nothing else."
)


def build_messages(
    task: Dict[str, Any],
    history: List[Dict[str, str]] | None = None,
    eval_hardening: bool = False,
) -> List[Dict[str, str]]:
    """Build the chat messages for the next generation step.

    `history` is the alternating assistant/user observation turns produced so
    far in THIS episode (the model's own outputs, never gold).

    `eval_hardening` (eval only) appends a strict output-format reminder to the
    system prompt to cut format-related parse failures. Training never sets it,
    so the trained policy's input distribution is unchanged.
    """
    system_content = SYSTEM_PROMPT + (_EVAL_HARDENING if eval_hardening else "")
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": build_user_content(task)},
    ]
    if history:
        messages.extend(history)
    return messages
