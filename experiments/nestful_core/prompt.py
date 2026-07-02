"""Unified, versioned prompt (v2) for train and eval.

Problem this fixes
------------------
Legacy training used ``SYSTEM_PROMPT`` only, while eval used
``SYSTEM_PROMPT + _EVAL_HARDENING`` (plus a lenient parser and one extra turn).
That train/eval mismatch means the policy was optimized under different
conditions than it was scored.

v2 decomposes the prompt into shared blocks so train and eval carry IDENTICAL
format + stop rules and differ only in a thin hardening tail:

    SYSTEM_PROMPT_BASE + REACT_TOOL_FORMAT_RULES + REACT_STOP_RULES + {TRAIN|EVAL}_HARDENING

The legacy ``SYSTEM_PROMPT`` / ``_EVAL_HARDENING`` in
``nestful_mtgrpo_minimal/prompt.py`` are left untouched for reproducibility;
``build_messages_v2`` is opt-in (a v2 run installs it via ``install_v2_prompt``).
"""
from __future__ import annotations

from typing import Any, Dict, List

from . import ensure_paths

ensure_paths()

# Re-export the canonical, unchanged helpers (observation truncation, tool
# response formatting, user content) so there is one implementation.
from prompt import (  # noqa: E402,F401
    build_user_content,
    format_tool_response,
    get_observation_limits,
    set_observation_limits,
)
import prompt as _legacy_prompt  # noqa: E402

# ---------------------------------------------------------------------------
#  Versioned prompt blocks (shared by train and eval)
# ---------------------------------------------------------------------------
TRAIN_PROMPT_VERSION = "v2.0-train"
EVAL_PROMPT_VERSION = "v2.0-eval"

SYSTEM_PROMPT_BASE = (
    "You are a tool-calling assistant. Solve problems step by step using the "
    "provided tools.\n"
)

REACT_TOOL_FORMAT_RULES = (
    "Tool-call format rules:\n"
    "- ALWAYS use the provided tools. NEVER solve any part of the problem "
    "mentally, and never answer from your own arithmetic — every value in the "
    "answer must come from a tool result you actually observed.\n"
    "- On every turn, use <think>...</think> to reason about the latest "
    "observation and which tool call to make next. Then emit exactly one tool call.\n"
    "- Every response must end with <tool_call_answer>[...]</tool_call_answer>. "
    "The closing tag MUST be spelled exactly </tool_call_answer>. Never reply "
    "with only <think> or plain text, and never stop before the closing tag.\n"
    "- Emit EXACTLY ONE non-empty tool call per turn (never zero, never several), "
    "except the final empty terminal output.\n"
    "- Use the EXACT argument names from the chosen tool's schema (arg_0, arg_1, "
    "… or the named parameters the schema lists). Never invent, drop, or rename "
    "arguments.\n"
    "- Label each call in order: \"label\": \"$var1\" for the first call, "
    "\"$var2\" for the second, and so on.\n"
    "- Reference a previous call's output by its label and output field, NOT the "
    "concrete number you saw: use $var1.result$ or $var1.output_0$. Only "
    "reference outputs that an EARLIER call actually produced.\n"
)

REACT_STOP_RULES = (
    "Stop / continuation rules:\n"
    "- NEVER emit the empty finish <tool_call_answer>[]</tool_call_answer> on the "
    "first turn, and never before you have received at least one real "
    "<tool_response>. The first turn MUST contain a real tool call.\n"
    "- After an intermediate observation, do NOT finish — continue with the next "
    "tool call until the task is fully solved.\n"
    "- Finish with <tool_call_answer>[]</tool_call_answer> ONLY when the most "
    "recent observation already contains enough information to be the final answer.\n"
)

_EXAMPLE = (
    "\nExample — What is (4 × 3) ÷ 2?\n"
    "[Turn 1 — no observations yet]\n"
    "<think>I need to first multiply 4 by 3.</think>\n"
    '<tool_call_answer>[{"name": "multiply", "arguments": {"arg_0": 4, "arg_1": 3}, "label": "$var1"}]</tool_call_answer>\n'
    "[Turn 2 — after receiving multiply result]\n"
    "<think>multiply returned 12. Now I divide 12 by 2.</think>\n"
    '<tool_call_answer>[{"name": "divide", "arguments": {"arg_0": "$var1.result$", "arg_1": 2}, "label": "$var2"}]</tool_call_answer>\n'
    "[Turn 3 — after receiving divide result (final answer)]\n"
    "<think>divide returned 6. That is the final answer.</think>\n"
    "<tool_call_answer>[]</tool_call_answer>\n"
)

# Thin hardening tails. Both only REINFORCE the shared rules above; eval adds the
# LaTeX/$ caution because the eval parser is lenient and prose answers collide
# with $varN$ references. Train keeps it minimal to stay close to deployment.
TRAIN_HARDENING = (
    "\nReminder: output ONLY the <tool_call_answer> tag content; keep reasoning to "
    "at most one short sentence before the call.\n"
)

EVAL_HARDENING = (
    "\n\nOUTPUT FORMAT (must follow exactly):\n"
    "- Reply with EXACTLY ONE tool call wrapped as "
    "<tool_call_answer>[{\"name\": ..., \"arguments\": {...}}]</tool_call_answer>.\n"
    "- Output ONLY that tag. Do NOT restate the answer in prose or bullet points.\n"
    "- Do NOT use LaTeX or $$...$$ / \\( \\) math. The $ character is reserved for "
    "variable references such as $var1.result$.\n"
    "- Keep reasoning to at most one short sentence, then emit the tool call.\n"
    "- If the latest tool result is already the final answer, reply with "
    "<tool_call_answer>[]</tool_call_answer> and nothing else.\n"
)


def build_system_prompt(role: str = "eval") -> str:
    core = SYSTEM_PROMPT_BASE + REACT_TOOL_FORMAT_RULES + REACT_STOP_RULES + _EXAMPLE
    tail = TRAIN_HARDENING if role == "train" else EVAL_HARDENING
    return core + tail


def prompt_versions() -> Dict[str, str]:
    return {
        "train_prompt_version": TRAIN_PROMPT_VERSION,
        "eval_prompt_version": EVAL_PROMPT_VERSION,
    }


def build_messages_v2(
    task: Dict[str, Any],
    history: List[Dict[str, str]] | None = None,
    role: str = "eval",
) -> List[Dict[str, str]]:
    """Drop-in replacement for ``prompt.build_messages`` using the v2 unified prompt.

    ``role`` is "train" or "eval"; both share BASE+FORMAT+STOP and differ only in
    the hardening tail. ``history`` is the model's own turns + real observations.
    """
    messages = [
        {"role": "system", "content": build_system_prompt(role)},
        {"role": "user", "content": build_user_content(task)},
    ]
    if history:
        messages.extend(history)
    return messages


def install_v2_prompt() -> None:
    """Monkeypatch the canonical ``prompt.build_messages`` to the v2 builder.

    Called by a v2 run BEFORE rollout/eval so train and eval share the unified
    prompt. The legacy ``eval_hardening`` boolean is mapped to role:
    ``eval_hardening=True`` -> role="eval", else role="train".
    """
    def _patched(task, history=None, eval_hardening=False):
        return build_messages_v2(task, history, role=("eval" if eval_hardening else "train"))

    _legacy_prompt.build_messages = _patched
