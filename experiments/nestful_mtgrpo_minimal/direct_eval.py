"""Direct-prompting (single-shot) evaluation, NESTFUL Table-1 style.

In the Direct paradigm the model receives the query + tools (+ a few ICL
examples) and must emit the ENTIRE sequence of function calls in one response,
using labels ($var1, $var2, ...) and variable references ($var1.result$) exactly
like the NESTFUL gold. This contrasts with our multi-turn ReAct rollout where the
model sees real observations and tends to paste concrete values (which cannot
match gold's nested references). Direct is what NESTFUL Table 1 reports.

This module only builds prompts, runs single-shot generation, and extracts the
full call sequence. Scoring is delegated to `nestful_official_score` so both
paradigms are measured with the identical official scorer.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from parser import parse_tool_calls_all

_HERE = os.path.dirname(os.path.abspath(__file__))
_ICL_PATH = os.path.join(_HERE, "data", "NESTFUL-main", "src", "icl_examples.json")

DIRECT_SYSTEM_PROMPT = (
    "You are a function-calling planner. Given a user query and a list of "
    "available tools, produce the COMPLETE ordered sequence of function calls "
    "required to answer the query.\n"
    "Rules:\n"
    "- Output ONLY one <tool_call_answer>[ ... ]</tool_call_answer> tag holding a "
    "JSON array of calls. No prose, no explanation, no final answer.\n"
    "- Each call is an object: {\"name\": <tool>, \"arguments\": {...}, "
    "\"label\": \"$varN\"} where N is the 1-based position of the call.\n"
    "- Use the EXACT argument names from the chosen tool's schema (e.g. arg_0, "
    "arg_1 for numeric tools, or named parameters like \"string\"/\"values\").\n"
    "- When an argument is the output of an earlier call, reference it as "
    "$varN.result$ (or the tool's declared output field, e.g. $varN.output_0$) "
    "instead of a concrete value.\n"
    "- Emit only the calls needed; do not compute the numeric result yourself."
)


def load_icl_examples(n: int = 1) -> List[Dict[str, Any]]:
    if n <= 0 or not os.path.exists(_ICL_PATH):
        return []
    with open(_ICL_PATH, "r", encoding="utf-8") as fh:
        examples = json.load(fh)
    return examples[:n]


def _format_tools(tools: List[Dict[str, Any]]) -> str:
    return json.dumps(tools, ensure_ascii=False, indent=2)


def _format_icl(examples: List[Dict[str, Any]]) -> str:
    blocks = []
    for ex in examples:
        answer = json.dumps(ex["output"], ensure_ascii=False)
        blocks.append(
            f"Query: {ex['input']}\n"
            f"Available tools (JSON):\n{_format_tools(ex['tools'])}\n"
            f"<tool_call_answer>{answer}</tool_call_answer>"
        )
    return "\n\n".join(blocks)


def build_direct_messages(
    task: Dict[str, Any], icl_examples: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, str]]:
    user_parts = []
    if icl_examples:
        user_parts.append("Here are some examples:\n" + _format_icl(icl_examples))
    user_parts.append(
        f"Now solve this one.\nQuery: {task['question']}\n"
        f"Available tools (JSON):\n{_format_tools(task['tools'])}"
    )
    return [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def run_direct_eval(
    tasks: List[Dict[str, Any]],
    generate_fn: Callable[[List[Dict[str, str]], int], Dict[str, Any]],
    *,
    num_icl: int = 1,
    max_new_tokens: int = 1024,
    progress: bool = True,
) -> Dict[str, Any]:
    """Run single-shot Direct prompting over tasks.

    `generate_fn(messages, max_new_tokens) -> {"text": str, ...}` is the same
    callable used by the ReAct rollout backend.

    Returns {"predicted": {task_id: [calls]}, "raw_text": {task_id: str}}.
    """
    icl = load_icl_examples(num_icl)
    predicted: Dict[str, List[Dict[str, Any]]] = {}
    raw_text: Dict[str, str] = {}

    iterator = tasks
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(tasks, desc="[direct_eval]", unit="task")
        except ImportError:
            pass

    for task in iterator:
        messages = build_direct_messages(task, icl)
        out = generate_fn(messages, max_new_tokens)
        text = out.get("text", "") if isinstance(out, dict) else str(out)
        calls = parse_tool_calls_all(text)
        predicted[task["task_id"]] = calls
        raw_text[task["task_id"]] = text

    return {"predicted": predicted, "raw_text": raw_text}
