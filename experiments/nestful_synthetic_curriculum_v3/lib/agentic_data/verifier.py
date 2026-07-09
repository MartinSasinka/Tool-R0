"""Verifier: deterministic execution first, LLM judge second.

The deterministic executor is the ONLY source of gold observations and the
gold answer — an LLM-claimed answer is never trusted (spec §3: "never trust
LLM gold answer without deterministic replay"). The LLM judge only assesses
naturalness / ambiguity / NESTFUL-likeness and can reject, but can NEVER
override a failed execution check.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..nestful_like_generator import TOOLS, execute_call
from .schema import question_leak_errors

JUDGE_MIN_QUALITY = 0.6


def execute_gold_trace(gold_calls: List[Dict[str, Any]]
                       ) -> Tuple[Optional[List[Any]], Optional[str]]:
    """Execute a candidate gold trace. Returns (observations, error)."""
    scope: Dict[str, Any] = {}
    observations: List[Any] = []
    for i, call in enumerate(gold_calls):
        name = call.get("name")
        if name not in TOOLS:
            return None, f"unknown tool '{name}' (call {i + 1})"
        args = call.get("arguments")
        if not isinstance(args, dict):
            return None, f"arguments of call {i + 1} not an object"
        expected = set(TOOLS[name]["params"].keys())
        if set(args.keys()) != expected:
            return None, (f"call {i + 1} ({name}) argument keys {sorted(args)} "
                          f"!= schema {sorted(expected)}")
        try:
            obs = execute_call(name, args, scope)
        except KeyError as exc:
            return None, f"unresolved reference in call {i + 1}: {exc}"
        except Exception as exc:  # noqa: BLE001
            return None, f"execution error in call {i + 1}: {type(exc).__name__}: {exc}"
        label = str(call.get("label", f"$var{i + 1}")).lstrip("$")
        scope[label] = obs
        observations.append(obs)
    return observations, None


def deterministic_verify(cand: Dict[str, Any]) -> Dict[str, Any]:
    """Run all deterministic gates on a normalized candidate.

    Returns {"ok", "reason", "detail", "observations", "gold_answer"}.
    """
    def fail(reason: str, detail: str) -> Dict[str, Any]:
        return {"ok": False, "reason": reason, "detail": detail,
                "observations": None, "gold_answer": None}

    calls = cand.get("gold_calls") or []
    observations, err = execute_gold_trace(calls)
    if err is not None:
        return fail("non_executable_gold_trace", err)
    gold_answer = observations[-1] if observations else None
    if gold_answer is None:
        return fail("null_answer", "executor produced None as final answer")
    if isinstance(gold_answer, str) and "$" in gold_answer:
        return fail("unresolved_var", f"final answer contains '$': {gold_answer!r}")
    # if the challenger claimed an answer, it must not be trusted — but a
    # blatant mismatch flags a confused/ambiguous task
    leaks = question_leak_errors(cand.get("question", ""))
    if leaks:
        return fail("metadata_leakage", "; ".join(leaks))
    # dependency sanity: multi-call tasks must actually chain (>=1 reference)
    if len(calls) >= 2:
        has_ref = any(isinstance(v, str) and v.startswith("$") and v.endswith("$")
                      for c in calls for v in (c.get("arguments") or {}).values())
        if not has_ref:
            return fail("invalid_schema",
                        "multi-call task with no $varN$ dependency between calls")
    return {"ok": True, "reason": None, "detail": None,
            "observations": observations, "gold_answer": gold_answer}


# ---------------------------------------------------------------------------
# LLM judge (secondary; style/ambiguity only)
# ---------------------------------------------------------------------------

def judge_messages(question: str, n_calls: int) -> list:
    system = (
        "You are a strict data-quality judge for tool-use training tasks in "
        "the style of the NESTFUL benchmark. Output STRICT JSON only."
    )
    user = (
        f"QUESTION ({n_calls} tool calls are required to solve it):\n"
        f"{question}\n\n"
        "Judge ONLY the question text (the executable trace is verified "
        "separately). Criteria:\n"
        "- nestful_like: concise (25-60 words), concrete everyday quantities, "
        "single paragraph, multi-step imperative/interrogative phrasing;\n"
        "- unambiguous: every needed input value is stated exactly once and "
        "the order of steps is implied clearly;\n"
        "- natural: reads like a real user request, not a template dump.\n\n"
        "OUTPUT: {\"nestful_like\": true|false, \"ambiguous\": true|false, "
        "\"natural\": true|false, \"quality_score\": 0.0-1.0, "
        "\"issues\": [\"...\"]}"
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def judge_verdict(parsed: Any) -> Dict[str, Any]:
    """Normalize judge output; judge failures never crash the loop."""
    if not isinstance(parsed, dict):
        return {"ok": True, "reason": None, "quality_score": None,
                "detail": "judge unparseable — skipped (deterministic gates rule)"}
    quality = parsed.get("quality_score")
    try:
        quality = float(quality) if quality is not None else None
    except (TypeError, ValueError):
        quality = None
    if parsed.get("ambiguous") is True:
        return {"ok": False, "reason": "ambiguous_question",
                "quality_score": quality,
                "detail": "; ".join(map(str, parsed.get("issues") or []))[:300]}
    if parsed.get("nestful_like") is False or (quality is not None
                                               and quality < JUDGE_MIN_QUALITY):
        return {"ok": False, "reason": "not_nestful_like",
                "quality_score": quality,
                "detail": "; ".join(map(str, parsed.get("issues") or []))[:300]}
    return {"ok": True, "reason": None, "quality_score": quality, "detail": None}
