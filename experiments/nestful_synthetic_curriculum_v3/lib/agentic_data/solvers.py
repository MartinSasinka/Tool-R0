"""Weak / strong solver prompting and DETERMINISTIC execution-based scoring.

The weak solver models the target training model (single attempt, small
budget, no scaffolding). The strong solver is the same cheap model with more
inference compute: multiple attempts + explicit planning scaffold, best score
kept (Autodata: weak and strong can be the same LLM in different modes).

Scores are deterministic (spec §6):
  1.0      executable win / solution-equivalent final answer
  0.5-0.8  correct prefix of the gold trace (partial credit by depth)
  0.0-0.4  under-call, wrong tool, wrong args, parse error
The LLM judge never changes these numbers.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .exec_bridge import execute_predicted_calls as _execute_predicted_via_executor

WEAK_MAX_TOKENS = 700
STRONG_MAX_TOKENS = 1400
STRONG_ATTEMPTS = 3
_NUM_TOL = 0.011  # gold answers are rounded to 2 decimals

# Solver-mode config (spec 7A). Defaults reproduce the historical behavior so
# cached runs stay replayable:
#   WEAK_SOLVER_MODE:   minimal (default)  = single attempt, 700 tokens, no
#                                            planning scaffold (current);
#                       handicapped        = 400 tokens, continuation-pressure
#                                            hint removed (weaker weak solver);
#   STRONG_SOLVER_MODE: scaffolded (default) = best-of-3 + planning scaffold;
#                       plain                = single attempt, no scaffold.
WEAK_SOLVER_MODES = ("minimal", "handicapped")
STRONG_SOLVER_MODES = ("scaffolded", "plain")


def weak_solver_mode() -> str:
    mode = os.environ.get("WEAK_SOLVER_MODE", "minimal")
    if mode not in WEAK_SOLVER_MODES:
        raise ValueError(f"WEAK_SOLVER_MODE={mode!r} not in {WEAK_SOLVER_MODES}")
    return mode


def strong_solver_mode() -> str:
    mode = os.environ.get("STRONG_SOLVER_MODE", "scaffolded")
    if mode not in STRONG_SOLVER_MODES:
        raise ValueError(f"STRONG_SOLVER_MODE={mode!r} not in {STRONG_SOLVER_MODES}")
    return mode


def solver_params(strong: bool) -> Dict[str, Any]:
    """Attempts / token budget / temperature for the configured solver mode."""
    if strong:
        mode = strong_solver_mode()
        return {"mode": mode,
                "attempts": STRONG_ATTEMPTS if mode == "scaffolded" else 1,
                "max_tokens": STRONG_MAX_TOKENS,
                "temperature": 0.7}
    mode = weak_solver_mode()
    return {"mode": mode,
            "attempts": 1,
            "max_tokens": 400 if mode == "handicapped" else WEAK_MAX_TOKENS,
            "temperature": 1.0}


def solver_messages(question: str, tools: List[Dict[str, Any]],
                    strong: bool, mode: Optional[str] = None
                    ) -> List[Dict[str, str]]:
    if mode is None:
        mode = strong_solver_mode() if strong else weak_solver_mode()
    tool_lines = []
    for t in tools:
        props = (t.get("parameters") or {}).get("properties") or {}
        params = ", ".join(f"{p}:{v.get('type', '?')}" for p, v in props.items())
        out = ", ".join((t.get("output_parameters") or {}).keys()) or "output"
        tool_lines.append(f"- {t['name']}({params}) -> {out}: {t.get('description', '')}")
    scaffold = ""
    if strong and mode == "scaffolded":
        scaffold = (
            "\nBefore answering, PLAN silently: identify every required step, "
            "which tool computes it, and which argument takes a previous "
            "result. Verify the call count matches the number of steps in the "
            "question. Then output only the JSON."
        )
    # the continuation-pressure hint is a (mild) planning aid — the
    # handicapped weak mode drops it to widen the weak/strong gap (spec 7A)
    pressure = (
        " Perform ALL steps the question asks for — do not stop after the "
        "first call, and do not answer directly without calls."
        if not (not strong and mode == "handicapped") else ""
    )
    system = (
        "You solve tasks by composing function calls. Output STRICT JSON only:\n"
        "{\"calls\": [{\"name\": \"tool\", \"arguments\": {...}, "
        "\"label\": \"$var1\"}], \"final_answer\": <value or null>}\n"
        "To use the result of an earlier call as an argument, write the string "
        "\"$varN.<output_key>$\" (N = 1-based call index)." + pressure + scaffold
    )
    user = f"TOOLS:\n" + "\n".join(tool_lines) + f"\n\nTASK:\n{question}"
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def parse_solver_output(parsed: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(parsed, dict):
        return None
    calls = parsed.get("calls")
    if not isinstance(calls, list):
        return None
    out = []
    for i, c in enumerate(calls):
        if not isinstance(c, dict) or not isinstance(c.get("name"), str):
            return None
        args = c.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(args, dict):
            return None
        out.append({"name": c["name"], "arguments": args,
                    "label": str(c.get("label") or f"$var{i + 1}")})
    return out


def _num_eq(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) <= _NUM_TOL
    except (TypeError, ValueError):
        return a == b


def _execute_predicted(calls: List[Dict[str, Any]]) -> Tuple[List[Any], Optional[str]]:
    """Execute a solver's predicted calls through the REAL synthetic
    executor (``exec_bridge.py``). A wrong argument value executes for real
    and returns the (wrong) observation; it NEVER falls back to the gold
    result — only schema/reference/runtime failures short-circuit here."""
    return _execute_predicted_via_executor(calls)


# Weak-failure taxonomy (diversity accounting). Every score_prediction status
# is one of these when the attempt is not a win:
#   no_tool_call, no_calls_right_answer, parse_error, wrong_tool, wrong_args,
#   invalid_reference, execution_error, under_call, correct_prefix_then_stop,
#   partial_prefix, correct_answer_wrong_trace, wrong_answer
FAILURE_TYPES = (
    "no_tool_call", "no_calls_right_answer", "parse_error", "wrong_tool",
    "wrong_args", "invalid_reference", "execution_error", "under_call",
    "correct_prefix_then_stop", "partial_prefix",
    "correct_answer_wrong_trace", "wrong_answer",
)


def score_prediction(predicted: Optional[List[Dict[str, Any]]],
                     final_answer: Any,
                     gold_calls: List[Dict[str, Any]],
                     gold_observations: List[Any],
                     gold_answer: Any) -> Dict[str, Any]:
    """Deterministic score for one solver attempt. Returns {score, status, ...}."""
    n_gold = len(gold_calls)
    if predicted is None:
        return {"score": 0.0, "status": "parse_error", "n_calls": 0}
    if not predicted:
        # direct answer without required calls only counts if trivially right —
        # and even then it is capped far below a win (we want tool USE)
        status = "no_calls_right_answer" if _num_eq(final_answer, gold_answer) \
            else "no_tool_call"
        return {"score": 0.2 if status == "no_calls_right_answer" else 0.0,
                "status": status, "n_calls": 0}

    obs, err = _execute_predicted(predicted)
    n_pred = len(predicted)

    # full win: executable to the end AND final value matches gold
    if err is None and obs and _num_eq(obs[-1], gold_answer):
        return {"score": 1.0, "status": "win", "n_calls": n_pred}
    if err is None and final_answer is not None and _num_eq(final_answer, gold_answer):
        return {"score": 1.0, "status": "solution_equivalent", "n_calls": n_pred}

    # partial credit: longest matching prefix vs gold observation sequence
    prefix = 0
    for i in range(min(len(obs), len(gold_observations))):
        if predicted[i]["name"] == gold_calls[i]["name"] \
                and _num_eq(obs[i], gold_observations[i]):
            prefix += 1
        else:
            break
    if prefix > 0 and prefix < n_gold:
        # 0.5 .. 0.8 by prefix depth; refined statuses for diversity accounting
        score = 0.5 + 0.3 * (prefix - 1) / max(1, n_gold - 1)
        if err is None and n_pred == prefix:
            status = "correct_prefix_then_stop"   # clean stop after good prefix
        elif err is None and n_pred < n_gold:
            status = "under_call"                 # under-called, extra divergence
        else:
            status = "partial_prefix"             # diverged / broke mid-trace
        return {"score": round(min(score, 0.8), 3), "status": status,
                "n_calls": n_pred, "prefix": prefix}

    # failures 0.0-0.4
    if final_answer is not None and _num_eq(final_answer, gold_answer):
        # right final answer but no usable trace — distinct category
        return {"score": 0.4, "status": "correct_answer_wrong_trace",
                "n_calls": n_pred}
    if err is not None and err.startswith("wrong_tool"):
        return {"score": 0.1, "status": "wrong_tool", "n_calls": n_pred}
    if err == "wrong_args":
        return {"score": 0.2, "status": "wrong_args", "n_calls": n_pred}
    if err == "invalid_reference":
        return {"score": 0.15, "status": "invalid_reference", "n_calls": n_pred}
    if err == "execution_error":
        return {"score": 0.1, "status": "execution_error", "n_calls": n_pred}
    if n_pred < n_gold:
        return {"score": 0.3, "status": "under_call", "n_calls": n_pred}
    return {"score": 0.4, "status": "wrong_answer", "n_calls": n_pred}


def best_of(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(scores, key=lambda s: s["score"]) if scores else \
        {"score": 0.0, "status": "parse_error", "n_calls": 0}
