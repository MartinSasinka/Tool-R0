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
from typing import Any, Dict, List, Optional, Tuple

from ..nestful_like_generator import TOOLS, execute_call

WEAK_MAX_TOKENS = 700
STRONG_MAX_TOKENS = 1400
STRONG_ATTEMPTS = 3
_NUM_TOL = 0.011  # gold answers are rounded to 2 decimals


def solver_messages(question: str, tools: List[Dict[str, Any]],
                    strong: bool) -> List[Dict[str, str]]:
    tool_lines = []
    for t in tools:
        props = (t.get("parameters") or {}).get("properties") or {}
        params = ", ".join(f"{p}:{v.get('type', '?')}" for p, v in props.items())
        out = ", ".join((t.get("output_parameters") or {}).keys()) or "output"
        tool_lines.append(f"- {t['name']}({params}) -> {out}: {t.get('description', '')}")
    scaffold = ""
    if strong:
        scaffold = (
            "\nBefore answering, PLAN silently: identify every required step, "
            "which tool computes it, and which argument takes a previous "
            "result. Verify the call count matches the number of steps in the "
            "question. Then output only the JSON."
        )
    system = (
        "You solve tasks by composing function calls. Output STRICT JSON only:\n"
        "{\"calls\": [{\"name\": \"tool\", \"arguments\": {...}, "
        "\"label\": \"$var1\"}], \"final_answer\": <value or null>}\n"
        "To use the result of an earlier call as an argument, write the string "
        "\"$varN.<output_key>$\" (N = 1-based call index). Perform ALL steps "
        "the question asks for — do not stop after the first call, and do not "
        "answer directly without calls." + scaffold
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
    scope: Dict[str, Any] = {}
    observations: List[Any] = []
    for i, call in enumerate(calls):
        name = call["name"]
        if name not in TOOLS:
            return observations, f"wrong_tool:{name}"
        expected = set(TOOLS[name]["params"].keys())
        if set(call["arguments"].keys()) != expected:
            return observations, "wrong_args"
        try:
            obs = execute_call(name, call["arguments"], scope)
        except Exception:  # noqa: BLE001
            return observations, "execution_error"
        scope[str(call.get("label", f"$var{i + 1}")).lstrip("$")] = obs
        observations.append(obs)
    return observations, None


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
        # 0.5 .. 0.8 by prefix depth
        score = 0.5 + 0.3 * (prefix - 1) / max(1, n_gold - 1)
        status = "under_call" if n_pred < n_gold and err is None else "partial_prefix"
        return {"score": round(min(score, 0.8), 3), "status": status,
                "n_calls": n_pred, "prefix": prefix}

    # failures 0.0-0.4
    if err is not None and err.startswith("wrong_tool"):
        return {"score": 0.1, "status": "wrong_tool", "n_calls": n_pred}
    if err == "wrong_args":
        return {"score": 0.2, "status": "wrong_args", "n_calls": n_pred}
    if err == "execution_error":
        return {"score": 0.1, "status": "execution_error", "n_calls": n_pred}
    if n_pred < n_gold:
        return {"score": 0.3, "status": "under_call", "n_calls": n_pred}
    return {"score": 0.4, "status": "wrong_answer", "n_calls": n_pred}


def best_of(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(scores, key=lambda s: s["score"]) if scores else \
        {"score": 0.0, "status": "parse_error", "n_calls": 0}
