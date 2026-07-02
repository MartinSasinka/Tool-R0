#!/usr/bin/env python3
"""GRPO rewards: Tool-R0 tags + IBM execution (eval-aligned)."""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore

from curricullum.data.exec_trajectory import execute_trajectory, get_ibm_registry  # noqa: E402
from curricullum.train.rewards_nestful import extract_completion_text, values_match
from nestful_evaluation.run import (
    IBMFunctionRegistry,
    _matches_gold,
    _normalize_call,
    parse_tool_calls,
)

_TAG_RE = re.compile(r"<tool_call_answer>(.*?)</tool_call_answer>", re.DOTALL | re.IGNORECASE)

DEFAULT_WEIGHTS = {
    "format": 0.20,
    "call_match": 0.30,
    "exec_match": 0.50,
}

_BATCH: Dict[str, float] = {}
_IBM: Optional[IBMFunctionRegistry] = None


def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))) == 0


def _ibm() -> Optional[IBMFunctionRegistry]:
    global _IBM
    if _IBM is None:
        _IBM = get_ibm_registry()
    return _IBM


def _loads_list(text: str) -> Optional[List[Dict[str, Any]]]:
    if not text:
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if isinstance(obj, list):
        return obj
    return None


def _parse_predicted_call(text: str) -> Optional[Dict[str, Any]]:
    calls, _ = parse_tool_calls(text)
    if not calls:
        m = _TAG_RE.search(text or "")
        if m:
            try:
                obj = json.loads(m.group(1).strip())
            except Exception:
                return None
            if isinstance(obj, list) and obj:
                return _normalize_call(obj[0])
            if isinstance(obj, dict):
                return _normalize_call(obj)
        return None
    return _normalize_call(calls[0])


def _call_soft_match(pred: Dict[str, Any], gold: Dict[str, Any]) -> float:
    if not pred or not gold:
        return 0.0
    score = 0.0
    if (pred.get("name") or "").strip() == (gold.get("name") or "").strip():
        score += 0.35
    pa = pred.get("arguments") or {}
    ga = gold.get("arguments") or {}
    if set(pa.keys()) == set(ga.keys()):
        score += 0.15
    if pa and ga:
        hits = sum(1 for k in ga if k in pa and values_match(pa[k], ga[k]))
        score += 0.5 * (hits / max(1, len(ga)))
    return min(1.0, score)


def _format_score(text: str) -> float:
    if not text or "<tool_call_answer>" not in text.lower():
        return 0.0
    score = 0.4
    if _parse_predicted_call(text):
        score += 0.6
    return score


def compute_toolr0_reward(
    completion: str,
    gold_calls: List[Dict[str, Any]],
    *,
    expected_result: Any,
    prefix_calls: Optional[List[Dict[str, Any]]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    w = weights or DEFAULT_WEIGHTS
    pred_call = _parse_predicted_call(completion)
    fmt = _format_score(completion)
    if not gold_calls:
        return 0.0
    gold_call = _normalize_call(gold_calls[0]) or gold_calls[0]
    call_m = _call_soft_match(pred_call or {}, gold_call) if pred_call else 0.0

    exec_m = 0.0
    ibm = _ibm()
    if ibm and pred_call:
        chain = list(prefix_calls or []) + [pred_call]
        final_value, _, err = execute_trajectory(chain, ibm_registry=ibm)
        if not err and expected_result is not None:
            exec_m = 1.0 if _matches_gold(final_value, expected_result) else 0.0
        elif not err and len(chain) == 1:
            exec_m = call_m  # fallback when no expected_result
    elif not ibm and pred_call and expected_result is not None:
        # No IBM at reward time: reward call structure only (train still runs).
        exec_m = call_m

    total = w["format"] * fmt + w["call_match"] * call_m + w["exec_match"] * exec_m
    return float(max(0.0, min(1.0, total)))


def build_toolr0_reward_func(weights: Optional[Dict[str, float]] = None):
    w = weights or DEFAULT_WEIGHTS

    def toolr0_reward_func(
        prompts,
        completions,
        gold_output,
        gold_answer,
        num_calls,
        turn_idx=None,
        expected_result=None,
        prefix_calls=None,
        **kwargs,
    ) -> List[float]:
        if not isinstance(gold_output, list):
            gold_output = [gold_output] if gold_output is not None else []
        if turn_idx is not None and not isinstance(turn_idx, list):
            turn_idx = [turn_idx] * len(completions)
        if expected_result is not None and not isinstance(expected_result, list):
            expected_result = [expected_result] * len(completions)
        if prefix_calls is not None and not isinstance(prefix_calls, list):
            prefix_calls = [prefix_calls] * len(completions)

        rewards: List[float] = []
        for i, completion in enumerate(completions):
            comp = extract_completion_text(completion)
            g_raw = gold_output[i] if i < len(gold_output) else "[]"
            g_calls = _loads_list(g_raw) if isinstance(g_raw, str) else g_raw
            if not isinstance(g_calls, list):
                g_calls = []

            exp = None
            if expected_result is not None and i < len(expected_result):
                raw_exp = expected_result[i]
                if isinstance(raw_exp, str) and raw_exp:
                    try:
                        exp = json.loads(raw_exp)
                    except Exception:
                        exp = raw_exp
                else:
                    exp = raw_exp

            prefix: List[Dict[str, Any]] = []
            if prefix_calls is not None and i < len(prefix_calls):
                p = prefix_calls[i]
                if isinstance(p, str) and p:
                    prefix = _loads_list(p) or []
                elif isinstance(p, list):
                    prefix = p

            rewards.append(
                compute_toolr0_reward(
                    comp,
                    g_calls,
                    expected_result=exp,
                    prefix_calls=prefix,
                    weights=w,
                )
            )
        return rewards

    return toolr0_reward_func
