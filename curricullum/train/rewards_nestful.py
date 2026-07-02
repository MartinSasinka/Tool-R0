#!/usr/bin/env python3
"""
rewards_nestful.py

Deterministic NESTFUL-style GRPO rewards for JSON tool-call trajectories.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore

_VAR_REF_RE = re.compile(r"^\$var_(\d+)(?:\.([A-Za-z_][\w]*))?\$$")

DEFAULT_WEIGHTS = {
    "format": 0.15,
    "call_count": 0.10,
    "tool_name": 0.15,
    "labels": 0.10,
    "argument_keys": 0.10,
    "argument_values": 0.15,
    "references": 0.15,
    "final_answer": 0.10,
}

_BATCH_STATS: Dict[str, float] = {}
_LAST_LOGGED_STEP: Optional[int] = None
_LOG_EVERY_STEPS = 1


def is_main_process() -> bool:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank) == 0
    return int(os.environ.get("RANK", "0")) == 0


def reset_batch_stats() -> None:
    _BATCH_STATS.clear()


def _loads_relaxed(text: str) -> Optional[Any]:
    if text is None:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            return None
    return None


def extract_completion_text(completion: Any) -> str:
    """Normalize TRL completion payloads to plain text."""
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
        return str(content) if content is not None else ""
    if isinstance(completion, list):
        if not completion:
            return ""
        if len(completion) == 1:
            return extract_completion_text(completion[0])
        texts = [extract_completion_text(item) for item in completion]
        texts = [t for t in texts if t]
        return texts[-1] if texts else ""
    return str(completion)


def parse_completion(text: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Return (parsed dict or None, truncated_heuristic)."""
    if not text or not str(text).strip():
        return None, True
    raw = str(text).strip()
    truncated = raw.endswith("...") or (
        raw.count("{") > raw.count("}") or raw.count("[") > raw.count("]")
    )
    obj = _loads_relaxed(raw)
    if not isinstance(obj, dict):
        return None, truncated
    return obj, truncated


def extract_predicted_calls(obj: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    output = obj.get("output")
    if isinstance(output, str):
        output = _loads_relaxed(output)
    if not isinstance(output, list):
        return None
    calls = []
    for item in output:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            calls.append(item)
    return calls if calls else None


def extract_predicted_answer(obj: Dict[str, Any]) -> Optional[str]:
    for key in ("answer", "final_answer", "gold_answer"):
        if key in obj and obj[key] is not None:
            val = obj[key]
            if isinstance(val, str):
                return val
            return json.dumps(val, ensure_ascii=False)
    return None


def normalize_string(s: Any) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def values_match(pred: Any, gold: Any, float_tol: float = 1e-5) -> bool:
    if pred == gold:
        return True
    if isinstance(pred, (int, float)) and isinstance(gold, (int, float)):
        return math.isclose(float(pred), float(gold), rel_tol=float_tol, abs_tol=float_tol)
    if isinstance(pred, str) and isinstance(gold, str):
        return normalize_string(pred) == normalize_string(gold)
    if isinstance(pred, (int, float)) and isinstance(gold, str):
        try:
            return values_match(pred, float(gold) if "." in gold else int(gold))
        except Exception:
            return normalize_string(pred) == normalize_string(gold)
    if isinstance(pred, str) and isinstance(gold, (int, float)):
        return values_match(gold, pred)
    return normalize_string(pred) == normalize_string(gold)


def score_format(obj: Optional[Dict[str, Any]], expected_calls: int = 1) -> float:
    if obj is None:
        return 0.0
    pred_calls = extract_predicted_calls(obj)
    pred_answer = extract_predicted_answer(obj)
    if pred_calls is None or len(pred_calls) == 0:
        return 0.0
    score = 0.5
    if pred_answer is not None:
        score += 0.5
    if expected_calls > 0 and len(pred_calls) != expected_calls:
        score = min(score, 0.5)
    return score


def score_call_count(pred_calls: Optional[List[Dict[str, Any]]], expected: int) -> float:
    if pred_calls is None:
        return 0.0
    if len(pred_calls) == expected:
        return 1.0
    if len(pred_calls) > expected:
        return max(0.0, 0.5 - 0.1 * (len(pred_calls) - expected))
    return max(0.0, len(pred_calls) / max(1, expected))


def score_tool_names(
    pred_calls: Optional[List[Dict[str, Any]]],
    gold_calls: List[Dict[str, Any]],
) -> float:
    if pred_calls is None or not gold_calls:
        return 0.0
    n = min(len(pred_calls), len(gold_calls))
    if n == 0:
        return 0.0
    matches = sum(
        1 for i in range(n) if pred_calls[i].get("name") == gold_calls[i].get("name")
    )
    return matches / len(gold_calls)


def score_labels(pred_calls: Optional[List[Dict[str, Any]]], expected: int) -> float:
    if pred_calls is None:
        return 0.0
    if len(pred_calls) != expected:
        return 0.0
    ok = 0
    for i, call in enumerate(pred_calls, start=1):
        label = call.get("label")
        if label == f"$var_{i}":
            ok += 1
    return ok / expected


def score_argument_keys(
    pred_calls: Optional[List[Dict[str, Any]]],
    gold_calls: List[Dict[str, Any]],
) -> float:
    if pred_calls is None or not gold_calls:
        return 0.0
    scores = []
    for i in range(min(len(pred_calls), len(gold_calls))):
        p_args = pred_calls[i].get("arguments") or {}
        g_args = gold_calls[i].get("arguments") or {}
        if not isinstance(p_args, dict) or not isinstance(g_args, dict):
            scores.append(0.0)
            continue
        if not g_args:
            scores.append(1.0 if not p_args else 0.5)
            continue
        pk, gk = set(p_args.keys()), set(g_args.keys())
        scores.append(len(pk & gk) / len(gk))
    if not scores:
        return 0.0
    return sum(scores) / len(gold_calls)


def extract_references_from_value(value: Any) -> List[Tuple[int, Optional[str]]]:
    refs: List[Tuple[int, Optional[str]]] = []
    if isinstance(value, str):
        m = _VAR_REF_RE.match(value.strip())
        if m:
            refs.append((int(m.group(1)), m.group(2)))
    elif isinstance(value, list):
        for item in value:
            refs.extend(extract_references_from_value(item))
    return refs


def score_references(
    pred_calls: Optional[List[Dict[str, Any]]],
    gold_calls: List[Dict[str, Any]],
    tools_by_name: Optional[Dict[str, Dict[str, Any]]] = None,
) -> float:
    del tools_by_name  # schema validation not available at reward time
    if pred_calls is None or not gold_calls:
        return 0.0

    def call_ref_score(call_idx: int, call: Dict[str, Any], gold_call: Dict[str, Any]) -> float:
        args = call.get("arguments") or {}
        g_args = gold_call.get("arguments") or {}
        if not isinstance(args, dict) or not isinstance(g_args, dict):
            return 0.0
        ref_pairs: List[Tuple[str, int, Optional[str]]] = []
        for k, v in args.items():
            if isinstance(v, str) and v.strip().startswith("$var_"):
                if _VAR_REF_RE.match(v.strip()) is None:
                    ref_pairs.append((k, -1, None))
                    continue
            for ref_idx, field in extract_references_from_value(v):
                ref_pairs.append((k, ref_idx, field))
        if not ref_pairs:
            return 1.0
        ok = 0.0
        for k, ref_idx, field in ref_pairs:
            if ref_idx < 1 or ref_idx >= call_idx:
                continue
            gold_v = g_args.get(k)
            gold_refs = extract_references_from_value(gold_v)
            pred_ok = any(r[0] == ref_idx and r[1] == field for r in gold_refs)
            if pred_ok:
                ok += 1.0
            elif field is None:
                ok += 0.5
        return ok / len(ref_pairs)

    scores = []
    for i, (pred, gold) in enumerate(zip(pred_calls, gold_calls), start=1):
        scores.append(call_ref_score(i, pred, gold))
    if len(scores) < len(gold_calls):
        scores.extend([0.0] * (len(gold_calls) - len(scores)))
    return sum(scores) / len(gold_calls) if gold_calls else 0.0


def score_argument_values(
    pred_calls: Optional[List[Dict[str, Any]]],
    gold_calls: List[Dict[str, Any]],
) -> float:
    if pred_calls is None or not gold_calls:
        return 0.0

    def value_score(a: Any, b: Any) -> float:
        refs_a = extract_references_from_value(a) if isinstance(a, str) else []
        refs_b = extract_references_from_value(b) if isinstance(b, str) else []
        if refs_a or refs_b:
            if refs_a == refs_b:
                return 1.0
            return 0.0
        return 1.0 if values_match(a, b) else 0.0

    scores = []
    for i in range(min(len(pred_calls), len(gold_calls))):
        p_args = pred_calls[i].get("arguments") or {}
        g_args = gold_calls[i].get("arguments") or {}
        if not isinstance(p_args, dict) or not isinstance(g_args, dict):
            scores.append(0.0)
            continue
        keys = set(g_args.keys())
        if not keys:
            scores.append(1.0)
            continue
        key_scores = [value_score(p_args.get(k), g_args.get(k)) for k in keys]
        scores.append(sum(key_scores) / len(key_scores))
    if not scores:
        return 0.0
    if len(scores) < len(gold_calls):
        scores.extend([0.0] * (len(gold_calls) - len(scores)))
    return sum(scores) / len(gold_calls)


def score_final_answer(pred_answer: Optional[str], gold_answer: str) -> float:
    if pred_answer is None or not gold_answer:
        return 0.0
    if values_match(pred_answer, gold_answer):
        return 1.0
    return 0.0


def compute_nestful_reward(
    pred_obj: Optional[Dict[str, Any]],
    gold_output: List[Dict[str, Any]],
    gold_answer: str,
    num_calls: int,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    w = weights or DEFAULT_WEIGHTS
    pred_calls = extract_predicted_calls(pred_obj) if pred_obj else None
    pred_answer = extract_predicted_answer(pred_obj) if pred_obj else None

    components = {
        "format": score_format(pred_obj, num_calls),
        "call_count": score_call_count(pred_calls, num_calls),
        "tool_name": score_tool_names(pred_calls, gold_output),
        "labels": score_labels(pred_calls, num_calls),
        "argument_keys": score_argument_keys(pred_calls, gold_output),
        "argument_values": score_argument_values(pred_calls, gold_output),
        "references": score_references(pred_calls, gold_output, {}),
        "final_answer": score_final_answer(pred_answer, gold_answer),
    }
    total = sum(w.get(k, 0.0) * v for k, v in components.items())
    total = float(max(0.0, min(1.0, total)))
    return total, components


def _accumulate_stats(key: str, value: float, count: int = 1) -> None:
    _BATCH_STATS[f"{key}_sum"] = _BATCH_STATS.get(f"{key}_sum", 0.0) + value
    _BATCH_STATS[f"{key}_count"] = _BATCH_STATS.get(f"{key}_count", 0.0) + count


def get_batch_stats_snapshot() -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = _BATCH_STATS.get("reward_count", 0.0)
    if n > 0:
        out["reward_mean"] = _BATCH_STATS.get("reward_sum", 0.0) / n
        out["invalid_json_rate"] = _BATCH_STATS.get("invalid_json_count", 0.0) / n
        out["truncated_rate"] = _BATCH_STATS.get("truncated_count", 0.0) / n
    return out


def _append_reward_debug(payload: Dict[str, float]) -> None:
    if not is_main_process():
        return
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "reward_components.jsonl")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def maybe_log_wandb(step: Optional[int], weights: Dict[str, float]) -> None:
    global _LAST_LOGGED_STEP
    if not is_main_process() or wandb is None or wandb.run is None:
        return
    if step is None:
        return
    if _LAST_LOGGED_STEP == step:
        return
    _LAST_LOGGED_STEP = step

    payload: Dict[str, float] = {"train/global_step": float(step)}
    n = _BATCH_STATS.get("reward_count", 0.0)
    if n <= 0:
        return

    payload["train/reward"] = _BATCH_STATS.get("reward_sum", 0.0) / n
    for comp in weights:
        ck = f"comp_{comp}_sum"
        cc = f"comp_{comp}_count"
        if _BATCH_STATS.get(cc, 0) > 0:
            payload[f"reward/{comp}"] = _BATCH_STATS[ck] / _BATCH_STATS[cc]

    inv = _BATCH_STATS.get("invalid_json_count", 0.0)
    trunc = _BATCH_STATS.get("truncated_count", 0.0)
    clen = _BATCH_STATS.get("completion_len_sum", 0.0)
    payload["metrics/invalid_json_rate"] = inv / n
    payload["metrics/truncated_rate"] = trunc / n
    payload["metrics/avg_completion_length"] = clen / n

    wandb.log(payload, step=step)
    _append_reward_debug(payload)
    reset_batch_stats()


def nestful_reward_func(
    prompts,
    completions,
    gold_output,
    gold_answer,
    num_calls,
    **kwargs,
) -> List[float]:
    del prompts
    weights = kwargs.get("reward_weights") or DEFAULT_WEIGHTS
    rewards: List[float] = []

    if gold_output is None:
        gold_outputs: List[Any] = []
    elif not isinstance(gold_output, list):
        gold_outputs = [gold_output]
    else:
        gold_outputs = gold_output

    if gold_answer is None:
        gold_answers: List[Any] = []
    elif not isinstance(gold_answer, list):
        gold_answers = [gold_answer]
    else:
        gold_answers = gold_answer

    if num_calls is None:
        num_calls_list: List[Any] = []
    elif not isinstance(num_calls, list):
        num_calls_list = [num_calls]
    else:
        num_calls_list = num_calls

    for idx, completion in enumerate(completions):
        try:
            comp_text = extract_completion_text(completion)
            g_out = gold_outputs[idx] if idx < len(gold_outputs) else []
            g_ans = gold_answers[idx] if idx < len(gold_answers) else ""
            n_call = int(num_calls_list[idx]) if idx < len(num_calls_list) else len(g_out)

            if isinstance(g_out, str):
                parsed_g = _loads_relaxed(g_out)
                g_out = parsed_g if isinstance(parsed_g, list) else []
            if not isinstance(g_out, list):
                g_out = []

            pred_obj, truncated = parse_completion(comp_text)
            if pred_obj is None:
                _accumulate_stats("invalid_json", 1.0)
            if truncated:
                _accumulate_stats("truncated", 1.0)
            _accumulate_stats("completion_len", float(len(comp_text or "")))

            total, components = compute_nestful_reward(
                pred_obj, g_out, str(g_ans), n_call, weights=weights
            )
            rewards.append(total)

            _accumulate_stats("reward", total)
            for k, v in components.items():
                _accumulate_stats(f"comp_{k}", v)
        except Exception:
            rewards.append(0.0)
            _accumulate_stats("invalid_json", 1.0)
            _accumulate_stats("reward", 0.0)

    step = kwargs.get("step")
    maybe_log_wandb(step, weights)
    return rewards


def build_curriculum_reward_func(weights: Dict[str, float]):
    """Return a TRL-compatible reward fn with explicit dataset column names."""

    def curriculum_reward_func(
        prompts,
        completions,
        gold_output,
        gold_answer,
        num_calls,
        **kwargs,
    ) -> List[float]:
        return nestful_reward_func(
            prompts,
            completions,
            gold_output,
            gold_answer,
            num_calls,
            reward_weights=weights,
            **kwargs,
        )

    return curriculum_reward_func
