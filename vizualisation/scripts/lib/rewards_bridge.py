"""Import GRPO reward helpers with local fallback."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

_REWARDS_SOURCE = "curricullum.train.rewards_nestful"
_USE_FALLBACK = False

try:
    from curricullum.train.rewards_nestful import (  # type: ignore
        extract_predicted_answer,
        extract_predicted_calls,
        extract_references_from_value,
        parse_completion,
        score_argument_keys,
        score_argument_values,
        score_call_count,
        score_final_answer,
        score_format,
        score_labels,
        score_references,
        score_tool_names,
        values_match,
    )
except Exception:
    _USE_FALLBACK = True

_VAR_REF_RE = re.compile(r"^\$var_(\d+)(?:\.([A-Za-z_][\w]*))?\$$")


def rewards_using_fallback() -> bool:
    return _USE_FALLBACK


def _loads_relaxed(text: str) -> Optional[Any]:
    if text is None:
        return None
    s = str(text).strip()
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


if _USE_FALLBACK:

    def parse_completion(text: str) -> Tuple[Optional[Dict[str, Any]], bool]:
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
        calls = [
            item
            for item in output
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
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

    def score_call_count(pred_calls, expected: int) -> float:
        if pred_calls is None:
            return 0.0
        if len(pred_calls) == expected:
            return 1.0
        if len(pred_calls) > expected:
            return max(0.0, 0.5 - 0.1 * (len(pred_calls) - expected))
        return max(0.0, len(pred_calls) / max(1, expected))

    def score_tool_names(pred_calls, gold_calls) -> float:
        if pred_calls is None or not gold_calls:
            return 0.0
        matches = sum(
            1
            for i in range(min(len(pred_calls), len(gold_calls)))
            if pred_calls[i].get("name") == gold_calls[i].get("name")
        )
        return matches / len(gold_calls)

    def score_labels(pred_calls, expected: int) -> float:
        if pred_calls is None or len(pred_calls) != expected:
            return 0.0
        ok = sum(
            1 for i, call in enumerate(pred_calls, start=1) if call.get("label") == f"$var_{i}"
        )
        return ok / expected

    def score_argument_keys(pred_calls, gold_calls) -> float:
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
        if len(scores) < len(gold_calls):
            scores.extend([0.0] * (len(gold_calls) - len(scores)))
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
        elif isinstance(value, dict):
            for item in value.values():
                refs.extend(extract_references_from_value(item))
        return refs

    def score_references(pred_calls, gold_calls, tools_by_name=None) -> float:
        del tools_by_name
        if pred_calls is None or not gold_calls:
            return 0.0

        def call_ref_score(call_idx, call, gold_call):
            args = call.get("arguments") or {}
            g_args = gold_call.get("arguments") or {}
            if not isinstance(args, dict) or not isinstance(g_args, dict):
                return 0.0
            ref_pairs = []
            for k, gv in g_args.items():
                for ref_idx, field in extract_references_from_value(gv):
                    ref_pairs.append((k, ref_idx, field))
            if not ref_pairs:
                return 1.0
            ok = 0.0
            for k, ref_idx, field in ref_pairs:
                pv = args.get(k)
                pred_refs = extract_references_from_value(pv)
                if any(r[0] == ref_idx and r[1] == field for r in pred_refs):
                    ok += 1.0
                elif field is None:
                    ok += 0.5
            return ok / len(ref_pairs)

        scores = [
            call_ref_score(i, pred, gold)
            for i, (pred, gold) in enumerate(zip(pred_calls, gold_calls), start=1)
        ]
        if len(scores) < len(gold_calls):
            scores.extend([0.0] * (len(gold_calls) - len(scores)))
        return sum(scores) / len(gold_calls) if gold_calls else 0.0

    def score_argument_values(pred_calls, gold_calls) -> float:
        if pred_calls is None or not gold_calls:
            return 0.0

        def value_score(a, b):
            refs_a = extract_references_from_value(a) if isinstance(a, str) else []
            refs_b = extract_references_from_value(b) if isinstance(b, str) else []
            if refs_a or refs_b:
                return 1.0 if refs_a == refs_b else 0.0
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

    def score_final_answer(pred_answer, gold_answer) -> float:
        if pred_answer is None or gold_answer is None or gold_answer == "":
            return 0.0
        return 1.0 if values_match(pred_answer, gold_answer) else 0.0
