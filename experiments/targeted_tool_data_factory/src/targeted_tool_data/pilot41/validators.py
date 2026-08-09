"""Pilot4.1 validators V9–V13."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from .graph_leak import analyze_graph_leak
from .query_render import FORBIDDEN_IMPLICIT_RE, query_template_fingerprint

SCHEMA_VERSION = "ttdf.pilot41.validators.v1"


def v9_graph_leak(record: Dict[str, Any]) -> Dict[str, Any]:
    mode = record.get("requested_query_mode") or record.get("query_mode") or ""
    result = analyze_graph_leak(record, query_mode=mode)
    return {
        "validator": "V9_GRAPH_LEAK",
        "passed": bool(result.get("passes_mode_budget")),
        "evidence": result,
        "warnings": list(result.get("warnings") or []),
    }


def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", text or "")


def v10_fact_preservation(record: Dict[str, Any],
                          contract: Optional[Dict[str, Any]] = None
                          ) -> Dict[str, Any]:
    contract = contract or record.get("semantic_contract") or {}
    question = str(record.get("question") or "")
    consts = contract.get("constants") or record.get("constants") or []
    # stringify constants the way the renderer does
    needed = []
    for c in consts:
        if isinstance(c, float) and c == int(c):
            needed.append(str(int(c)))
        else:
            needed.append(str(c))
    q_nums = set(_extract_numbers(question))
    missing = [n for n in needed if _extract_numbers(n) and
               not any(x in q_nums for x in _extract_numbers(n))]
    # extra numbers beyond contract (allow mild formatting)
    needed_nums = set()
    for n in needed:
        needed_nums.update(_extract_numbers(n))
    extra = sorted(q_nums - needed_nums)
    units = [str(u).lower() for u in (contract.get("units") or [])]
    unit_ok = True
    unit_issues = []
    # if contract declares units, changing to a conflicting unit token is bad;
    # we only flag clearly swapped temperature/duration words
    bad_swaps = [("celsius", "fahrenheit"), ("days", "seconds"),
                 ("eur", "usd")]
    q_low = question.lower()
    for a, b in bad_swaps:
        if a in units and b in q_low and a not in q_low:
            unit_ok = False
            unit_issues.append(f"{a}_became_{b}")
    target = (contract.get("target_variable") or {}).get("role") or ""
    target_ok = (not target) or (target.replace("_", " ") in q_low) or (
        any(w in q_low for w in target.split("_") if len(w) > 3))
    # tool names
    tool_hits = []
    for t in record.get("tools") or []:
        name = (t.get("name") if isinstance(t, dict) else None) or ""
        if name and re.search(rf"\b{re.escape(name)}\b", question):
            tool_hits.append(name)
    passed = (not missing) and unit_ok and target_ok and not tool_hits
    # extra numbers are soft warnings (dates/counts in NL can appear)
    warnings = []
    if extra:
        warnings.append(f"extra_numbers:{','.join(extra[:8])}")
    if missing:
        warnings.append(f"missing_constants:{missing}")
    if tool_hits:
        warnings.append(f"tool_names:{tool_hits}")
    return {
        "validator": "V10_FACT_PRESERVATION",
        "passed": passed,
        "evidence": {
            "missing_constants": missing,
            "extra_numbers": extra,
            "unit_ok": unit_ok,
            "unit_issues": unit_issues,
            "target_ok": target_ok,
            "tool_name_hits": tool_hits,
        },
        "warnings": warnings,
    }


def v11_query_mode_compliance(record: Dict[str, Any]) -> Dict[str, Any]:
    mode = record.get("requested_query_mode") or record.get("query_mode") or ""
    question = str(record.get("question") or "")
    implicit = mode in ("GOAL_BASED_IMPLICIT", "DOMAIN_GROUNDED_IMPLICIT",
                        "SEMI_IMPLICIT", "OPERATION_EXPLICIT_GRAPH_IMPLICIT")
    leaks = []
    if implicit and FORBIDDEN_IMPLICIT_RE.search(question):
        leaks.append("forbidden_implicit_phrase")
    words = len(question.split())
    if words > 120:
        leaks.append("over_max_words")
    if words < 4:
        leaks.append("under_min_words")
    v9 = v9_graph_leak(record)
    if not v9["passed"]:
        leaks.append("graph_leak_mode_fail")
    return {
        "validator": "V11_QUERY_MODE_COMPLIANCE",
        "passed": not leaks,
        "evidence": {"mode": mode, "word_count": words, "failures": leaks},
        "warnings": leaks,
    }


def v12_llm_semantic_alignment(record: Dict[str, Any],
                               critic: Optional[Dict[str, Any]] = None
                               ) -> Dict[str, Any]:
    """Validate critic structured response; PASS if no critic was required."""
    critic = critic or record.get("llm_critic") or {}
    if not critic:
        return {
            "validator": "V12_LLM_SEMANTIC_ALIGNMENT",
            "passed": True,
            "evidence": {"skipped": True, "reason": "no_critic_response"},
            "warnings": ["critic_skipped"],
        }
    required = ["facts_preserved", "target_preserved", "units_preserved",
                "no_new_conditions", "graph_not_disclosed", "verdict"]
    missing = [k for k in required if k not in critic]
    verdict = str(critic.get("verdict") or "").upper()
    passed = (not missing) and verdict == "PASS" and all(
        critic.get(k) is True for k in (
            "facts_preserved", "target_preserved", "units_preserved",
            "no_new_conditions", "graph_not_disclosed"))
    return {
        "validator": "V12_LLM_SEMANTIC_ALIGNMENT",
        "passed": passed,
        "evidence": {"verdict": verdict, "missing_fields": missing,
                     "failure_reasons": critic.get("failure_reasons") or []},
        "warnings": list(critic.get("failure_reasons") or []),
    }


def v13_template_diversity(records: Sequence[Dict[str, Any]], *,
                           max_top1_share: float = 0.12,
                           max_exact_dup_rate: float = 0.0
                           ) -> Dict[str, Any]:
    texts = [str(r.get("question") or "") for r in records]
    exact = Counter(texts)
    n = len(texts) or 1
    exact_dup_rate = round(sum(c - 1 for c in exact.values() if c > 1) / n, 4)
    skeletons = Counter(query_template_fingerprint(t) for t in texts)
    top1 = (skeletons.most_common(1)[0][1] / n) if skeletons else 0.0
    top5 = sum(c for _, c in skeletons.most_common(5)) / n
    # per-mode top1
    by_mode: Dict[str, Counter] = {}
    for r in records:
        m = str(r.get("requested_query_mode") or "UNK")
        by_mode.setdefault(m, Counter())[
            query_template_fingerprint(str(r.get("question") or ""))] += 1
    mode_top1 = {m: (c.most_common(1)[0][1] / (sum(c.values()) or 1))
                 for m, c in by_mode.items()}
    stages = sum(1 for t in texts if "the stages are related" in t.lower())
    passed = (exact_dup_rate <= max_exact_dup_rate
              and top1 <= max_top1_share
              and stages == 0)
    return {
        "validator": "V13_TEMPLATE_DIVERSITY",
        "passed": passed,
        "evidence": {
            "n": n,
            "exact_duplicate_rate": exact_dup_rate,
            "top1_skeleton_share": round(top1, 4),
            "top5_skeleton_share": round(top5, 4),
            "n_distinct_skeletons": len(skeletons),
            "mode_top1_skeleton_share": {k: round(v, 4)
                                         for k, v in mode_top1.items()},
            "stages_related_count": stages,
        },
        "warnings": ([] if passed else ["diversity_constraint_failed"]),
    }


def validate_query_record(record: Dict[str, Any], *,
                          run_v12: bool = True) -> Dict[str, Any]:
    layers = [v9_graph_leak(record), v10_fact_preservation(record),
              v11_query_mode_compliance(record)]
    if run_v12:
        layers.append(v12_llm_semantic_alignment(record))
    passed = all(L["passed"] for L in layers)
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": passed,
        "layers": {L["validator"]: L for L in layers},
    }
