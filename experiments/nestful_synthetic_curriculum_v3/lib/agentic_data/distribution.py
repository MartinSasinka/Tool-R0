"""Corpus distribution statistics + distance-to-NESTFUL scoring.

Shared by the agentic builder and scripts/data/score_dataset_quality.py.
Same definitions as the deterministic v4 audit so numbers are comparable.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List

_VAR_RE = re.compile(r"\$[A-Za-z_]\w*(\.\w+)?\$")

DIMENSIONS = ("call_count_dist", "offered_tools_dist", "tool_arity_dist",
              "arg_type_dist", "answer_type_dist")

# Reported (not L1-scored against NESTFUL — the tool/domain vocabulary is
# ours, not NESTFUL's, so a distance comparison there would be meaningless)
# per the 2026-07-11 diversity-gate audit: tool-family and question-template
# distribution, in addition to the dimensions above.
_FIRST_THEN_RE = re.compile(r"\bfirst\b.{0,80}\bthen\b", re.IGNORECASE)
_ENUMERATED_RE = re.compile(r"(^|\s)(1\)|step\s*1\b)", re.IGNORECASE)
_CONDITIONAL_RE = re.compile(r"\b(if|whenever|assuming|given that)\b", re.IGNORECASE)


def question_template_bucket(question: str) -> str:
    """Coarse heuristic classification of question phrasing style (spec:
    guard against the pilot's 'First ... then ...' monoculture)."""
    q = question or ""
    if _ENUMERATED_RE.search(q):
        return "enumerated"
    if _FIRST_THEN_RE.search(q):
        return "first_then"
    if _CONDITIONAL_RE.search(q):
        return "conditional"
    if q.rstrip().endswith("?"):
        return "interrogative"
    return "narrative_other"


def _tool_domain(name: str) -> str:
    try:
        from ..nestful_like_generator import TOOLS
        return TOOLS.get(name, {}).get("domain", "unknown")
    except ImportError:
        return "unknown"


def _coerce(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return v
    return v


def norm_row(row: Dict[str, Any]) -> Dict[str, Any]:
    q = row.get("question") or row.get("input") or ""
    tools = _coerce(row.get("tools")) or []
    calls = _coerce(row.get("gold_calls") or row.get("output")) or []
    return {"sample_id": str(row.get("sample_id") or ""), "question": str(q),
            "tools": tools if isinstance(tools, list) else [],
            "gold_calls": calls if isinstance(calls, list) else [],
            "gold_answer": row.get("gold_answer"),
            "motif_type": row.get("motif_type")}


def _tool_arity(tool: Dict[str, Any]) -> int:
    params = tool.get("parameters") or {}
    props = params.get("properties", params)
    return len(props) if isinstance(props, dict) else 0


def _arg_type(v: Any) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "reference" if _VAR_RE.fullmatch(v.strip() or "") else "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "other"


def _answer_type(v: Any) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "scalar"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "object"
    return "null"


def corpus_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    call_counts: Counter = Counter()
    offered: Counter = Counter()
    arity: Counter = Counter()
    arg_types: Counter = Counter()
    ans_types: Counter = Counter()
    motifs: Counter = Counter()
    used_tools: Counter = Counter()
    tool_families: Counter = Counter()
    templates: Counter = Counter()
    q_lens: List[int] = []
    for raw in rows:
        r = norm_row(raw)
        call_counts[min(len(r["gold_calls"]), 8)] += 1
        offered[min(len(r["tools"]), 30)] += 1
        for t in r["tools"]:
            if isinstance(t, dict):
                arity[min(_tool_arity(t), 6)] += 1
        for c in r["gold_calls"]:
            if isinstance(c, dict):
                name = str(c.get("name"))
                used_tools[name] += 1
                tool_families[_tool_domain(name)] += 1
                for v in (c.get("arguments") or {}).values():
                    arg_types[_arg_type(v)] += 1
        ans_types[_answer_type(r["gold_answer"])] += 1
        if r["motif_type"]:
            motifs[str(r["motif_type"])] += 1
        templates[question_template_bucket(r["question"])] += 1
        q_lens.append(len(r["question"].split()))
    n = len(rows) or 1
    return {
        "n_rows": len(rows),
        "call_count_dist": dict(sorted(call_counts.items())),
        "offered_tools_dist": dict(sorted(offered.items())),
        "tool_arity_dist": dict(sorted(arity.items())),
        "arg_type_dist": dict(sorted(arg_types.items())),
        "answer_type_dist": dict(sorted(ans_types.items())),
        "motif_dist": dict(motifs.most_common()),
        "tool_family_dist": dict(tool_families.most_common()),
        "question_template_dist": dict(templates.most_common()),
        "used_tools_top": dict(used_tools.most_common(15)),
        "mean_question_words": round(sum(q_lens) / len(q_lens), 1) if q_lens else None,
        "dominance": {
            "motif": round(max(motifs.values()) / n, 4) if motifs else None,
            "answer_type": round(max(ans_types.values()) / n, 4) if ans_types else None,
            "tool_family": round(max(tool_families.values()) / sum(tool_families.values()), 4)
            if tool_families else None,
            "question_template": round(max(templates.values()) / n, 4) if templates else None,
        },
    }


def l1_distance(d1: Dict, d2: Dict) -> float:
    """Total-variation distance between count distributions (0..1)."""
    keys = set(d1) | set(d2)
    n1, n2 = sum(d1.values()) or 1, sum(d2.values()) or 1
    return round(0.5 * sum(abs(d1.get(k, 0) / n1 - d2.get(k, 0) / n2) for k in keys), 4)


def distance_report(stats_by_corpus: Dict[str, Dict[str, Any]],
                    reference: str = "nestful") -> Dict[str, Any]:
    """Per-dimension distance of every corpus to the reference corpus."""
    ref = stats_by_corpus[reference]
    out: Dict[str, Any] = {"reference": reference, "dimensions": {}}
    for dim in DIMENSIONS:
        out["dimensions"][dim] = {
            name: l1_distance(stats[dim], ref[dim])
            for name, stats in stats_by_corpus.items() if name != reference
        }
    means = {}
    for name in stats_by_corpus:
        if name == reference:
            continue
        vals = [out["dimensions"][dim][name] for dim in DIMENSIONS]
        means[name] = round(sum(vals) / len(vals), 4)
    out["mean_distance"] = means
    return out
