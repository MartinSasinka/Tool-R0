"""Graph-leak metrics and Pilot4 language audit (V9 foundation).

Separates operation leakage from topology disclosure: a question that never
names tools can still reveal call count, stage order and every dependency edge.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..repro import stamp, write_csv, write_json, write_text

SCHEMA_VERSION = "ttdf.graph_leak.v1"

STAGE_LABEL_RE = re.compile(
    r"\b(?:stage|step)\s*(\d+)\b|\b(?:first|second|third|fourth|fifth|"
    r"sixth|seventh|eighth|ninth|tenth)\b",
    re.I)
STAGES_RELATED_RE = re.compile(
    r"the stages are related|the remaining \d+ intermediate|"
    r"the available figures are|the figure from stage|"
    r"derives? (?:from|how|after)|follow(?:s)? from the figure",
    re.I)
CALL_COUNT_RE = re.compile(
    r"\b(?:in|after|using|over)\s+(\d+)\s+(?:steps?|stages?|operations?|"
    r"calls?|tools?)\b|\b(\d+)\s*-\s*step\b",
    re.I)
PREV_RESULT_RE = re.compile(
    r"\b(?:the )?(?:previous|prior|earlier) (?:result|value|figure|output)\b|"
    r"\buse the result\b|\bresult of (?:the )?(?:previous|prior|step|stage)\b|"
    r"\bthe figure from\b|\bonce completed\b|\bafter that\b",
    re.I)
FAN_PHRASE_RE = re.compile(
    r"\b(?:both|each of|combine(?:s|d)?|merge(?:s|d)?|join(?:s|ed)?)\b.*"
    r"\b(?:stage|step|figure|result)|\bfeeds?\b.*\b(?:two|both)\b",
    re.I)
PROCEDURAL_LEAD_RE = re.compile(
    r"\b(?:first[,:]|then |next |finally[,:]|after that|to begin|"
    r"step one|lastly)\b",
    re.I)

GRAPH_LEAK_THRESHOLDS = {
    "GRAPH_EXPLICIT": {"max_edge_coverage": 1.0, "max_stage_labels": 99,
                       "allow_call_count": True},
    "OPERATION_EXPLICIT_GRAPH_IMPLICIT": {"max_edge_coverage": 0.15,
                                          "max_stage_labels": 0,
                                          "allow_call_count": False},
    "SEMI_IMPLICIT": {"max_edge_coverage": 0.10, "max_stage_labels": 0,
                      "allow_call_count": False},
    "GOAL_BASED_IMPLICIT": {"max_edge_coverage": 0.05, "max_stage_labels": 0,
                            "allow_call_count": False},
    "DOMAIN_GROUNDED_IMPLICIT": {"max_edge_coverage": 0.05, "max_stage_labels": 0,
                                 "allow_call_count": False},
    # Pilot4 legacy labels
    "PROCEDURAL_EXPLICIT": {"max_edge_coverage": 1.0, "max_stage_labels": 99,
                            "allow_call_count": True},
    "PROCEDURAL_PARTIAL": {"max_edge_coverage": 0.5, "max_stage_labels": 4,
                           "allow_call_count": True},
}


def _question(row: Dict[str, Any]) -> str:
    return str(row.get("question") or row.get("query") or "")


def _gold_calls(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls = row.get("gold_calls") or row.get("output") or []
    if isinstance(calls, list):
        return [c for c in calls if isinstance(c, dict)]
    return []


def _n_edges_from_calls(calls: Sequence[Dict[str, Any]]) -> int:
    n = 0
    for c in calls:
        args = c.get("arguments") or c.get("parameters") or {}
        if not isinstance(args, dict):
            continue
        for v in args.values():
            if isinstance(v, str) and ("$var" in v or "result" in v.lower()):
                n += 1
            elif isinstance(v, list):
                n += sum(1 for x in v if isinstance(x, str) and "$var" in x)
    return n


def _disclosed_edges(text: str, n_calls: int) -> int:
    """Count stage-to-stage dependency mentions in the question text."""
    if n_calls <= 0:
        return 0
    hits = 0
    # "stage i ... stage j" / "figure from stage N"
    for m in re.finditer(
            r"stage\s*(\d+).*?(?:from|using|of)\s*(?:stage|step|the figure from stage)\s*(\d+)",
            text, re.I | re.S):
        hits += 1
    hits += len(re.findall(r"the figure from stage\s*\d+", text, re.I))
    hits += len(re.findall(r"result of (?:step|stage)\s*\d+", text, re.I))
    # "stages are related" block implies ~n_calls-1 disclosed edges when present
    if STAGES_RELATED_RE.search(text) and n_calls >= 2:
        hits = max(hits, n_calls - 1)
    return hits


def analyze_graph_leak(row: Dict[str, Any],
                       query_mode: Optional[str] = None) -> Dict[str, Any]:
    text = _question(row)
    calls = _gold_calls(row)
    n_calls = int(row.get("call_count") or row.get("num_calls") or len(calls) or 0)
    stage_labels = len(STAGE_LABEL_RE.findall(text))
    n_edges = max(_n_edges_from_calls(calls), max(n_calls - 1, 0))
    disclosed = _disclosed_edges(text, n_calls)
    edge_cov = round(disclosed / n_edges, 4) if n_edges else 0.0
    call_count_disclosed = bool(CALL_COUNT_RE.search(text)) or (
        STAGES_RELATED_RE.search(text) is not None and n_calls >= 2)
    ref_src = len(PREV_RESULT_RE.findall(text))
    ref_cov = round(min(1.0, ref_src / max(n_edges, 1)), 4)
    fan = 1.0 if FAN_PHRASE_RE.search(text) else 0.0
    procedural = len(PROCEDURAL_LEAD_RE.findall(text))

    if edge_cov >= 0.85 or (call_count_disclosed and stage_labels >= n_calls >= 2):
        gclass = "COMPLETE"
    elif edge_cov >= 0.5 or stage_labels >= 3:
        gclass = "HIGH"
    elif edge_cov >= 0.2 or stage_labels >= 1 or call_count_disclosed:
        gclass = "MEDIUM"
    elif edge_cov > 0 or ref_src > 0 or procedural >= 2:
        gclass = "LOW"
    else:
        gclass = "NONE"

    mode = query_mode or row.get("requested_query_mode") or row.get("query_mode") \
        or row.get("classified_query_mode") or ""
    thr = GRAPH_LEAK_THRESHOLDS.get(str(mode), {})
    passes = True
    warnings: List[str] = []
    if thr:
        if stage_labels > thr.get("max_stage_labels", 99):
            passes = False
            warnings.append("stage_label_count_exceeds_mode")
        if edge_cov > thr.get("max_edge_coverage", 1.0):
            passes = False
            warnings.append("graph_edge_coverage_exceeds_mode")
        if call_count_disclosed and not thr.get("allow_call_count", True):
            passes = False
            warnings.append("call_count_disclosed_in_implicit_mode")

    return {
        "schema_version": SCHEMA_VERSION,
        "operation_leakage": None,  # filled by caller when available
        "operation_sequence_leakage": None,
        "call_count_disclosed": call_count_disclosed,
        "n_graph_edges_disclosed": disclosed,
        "graph_edge_coverage": edge_cov,
        "reference_source_coverage": ref_cov,
        "stage_label_count": stage_labels,
        "fan_in_leakage": fan,
        "fan_out_leakage": fan,
        "intermediate_variable_leakage": ref_src,
        "procedural_lead_count": procedural,
        "stages_related_phrase": bool(STAGES_RELATED_RE.search(text)),
        "graph_leak_class": gclass,
        "query_mode": mode,
        "passes_mode_budget": passes,
        "warnings": warnings,
        "gold_call_count": n_calls,
    }


def audit_dataset(rows: Sequence[Dict[str, Any]], *,
                  label: str) -> Dict[str, Any]:
    per: List[Dict[str, Any]] = []
    classes: Counter = Counter()
    stages_related = 0
    for i, row in enumerate(rows):
        a = analyze_graph_leak(row)
        a["task_id"] = row.get("task_id") or row.get("sample_id") or f"row_{i}"
        a["dataset"] = label
        per.append(a)
        classes[a["graph_leak_class"]] += 1
        stages_related += int(a["stages_related_phrase"])
    n = len(per) or 1
    return {
        "dataset": label,
        "n_tasks": len(per),
        "graph_leak_class_distribution": dict(classes),
        "mean_graph_edge_coverage": round(
            sum(p["graph_edge_coverage"] for p in per) / n, 4),
        "mean_stage_label_count": round(
            sum(p["stage_label_count"] for p in per) / n, 4),
        "call_count_disclosed_rate": round(
            sum(1 for p in per if p["call_count_disclosed"]) / n, 4),
        "stages_related_phrase_rate": round(stages_related / n, 4),
        "high_or_complete_rate": round(
            sum(1 for p in per if p["graph_leak_class"] in ("HIGH", "COMPLETE"))
            / n, 4),
        "per_task": per,
    }


def run_pilot4_language_audit(repo_root: Path, out_dir: Path, *,
                              pilot4_canonical: Optional[Path] = None,
                              cli_args: Optional[Sequence[str]] = None
                              ) -> Dict[str, Any]:
    """Read-only audit of frozen Pilot4; never writes into pilot4_profile_safe."""
    module = repo_root / "experiments" / "targeted_tool_data_factory"
    path = pilot4_canonical or (
        module / "outputs" / "pilot4_profile_safe" / "canonical.jsonl")
    rows: List[Dict[str, Any]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    train = [r for r in rows if r.get("split") == "train"] or rows
    selected = rows
    agg_train = audit_dataset(train, label="pilot4_train")
    agg_sel = audit_dataset(selected, label="pilot4_selected")
    # drop per_task from summary json (keep CSV)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source": str(path),
        "n_source_rows": len(rows),
        "train": {k: v for k, v in agg_train.items() if k != "per_task"},
        "selected": {k: v for k, v in agg_sel.items() if k != "per_task"},
        "finding": (
            "Pilot4 GOAL_BASED_IMPLICIT frequently discloses the dependency "
            "graph via 'The stages are related as follows' and per-stage "
            "derives-from clauses; these are graph-explicit, not goal-implicit."
        ),
        "provenance": stamp(repo_root, schema_version=SCHEMA_VERSION,
                            cli_args=cli_args, input_paths=[path]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "PILOT4_LANGUAGE_AUDIT.json", summary)
    write_csv(out_dir / "PILOT4_GRAPH_LEAK_PER_TASK.csv",
              [{k: p[k] for k in p if k != "warnings"}
               for p in agg_train["per_task"][:2000]])
    md = [
        "# Pilot4 language / graph-leak audit",
        "",
        f"Source: `{path}` (read-only).",
        "",
        summary["finding"],
        "",
        "## Train split",
        "",
        f"- n: {agg_train['n_tasks']}",
        f"- stages_related_phrase_rate: {agg_train['stages_related_phrase_rate']}",
        f"- mean_graph_edge_coverage: {agg_train['mean_graph_edge_coverage']}",
        f"- call_count_disclosed_rate: {agg_train['call_count_disclosed_rate']}",
        f"- high_or_complete_rate: {agg_train['high_or_complete_rate']}",
        f"- class dist: {json.dumps(agg_train['graph_leak_class_distribution'])}",
        "",
        "## Selected set",
        "",
        f"- n: {agg_sel['n_tasks']}",
        f"- stages_related_phrase_rate: {agg_sel['stages_related_phrase_rate']}",
        f"- mean_graph_edge_coverage: {agg_sel['mean_graph_edge_coverage']}",
        f"- high_or_complete_rate: {agg_sel['high_or_complete_rate']}",
        "",
    ]
    write_text(out_dir / "PILOT4_LANGUAGE_AUDIT.md", "\n".join(md))
    return summary
