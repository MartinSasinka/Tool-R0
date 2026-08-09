"""V1-V6 for the pilot4 record shape, plus the V1-V8 orchestrator.

The pilot3 validators operate on ``TaskRecord`` pydantic objects; pilot4 records
carry three explicit layers instead, so the same checks are re-expressed here
against the new shape. The semantics of each layer are unchanged:

    V1 schema        callable, resolvable, type-correct
    V2 execution     the oracle really is the oracle, and it replays
    V3 semantic      constants readable, answer and intermediates not leaked
    V4 minimal path  no shorter program reaches the same answer (bounded)
    V5 duplication   pool-level dedup and target-set contamination
    V6 distribution  no template / cell / family dominates the pool
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .. import registry as reg
from ..executor import ExecutionError, execute, replay_consistent
from ..graph import REF
from ..schemas import GraphNode, SemanticProgram
from ..util import short_hash

SCHEMA_VERSION = "ttdf.pilot4.validation.v1"

_REF_RE = re.compile(r"^\$(var_?\d+)\.([A-Za-z0-9_]+)\$$")


def _program_from_record(rec: Dict[str, Any]) -> SemanticProgram:
    sp = rec.get("semantic_program") or {}
    nodes = [GraphNode(node_id=n["node_id"], semantic_id=n["primitive_id"],
                       inputs=n["inputs"], output_type=n["output_type"])
             for n in sp.get("nodes", [])]
    return SemanticProgram(nodes=nodes, sink=sp.get("sink", ""),
                           motif=rec.get("pattern_family", ""),
                           depth=int((rec.get("structural_features") or {}).get("depth", 0)))


def v1_schema(rec: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    tools = {t["name"]: t for t in rec.get("tools") or []}
    calls = rec.get("gold_calls") or []
    labels = [c.get("label") for c in calls]
    if len(set(labels)) != len(labels):
        errs.append("duplicate call labels")
    if len({t for t in tools}) != len(rec.get("tools") or []):
        errs.append("duplicate tool names in the offered set")

    seen_labels: Set[str] = set()
    for i, call in enumerate(calls):
        name = call.get("name")
        tool = tools.get(name)
        if tool is None:
            errs.append(f"call {i}: tool {name!r} not in the offered set")
            continue
        props = ((tool.get("parameters") or {}).get("properties")) or {}
        for arg, value in (call.get("arguments") or {}).items():
            if arg not in props:
                errs.append(f"call {i}: argument {arg!r} not in the schema of {name}")
                continue
            declared = props[arg].get("type")
            if isinstance(value, str) and _REF_RE.match(value):
                src = _REF_RE.match(value).group(1)
                if src not in seen_labels:
                    errs.append(f"call {i}: reference {value} used before it is defined")
            elif isinstance(value, list):
                if declared != "array":
                    errs.append(f"call {i}: list passed to non-array {arg!r}")
                if any(isinstance(x, str) and _REF_RE.match(x) for x in value):
                    errs.append(f"call {i}: reference inside an array argument")
            elif declared == "array":
                errs.append(f"call {i}: array parameter {arg!r} got a scalar")
        seen_labels.add(str(call.get("label")).lstrip("$"))
    return errs


def v2_execution(rec: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    try:
        prog = _program_from_record(rec)
        observations, answer = execute(prog)
    except (ExecutionError, KeyError, ValueError, TypeError) as exc:
        return [f"program does not execute: {exc}"]
    if list(observations) != list(rec.get("oracle_observations") or []):
        errs.append("recorded observations differ from a fresh execution")
    if answer != rec.get("gold_answer"):
        errs.append("recorded answer differs from a fresh execution")
    if not replay_consistent(prog, 2):
        errs.append("execution is not replay-deterministic")
    return errs


def v3_semantic(rec: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    question = str(rec.get("question") or "")
    if len(rec.get("gold_calls") or []) < 2:
        errs.append("fewer than two gold calls")
    ans = _as_text(rec.get("gold_answer"))
    if ans and _appears(ans, question):
        errs.append("the final answer appears verbatim in the question")
    obs = list(rec.get("oracle_observations") or [])
    for i, o in enumerate(obs[:-1]):
        text = _as_text(o)
        if text and len(text) >= 3 and _appears(text, question):
            errs.append(f"intermediate observation {i} leaks into the question")
    # every direct constant must be readable from the question
    missing = 0
    for node in (rec.get("semantic_program") or {}).get("nodes", []):
        for value in (node.get("inputs") or {}).values():
            if isinstance(value, dict) and REF in value:
                continue
            text = _as_text(value)
            if text and not _appears(text, question):
                missing += 1
    if missing:
        errs.append(f"{missing} direct constants are not present in the question")
    return errs


def _appears(needle: str, haystack: str) -> bool:
    """Substring matching would flag 12 inside 1200 or 12.5; require token
    boundaries, while still allowing a sentence-final period after a number."""
    return re.search(rf"(?<!\w)(?<!\d\.){re.escape(needle)}(?!\w)(?!\.\d)",
                     haystack) is not None


def _as_text(v: Any) -> str:
    if isinstance(v, bool):
        return ""
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    if isinstance(v, (int, float, str)):
        return str(v)
    return ""


def v4_minimal_path(rec: Dict[str, Any], *, max_depth: int = 2,
                    max_evals: int = 4000) -> Tuple[List[str], Dict[str, Any]]:
    """Bounded search for a shorter program producing the same answer."""
    target = rec.get("gold_answer")
    if not isinstance(target, (int, float)) or isinstance(target, bool):
        return [], {"searched": False, "reason": "non-numeric answer"}
    consts: List[float] = []
    for node in (rec.get("semantic_program") or {}).get("nodes", []):
        for value in (node.get("inputs") or {}).values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                consts.append(float(value))
    consts = sorted(set(consts))[:8]
    sids = sorted({n["primitive_id"] for n in
                   (rec.get("semantic_program") or {}).get("nodes", [])})
    evals = 0
    frontier = list(consts)
    for _depth in range(max_depth):
        new_vals: List[float] = []
        for sid in sids:
            prim = reg.get(sid)
            numeric = [i for i, (_n, t, _s) in enumerate(prim.params)
                       if t in (reg.NUM, reg.INT)]
            if len(numeric) != len(prim.params) or not numeric:
                continue
            for a in frontier:
                for b in (frontier if len(numeric) > 1 else [None]):
                    evals += 1
                    if evals > max_evals:
                        return [], {"searched": True, "exhausted": True,
                                    "evals": evals}
                    args = [a] if b is None else [a, b]
                    if len(args) != len(prim.params):
                        continue
                    try:
                        out = prim.fn(**{p[0]: v for p, v in zip(prim.params, args)})
                    except Exception:  # noqa: BLE001
                        continue
                    if isinstance(out, (int, float)) and not isinstance(out, bool):
                        if abs(float(out) - float(target)) <= 1e-6:
                            return ([f"a shorter path of depth <= {_depth + 1} "
                                     f"reaches the same answer"],
                                    {"searched": True, "shortcut_depth": _depth + 1,
                                     "evals": evals})
                        new_vals.append(float(out))
        frontier = sorted(set(frontier + new_vals))[:60]
    return [], {"searched": True, "evals": evals, "shortcut_depth": None}


def v5_dedup(records: Sequence[Dict[str, Any]],
             target_questions: Optional[Set[str]] = None) -> Dict[str, Any]:
    seen_exact: Dict[str, str] = {}
    seen_norm: Dict[str, str] = {}
    duplicates: List[str] = []
    contaminated: List[str] = []
    for rec in records:
        q = str(rec.get("question") or "")
        norm = re.sub(r"\s+", " ", q.lower()).strip()
        key = short_hash([norm, rec.get("tool_combination_hash")])
        if key in seen_exact:
            duplicates.append(rec["task_id"])
        seen_exact[key] = rec["task_id"]
        seen_norm.setdefault(norm, rec["task_id"])
        if target_questions and norm in target_questions:
            contaminated.append(rec["task_id"])
    return {
        "layer": "V5_DEDUP",
        "n_records": len(records),
        "n_duplicates": len(duplicates),
        "duplicate_ids": duplicates[:20],
        "n_contaminated": len(contaminated),
        "contaminated_ids": contaminated[:20],
        "passed": not duplicates and not contaminated,
    }


def v6_distribution(records: Sequence[Dict[str, Any]], *,
                    template_max_share: float = 0.06,
                    cell_max_share: float = 0.05,
                    family_max_share: float = 0.02) -> Dict[str, Any]:
    n = max(len(records), 1)
    skeletons = Counter(r.get("query_skeleton") for r in records)
    cells = Counter(r.get("generation_cell") for r in records)
    families = Counter(r.get("program_family_id") for r in records)
    warnings = []
    for label, counter, cap in (("query_skeleton", skeletons, template_max_share),
                                ("generation_cell", cells, cell_max_share),
                                ("program_family", families, family_max_share)):
        top, cnt = counter.most_common(1)[0] if counter else ("", 0)
        if cnt / n > cap:
            warnings.append(f"{label} {top!r} holds {cnt / n:.3f} > {cap}")
    return {
        "layer": "V6_DISTRIBUTION",
        "n_records": len(records),
        "top_query_skeleton_share": round(
            (skeletons.most_common(1)[0][1] / n) if skeletons else 0.0, 4),
        "top_cell_share": round((cells.most_common(1)[0][1] / n) if cells else 0.0, 4),
        "top_family_share": round(
            (families.most_common(1)[0][1] / n) if families else 0.0, 4),
        "warnings": warnings,
        "passed": not warnings,
    }


def validate_record(rec: Dict[str, Any], *, run_v4: bool = False) -> Dict[str, Any]:
    """Per-record layers. V7/V8 were already attached at render time."""
    layers: Dict[str, Any] = {}
    for name, fn in (("V1", v1_schema), ("V2", v2_execution), ("V3", v3_semantic)):
        errs = fn(rec)
        layers[name] = {"passed": not errs, "reasons": errs[:6]}
    if run_v4:
        errs, meta = v4_minimal_path(rec)
        layers["V4"] = {"passed": not errs, "reasons": errs, **meta}
    else:
        layers["V4"] = {"passed": True, "skipped": True,
                        "reason": "bounded minimal-path search disabled for this run"}
    existing = rec.get("validation") or {}
    for name in ("V7", "V8"):
        if name in existing:
            layers[name] = existing[name]
    passed = all(v.get("passed", v.get("passes_target_bucket", True))
                 for k, v in layers.items() if k != "V7")
    # V7 failures are labelled, not fatal: an explicit query is still a valid
    # task, it just belongs to a different query-mode quota
    return {"schema_version": SCHEMA_VERSION, "layers": layers, "passed": passed,
            "v7_in_target_bucket": layers.get("V7", {}).get("passes_target_bucket", True)}
