"""Multi-layer validation V1–V6 (DESIGN.md §14) + bounded minimal-path and
shortcut search (§13) + contamination audit (§5).

Hard gates: 100 % deterministic replay, 0 unresolved refs, 0 schema errors,
0 oracle mismatch, 0 exact target overlap, 0 NaN/Inf.
"""
from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import jsonschema

from .. import registry as reg
from ..executor import execute, replay_consistent
from ..render import tool_to_jsonschema
from ..schemas import SemanticProgram, TaskRecord
from ..util import is_numeric_string, is_reference, normalize_query, sha256_obj

try:
    from rapidfuzz import fuzz
    _HAVE_RF = True
except ImportError:                                   # pragma: no cover
    _HAVE_RF = False


# ── V1: schema & format ───────────────────────────────────────────────────
def v1_schema(rec: TaskRecord) -> List[str]:
    errs = []
    names = [t.name for t in rec.offered_tools]
    if len(names) != len(set(names)):
        errs.append("duplicate tool names in offered set")
    labels = [c.label for c in rec.canonical_calls]
    if len(labels) != len(set(labels)):
        errs.append("duplicate call labels")
    offered = set(names)
    for c in rec.canonical_calls:
        if c.name not in offered:
            errs.append(f"gold call {c.name} not in offered set")
    # a reference nested inside an ARRAY argument cannot be resolved by the
    # trainer's executor (and NESTFUL never produces one) -> hard error
    for c in rec.canonical_calls:
        for pname, v in c.arguments.items():
            if isinstance(v, list) and any(is_reference(x) for x in v):
                errs.append(f"{c.name}.{pname}: reference nested in array argument")
    defined: Set[str] = set()
    for c in rec.canonical_calls:
        for v in _iter_args(c.arguments):
            if is_reference(v):
                key = v.strip().strip("$").split(".")[0]
                if key not in defined:
                    errs.append(f"unresolved reference {v} in {c.name}")
        defined.add(c.label.strip("$"))
    for t in rec.offered_tools:
        try:
            js = tool_to_jsonschema(t)
            jsonschema.Draft202012Validator.check_schema(js["parameters"])
        except jsonschema.SchemaError as exc:
            errs.append(f"invalid JSON schema for {t.name}: {exc}")
    # typed gold arguments must satisfy the rendered schema types
    spec_by_name = {t.name: t for t in rec.offered_tools}
    for c in rec.canonical_calls:
        spec = spec_by_name.get(c.name)
        if not spec:
            continue
        for p in spec.params:
            if p.name not in c.arguments:
                errs.append(f"{c.name}: missing required arg {p.name}")
                continue
            v = c.arguments[p.name]
            if is_reference(v):
                continue
            if p.enum and v not in p.enum:
                errs.append(f"{c.name}.{p.name}: {v!r} not in enum")
            elif p.type == "integer" and not (isinstance(v, int) and not isinstance(v, bool)):
                errs.append(f"{c.name}.{p.name}: expected integer, got {type(v).__name__}")
            elif p.type == "number" and not isinstance(v, (int, float)):
                errs.append(f"{c.name}.{p.name}: expected number")
            elif p.type == "string" and not isinstance(v, str):
                errs.append(f"{c.name}.{p.name}: expected string")
            elif p.type == "array" and not isinstance(v, list):
                errs.append(f"{c.name}.{p.name}: expected array")
    return errs


def _iter_args(args: Dict[str, Any]):
    for v in args.values():
        if isinstance(v, list):
            yield from v
        else:
            yield v


# ── V2: actual execution + replay ─────────────────────────────────────────
def v2_execution(rec: TaskRecord) -> List[str]:
    errs = []
    try:
        obs, ans = execute(rec.semantic_program)
    except Exception as exc:
        return [f"execution failed: {exc}"]
    if obs != rec.oracle_observations:
        errs.append("observations mismatch vs record")
    if ans != rec.gold_answer:
        errs.append("final answer mismatch vs record")
    if not replay_consistent(rec.semantic_program, n=2):
        errs.append("replay not deterministic")
    for x in obs + [ans]:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            errs.append("NaN/Inf in oracle values")
    return errs


# ── V3: semantic consistency ──────────────────────────────────────────────
def _fmt_variants(v: Any) -> List[str]:
    if isinstance(v, str):
        return [v, f"'{v}'"]
    if isinstance(v, float) and v == int(v):
        return [str(int(v)), str(v)]
    if isinstance(v, list):
        return ["[" + ", ".join(str(int(x) if isinstance(x, float) and x == int(x) else x)
                                for x in v) + "]"]
    return [str(v)]


def _contains_number_token(q: str, s: str) -> bool:
    """True if s occurs in q as a standalone numeric token (digit-boundary
    guarded, so '209.4' does not match inside '209.4125')."""
    return bool(re.search(rf"(?<![\d.]){re.escape(s)}(?!\d)(?!\.\d)", q))


def v3_semantic(rec: TaskRecord) -> List[str]:
    errs = []
    q = rec.query
    # every direct constant must be readable from the query
    for c in rec.canonical_calls:
        for v in _iter_args(c.arguments):
            if is_reference(v) or isinstance(v, bool):
                continue
            if isinstance(v, (int, float, str)) and not any(s in q for s in _fmt_variants(v)):
                errs.append(f"direct constant {v!r} not present in query")
    # answer must not be present in the query
    for s in _fmt_variants(rec.gold_answer):
        if len(s) >= 2 and _contains_number_token(q, s):
            errs.append(f"oracle answer {s!r} appears in query")
    # intermediate observations must not leak into the query — except
    # pass-through nodes whose observation trivially equals one of their own
    # direct inputs (e.g. parse_number('26181') -> 26181).
    nodes = rec.semantic_program.nodes
    for i, o in enumerate(rec.oracle_observations[:-1]):
        own_inputs = set()
        if i < len(nodes):
            for v in nodes[i].inputs.values():
                if not (isinstance(v, dict) or isinstance(v, list)):
                    own_inputs.update(_fmt_variants(v))
        if set(_fmt_variants(o)) & own_inputs:
            continue
        for s in _fmt_variants(o):
            if len(s) >= 4 and _contains_number_token(q, s):
                errs.append(f"intermediate result {s!r} leaks into query")
    if rec.call_count < 2:
        errs.append("trivial single-call task")
    return sorted(set(errs))


# ── V4: bounded minimal-path & shortcut search ────────────────────────────
def _val_key(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, list):
        return tuple(_val_key(x) for x in v)
    return v


def _match(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        try:
            return abs(float(a) - float(b)) <= tol
        except OverflowError:
            # a shortcut candidate that ran an exponent primitive can exceed the
            # float range; such a value cannot equal a finite gold answer
            return isinstance(a, int) and isinstance(b, int) and a == b
    return a == b


def minimal_path_search(rec: TaskRecord, *, tol: float = 1e-6,
                        max_depth: int = 3, max_evals: int = 20000) -> Dict[str, Any]:
    """Value-based bounded search for a shorter valid path over the OFFERED
    tools. Deterministic, CPU-cheap. Lists are only usable as direct
    constants (documented bound)."""
    direct: List[Any] = []
    for c in rec.canonical_calls:
        for v in _iter_args(c.arguments):
            if not is_reference(v):
                direct.append(v)
    for c in rec.canonical_calls:          # raw list constants stay usable
        for v in c.arguments.values():
            if isinstance(v, list) and not any(is_reference(x) for x in v):
                direct.append(v)
    answer = rec.gold_answer
    offset = 0
    if isinstance(answer, bool):
        # A one-bit answer collides with unrelated computations by chance, so
        # value-based path equivalence is uninformative for it. Audit the
        # value the predicate consumes instead: if THAT is not reachable in
        # fewer calls, no genuine shortcut exists.
        if len(rec.oracle_observations) >= 2:
            answer = rec.oracle_observations[-2]
            offset = 1
    depth_limit = min(rec.call_count - 1 - offset, max_depth)
    result = {"minimal_found": None, "single_call_shortcut": False,
              "evals": 0, "budget_exhausted": False,
              "audited_value": "predicate_input" if offset else "final_answer"}
    if depth_limit < 1:
        result["minimal_found"] = rec.call_count
        return result

    prims = {t.name: (reg.get(t.semantic_id), t) for t in rec.offered_tools
             if t.semantic_id in reg.all_primitives()}
    values: List[Any] = list({_val_key(v): v for v in direct}.values())
    evals = 0
    for depth in range(1, depth_limit + 1):
        new_vals: List[Any] = []
        nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        ints = [v for v in nums if isinstance(v, int)]
        strs = [v for v in values if isinstance(v, str)]
        lists = [v for v in values if isinstance(v, list)]
        for name, (prim, _spec) in sorted(prims.items()):
            pools = []
            for (_pn, ptype, _sem) in prim.params:
                if ptype == reg.NUM:
                    pools.append(nums[:20])
                elif ptype == reg.INT:
                    pools.append(ints[:12])
                elif ptype == reg.STR:
                    pools.append(strs[:12])
                elif ptype == reg.ARR:
                    pools.append(lists[:6])
                elif ptype.startswith("enum:"):
                    pools.append(ptype[5:].split(","))
                else:
                    pools.append([])
            if any(not p for p in pools):
                continue
            for combo in itertools.product(*pools):
                evals += 1
                if evals > max_evals:
                    result["budget_exhausted"] = True
                    result["evals"] = evals
                    return result
                try:
                    out = prim.fn(**{pn: cv for (pn, _t, _s), cv
                                     in zip(prim.params, combo)})
                except Exception:
                    continue
                if isinstance(out, float) and (math.isnan(out) or math.isinf(out)):
                    continue
                if _match(out, answer, tol):
                    result["minimal_found"] = depth + offset
                    result["single_call_shortcut"] = (depth + offset == 1)
                    result["evals"] = evals
                    return result
                new_vals.append(out)
        seen = {_val_key(v) for v in values}
        for v in new_vals:
            k = _val_key(v)
            if k not in seen and len(values) < 400:
                seen.add(k)
                values.append(v)
    result["evals"] = evals
    return result


def v4_minimal_path(rec: TaskRecord, *, tol: float, max_depth: int,
                    max_evals: int) -> Tuple[List[str], Dict[str, Any]]:
    res = minimal_path_search(rec, tol=tol, max_depth=max_depth, max_evals=max_evals)
    errs = []
    if res["single_call_shortcut"]:
        errs.append("single offered tool solves the whole task")
    elif res["minimal_found"] is not None and res["minimal_found"] < rec.call_count:
        errs.append(f"shorter valid path found: {res['minimal_found']} < {rec.call_count}")
    res["declared"] = rec.call_count
    res["minimal_valid_call_count"] = (res["minimal_found"]
                                       if res["minimal_found"] is not None
                                       else rec.call_count)
    return errs, res


# ── V5: dedup + contamination ─────────────────────────────────────────────
def dedup_pool(records: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Returns {task_id: [reasons]} for duplicates to drop (keeps first)."""
    drops: Dict[str, List[str]] = {}
    seen_exact: Dict[str, str] = {}
    seen_norm: Dict[str, str] = {}
    seen_prog: Dict[str, str] = {}
    for r in records:
        tid = r["task_id"]
        q = r["query"]
        nq = normalize_query(q)
        prog_key = sha256_obj([
            [n["semantic_id"] for n in r["semantic_program"]["nodes"]],
            r["argument_skeleton_hash"], r["value_seed"]])
        if q in seen_exact:
            drops.setdefault(tid, []).append(f"exact duplicate of {seen_exact[q]}")
        else:
            seen_exact[q] = tid
        if nq in seen_norm and tid not in drops:
            drops.setdefault(tid, []).append(f"normalized duplicate of {seen_norm[nq]}")
        else:
            seen_norm.setdefault(nq, tid)
        if prog_key in seen_prog and tid not in drops:
            drops.setdefault(tid, []).append(f"program duplicate of {seen_prog[prog_key]}")
        else:
            seen_prog.setdefault(prog_key, tid)
    return drops


def contamination_check(records: List[Dict[str, Any]], blocklist: Dict[str, Any],
                        *, ratio_threshold: int = 90,
                        check_skeletons_tracks: Tuple[str, ...] = ("A",)
                        ) -> Dict[str, List[str]]:
    """Returns {task_id: [reasons]} for contaminated records."""
    bad: Dict[str, List[str]] = {}
    target_norm = blocklist["normalized"]
    target_exact = blocklist["exact"]
    target_skel = blocklist["skeletons"]
    target_queries = blocklist["queries"]
    for r in records:
        tid = r["task_id"]
        q = r["query"]
        if q in target_exact:
            bad.setdefault(tid, []).append("exact target query overlap")
        if normalize_query(q) in target_norm:
            bad.setdefault(tid, []).append("normalized target query overlap")
        if r.get("track") in check_skeletons_tracks:
            skel = tuple(c["name"] for c in r["canonical_calls"])
            if skel in target_skel:
                bad.setdefault(tid, []).append("gold tool-call skeleton overlap with target")
        if _HAVE_RF and tid not in bad:
            for tq in target_queries:
                if abs(len(tq) - len(q)) > max(len(q), len(tq)) * 0.5:
                    continue
                if fuzz.ratio(q, tq) >= ratio_threshold:
                    bad.setdefault(tid, []).append(
                        f"near-duplicate of target query (ratio>={ratio_threshold})")
                    break
    return bad


# ── V6: pool-level distribution & shortcut audit ──────────────────────────
def v6_distribution(records: List[Dict[str, Any]], *, template_max: float,
                    cell_max: float,
                    justified_cells: Optional[Set[str]] = None) -> Dict[str, Any]:
    n = max(len(records), 1)
    templates = Counter(r["template_id"] for r in records)
    cells = Counter(r["generation_cell_id"] for r in records)
    families = Counter(r["semantic_program_family"] for r in records)
    tools = Counter(name for r in records
                    for name in {c["name"] for c in r["canonical_calls"]})
    warnings = []
    for t, c in templates.most_common(5):
        if c / n > template_max:
            warnings.append(f"template {t} share {c / n:.3f} > {template_max}")
    for cl, c in cells.most_common(5):
        if c / n > cell_max and cl not in (justified_cells or set()):
            warnings.append(f"cell {cl} share {c / n:.3f} > {cell_max}")
    return {
        "n": n,
        "template_top": templates.most_common(8),
        "cell_top": cells.most_common(8),
        "family_top": families.most_common(5),
        "gold_tool_top": tools.most_common(10),
        "template_max_share": max((c / n for c in templates.values()), default=0),
        "cell_max_share": max((c / n for c in cells.values()), default=0),
        "warnings": warnings,
    }


# ── orchestration per record ──────────────────────────────────────────────
def validate_record(rec: TaskRecord, thresholds: Dict[str, Any]) -> Dict[str, Any]:
    layers: Dict[str, Any] = {}
    errs1 = v1_schema(rec)
    layers["V1"] = {"passed": not errs1, "reasons": errs1}
    if errs1:
        errs2 = ["skipped (V1 failed)"]
        layers["V2"] = {"passed": False, "reasons": errs2}
    else:
        errs2 = v2_execution(rec)
        layers["V2"] = {"passed": not errs2, "reasons": errs2}
    errs3 = v3_semantic(rec)
    layers["V3"] = {"passed": not errs3, "reasons": errs3}
    if not errs1 and not errs2:
        errs4, mp = v4_minimal_path(
            rec, tol=float(thresholds.get("answer_tolerance", 1e-6)),
            max_depth=int(thresholds.get("minimal_path_max_depth", 3)),
            max_evals=int(thresholds.get("minimal_path_max_evals", 20000)))
        layers["V4"] = {"passed": not errs4, "reasons": errs4, "search": mp}
    else:
        layers["V4"] = {"passed": False, "reasons": ["skipped (V1/V2 failed)"]}
    passed = all(layers[k]["passed"] for k in ("V1", "V2", "V3", "V4"))
    return {"passed": passed, "layers": layers}
