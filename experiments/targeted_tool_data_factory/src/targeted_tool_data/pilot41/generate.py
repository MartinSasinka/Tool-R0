"""Workflow-first semantic candidate generation for Pilot4.1."""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import registry as reg
from ..capability import family_of
from ..executor import execute, executor_hash
from ..pilot4.distractors import build_offered_set
from ..pilot4.patterns import MIN_CALLS, PatternError, generate_program
from ..pilot4.program import make_spec
from ..pilot4.surface_render import pick_surfaces, render_calls, surface_signature
from ..util import short_hash
from . import QUERY_MODES, SCHEMA_VERSION
from .cells import BUCKET_CALLS, Cell41
from .primitive_semantics import semantics_for
from .query_render import (build_semantic_contract, query_template_fingerprint,
                           render_query)
from .semantic_edge import validate_program_edges
from .validators import validate_query_record
from .workflows import pick_workflow, workflows_by_id

SKILL_SUBTYPES = [
    "single_join", "double_join", "early_reuse", "late_reuse",
    "short_reference", "long_reference", "same_tool_repeated",
    "same_family_repeated", "cross_type_transition", "parallel_independent",
    "parallel_shared_source", "late_aggregation",
]


def skill_subtypes(features: Dict[str, Any], pattern: str,
                   nodes: Sequence[Dict[str, Any]]) -> List[str]:
    out = []
    if features.get("n_joins", 0) == 1:
        out.append("single_join")
    if features.get("n_joins", 0) >= 2:
        out.append("double_join")
    if features.get("n_reused_outputs", 0):
        out.append("early_reuse" if pattern == "REUSE_EARLY_OUTPUT" else "late_reuse")
    if features.get("mean_reference_distance", 0) <= 1.2:
        out.append("short_reference")
    if features.get("n_late_references", 0):
        out.append("long_reference")
    prims = [n.get("primitive_id") for n in nodes]
    if len(prims) != len(set(prims)):
        out.append("same_tool_repeated")
    fams = [n.get("capability_family") for n in nodes]
    if len(fams) != len(set(fams)):
        out.append("same_family_repeated")
    if features.get("n_type_transitions", 0):
        out.append("cross_type_transition")
    if features.get("n_roots", 0) >= 2:
        out.append("parallel_independent")
    if features.get("n_fan_out_nodes", 0):
        out.append("parallel_shared_source")
    if pattern in ("TWO_STAGE_AGGREGATION", "NESTED_AGGREGATION"):
        out.append("late_aggregation")
    return out or ["single_join"]


def _bucket_for(n: int) -> str:
    if n <= 2:
        return "2"
    if n == 3:
        return "3"
    if n == 4:
        return "4"
    if n == 5:
        return "5"
    return "6+"


def build_semantic_candidate(cell: Cell41, index: int, seed: int,
                             ) -> Optional[Dict[str, Any]]:
    rng = random.Random((seed ^ index ^ hash(cell.cell_id)) & 0xFFFFFFFF)
    target = rng.choice(BUCKET_CALLS[cell.call_bucket])
    pattern = cell.pattern_family
    if MIN_CALLS.get(pattern, 2) > target:
        pattern = "LINEAR_CHAIN"
    try:
        result = generate_program(pattern, target, rng, answer_kind="",
                                  transformations=[])
    except (PatternError, Exception):
        return None
    spec = make_spec(result)
    nodes = [
        {"node_id": nd.node_id, "primitive_id": nd.semantic_id,
         "capability_family": family_of(nd.semantic_id),
         "output_type": nd.output_type, "inputs": nd.inputs,
         "output_role": semantics_for(nd.semantic_id).output_role}
        for nd in spec.program.nodes
    ]
    wf = pick_workflow(rng, domain=cell.workflow_domain, n_calls=spec.call_count,
                       pattern=pattern)
    if cell.workflow_ids:
        catalog = workflows_by_id()
        for wid in cell.workflow_ids:
            if wid in catalog:
                wf = catalog[wid]
                break
    edge_report = validate_program_edges(
        nodes, spec.edges,
        workflow_context={"workflow_family": wf.domain, "domain": wf.domain})
    if not edge_report["all_accepted"] and edge_report["rejection_rate"] > 0.34:
        # allow mild GenericScalar chains inside commerce/numeric domains
        if wf.domain not in ("commerce", "personal_finance", "rates_and_ratios",
                             "measurement", "statistics", "data_summary",
                             "resource_allocation", "inventory", "geometry",
                             "quality_control", "threshold_decision",
                             "scheduling", "travel_distance", "time_duration"):
            return None

    task_id = f"p41_{short_hash([cell.cell_id, index, seed, spec.semantic_program_id])[:12]}"
    contract = build_semantic_contract(
        task_id, wf, nodes, spec.answer, query_mode=cell.query_mode, rng=rng)

    # surface first (oracle already known)
    track_code = "A" if cell.track.startswith("A") else "G"
    surface_track = "A_NATIVE" if track_code == "A" else "G_GENERAL"
    try:
        tools_by_sid = pick_surfaces(spec, surface_track, rng)
        calls = render_calls(spec, tools_by_sid, surface_track)
        offered = build_offered_set(spec, tools_by_sid, surface_track, rng,
                                    max(10, len(tools_by_sid) + 4),
                                    cell.distractor_profile)
    except Exception:
        return None

    def _tool_to_dict(t: Any) -> Dict[str, Any]:
        return {
            "name": t.name,
            "description": getattr(t, "description", ""),
            "parameters": [{"name": p.name, "type": p.type,
                           "description": getattr(p, "description", "")}
                          for p in t.params],
            "output_field": t.output_field,
            "is_distractor": bool(getattr(t, "is_distractor", False)),
        }

    tool_dicts = [_tool_to_dict(t) for t in offered["offered_tools"]]
    call_dicts = [{"name": c.name, "arguments": c.arguments, "label": c.label}
                  for c in calls]

    question, skeleton = render_query(contract, cell.query_mode, rng)
    rec: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "generation_cell": cell.cell_id,
        "cell_tier": cell.tier,
        "workflow_id": wf.workflow_id,
        "workflow_domain": wf.domain,
        "call_bucket": _bucket_for(spec.call_count),
        "call_count": spec.call_count,
        "pattern_family": pattern,
        "skill_subtypes": skill_subtypes(spec.features, pattern, nodes),
        "requested_query_mode": cell.query_mode,
        "surface_track": surface_track,
        "track": track_code,
        "question": question,
        "query_skeleton": skeleton,
        "query_template_family": query_template_fingerprint(question),
        "query_source": "deterministic_v41",
        "semantic_contract": contract,
        "semantic_program_id": spec.semantic_program_id,
        "program_family_id": spec.program_family_id,
        "graph_template_id": spec.graph_template_id,
        "semantic_program": spec.as_dict(),
        "gold_calls": call_dicts,
        "gold_answer": spec.answer,
        "oracle_observations": list(spec.observations),
        "tools": tool_dicts,
        "distractors": offered.get("distractor_records") or [],
        "distractor_levels": offered.get("distractor_levels") or [],
        "hard_distractor_count": offered.get("hard_distractor_count", 0),
        "schema_compatible_distractor_count": offered.get(
            "schema_compatible_distractor_count", 0),
        "offered_tool_count": len(tool_dicts),
        "capability_families": list(spec.capability_families),
        "structural_features": dict(spec.features),
        "semantic_edge_report": {
            "n_edges": edge_report["n_edges"],
            "n_accepted": edge_report["n_accepted"],
            "rejection_rate": edge_report["rejection_rate"],
            "all_accepted": edge_report["all_accepted"],
        },
        "difficulty_band": cell.difficulty_band,
        "executor_hash": executor_hash(),
        "generation_seed": seed,
        "paired_with": None,
        "surface_signature": surface_signature(tools_by_sid),
        "tool_combination_hash": short_hash(
            sorted(t.name for t in tools_by_sid.values())),
        "constants": list(contract.get("constants") or []),
    }
    # re-exec sanity
    try:
        obs, ans = execute(spec.program)
        if ans != spec.answer:
            return None
    except Exception:
        return None

    qval = validate_query_record(rec, run_v12=False)
    rec["query_validation"] = qval
    if not qval["passed"]:
        return None
    return rec


def generate_semantic_pool(cells: Sequence[Cell41], *,
                           candidate_target: int,
                           seed: int,
                           max_attempts_factor: int = 40,
                           ) -> List[Dict[str, Any]]:
    """Generate executable, query-validated semantic candidates."""
    # weight cells by target_count
    weights = [max(c.target_count, 1) for c in cells]
    total_w = sum(weights) or 1
    quotas = [max(1, int(round(candidate_target * w / total_w))) for w in weights]
    # fix rounding
    while sum(quotas) < candidate_target:
        quotas[sum(quotas) % len(quotas)] += 1
    while sum(quotas) > candidate_target and any(q > 1 for q in quotas):
        i = max(range(len(quotas)), key=lambda j: quotas[j])
        quotas[i] -= 1

    out: List[Dict[str, Any]] = []
    seen_prog = set()
    for cell, need in zip(cells, quotas):
        got = 0
        attempts = 0
        limit = need * max_attempts_factor
        while got < need and attempts < limit:
            attempts += 1
            rec = build_semantic_candidate(cell, attempts + got * 17, seed)
            if rec is None:
                continue
            pid = rec["semantic_program_id"]
            # allow limited reuse across query modes but prefer novelty
            key = (pid, rec["requested_query_mode"], rec["surface_track"])
            if key in seen_prog:
                continue
            seen_prog.add(key)
            out.append(rec)
            got += 1
    return out


def select_render_shortlist(candidates: Sequence[Dict[str, Any]], *,
                            target: int = 2000,
                            seed: int = 0) -> List[Dict[str, Any]]:
    """Structural diversity shortlist before LLM rendering."""
    rng = random.Random(seed)
    pool = list(candidates)
    rng.shuffle(pool)
    # prefer core tier, cover buckets/modes/workflows
    selected: List[Dict[str, Any]] = []
    seen_fam = set()
    bucket_cap = {
        "2": int(target * 0.33), "3": int(target * 0.22),
        "4": int(target * 0.135), "5": int(target * 0.095),
        "6+": int(target * 0.22),
    }
    bucket_got = {b: 0 for b in bucket_cap}
    # pass 1: core
    for rec in sorted(pool, key=lambda r: (0 if r.get("cell_tier") == "CORE_PROFILE" else 1,
                                           r["task_id"])):
        if len(selected) >= target:
            break
        b = rec["call_bucket"]
        if bucket_got[b] >= bucket_cap.get(b, target):
            continue
        fam = rec["program_family_id"]
        if fam in seen_fam and rng.random() < 0.7:
            continue
        selected.append(rec)
        seen_fam.add(fam)
        bucket_got[b] += 1
    # fill
    for rec in pool:
        if len(selected) >= target:
            break
        if rec in selected:
            continue
        b = rec["call_bucket"]
        if bucket_got[b] >= int(bucket_cap.get(b, target) * 1.15):
            continue
        selected.append(rec)
        bucket_got[b] += 1
    return selected[:target]
