"""Candidate generation: SemanticProgram -> query -> surface -> distractors.

One ``ProgramSpec`` can produce several paired task variants. Variants share
the program family id, the oracle and the topology, so the split's union-find
keeps them together.
"""
from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .. import registry as reg
from ..capability import family_of
from ..executor import executor_hash
from ..profile_v2 import surface_features
from ..query_realism import audit_task
from ..util import arg_type_of, short_hash
from . import PILOT4_VERSION, SCHEMA_VERSION
from .cells import BUCKET_CALLS, Cell
from .difficulty import build_signature, difficulty_band, difficulty_score
from .distractors import build_offered_set
from .patterns import MIN_CALLS, PatternError, generate_program
from .program import ProgramSpec, make_spec
from .query_render import (answer_leaks_into_query, goal_is_underspecified,
                           render_query)
from .surface_render import (REFERENCE_PROFILES, pick_surfaces, render_calls,
                             surface_signature)
from .validators import v7_plan_leak, v8_distractor_validity

# Transformations are drawn per candidate so two tasks from the same pattern
# family rarely share a topology inside a long call bucket.
_TRANSFORM_BUDGET = {"2": 0, "3": 0, "4": 1, "5": 1, "6+": 2}
# How many nodes each transformation appends, so the final call count still
# lands inside the cell's bucket.
_TRANSFORM_GROWTH = {
    "REUSE_OUTPUT": 1, "ADD_LATE_JOIN": 2, "ADD_SECOND_JOIN": 2,
    "SPLIT_BRANCH": 2, "ADD_PARALLEL_BRANCH": 2, "INSERT_NODE_ON_EDGE": 1,
    "CHANGE_TYPE_PATH": 1, "EXTEND_CRITICAL_PATH": 1,
}
_TRANSFORM_POOL = list(_TRANSFORM_GROWTH)
_BUCKET_MAX_CALLS = {"2": 2, "3": 3, "4": 4, "5": 5, "6+": 8}


def _target_calls(cell: Cell, rng: random.Random) -> int:
    return rng.choice(BUCKET_CALLS[cell.call_bucket])


def _pick_transforms(cell: Cell, rng: random.Random, target: int) -> List[str]:
    """Pick transformations whose combined growth still fits the target."""
    budget = _TRANSFORM_BUDGET.get(cell.call_bucket, 0)
    if budget <= 0 or rng.random() < 0.3:
        return []
    room = target - max(2, MIN_CALLS.get(cell.pattern_family, 2))
    picks: List[str] = []
    for name in rng.sample(_TRANSFORM_POOL, k=len(_TRANSFORM_POOL)):
        if len(picks) >= budget:
            break
        growth = _TRANSFORM_GROWTH[name]
        if growth <= room:
            picks.append(name)
            room -= growth
    return picks


def build_program_for_cell(cell: Cell, rng: random.Random) -> ProgramSpec:
    """Programs are built to the *base* call count; transforms add nodes."""
    target = _target_calls(cell, rng)
    transforms = _pick_transforms(cell, rng, target)
    growth = sum(_TRANSFORM_GROWTH[t] for t in transforms)
    base = max(MIN_CALLS.get(cell.pattern_family, 2), target - growth)
    result = generate_program(
        cell.pattern_family, base, rng,
        capability_mix=cell.capability_mix,
        answer_kind="", transformations=transforms)
    spec = make_spec(result)
    if spec.call_count > _BUCKET_MAX_CALLS[cell.call_bucket]:
        raise PatternError(
            f"{cell.cell_id}: {spec.call_count} calls exceeds bucket "
            f"{cell.call_bucket}")
    return spec


def render_variant(spec: ProgramSpec, cell: Cell, rng: random.Random, *,
                   track: Optional[str] = None,
                   query_mode: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """One (surface track, query mode) rendering of a program."""
    track = track or cell.track
    query_mode = query_mode or cell.query_mode

    rendered = render_query(spec, query_mode, rng)
    if answer_leaks_into_query(rendered["query"], spec.answer):
        return None

    tools_by_sid = pick_surfaces(spec, track, rng)
    calls = render_calls(spec, tools_by_sid, track)
    lo, hi = cell.offered_tool_range
    offered_n = max(rng.randint(lo, hi), len(tools_by_sid) + 3)
    offered = build_offered_set(spec, tools_by_sid, track, rng, offered_n,
                                cell.distractor_profile)

    gold_sids = [nd.semantic_id for nd in spec.program.nodes]
    audit = audit_task(rendered["query"], gold_sids)
    v7 = v7_plan_leak(rendered["query"], gold_sids, query_mode,
                      goal_underspecified=goal_is_underspecified(spec))
    v8 = v8_distractor_validity(offered["distractor_records"])

    tool_dicts = [_tool_to_dict(t) for t in offered["offered_tools"]]
    call_dicts = [{"name": c.name, "arguments": c.arguments, "label": c.label}
                  for c in calls]
    sfeat = surface_features(tool_dicts, call_dicts)

    sig = build_signature(
        features=spec.features, query_audit=audit, track=track,
        schema_complexity=sfeat["schema_complexity"],
        repeated_tool_count=sfeat["repeated_tool_count"],
        reference_format=REFERENCE_PROFILES[track],
        offered_tool_count=len(offered["offered_tools"]),
        distractor_summary=offered)

    arg_types: List[str] = []
    n_ref_args = 0
    for c in calls:
        for v in c.arguments.values():
            t = arg_type_of(v)
            arg_types.append(t)
            n_ref_args += (t == "reference")

    variant_id = short_hash([spec.semantic_program_id, track, query_mode,
                             rendered["template_id"], surface_signature(tools_by_sid)])
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": f"ttdf4_{variant_id}",
        "generator_version": PILOT4_VERSION,
        # ── layer 1: semantic program
        "semantic_program_id": spec.semantic_program_id,
        "program_family_id": spec.program_family_id,
        "graph_template_id": spec.graph_template_id,
        "pattern_family": spec.pattern_family,
        "transformations": list(spec.transformations),
        "semantic_program": spec.as_dict(),
        "call_count": spec.call_count,
        "call_bucket": cell.call_bucket,
        "capability_families": sorted(set(spec.capability_families)),
        "oracle_observations": list(spec.observations),
        "gold_answer": spec.answer,
        "answer_type": arg_type_of(spec.answer),
        # ── layer 2: query
        "question": rendered["query"],
        "template_id": rendered["template_id"],
        "query_skeleton": rendered["query_skeleton"],
        "requested_query_mode": query_mode,
        "classified_query_mode": audit["query_mode"],
        "query_audit": {k: audit[k] for k in (
            "exact_operation_coverage", "lexical_operation_coverage",
            "implicit_operation_rate", "sequence_leakage", "lcs_ratio",
            "kendall_agreement", "procedural_cue_count", "confidence")},
        # ── layer 3: surface
        "surface_track": track,
        "track": "A" if track == "A_NATIVE" else "G",
        "reference_profile": REFERENCE_PROFILES[track],
        "surface_signature": surface_signature(tools_by_sid),
        "tools": tool_dicts,
        "gold_calls": call_dicts,
        "offered_tool_count": len(offered["offered_tools"]),
        "relevant_tool_count": len(tools_by_sid),
        "gold_tool_positions": offered["gold_tool_positions"],
        "distractors": offered["distractor_records"],
        "hard_distractor_count": offered["hard_distractor_count"],
        "easy_distractor_count": offered["easy_distractor_count"],
        "same_family_distractor_count": offered["same_family_distractor_count"],
        "schema_compatible_distractor_count":
            offered["schema_compatible_distractor_count"],
        "distractor_levels": offered["distractor_levels"],
        "tool_combination_hash": "tc4_" + short_hash(
            sorted(t.name for t in tools_by_sid.values())),
        # ── difficulty + structure
        "difficulty_signature": sig,
        "difficulty_band": difficulty_band(sig),
        "difficulty_score": difficulty_score(sig),
        "structural_features": dict(spec.features),
        "surface_features": sfeat,
        "argument_type_pattern": arg_types,
        "reference_arg_share": round(n_ref_args / max(len(arg_types), 1), 4),
        # ── validation + provenance
        "validation": {"V7": v7, "V8": v8},
        "generation_cell": cell.cell_id,
        "generation_cell_meta": {
            "call_bucket": cell.call_bucket, "pattern_family": cell.pattern_family,
            "query_mode": cell.query_mode, "track": cell.track,
            "capability_mix": cell.capability_mix_name,
            "distractor_profile": cell.distractor_profile,
            "difficulty_band": cell.difficulty_band,
        },
        "target_skill": cell.target_skill,
        "target_failure_mode": cell.target_failure_skill,
        "registry_hash": reg.registry_hash(),
        "executor_hash": executor_hash(),
    }


def _tool_to_dict(spec) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    required: List[str] = []
    for p in spec.params:
        entry: Dict[str, Any] = {"type": p.type, "description": p.description}
        if p.type == "array":
            entry["items"] = {"type": p.items_type or "number"}
        if p.enum:
            entry["enum"] = p.enum
        props[p.name] = entry
        if p.required:
            required.append(p.name)
    return {
        "name": spec.name,
        "description": spec.description,
        "parameters": {"type": "object", "properties": props, "required": required},
        "output_parameters": {spec.output_field: {
            "type": spec.output_type,
            "description": spec.output_description or spec.description}},
        "is_distractor": spec.is_distractor,
        "distractor_type": spec.distractor_type,
    }


def generate_candidates(cells: Sequence[Cell], n_total: int, seed: int, *,
                        paired_variant_rate: float = 0.25,
                        max_attempts_per_slot: int = 6
                        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Deterministic candidate pool. Same seed + same commit => same pool."""
    out: List[Dict[str, Any]] = []
    stats: Dict[str, Dict[str, int]] = {}
    failures: Dict[str, int] = {}

    for cell in cells:
        want = max(1, int(round(cell.quota_weight * n_total)))
        got = tried = 0
        while got < want and tried < want * max_attempts_per_slot:
            rng = random.Random(f"p4:{seed}:{cell.cell_id}:{tried}")
            tried += 1
            try:
                spec = build_program_for_cell(cell, rng)
            except PatternError as exc:
                failures[f"program:{type(exc).__name__}"] = \
                    failures.get(f"program:{type(exc).__name__}", 0) + 1
                continue
            variant = render_variant(spec, cell, rng)
            if variant is None:
                failures["query:answer_leak"] = failures.get("query:answer_leak", 0) + 1
                continue
            out.append(variant)
            got += 1
            # paired rendering: the same program under the other track / a
            # different query mode, kept in the same split by family id
            if rng.random() < paired_variant_rate:
                other_track = ("G_GENERAL" if cell.track == "A_NATIVE"
                               else "A_NATIVE")
                other_mode = ("GOAL_BASED_IMPLICIT"
                              if cell.query_mode != "GOAL_BASED_IMPLICIT"
                              else "SEMI_IMPLICIT")
                pair = render_variant(spec, cell, rng, track=other_track,
                                      query_mode=other_mode)
                if pair is not None:
                    pair["paired_with"] = variant["task_id"]
                    variant["paired_with"] = pair["task_id"]
                    out.append(pair)
                    got += 1
        stats[cell.cell_id] = {"requested": want, "generated": got, "attempts": tried}
    return out, {"per_cell": stats, "failure_reasons": failures,
                 "n_candidates": len(out)}
