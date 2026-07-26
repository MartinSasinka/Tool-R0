"""Failure-driven generation cells + program-first candidate factory.

Cells are derived from the TargetProfile (structure) and the
StudentFailureProfile (what Qwen3-4B actually gets wrong) — DESIGN.md §7.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .. import GENERATOR_VERSION
from .. import registry as reg
from ..executor import ExecutionError, execute, executor_hash, question_constants
from ..graph import (GraphBuildError, argument_skeleton, build_program,
                     build_program_v2, graph_template_id, is_acyclic,
                     program_family)
from ..plausibility import ARTIFICIAL, analyze
from ..render import pick_surfaces, render_calls, render_query
from ..distractors import build_offered_set
from ..schemas import GenerationCell, TaskRecord
from ..util import arg_type_of, short_hash, sha256_obj

_DISTRACTOR_ROTATION = [
    "same_signature_different_semantics",
    "near_semantics",
    "similar_name",
]

_SKILL_BY_BUCKET = {
    "2": ("continuation_after_observation", "wrong_second_tool_after_correct_prefix"),
    "3": ("variable_planning", "too_few_calls"),
    "4": ("variable_planning", "too_few_calls"),
    "5": ("long_horizon_planning", "premature_stop"),
    "6+": ("long_horizon_planning", "premature_stop"),
}

# "selection" (max/min merge) is deliberately absent: a selection sink's
# value always equals one of its inputs, so such tasks can never pass the
# duplicate-observation guard nor the value-based minimal-path audit
# (documented in LIMITATIONS.md; selection primitives serve as distractors).
_MOTIFS_BY_BUCKET = {
    "2": ["linear"],
    "3": ["linear", "fan_in", "branch_aggregate"],
    "4": ["linear", "fan_in", "branch_aggregate"],
    "5": ["linear", "fan_in", "branch_aggregate"],
    "6+": ["linear", "fan_in", "branch_aggregate"],
}

_CALLS_BY_BUCKET = {"2": [2], "3": [3], "4": [4], "5": [5], "6+": [6, 7, 8]}


def derive_call_bucket_shares(profile_call_dist: Dict[str, float],
                              failure_profile: Dict[str, Any]) -> Dict[str, float]:
    """Start from the target profile; oversample 2-call ONLY if the measured
    student failure profile shows 2-call as the weakest bucket (D07)."""
    shares = dict(profile_call_dist)
    wr = (failure_profile or {}).get("win_rate_by_call_bucket", {})
    others = [k for k in shares if k != "2"]
    if wr and others and min(wr, key=lambda k: wr[k]) == "2":
        boost = 0.045
        shares["2"] = shares.get("2", 0.0) + boost
        # take the boost from the largest non-2 bucket
        donor = max(others, key=lambda k: shares[k])
        shares[donor] = max(shares[donor] - boost, 0.01)
    total = sum(shares.values())
    return {k: v / total for k, v in shares.items()}


def build_cells(profile: Any, cfg: Dict[str, Any], tracks: List[str],
                adaptation_ratio: float) -> List[GenerationCell]:
    call_shares = derive_call_bucket_shares(
        profile.call_count_dist, profile.student_failure_profile)
    # motif shares restricted to generatable motifs, renormalized from profile
    motif_prof = {k: v for k, v in profile.motif_dist.items()
                  if k in ("linear", "fan_in", "branch_aggregate")}
    motif_prof.setdefault("linear", 0.4)
    ns_quota = max(0.06, min(0.15, profile.answer_type_dist.get("string", 0.0)
                             + profile.numeric_string_rate))
    cells: List[GenerationCell] = []
    track_shares = {"A": adaptation_ratio, "G": 1 - adaptation_ratio}
    for track in ["A", "G"]:
        if (track == "A" and "adaptation" not in tracks) or \
           (track == "G" and "generalization" not in tracks):
            continue
        mode = "adaptation" if track == "A" else "generalization"
        for bucket, bshare in call_shares.items():
            motifs = _MOTIFS_BY_BUCKET[bucket]
            mshares = {m: motif_prof.get(m, 0.15) for m in motifs}
            msum = sum(mshares.values())
            skill, failure = _SKILL_BY_BUCKET[bucket]
            for mi, motif in enumerate(motifs):
                w = track_shares[track] * bshare * (mshares[motif] / msum)
                # subdivide by distractor type so no cell exceeds ~8 %
                n_sub = max(1, round(w / 0.055))
                for si in range(n_sub):
                    dt = _DISTRACTOR_ROTATION[(mi + si) % len(_DISTRACTOR_ROTATION)]
                    sk, fl = skill, failure
                    if dt == "similar_name" or si % 3 == 2:
                        sk, fl = "tool_catalog_search", "distractor_confusion"
                    cells.append(GenerationCell(
                        generation_cell_id=f"{track}_{bucket.replace('+', 'p')}call_{motif}_{sk[:12]}_{si:02d}",
                        track=track, mode=mode,
                        call_count=_CALLS_BY_BUCKET[bucket][0], motif=motif,
                        target_skill=sk, target_failure=fl,
                        direct_argument_rate=round(profile.direct_arg_share, 3),
                        numeric_string=False, reference_usage=True,
                        offered_tools_bucket=["small", "medium", "large"][(mi + si) % 3],
                        hard_distractor_type=dt,
                        quota_weight=w / n_sub))
            # one numeric-string cell per (track, bucket)
            cells.append(GenerationCell(
                generation_cell_id=f"{track}_{bucket.replace('+', 'p')}call_ns_numstring_00",
                track=track, mode=mode,
                call_count=_CALLS_BY_BUCKET[bucket][0], motif="linear",
                target_skill="argument_typing",
                target_failure="numeric_string_confusion",
                direct_argument_rate=round(profile.direct_arg_share, 3),
                numeric_string=True, reference_usage=True,
                offered_tools_bucket="medium",
                hard_distractor_type="near_semantics",
                quota_weight=track_shares[track] * bshare * ns_quota))
    total = sum(c.quota_weight for c in cells)
    for c in cells:
        c.quota_weight = c.quota_weight / total
    return cells


# ══════════════════════════════════════════════════════════════════════════
#  Engine v2 cells (pilot2)
# ══════════════════════════════════════════════════════════════════════════

# branch_aggregate needs >=3 independent branches to be distinguishable from
# fan_in (an aggregate over 2 branches has indegree 2 and is classified as
# fan_in by the profiler), so it starts at the 4-call bucket.
_MOTIFS_BY_BUCKET_V2 = {
    "2": ["linear"],
    "3": ["linear", "fan_in"],
    "4": ["linear", "fan_in", "branch_aggregate"],
    "5": ["linear", "fan_in", "branch_aggregate"],
    "6+": ["linear", "fan_in", "branch_aggregate"],
}
# fan-in is deliberately oversampled inside the multi-call buckets: NESTFUL
# dev is ~43 % fan-in and only >=3-call tasks can carry it at all.
_MOTIF_WEIGHTS_V2 = {
    "2": {"linear": 1.0},
    "3": {"linear": 0.36, "fan_in": 0.64},
    "4": {"linear": 0.27, "fan_in": 0.58, "branch_aggregate": 0.15},
    "5": {"linear": 0.25, "fan_in": 0.60, "branch_aggregate": 0.15},
    "6+": {"linear": 0.25, "fan_in": 0.60, "branch_aggregate": 0.15},
}


def target_answer_kind_shares(profile: Any, cfg: Dict[str, Any]) -> Dict[str, float]:
    """Answer-kind quotas: the target's own answer-type distribution, with the
    float share pinned into the configured band (pilot1 produced ~97 % float)."""
    ak = dict((cfg.get("generation", {}) or {}).get("answer_kind_shares") or {})
    if ak:
        total = sum(ak.values())
        return {k: v / total for k, v in ak.items()}
    dist = dict(profile.answer_type_dist or {})
    shares = {
        "float": dist.get("float", 0.78),
        "int": dist.get("int", 0.05),
        "string": dist.get("string", 0.05),
        "list": dist.get("list", 0.07),
        "bool": dist.get("bool", 0.02),
        "numeric_string": max(dist.get("numeric_string", 0.0),
                              profile.numeric_string_rate * 0.5, 0.02),
    }
    total = sum(shares.values())
    return {k: v / total for k, v in shares.items()}


def _assign_hard_distractors(cells: List[GenerationCell], hard_share: float) -> None:
    """Give exactly `hard_share` of the pool WEIGHT adversarial offered sets;
    the rest keeps an ordinary menu, so the student is not trained only in a
    maximally conflicting tool environment (pilot1 hardcoded 100 %)."""
    order = sorted(cells, key=lambda c: short_hash(c.generation_cell_id))
    acc = 0.0
    for c in order:
        if acc + 0.5 * c.quota_weight <= hard_share:
            c.hard_distractors = True
            acc += c.quota_weight
        else:
            c.hard_distractors = False
            c.hard_distractor_type = None


def _assign_answer_kinds(cells: List[GenerationCell],
                         shares: Dict[str, float]) -> None:
    """Greedy deterministic assignment of answer kinds to cells so the pooled
    weight per kind matches `shares`. A typed sink costs one call, so fan_in /
    branch_aggregate cells need >= 4 calls to keep their motif."""
    remaining = {k: v for k, v in shares.items() if k != "float"}
    eligible = sorted(
        [c for c in cells
         if c.motif == "linear"
         or (c.motif == "fan_in" and c.call_count >= 4)
         or (c.motif == "branch_aggregate" and c.call_count >= 5)],
        key=lambda c: (-c.quota_weight, c.generation_cell_id))
    for c in eligible:
        # only assign when the cell fits the remaining quota: otherwise the
        # greedy pass overshoots the typed share by one cell weight per kind
        fits = [k for k, v in remaining.items() if v >= 0.5 * c.quota_weight]
        if not fits:
            continue        # a smaller cell may still fit a small quota
        kind = max(fits, key=lambda k: remaining[k])
        c.answer_kind = kind
        remaining[kind] -= c.quota_weight
        if remaining[kind] <= 0:
            del remaining[kind]


def build_cells_v2(profile: Any, cfg: Dict[str, Any], tracks: List[str],
                   adaptation_ratio: float) -> List[GenerationCell]:
    """pilot2 cells: measured motif weights, answer-kind quotas and a
    configurable hard-distractor share (pilot1 hardcoded 100 %)."""
    gcfg = cfg.get("generation", {}) or {}
    hard_share = float(gcfg.get("hard_distractor_share", 0.8))
    call_shares = derive_call_bucket_shares(
        profile.call_count_dist, profile.student_failure_profile)
    ns_quota = 0.0
    cells: List[GenerationCell] = []
    track_shares = {"A": adaptation_ratio, "G": 1 - adaptation_ratio}
    for track in ["A", "G"]:
        if (track == "A" and "adaptation" not in tracks) or \
           (track == "G" and "generalization" not in tracks):
            continue
        mode = "adaptation" if track == "A" else "generalization"
        for bucket, bshare in call_shares.items():
            motifs = _MOTIFS_BY_BUCKET_V2[bucket]
            mshares = _MOTIF_WEIGHTS_V2[bucket]
            msum = sum(mshares[m] for m in motifs)
            skill, failure = _SKILL_BY_BUCKET[bucket]
            for mi, motif in enumerate(motifs):
                w = track_shares[track] * bshare * (mshares[motif] / msum)
                # finer subdivision than pilot1: quota control over answer
                # kinds and hard-distractor share needs small cells
                n_sub = max(1, round(w / 0.03))
                for si in range(n_sub):
                    dt = _DISTRACTOR_ROTATION[(mi + si) % len(_DISTRACTOR_ROTATION)]
                    sk, fl = skill, failure
                    if dt == "similar_name" or si % 3 == 2:
                        sk, fl = "tool_catalog_search", "distractor_confusion"
                    cells.append(GenerationCell(
                        generation_cell_id=(
                            f"{track}_{bucket.replace('+', 'p')}call_{motif}_"
                            f"{sk[:12]}_{si:02d}"),
                        track=track, mode=mode,
                        call_count=_CALLS_BY_BUCKET[bucket][0], motif=motif,
                        target_skill=sk, target_failure=fl,
                        direct_argument_rate=round(profile.direct_arg_share, 3),
                        numeric_string=False, reference_usage=True,
                        offered_tools_bucket=["small", "medium", "large"][(mi + si) % 3],
                        hard_distractor_type=dt,
                        hard_distractors=True,
                        answer_kind="float",
                        quota_weight=w / n_sub))
    total = sum(c.quota_weight for c in cells)
    for c in cells:
        c.quota_weight = c.quota_weight / total
    _assign_hard_distractors(cells, hard_share)
    _assign_answer_kinds(cells, target_answer_kind_shares(profile, cfg))
    # numeric_string cells are expressed through answer_kind now
    for c in cells:
        if c.answer_kind == "numeric_string":
            c.numeric_string = True
            c.target_skill = "argument_typing"
            c.target_failure = "numeric_string_confusion"
    return cells


def _offered_count(bucket: str, rng: random.Random,
                   buckets_cfg: Dict[str, List[int]]) -> int:
    lo, hi = buckets_cfg.get(bucket, [10, 12])
    lo = max(lo, 8)
    return rng.randint(lo, min(hi, 18))


def _answer_str(ans: Any) -> str:
    if isinstance(ans, float) and ans == int(ans):
        return str(int(ans))
    return str(ans)


def make_candidate(cell: GenerationCell, index: int, seed: int,
                   conventions: Dict[str, Any],
                   offered_buckets_cfg: Dict[str, List[int]],
                   profile_version: str, registry_hash_val: str,
                   config_hash: str, engine: str = "v1") -> Optional[TaskRecord]:
    """Deterministic candidate for (cell, index). None if all attempts fail."""
    base = f"{seed}:{cell.generation_cell_id}:{index}"
    for attempt in range(10):
        rng = random.Random(f"{base}:{attempt}")
        # call count: 6+ bucket samples 6-8
        cc = cell.call_count
        if cell.generation_cell_id.split("_")[1].startswith("6"):
            cc = rng.choice([6, 6, 7, 8])
        cell_local = cell.model_copy(update={"call_count": cc})
        try:
            if engine == "v2":
                prog = build_program_v2(cell_local, rng)
                # the motif is measured from the built graph; a cell whose
                # intent was not realised must not be counted as that motif
                if prog.motif != cell_local.motif:
                    continue
            else:
                prog = build_program(cell_local, rng)
            if not is_acyclic(prog):
                continue
            observations, answer = execute(prog)
        except (GraphBuildError, ExecutionError):
            continue

        plaus = analyze(prog)
        if engine == "v2" and plaus["plausibility_class"] == ARTIFICIAL:
            continue        # engine v2 never emits unit-incoherent chains
        if engine == "v2" and cell_local.answer_kind != "float":
            if arg_type_of(answer) != cell_local.answer_kind:
                continue
            if isinstance(answer, list) and len(answer) < 2:
                continue    # degenerate list answer

        # shortcut guard: no observation (incl. the answer) may equal a
        # numeric direct constant of the task — otherwise the step (or the
        # whole task) collapses into a shorter path (V4 would reject anyway;
        # rejecting here keeps cell coverage healthy).
        consts_num = {round(float(v), 9) for v in question_constants(prog)
                      if isinstance(v, (int, float)) and not isinstance(v, bool)}
        obs_num = [round(float(o), 9) for o in observations
                   if isinstance(o, (int, float)) and not isinstance(o, bool)]
        if any(o in consts_num for o in obs_num):
            continue
        # duplicated intermediate values make references ambiguous
        if len(set(obs_num)) != len(obs_num):
            continue

        track = cell.track
        # engine v2 keeps each surface's own parameter names so a tool NAME
        # has one global signature (trainer executor requirement)
        param_style = "as_surface" if engine == "v2" else "semantic"
        label_style = "$var{i}"
        if track == "A":
            if engine != "v2":
                param_style = rng.choice(conventions.get("param_styles", ["semantic"]))
            label_style = rng.choice(conventions.get("label_styles", ["$var{i}"]))

        tools_by_sid = pick_surfaces(prog, track, rng, param_style)
        calls = render_calls(prog, tools_by_sid, label_style)
        with_irr = cell.target_skill == "tool_catalog_search" and rng.random() < 0.4
        query, template_id, para_family_base = render_query(prog, rng, with_irrelevant=with_irr)

        # answer must not appear in the query (shortcut guard; V3 re-checks)
        if _answer_str(answer) in query:
            continue

        gold_specs = list(tools_by_sid.values())
        offered_n = max(_offered_count(cell.offered_tools_bucket, rng, offered_buckets_cfg),
                        len(gold_specs) + 3)
        offered, gold_pos, dsims = build_offered_set(
            gold_specs, track, rng, offered_n, cell.hard_distractor_type,
            param_style=param_style)

        arg_types: List[str] = []
        ref_pattern_parts: List[str] = []
        ns_args = 0
        for c in calls:
            part = []
            for v in c.arguments.values():
                t = arg_type_of(v)
                arg_types.append(t)
                ns_args += (t == "numeric_string")
                part.append("r" if t == "reference" else "d")
            ref_pattern_parts.append(",".join(part))
        n_args = len(arg_types) or 1
        family = program_family(prog)
        value_seed = rng.randint(0, 2**31 - 1)
        hard_n = sum(1 for t in offered if t.distractor_type not in (None, "easy") and t.is_distractor)
        easy_n = sum(1 for t in offered if t.distractor_type == "easy")

        rec = TaskRecord(
            task_id=f"ttdf_{short_hash([base, attempt])}",
            track=track, mode=cell.mode,
            generation_cell_id=cell.generation_cell_id,
            target_skill=cell.target_skill,
            target_failure_mode=cell.target_failure,
            query=query, template_id=template_id,
            paraphrase_family="para_" + short_hash([family, value_seed]),
            offered_tools=offered,
            offered_tool_count=len(offered),
            relevant_tool_count=len(gold_specs),
            hard_distractor_count=hard_n,
            easy_distractor_count=easy_n,
            distractor_types=sorted({t.distractor_type for t in offered
                                     if t.is_distractor and t.distractor_type}),
            distractor_similarity=dsims,
            gold_tool_positions=gold_pos,
            semantic_program=prog,
            graph_template_id=graph_template_id(prog),
            semantic_program_family=family,
            motif=prog.motif, call_count=len(calls),
            dependency_depth=prog.depth,
            canonical_calls=calls,
            oracle_observations=observations,
            gold_answer=answer,
            answer_type=arg_type_of(answer),
            plausibility_class=plaus["plausibility_class"],
            unit_trace=plaus["unit_trace"],
            sink_unit=plaus["sink_unit"],
            argument_type_pattern=arg_types,
            reference_pattern="|".join(ref_pattern_parts),
            reference_arg_share=round(sum(1 for t in arg_types if t == "reference") / n_args, 4),
            numeric_string_args=ns_args,
            output_schema_pattern=[t.output_type for t in gold_specs],
            value_seed=value_seed,
            argument_skeleton_hash=argument_skeleton(prog),
            tool_combination_hash="tc_" + short_hash(sorted(t.name for t in gold_specs)),
            generator_version=GENERATOR_VERSION,
            profile_version=profile_version,
            registry_hash=registry_hash_val,
            executor_hash=executor_hash(),
            config_hash=config_hash,
            provenance={"cell_index": index, "attempt": attempt,
                        "seed_base": base, "paraphrase_family_base": para_family_base,
                        "engine": engine, "cell_answer_kind": cell.answer_kind,
                        "cell_hard_distractors": cell.hard_distractors},
        )
        return rec
    return None


def generate_pool(cells: List[GenerationCell], n_total: int, seed: int,
                  conventions: Dict[str, Any],
                  offered_buckets_cfg: Dict[str, List[int]],
                  profile_version: str, config_hash: str,
                  only_cells: Optional[List[str]] = None,
                  start_index: int = 0,
                  engine: str = "v1") -> Tuple[List[TaskRecord], Dict[str, Dict[str, int]]]:
    reg_hash = reg.registry_hash()
    out: List[TaskRecord] = []
    stats: Dict[str, Dict[str, int]] = {}
    for cell in cells:
        if only_cells and cell.generation_cell_id not in only_cells:
            continue
        n_cell = max(1, round(cell.quota_weight * n_total))
        got, tried = 0, 0
        while got < n_cell and tried < n_cell * 4:
            rec = make_candidate(cell, start_index + tried, seed, conventions,
                                 offered_buckets_cfg, profile_version, reg_hash,
                                 config_hash, engine=engine)
            tried += 1
            if rec is not None:
                out.append(rec)
                got += 1
        stats[cell.generation_cell_id] = {"requested": n_cell, "generated": got,
                                          "attempts": tried}
    return out, stats


def record_to_canonical(rec: Dict[str, Any]) -> Dict[str, Any]:
    """TaskRecord dict -> canonical profiling row (shared featurizer input)."""
    tools = []
    for t in rec["offered_tools"]:
        tools.append({
            "name": t["name"], "description": t["description"],
            "param_types": {p["name"]: p["type"] for p in t["params"]},
            "output_fields": [t["output_field"]],
        })
    return {
        "query": rec["query"],
        "calls": [{"name": c["name"], "arguments": c["arguments"], "label": c["label"]}
                  for c in rec["canonical_calls"]],
        "tools": tools,
        "gold_answer": rec["gold_answer"],
    }
