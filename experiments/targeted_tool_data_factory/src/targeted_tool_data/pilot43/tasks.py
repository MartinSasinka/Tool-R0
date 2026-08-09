"""Candidate identity, derived properties and the exported task record.

A candidate is identified by ``(workflow_id, plan_id, surface_track, seed)`` and
nothing else: :func:`rebuild` re-derives the whole instance from that tuple and
refuses to continue if the program fingerprint differs from the one recorded at
generation time. That keeps every intermediate file small, makes each stage
resumable, and turns "is this dataset reproducible?" into a check the pipeline
runs on itself at every stage rather than a claim in a report.

Everything a record states about structure is derived from the built program
(:mod:`.patterns`, :mod:`.program`), never from a plan label. The exported
``declared.*`` block exists so the independent audit has something to disagree
with; it is filled from the derived values, so a disagreement means the producer
and the auditor read the same bytes differently, which is exactly what the
producer/audit agreement gate is for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from ..repro import sha256_obj
from . import RUN_ID, SCHEMA_VERSION, GENERATOR_VERSION
from . import semtypes as st
from .blueprints import Blueprint, Plan, by_id
from .build import BuildError, Instance, instantiate
from .distractors import hard_aliases
from .ops import CODING_FAMILIES, build_ops
from .program import gold_calls, program_summary


@dataclass(frozen=True)
class CandidateId:
    workflow_id: str
    plan_id: str
    track: str
    seed: int

    @property
    def task_id(self) -> str:
        return "p43_" + sha256_obj([self.workflow_id, self.plan_id, self.track,
                                    self.seed])[:16]

    def as_dict(self) -> Dict[str, Any]:
        return {"workflow_id": self.workflow_id, "plan_id": self.plan_id,
                "surface_track": self.track, "seed": self.seed,
                "task_id": self.task_id}


def program_fingerprint(inst: Instance) -> str:
    """Hash of the executed program *and* its values: the reproducibility anchor."""
    return sha256_obj({
        "nodes": [(nd.node_id, nd.op, _hashable(nd.args))
                  for nd in inst.program.nodes],
        "sink": inst.program.sink,
        "answer": _hashable(inst.answer),
    })


def semantic_program_id(inst: Instance) -> str:
    """Identity of the *shape*: same plan and same op binding, any values."""
    return "sp_" + sha256_obj({
        "workflow": inst.workflow_id, "plan": inst.plan_id,
        "binding": [(nd.node_id, nd.op) for nd in inst.program.nodes],
        "sink": inst.program.sink,
    })[:16]


def _hashable(value: Any) -> Any:
    from .program import Ref
    if isinstance(value, Ref):
        return f"@{value.node_id}"
    if isinstance(value, dict):
        return {k: _hashable(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_hashable(v) for v in value]
    return value


# ── derived candidate properties ─────────────────────────────────────────
def primitive_sequence(inst: Instance) -> Tuple[str, ...]:
    return tuple(nd.op for nd in inst.program.nodes)


def capability_sequence(inst: Instance) -> Tuple[str, ...]:
    ops = build_ops()
    return tuple(ops[nd.op].capability for nd in inst.program.nodes)


def normalized_capability_sequence(inst: Instance) -> Tuple[str, ...]:
    ops = build_ops()
    return tuple(ops[nd.op].family for nd in inst.program.nodes)


def capability_families(inst: Instance) -> Tuple[str, ...]:
    return tuple(sorted(set(normalized_capability_sequence(inst))))


def coding_share(inst: Instance) -> float:
    ops = build_ops()
    used = [nd.op for nd in inst.program.nodes]
    coding = [pid for pid in used if ops[pid].coding_like]
    return round(len(coding) / max(1, len(used)), 4)


def call_bucket(call_count: int) -> str:
    return str(call_count) if call_count <= 5 else "6+"


def difficulty_band(inst: Instance) -> str:
    """Coarse difficulty from graph shape and answer type, not from a label."""
    feats = inst.graph_features
    score = (inst.call_count
             + 2 * int(feats.get("n_join_nodes", 0) > 1)
             + int(feats.get("n_reused_outputs", 0) > 0)
             + int(feats.get("n_late_edges", 0) > 0)
             + int(inst.answer_type in ("list", "object", "category")))
    if score <= 4:
        return "easy"
    if score <= 7:
        return "medium"
    if score <= 10:
        return "hard"
    return "very_hard"


def structural_skills(inst: Instance) -> Tuple[str, ...]:
    """Structural skills this instance actually exercises (from the graph)."""
    return tuple(inst.actual_patterns)


def cell_id(inst: Instance) -> str:
    """A teachable-skill cell: capability areas x shape x answer type x length.

    Deliberately coarser than the metadata tuple. Pilot4.2's cells were exact
    metadata combinations, which produced singletons that teach nothing and made
    "cell support" meaningless.
    """
    fams = capability_families(inst)
    head = "+".join(fams[:2]) if fams else "none"
    return (f"{head}|{inst.actual_primary_pattern}|{inst.answer_type}|"
            f"{call_bucket(inst.call_count)}")


def candidate_row(inst: Instance, cid: CandidateId) -> Dict[str, Any]:
    """The per-candidate record written to ``semantic_candidates.jsonl``."""
    ops = build_ops()
    calls = gold_calls(inst.program, inst.track)
    return {
        **cid.as_dict(),
        "run_id": RUN_ID,
        "domain": inst.domain,
        "semantic_program_id": semantic_program_id(inst),
        "workflow_instance_id": inst.workflow_instance_id(),
        "program_fingerprint": program_fingerprint(inst),
        "call_count": inst.call_count,
        "call_bucket": call_bucket(inst.call_count),
        "answer_type": inst.answer_type,
        "boolean_label": (bool(inst.answer) if inst.answer_type == "boolean"
                          else None),
        "boolean_band": inst.boolean_band,
        "category_band": inst.category_band,
        "actual_primary_pattern": inst.actual_primary_pattern,
        "actual_patterns": list(inst.actual_patterns),
        "graph_features": inst.graph_features,
        "primitives": list(primitive_sequence(inst)),
        "primitive_sequence": "->".join(primitive_sequence(inst)),
        "capability_sequence": "->".join(capability_sequence(inst)),
        "normalized_capability_sequence": "->".join(
            normalized_capability_sequence(inst)),
        "capability_families": list(capability_families(inst)),
        "coding_like": any(ops[nd.op].coding_like for nd in inst.program.nodes),
        "coding_call_share": coding_share(inst),
        "coding_families": sorted({ops[nd.op].family
                                   for nd in inst.program.nodes
                                   if ops[nd.op].family in CODING_FAMILIES}),
        "gold_capabilities": [c["capability"] for c in calls],
        "difficulty_band": difficulty_band(inst),
        "cell_id": cell_id(inst),
        "structural_skills": list(structural_skills(inst)),
        "answer_preview": _preview(inst.answer),
        "n_roles": len(inst.role_values),
    }


def _preview(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 120 else text[:117] + "..."


def rebuild(row: Dict[str, Any]) -> Tuple[Instance, Blueprint, Plan]:
    """Re-derive an instance from its identity and verify it did not drift."""
    bp = by_id(row["workflow_id"])
    plan = next((p for p in bp.plans if p.plan_id == row["plan_id"]), None)
    if plan is None:
        raise BuildError(f"unknown plan {row['workflow_id']}/{row['plan_id']}")
    inst = instantiate(bp, plan, int(row["seed"]), track=row["surface_track"])
    got = program_fingerprint(inst)
    want = row.get("program_fingerprint")
    if want and got != want:
        raise BuildError(f"non-reproducible candidate {row.get('task_id')}: "
                         f"fingerprint {got} != recorded {want}")
    return inst, bp, plan


# ── exported task record ─────────────────────────────────────────────────
def task_record(*, row: Dict[str, Any], inst: Instance, bp: Blueprint,
                plan: Plan, query: Dict[str, Any], offered: Dict[str, Any],
                validation: Dict[str, Any], verifier: Dict[str, Any],
                tier: str, split: str, requested_skill: str = "") -> Dict[str, Any]:
    """One line of the exported dataset (``ttdf.pilot43.task.v1``)."""
    calls = gold_calls(inst.program, inst.track, inst.observations)
    summary = program_summary(inst.program)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "generator_version": GENERATOR_VERSION,
        "task_id": row["task_id"],
        "seed": row["seed"],

        "workflow_id": inst.workflow_id,
        "domain": inst.domain,
        "plan_id": inst.plan_id,
        "workflow_instance_id": inst.workflow_instance_id(),
        "semantic_program_id": row["semantic_program_id"],
        "program_fingerprint": row["program_fingerprint"],
        "natural_user_goal": bp.natural_user_goal,
        "plan_intent": plan.intent,

        "question": query["query"],
        "requested_query_mode": query["requested_mode"],
        "actual_query_mode": query["actual_mode"],
        "query_source": query["source"],
        "query_renderer": query.get("renderer", ""),
        "query_fingerprints": query.get("fingerprints", {}),

        "gold_calls": calls,
        "gold_answer": inst.answer,
        "answer_type": inst.answer_type,
        "boolean_label": row.get("boolean_label"),
        "call_count": inst.call_count,
        "call_bucket": call_bucket(inst.call_count),
        "surface_track": inst.track,

        "tools": offered["tools"],
        "offered_tool_count": offered["offered_tool_count"],
        "distractor_profile": {
            "gold_tool_count": offered["gold_tool_count"],
            "distractor_count": offered["distractor_count"],
            "hard": offered["hard_distractor_count"],
            "medium": offered["medium_distractor_count"],
            "easy": offered["easy_distractor_count"],
            "hard_rejected_as_alias": hard_aliases(
                [nd.op for nd in inst.program.nodes],
                offered.get("rejected_distractors", ())),
        },

        "stated_facts": [
            {"role": r.name, "description": r.description, "hint": r.hint,
             "semantic_type": r.sem, "value": inst.role_values[r.name]}
            for r in plan.roles],

        # what the producer says the task is; the independent audit recomputes all
        # of it from gold_calls and must agree
        "declared": {
            "structural_pattern": inst.actual_primary_pattern,
            "satisfied_patterns": list(inst.actual_patterns),
            "requested_structural_skill": requested_skill,
            "graph_features": inst.graph_features,
            "capability_families": list(capability_families(inst)),
            "primitives": list(primitive_sequence(inst)),
            "primitive_sequence": row["primitive_sequence"],
            "normalized_capability_sequence":
                row["normalized_capability_sequence"],
            "coding_like": row["coding_like"],
            "coding_call_share": row["coding_call_share"],
            "call_count": inst.call_count,
            "answer_type": inst.answer_type,
            "depth": inst.graph_features.get("depth"),
            "program_summary": summary,
        },

        "validation": validation,
        "verifier": verifier,

        "cell_id": row["cell_id"],
        "cell_tier": tier,
        "difficulty_band": row["difficulty_band"],
        "split": split,
    }


def nestful_compat(rec: Dict[str, Any]) -> Dict[str, Any]:
    """The NESTFUL-shaped view: question, tools, gold calls, answer only."""
    return {
        "sample_id": rec["task_id"],
        "input": rec["question"],
        "tools": [{"name": t["name"], "description": t["description"],
                   "parameters": t["parameters"],
                   "output_parameters": {t["output_field"]: {
                       "type": t["output_type"], "description": "result"}}}
                  for t in rec["tools"]],
        "output": [{"name": c["name"], "arguments": c["arguments"],
                    "label": c["label"]} for c in rec["gold_calls"]],
        "gold_answer": rec["gold_answer"],
        "answer_type": rec["answer_type"],
        "n_calls": rec["call_count"],
    }


def value_realism_flags(inst: Instance, plan: Plan) -> List[str]:
    """Value-level rejections that type checking cannot catch."""
    flags: List[str] = []
    for role in plan.roles:
        value = inst.role_values[role.name]
        sem = role.sem
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if sem == st.MONEY and value <= 0:
                flags.append(f"{role.name}: non-positive money")
            if sem == st.PERCENTAGE and not -100.0 <= float(value) <= 400.0:
                flags.append(f"{role.name}: implausible percentage {value}")
            if sem == st.RATIO and not -5.0 <= float(value) <= 5.0:
                flags.append(f"{role.name}: implausible ratio {value}")
            if sem in st.PHYSICAL and sem not in (st.TEMP_C, st.TEMP_F) \
                    and value <= 0:
                flags.append(f"{role.name}: non-positive {sem}")
            if sem in (st.COUNT, st.QUANTITY) and value < 0:
                flags.append(f"{role.name}: negative count")
        if isinstance(value, str):
            if sem == st.PATH and ("/" not in value and "\\" not in value):
                flags.append(f"{role.name}: path without a separator")
            if sem == st.URL and "://" not in value:
                flags.append(f"{role.name}: url without a scheme")
            if not value.strip():
                flags.append(f"{role.name}: blank string")
        if isinstance(value, list):
            if not value:
                flags.append(f"{role.name}: empty list")
            elif len({repr(v) for v in value}) == 1 and len(value) > 2:
                flags.append(f"{role.name}: constant list")
    for nid, value in inst.observations.items():
        if isinstance(value, float) and abs(value) > 1e12:
            flags.append(f"{nid}: runaway magnitude")
    flags.extend(_date_order_flags(inst, plan))
    flags.extend(_integer_comparison_flags(inst))
    return flags


def _date_order_flags(inst: Instance, plan: Plan) -> List[str]:
    """A period that ends before it starts (spec 10: no impossible situations)."""
    from .build import DATE_END_MARKERS, DATE_START_MARKERS

    starts, ends = [], []
    for role in plan.roles:
        value = inst.role_values.get(role.name)
        if role.sem != st.DATE or not isinstance(value, str):
            continue
        name = role.name.lower()
        if any(m in name for m in DATE_START_MARKERS):
            starts.append((role.name, value))
        elif any(m in name for m in DATE_END_MARKERS):
            ends.append((role.name, value))
    return [f"{end_name}: ends {end} before {start_name} {start}"
            for start_name, start in starts for end_name, end in ends
            if end < start]


def _integer_comparison_flags(inst: Instance) -> List[str]:
    """A whole quantity judged against a fractional limit.

    A weekday index compared against 2.09, or a permission mask against 352.15,
    executes perfectly and describes a situation that cannot exist.
    """
    from .ops import build_ops
    from .program import Ref

    ops = build_ops()
    flags: List[str] = []
    for nd in inst.program.nodes:
        op = ops[nd.op]
        if op.capability not in ("comparison.at_least", "comparison.greater",
                                 "comparison.less_than", "validation.in_range",
                                 "validation.list_limit"):
            continue
        subject = nd.args.get(op.params[0].name)
        observed = (inst.observations.get(subject.node_id)
                    if isinstance(subject, Ref) else subject)
        values = observed if isinstance(observed, list) else [observed]
        integral = values and all(isinstance(v, int) and not isinstance(v, bool)
                                  for v in values)
        if not integral:
            continue
        for param in op.params[1:]:
            limit = nd.args.get(param.name)
            if isinstance(limit, float) and limit != int(limit):
                flags.append(f"{nd.node_id}: whole quantity judged against "
                             f"{limit}")
    return flags


def domain_capability_requirements(bp: Blueprint) -> Tuple[str, ...]:
    """Capability families the workflow's own domain claim commits it to.

    Renaming an arithmetic workflow ``file_processing`` is the Pilot4.2 defect this
    exists to block: the domain name has to be backed by gold calls in the matching
    capability family.
    """
    return DOMAIN_REQUIREMENTS.get(bp.domain, ())


#: domain -> capability families that must appear in the actual gold calls.
DOMAIN_REQUIREMENTS: Dict[str, Tuple[str, ...]] = {
    "list_processing": ("list",),
    "dictionary_processing": ("dictionary",),
    "record_processing": ("record",),
    "text_processing": ("string",),
    "string_parsing": ("string",),
    "file_processing": ("path",),
    "path_processing": ("path",),
    "url_processing": ("url",),
    "date_time": ("date",),
    "scheduling": ("date", "duration"),
    "validation": ("validation",),
    "classification": ("classification",),
    "boolean_logic": ("boolean",),
    "bitwise_realistic": ("bitwise",),
    "formatting": ("format",),
    "geometry": ("geometry",),
    "statistics": ("statistics",),
    "data_summary": ("statistics",),
    "unit_conversion": ("unit_conversion",),
    "measurement": ("unit_conversion", "statistics", "geometry"),
    "rates_and_ratios": ("rates",),
    "threshold_decision": ("comparison", "decision", "boolean"),
    "quality_control": ("comparison", "statistics", "validation"),
    "lookup": ("dictionary", "record", "list"),
    "aggregation": ("statistics", "list", "arithmetic"),
}


def domain_claim_satisfied(bp: Blueprint, families: Sequence[str]) -> bool:
    required = domain_capability_requirements(bp)
    if not required:
        return True
    return any(f in families for f in required)
