"""Pilot4 validators V7 (plan leak) and V8 (distractor validity).

V1-V6 continue to live in ``targeted_tool_data.validation`` and are reused
unchanged; these two layers cover what pilot3 could not express.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .. import registry as reg
from ..capability import behaviourally_equivalent, signatures_compatible
from ..query_realism import audit_task
from .distractors import HARD_LEVELS

SCHEMA_VERSION = "ttdf.pilot4.validation.v1"

# Per query mode: allowed ranges for the leakage metrics. Explicit tasks are
# kept, but they are labelled and quota-controlled rather than silently mixed
# in with the implicit ones.
QUERY_MODE_BUDGETS: Dict[str, Dict[str, Any]] = {
    "PROCEDURAL_EXPLICIT": {
        "lexical_operation_coverage": (0.6, 1.0),
        "sequence_leakage": (0.3, 1.0),
        "procedural_cue_count": (0, 40),
    },
    "PROCEDURAL_PARTIAL": {
        "lexical_operation_coverage": (0.4, 0.9),
        "sequence_leakage": (0.2, 0.85),
        "procedural_cue_count": (0, 20),
    },
    "SEMI_IMPLICIT": {
        "lexical_operation_coverage": (0.0, 0.6),
        "sequence_leakage": (0.0, 0.6),
        "procedural_cue_count": (0, 6),
    },
    "GOAL_BASED_IMPLICIT": {
        "lexical_operation_coverage": (0.0, 0.35),
        "sequence_leakage": (0.0, 0.35),
        "procedural_cue_count": (0, 3),
    },
}


def v7_plan_leak(question: str, gold_primitive_ids: Sequence[str],
                 target_mode: str, *, goal_underspecified: bool = False
                 ) -> Dict[str, Any]:
    """Measure how much of the gold plan the question reveals."""
    audit = audit_task(question, gold_primitive_ids)
    budget = QUERY_MODE_BUDGETS.get(target_mode)
    warnings: List[str] = []
    passes = True
    if budget is None:
        warnings.append(f"no budget defined for target mode {target_mode!r}")
        passes = False
    else:
        for metric, (lo, hi) in budget.items():
            val = audit[metric] if metric in audit else 0
            if not (lo <= val <= hi):
                passes = False
                warnings.append(
                    f"{metric}={val} outside [{lo}, {hi}] for {target_mode}")
    if target_mode != audit["query_mode"]:
        warnings.append(
            f"classifier says {audit['query_mode']}, renderer requested {target_mode}")
    if target_mode in ("SEMI_IMPLICIT", "GOAL_BASED_IMPLICIT"):
        if audit["evidence_flags"]["one_text_step_per_gold_call"]:
            passes = False
            warnings.append("one text step per gold call is forbidden in this mode")
        if audit["evidence_flags"]["has_step_numbers"]:
            passes = False
            warnings.append("explicit step numbering is forbidden in this mode")
    if goal_underspecified and target_mode == "GOAL_BASED_IMPLICIT":
        # not a hard failure: the task is still well posed structurally, but a
        # purely generic goal makes the intended operation harder to infer
        warnings.append("goal_underspecified: sink primitive has no semantic cue")
    return {
        "goal_underspecified": bool(goal_underspecified),
        "layer": "V7_PLAN_LEAK",
        "query_mode": audit["query_mode"],
        "target_query_mode": target_mode,
        "classifier_confidence": audit["confidence"],
        "exact_operation_coverage": audit["exact_operation_coverage"],
        "lexical_operation_coverage": audit["lexical_operation_coverage"],
        "sequence_leakage": audit["sequence_leakage"],
        "procedural_cues": audit["procedural_cue_count"],
        "passes_target_bucket": passes,
        "warnings": warnings,
    }


def v8_distractor_validity(distractor_records: Sequence[Dict[str, Any]],
                           node_inputs: Optional[Sequence[Dict[str, Any]]] = None
                           ) -> Dict[str, Any]:
    """Confirm no distractor is a hidden alias of the tool it targets.

    Schema compatibility is checked from the recorded metadata; equivalence is
    re-tested by execution on the primitive's own input domain, so a distractor
    that merely *looks* different is rejected.
    """
    problems: List[str] = []
    checked = 0
    hidden_aliases: List[str] = []
    for rec in distractor_records:
        gold = rec.get("target_gold_primitive")
        cand = rec.get("distractor_primitive")
        if not gold or not cand:
            problems.append(f"{rec.get('distractor_tool')}: missing primitive ids")
            continue
        try:
            gp, cp = reg.get(gold), reg.get(cand)
        except KeyError:
            problems.append(f"{cand}: not in registry")
            continue
        level = rec.get("difficulty_level")
        if level in HARD_LEVELS:
            if not (rec.get("input_types_compatible") and rec.get("output_type_compatible")):
                problems.append(f"{cand}: marked hard but schema-incompatible with {gold}")
            if behaviourally_equivalent(gp, cp):
                hidden_aliases.append(f"{cand} == {gold}")
        elif level == "EASY_TYPE_INCOMPATIBLE" and signatures_compatible(gp, cp):
            problems.append(f"{cand}: marked easy but fully substitutable for {gold}")
        if not rec.get("reason_incorrect"):
            problems.append(f"{cand}: missing reason_incorrect")
        checked += 1
    passed = not problems and not hidden_aliases
    return {
        "layer": "V8_DISTRACTOR_VALIDITY",
        "n_distractors_checked": checked,
        "n_hidden_aliases": len(hidden_aliases),
        "hidden_aliases": hidden_aliases[:10],
        "problems": problems[:10],
        "passed": passed,
    }
