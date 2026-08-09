"""Pilot4.3 -- workflow-first synthetic tool-use dataset (``pilot4_3_nestful_final``).

The Pilot4.2 audit showed three structural defects that no amount of extra
metadata could repair:

* the dependency graph was a short arithmetic chain while the record declared a
  rich structural pattern,
* "generic/coding" was a workflow label over arithmetic primitives,
* every reported metric was read back from producer-side labels instead of the
  exported content.

Pilot4.3 therefore inverts the pipeline. A workflow blueprint owns an explicit
capability plan *with edges*; the plan is what produces the program; every
reported structural, capability and answer property is recomputed from the
built graph (producer side) and again from the exported JSONL by a separate
audit package that shares no code with the producer.
"""
from __future__ import annotations

SCHEMA_VERSION = "ttdf.pilot43.task.v1"
RUN_ID = "pilot4_3_nestful_final"
GENERATOR_VERSION = "pilot43.generator.v1"
PROMPT_VERSION_PREFIX = "pilot43"

# ── selection tiers (hard quotas, never cross-filled) ────────────────────
TIER_PROFILE_CORE = "PROFILE_CORE"
TIER_LONG_HORIZON = "LONG_HORIZON_ENRICHMENT"
TIER_CAPABILITY = "CAPABILITY_ENRICHMENT"
TIER_CHALLENGE = "CHALLENGE"
TIERS = (TIER_PROFILE_CORE, TIER_LONG_HORIZON, TIER_CAPABILITY, TIER_CHALLENGE)

TIER_TARGETS = {
    TIER_PROFILE_CORE: 3000,
    TIER_LONG_HORIZON: 1200,
    TIER_CAPABILITY: 600,
    TIER_CHALLENGE: 200,
}
TRAIN_MASTER_TARGET = 5000
HELDOUT_TARGET = 1000
RESERVE_TARGET = 1000
SELECTED_TARGET = 7000

HELDOUT_PARTS = {
    "standard_profile": 400,
    "workflow_family": 150,
    "program_plan": 100,
    "actual_topology": 100,
    "query_template": 100,
    "capability_combination": 100,
    "surface": 50,
}

# ── query modes (ordered from most to least explicit) ────────────────────
QUERY_MODES = (
    "GRAPH_EXPLICIT",
    "OPERATION_EXPLICIT_GRAPH_IMPLICIT",
    "SEMI_IMPLICIT",
    "GOAL_BASED_IMPLICIT",
    "DOMAIN_GROUNDED_IMPLICIT",
)
IMPLICIT_MODES = ("GOAL_BASED_IMPLICIT", "DOMAIN_GROUNDED_IMPLICIT")
LLM_WRITER_MODES = ("DOMAIN_GROUNDED_IMPLICIT", "GOAL_BASED_IMPLICIT", "SEMI_IMPLICIT")

QUERY_MODE_TARGETS = {           # train-master share targets (low, high)
    "DOMAIN_GROUNDED_IMPLICIT": (0.45, 0.55),
    "GOAL_BASED_IMPLICIT": (0.20, 0.25),
    "SEMI_IMPLICIT": (0.15, 0.20),
    "OPERATION_EXPLICIT_GRAPH_IMPLICIT": (0.05, 0.08),
    "GRAPH_EXPLICIT": (0.00, 0.03),
}

# ── the 15 structural pattern families (properties of the actual DAG) ────
STRUCTURAL_PATTERNS = (
    "LINEAR_CHAIN",
    "FAN_IN_SINGLE",
    "FAN_IN_MULTIPLE",
    "FAN_OUT",
    "DIAMOND",
    "PARALLEL_THEN_MERGE",
    "REUSE_EARLY_OUTPUT",
    "LATE_REFERENCE",
    "TWO_STAGE_AGGREGATION",
    "MULTI_JOIN",
    "ALTERNATING_BRANCH_CHAIN",
    "MIXED_INDEPENDENT_DEPENDENT",
    "REPEATED_PRIMITIVE",
    "TYPE_TRANSITION_CHAIN",
    "NESTED_AGGREGATION",
)

ANSWER_TYPES = ("float", "integer", "boolean", "string", "list", "object", "category")

SURFACE_TRACKS = ("A_NATIVE", "G_GENERAL_1", "G_GENERAL_2")

CALL_BUCKETS = ("2", "3", "4", "5", "6+")

# PROFILE_CORE call-count targets (NESTFUL dev-200 derived, +- tolerance in pp)
PROFILE_CALL_TARGETS = {
    "2": (0.330, 0.020),
    "3": (0.220, 0.020),
    "4": (0.135, 0.015),
    "5": (0.095, 0.015),
    "6+": (0.220, 0.020),
}
LONG_HORIZON_CALL_TARGETS = {"4": 0.10, "5": 0.20, "6+": 0.70}
LONG_HORIZON_DEEP_MIX = {6: (0.25, 0.35), 7: (0.20, 0.30), 8: (0.15, 0.25),
                         9: (0.08, 0.15), 10: (0.05, 0.10)}

TRAINING_READINESS_KEYS = (
    "IMPLEMENTATION_COMPLETE",
    "GENERATION_COMPLETE",
    "AUTOMATED_GATES_PASSED",
    "INDEPENDENT_AUDIT_PASSED",
    "LLM_VALIDATED",
    "HUMAN_REVIEW_PENDING",
    "HUMAN_VALIDATED",
    "GRPO_PROBE_PENDING",
    "GRPO_SIGNAL_READY",
    "TRAINING_READY",
)
