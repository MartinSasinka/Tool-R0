"""Pilot4.1: semantic workflows, non-leaking queries, staged OpenRouter render."""
from __future__ import annotations

SCHEMA_VERSION = "ttdf.pilot41.task.v1"
RUN_ID = "pilot4_1_profile_safe"

QUERY_MODES = [
    "GRAPH_EXPLICIT",
    "OPERATION_EXPLICIT_GRAPH_IMPLICIT",
    "SEMI_IMPLICIT",
    "GOAL_BASED_IMPLICIT",
    "DOMAIN_GROUNDED_IMPLICIT",
]

CELL_TIERS = [
    "CORE_PROFILE",
    "STRUCTURAL_ENRICHMENT",
    "CAPABILITY_ENRICHMENT",
    "CHALLENGE",
]
