"""Fully self-contained independent audit of exported tool-use datasets.

This package re-derives dataset structure, patterns and statistics from the
exported JSONL content alone. It imports ONLY the Python standard library: no
module from ``targeted_tool_data`` (in particular none of ``pilot42``,
``pilot43``, ``graph``, ``profile`` or any producer-side classifier) may be
imported from here, so its verdict is independent of the code that produced the
data. Exported *data* files such as ``primitive_registry.json`` are read as
data.
"""
from __future__ import annotations

from .audit import CSV_COLUMNS, RATE_TOLERANCE, audit_export, read_jsonl
from .graph_recon import (
    REF_RE,
    Graph,
    ReconError,
    iter_refs,
    literal_arguments,
    normalize_label,
    parse_ref,
    reconstruct,
)
from .metrics import (
    answer_type_of,
    boolean_balance,
    capability_usage,
    concentration,
    duplicate_rates,
    get_path,
    intent_signature,
    lexical_skeleton,
    normalize_answer_type,
    numeric_literal_stats,
    primitive_sequences,
    primitive_usage,
    query_fingerprints,
    recompute_call_count,
    split_overlap,
    surface_names,
    surface_to_primitive_from_tools,
    tv_distance,
)
from .pattern_rules import (
    ALL_PATTERNS,
    KIND_DEPENDENT_PATTERNS,
    PATTERN_PRIORITY,
    VALUE_KIND,
    late_threshold_for,
    primary_pattern,
    satisfied_patterns,
    undecidable_patterns,
)

__all__ = [
    "ALL_PATTERNS",
    "CSV_COLUMNS",
    "Graph",
    "KIND_DEPENDENT_PATTERNS",
    "PATTERN_PRIORITY",
    "RATE_TOLERANCE",
    "REF_RE",
    "ReconError",
    "VALUE_KIND",
    "answer_type_of",
    "audit_export",
    "boolean_balance",
    "capability_usage",
    "concentration",
    "duplicate_rates",
    "get_path",
    "intent_signature",
    "iter_refs",
    "late_threshold_for",
    "lexical_skeleton",
    "literal_arguments",
    "normalize_answer_type",
    "normalize_label",
    "numeric_literal_stats",
    "parse_ref",
    "primary_pattern",
    "primitive_sequences",
    "primitive_usage",
    "query_fingerprints",
    "read_jsonl",
    "recompute_call_count",
    "reconstruct",
    "satisfied_patterns",
    "split_overlap",
    "surface_names",
    "surface_to_primitive_from_tools",
    "tv_distance",
    "undecidable_patterns",
]
