"""Unit tests for the fully self-contained Pilot4.3 independent audit package."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pytest

from analysis.pilot43_independent_audit import (
    Graph,
    ReconError,
    duplicate_rates,
    late_threshold_for,
    lexical_skeleton,
    parse_ref,
    primary_pattern,
    query_fingerprints,
    reconstruct,
    satisfied_patterns,
    split_overlap,
    tv_distance,
    undecidable_patterns,
)
from analysis.pilot43_independent_audit.audit import audit_export
from analysis.pilot43_independent_audit.metrics import (
    VALUE_KIND,
    answer_type_of,
    concentration,
    numeric_literal_stats,
    primitive_usage,
    recompute_call_count,
    surface_to_primitive_from_tools,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def ref(index: int, field: str = "output_0") -> str:
    """Reference string pointing at the 0-based node ``index``."""
    return f"$var_{index + 1}.{field}$"


def build(specs: Sequence[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Build a ``gold_calls`` list from ``(name, arguments)`` pairs."""
    return [
        {"name": name, "arguments": args, "label": f"$var_{i + 1}"}
        for i, (name, args) in enumerate(specs)
    ]


def graph_of(specs: Sequence[Tuple[str, Dict[str, Any]]]) -> Graph:
    """Reconstruct the DAG of a spec list."""
    return reconstruct(build(specs))


def kinds(graph: Graph, *values: str) -> List[str]:
    """Per-node kinds, padded with ``integer`` so kinds are fully known."""
    out = list(values)
    return out + ["integer"] * (graph.n - len(out))


LINEAR3 = [
    ("a", {"x": 1}),
    ("b", {"x": ref(0)}),
    ("c", {"x": ref(1)}),
]
FAN_IN = [
    ("a", {"x": 1}),
    ("b", {"x": 2}),
    ("c", {"p": ref(0), "q": ref(1)}),
]
FAN_IN_3 = [
    ("a", {"x": 1}),
    ("b", {"x": 2}),
    ("c", {"x": 3}),
    ("d", {"p": ref(0), "q": ref(1), "r": ref(2)}),
]
FAN_OUT = [
    ("a", {"x": 1}),
    ("b", {"x": ref(0)}),
    ("c", {"x": ref(0)}),
]
DIAMOND5 = [
    ("a", {"x": 1}),
    ("b", {"x": ref(0)}),
    ("c", {"x": ref(0)}),
    ("d", {"p": ref(1), "q": ref(2)}),
    ("e", {"x": ref(3)}),
]
TWO_JOINS_WITH_ROOT_BETWEEN = [
    ("a", {"x": 1}),
    ("b", {"x": 2}),
    ("c", {"p": ref(0), "q": ref(1)}),
    ("d", {"x": 3}),
    ("e", {"p": ref(2), "q": ref(3)}),
]
TWO_JOINS_NO_ROOT_BETWEEN = [
    ("a", {"x": 1}),
    ("b", {"x": 2}),
    ("c", {"p": ref(0), "q": ref(1)}),
    ("d", {"p": ref(2), "q": ref(0)}),
]
LATE_EDGE = [
    ("a", {"x": 1}),
    ("b", {"x": ref(0)}),
    ("c", {"x": ref(1)}),
    ("d", {"x": ref(0)}),
]


# ---------------------------------------------------------------------------
# reference parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$var_3.output_0$", ("var_3", "output_0")),
        ("$var3.output_0$", ("var_3", "output_0")),
        ("$var_3$", ("var_3", "")),
        ("$var3$", ("var_3", "")),
    ],
)
def test_parse_ref_all_four_forms(text: str, expected: Tuple[str, str]) -> None:
    assert parse_ref(text) == expected


@pytest.mark.parametrize(
    "text", ["var_3", "$var_3", "var_3$", "$node_3.output_0$", "", "$var_a$", 7, None, 1.5]
)
def test_parse_ref_rejects_non_references(text: Any) -> None:
    assert parse_ref(text) is None


def test_parse_ref_tolerates_surrounding_whitespace() -> None:
    assert parse_ref("  $var_12.result$  ") == ("var_12", "result")


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------


def test_reconstruct_walks_nested_lists_and_dicts() -> None:
    calls = build(
        [
            ("a", {"x": 1}),
            ("b", {"x": 2}),
            ("c", {"items": [{"deep": [ref(0)]}, 5], "other": {"k": ref(1)}}),
        ]
    )
    graph = reconstruct(calls)
    assert graph.edges == [(0, 2), (1, 2)]
    assert graph.parents[2] == [0, 1]


def test_reconstruct_deduplicates_repeated_references() -> None:
    calls = build([("a", {"x": 1}), ("b", {"p": ref(0), "q": ref(0), "r": "$var1$"})])
    graph = reconstruct(calls)
    assert graph.edges == [(0, 1)]
    assert graph.features()["n_edges"] == 1


def test_reconstruct_rejects_forward_reference() -> None:
    calls = build([("a", {"x": ref(1)}), ("b", {"x": 1})])
    with pytest.raises(ReconError, match="forward or self reference"):
        reconstruct(calls)


def test_reconstruct_rejects_self_reference() -> None:
    calls = build([("a", {"x": ref(0)})])
    with pytest.raises(ReconError, match="forward or self reference"):
        reconstruct(calls)


def test_reconstruct_rejects_unknown_label() -> None:
    calls = build([("a", {"x": 1}), ("b", {"x": "$var_9.output_0$"})])
    with pytest.raises(ReconError, match="unknown label"):
        reconstruct(calls)


def test_reconstruct_rejects_duplicate_labels() -> None:
    calls = [
        {"name": "a", "arguments": {"x": 1}, "label": "$var_1"},
        {"name": "b", "arguments": {"x": 2}, "label": "$var1"},
    ]
    with pytest.raises(ReconError, match="duplicate call label"):
        reconstruct(calls)


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


def test_features_on_hand_built_five_node_diamond() -> None:
    graph = graph_of(DIAMOND5)
    feats = graph.features()
    assert set(feats) == {
        "n_nodes",
        "n_edges",
        "indegree",
        "outdegree",
        "n_roots",
        "n_leaves",
        "depth",
        "critical_path",
        "n_join_nodes",
        "n_multi_parent_nodes",
        "n_fan_out_nodes",
        "n_reused_outputs",
        "n_late_edges",
        "reference_distances",
        "mean_reference_distance",
        "max_reference_distance",
        "n_parallel_branches",
        "n_independent_roots",
        "has_cycle",
    }
    assert feats["n_nodes"] == 5
    assert feats["n_edges"] == 5
    assert feats["indegree"] == [0, 1, 1, 2, 1]
    assert feats["outdegree"] == [2, 1, 1, 1, 0]
    assert feats["n_roots"] == 1
    assert feats["n_leaves"] == 1
    assert feats["depth"] == 4
    assert feats["critical_path"] == [0, 1, 3, 4]
    assert feats["n_join_nodes"] == 1
    assert feats["n_multi_parent_nodes"] == 1
    assert feats["n_fan_out_nodes"] == 1
    assert feats["n_reused_outputs"] == 1
    assert feats["n_late_edges"] == 0
    assert feats["reference_distances"] == [1, 2, 2, 1, 1]
    assert feats["mean_reference_distance"] == pytest.approx(1.4)
    assert feats["max_reference_distance"] == 2
    assert feats["n_parallel_branches"] == 1
    assert feats["n_independent_roots"] == 1
    assert feats["has_cycle"] is False


def test_features_single_node_has_depth_one() -> None:
    feats = graph_of([("a", {"x": 1})]).features()
    assert feats["depth"] == 1
    assert feats["critical_path"] == [0]
    assert feats["n_parallel_branches"] == 1
    assert feats["mean_reference_distance"] == 0.0


def test_features_parallel_branches_counts_extra_roots() -> None:
    assert graph_of(FAN_IN).features()["n_parallel_branches"] == 2
    assert graph_of(FAN_IN_3).features()["n_parallel_branches"] == 3
    assert graph_of(LINEAR3).features()["n_parallel_branches"] == 1


def test_features_late_edges_and_distances() -> None:
    feats = graph_of(LATE_EDGE).features()
    assert feats["reference_distances"] == [1, 3, 1]
    assert feats["n_late_edges"] == 1
    assert feats["max_reference_distance"] == 3


def test_critical_path_prefers_smallest_indices_on_ties() -> None:
    # Two equally long paths 0->1->3 and 0->2->3; the smallest-index parent wins.
    graph = graph_of(
        [
            ("a", {"x": 1}),
            ("b", {"x": ref(0)}),
            ("c", {"x": ref(0)}),
            ("d", {"p": ref(1), "q": ref(2)}),
        ]
    )
    assert graph.critical_path() == [0, 1, 3]


# ---------------------------------------------------------------------------
# pattern invariants: one positive and one negative case each
# ---------------------------------------------------------------------------


def sat(specs: Sequence[Tuple[str, Dict[str, Any]]], *kind_values: str) -> set:
    """Satisfied patterns for a spec list with fully known node kinds."""
    graph = graph_of(specs)
    return satisfied_patterns(graph, kinds(graph, *kind_values))


def test_linear_chain_positive_and_negative() -> None:
    assert "LINEAR_CHAIN" in sat(LINEAR3)
    assert "LINEAR_CHAIN" not in sat(DIAMOND5)


def test_fan_in_single_positive_and_negative() -> None:
    assert "FAN_IN_SINGLE" in sat(FAN_IN)
    assert "FAN_IN_SINGLE" not in sat(LINEAR3)
    assert "FAN_IN_SINGLE" not in sat(FAN_IN_3)


def test_fan_in_multiple_positive_and_negative() -> None:
    assert "FAN_IN_MULTIPLE" in sat(FAN_IN_3)
    assert "FAN_IN_MULTIPLE" in sat(TWO_JOINS_WITH_ROOT_BETWEEN)
    assert "FAN_IN_MULTIPLE" not in sat(FAN_IN)


def test_fan_out_positive_and_negative() -> None:
    assert "FAN_OUT" in sat(FAN_OUT)
    assert "FAN_OUT" not in sat(LINEAR3)


def test_diamond_positive_and_negative() -> None:
    assert "DIAMOND" in sat(DIAMOND5)
    assert "DIAMOND" not in sat(FAN_OUT)


def test_parallel_then_merge_positive_and_negative() -> None:
    assert "PARALLEL_THEN_MERGE" in sat(FAN_IN)
    assert "PARALLEL_THEN_MERGE" not in sat(DIAMOND5)


def test_reuse_early_output_positive_and_negative() -> None:
    assert "REUSE_EARLY_OUTPUT" in sat(FAN_OUT)
    assert "REUSE_EARLY_OUTPUT" not in sat(LINEAR3)


def test_late_reference_positive_and_negative() -> None:
    assert "LATE_REFERENCE" in sat(LATE_EDGE)
    assert "LATE_REFERENCE" not in sat(LINEAR3)


def test_late_threshold_grows_for_long_programs() -> None:
    assert late_threshold_for(2) == 3
    assert late_threshold_for(7) == 3
    assert late_threshold_for(8) == 4
    assert late_threshold_for(12) == 4
    graph = graph_of(LATE_EDGE)
    assert "LATE_REFERENCE" not in satisfied_patterns(
        graph, kinds(graph), late_threshold=4
    )


def test_two_stage_aggregation_positive_and_negative() -> None:
    assert "TWO_STAGE_AGGREGATION" in sat(TWO_JOINS_WITH_ROOT_BETWEEN)
    assert "TWO_STAGE_AGGREGATION" not in sat(FAN_IN)


def test_two_stage_aggregation_via_list_literal_collapse() -> None:
    specs = [
        ("sum_values", {"values": [1, 2, 3]}),
        ("mean_values", {"values": [4, 5, 6], "extra": ref(0)}),
    ]
    assert "TWO_STAGE_AGGREGATION" in sat(specs, "integer", "integer")
    # An identical shape whose outputs are lists does not collapse anything.
    assert "TWO_STAGE_AGGREGATION" not in sat(specs, "list", "list")


def test_a_reduction_over_a_parents_list_aggregates_like_a_literal_one() -> None:
    # var_1 emits a list, var_2 collapses it, var_3 collapses a written-out list
    # and merges. Both collapses must count, or the producer and the audit would
    # disagree on every plan that builds its list instead of stating it.
    specs = [
        ("collect", {"x": 1}),
        ("sum_values", {"values": ref(0)}),
        ("mean_values", {"values": [4, 5, 6], "extra": ref(1)}),
    ]
    graph = graph_of(specs)
    assert "TWO_STAGE_AGGREGATION" in satisfied_patterns(
        graph, ["list", "integer", "integer"]
    )
    # the same shape where nothing is ever a collection collapses nothing
    assert "TWO_STAGE_AGGREGATION" not in satisfied_patterns(
        graph_of([("a", {"x": 1}), ("b", {"x": ref(0)})]), ["integer", "integer"]
    )


def test_node_kinds_come_from_the_observations_the_export_ships() -> None:
    from analysis.pilot43_independent_audit.audit import node_value_kinds_for

    rec = {
        "gold_calls": [
            {"name": "collect", "arguments": {}, "observation": [1, 2, 3]},
            {"name": "sum_values", "arguments": {}, "observation": 6},
        ],
        "gold_answer": 6,
    }
    assert node_value_kinds_for(rec, 2, "from_calls", "observation") == [
        "list", "integer"]
    # sink_only is still honest about what it does not know
    assert node_value_kinds_for(rec, 2, "sink_only") == ["unknown", "integer"]
    # a record without observations must not silently claim a kind
    bare = {"gold_calls": [{"name": "a", "arguments": {}}], "gold_answer": 1}
    assert node_value_kinds_for(bare, 1, "from_calls", "observation") == ["unknown"]


def test_multi_join_positive_and_negative() -> None:
    assert "MULTI_JOIN" in sat(TWO_JOINS_WITH_ROOT_BETWEEN)
    assert "MULTI_JOIN" not in sat(FAN_IN)


def test_alternating_branch_chain_positive_and_negative() -> None:
    assert "ALTERNATING_BRANCH_CHAIN" in sat(TWO_JOINS_WITH_ROOT_BETWEEN)
    assert "ALTERNATING_BRANCH_CHAIN" not in sat(TWO_JOINS_NO_ROOT_BETWEEN)


def test_mixed_independent_dependent_positive_and_negative() -> None:
    assert "MIXED_INDEPENDENT_DEPENDENT" in sat(FAN_IN)
    assert "MIXED_INDEPENDENT_DEPENDENT" not in sat(LINEAR3)
    assert "MIXED_INDEPENDENT_DEPENDENT" not in sat(
        [("a", {"x": 1}), ("b", {"x": 2})]
    )


def test_repeated_primitive_positive_and_negative() -> None:
    distinct_provenance = [
        ("add", {"a": 1, "b": 2}),
        ("add", {"a": ref(0), "b": 3}),
    ]
    same_provenance = [
        ("add", {"a": 1, "b": 2}),
        ("add", {"a": 1, "b": 2}),
    ]
    assert "REPEATED_PRIMITIVE" in sat(distinct_provenance)
    assert "REPEATED_PRIMITIVE" not in sat(same_provenance)
    assert "REPEATED_PRIMITIVE" not in sat(LINEAR3)


def test_type_transition_chain_positive_and_negative() -> None:
    linear4 = [
        ("a", {"x": 1}),
        ("b", {"x": ref(0)}),
        ("c", {"x": ref(1)}),
        ("d", {"x": ref(2)}),
    ]
    assert "TYPE_TRANSITION_CHAIN" in sat(linear4, "integer", "float", "string", "string")
    assert "TYPE_TRANSITION_CHAIN" not in sat(
        linear4, "integer", "integer", "integer", "integer"
    )
    assert "TYPE_TRANSITION_CHAIN" not in sat(
        linear4, "integer", "integer", "integer", "boolean"
    )


def test_type_transition_chain_is_undecidable_with_unknown_kinds() -> None:
    graph = graph_of(LINEAR3)
    unknown = ["unknown", "unknown", "boolean"]
    assert "TYPE_TRANSITION_CHAIN" not in satisfied_patterns(graph, unknown)
    assert undecidable_patterns(unknown) == {"TYPE_TRANSITION_CHAIN"}
    assert undecidable_patterns(["integer", "float", "boolean"]) == set()


def test_nested_aggregation_positive_and_negative() -> None:
    assert "NESTED_AGGREGATION" in sat(TWO_JOINS_WITH_ROOT_BETWEEN)
    assert "NESTED_AGGREGATION" not in sat(FAN_IN)


def test_primary_pattern_uses_priority_order() -> None:
    assert primary_pattern({"LINEAR_CHAIN"}) == "LINEAR_CHAIN"
    assert primary_pattern({"LINEAR_CHAIN", "FAN_OUT"}) == "FAN_OUT"
    assert primary_pattern({"DIAMOND", "FAN_OUT", "MULTI_JOIN"}) == "MULTI_JOIN"
    assert primary_pattern(set()) == "UNCLASSIFIED"


def test_empty_graph_satisfies_nothing() -> None:
    graph = reconstruct([])
    assert satisfied_patterns(graph, []) == set()
    assert graph.features()["n_nodes"] == 0
    assert graph.features()["n_parallel_branches"] == 0


# ---------------------------------------------------------------------------
# value kinds and metrics
# ---------------------------------------------------------------------------


def test_value_kind_checks_bool_before_int() -> None:
    assert VALUE_KIND(True) == "boolean"
    assert VALUE_KIND(False) == "boolean"
    assert VALUE_KIND(1) == "integer"
    assert VALUE_KIND(1.5) == "float"
    assert VALUE_KIND("x") == "string"
    assert VALUE_KIND([1]) == "list"
    assert VALUE_KIND({"a": 1}) == "object"
    assert VALUE_KIND(None) == "null"


def test_answer_type_and_call_count_recomputation() -> None:
    rec = {"gold_answer": True, "gold_calls": build(LINEAR3), "call_count": 99}
    assert answer_type_of(rec) == "boolean"
    assert recompute_call_count(rec) == 3


def test_duplicate_rates_counts_repeated_questions() -> None:
    records = [
        {"question": "What is the total for 10 and 20?"},
        {"question": "What is the total for 10 and 20?"},
        {"question": "Compute the average of 3 and 4."},
    ]
    rates = duplicate_rates(records)
    assert rates["n"] == 3
    assert rates["n_distinct_exact"] == 2
    assert rates["exact_duplicate_rate"] == pytest.approx(2 / 3)
    # The two "total" queries share a skeleton with the third only if the
    # lexical template matches; here they do not.
    assert rates["n_distinct_skeleton"] == 2
    assert rates["top1_skeleton_share"] == pytest.approx(2 / 3)


def test_duplicate_rates_all_unique() -> None:
    records = [{"question": f"Query number {i} about a thing?"} for i in range(4)]
    rates = duplicate_rates(records)
    assert rates["exact_duplicate_rate"] == 0.0
    assert rates["n_distinct_exact"] == 4
    # Same template, different numbers -> one skeleton.
    assert rates["n_distinct_skeleton"] == 1
    assert rates["top1_skeleton_share"] == pytest.approx(1.0)


def test_lexical_skeleton_normalises_values() -> None:
    skeleton = lexical_skeleton('Pay $1,200.50 for "widget" at https://x.example/y, ok?')
    assert "<cur>" in skeleton
    assert "<n>" in skeleton
    assert "<str>" in skeleton
    assert skeleton.endswith("ok?")
    assert "," not in skeleton


def test_query_fingerprints_are_stable_hashes() -> None:
    first = query_fingerprints("Total of 10 and 20?")
    second = query_fingerprints("Total of 30 and 40?")
    assert first["exact"] != second["exact"]
    assert first["skeleton_hash"] == second["skeleton_hash"]
    assert first["intent_hash"] == second["intent_hash"]
    assert len(first["exact"]) == 64


def test_tv_distance() -> None:
    assert tv_distance({"a": 1, "b": 1}, {"a": 1, "b": 1}) == pytest.approx(0.0)
    assert tv_distance({"a": 1, "b": 1}, {"a": 1, "b": 3}) == pytest.approx(0.25)
    assert tv_distance({"a": 1}, {"b": 1}) == pytest.approx(1.0)
    assert tv_distance({}, {}) == pytest.approx(0.0)
    assert tv_distance({"a": 0}, {"a": 1}) == pytest.approx(1.0)


def test_concentration_shares() -> None:
    from collections import Counter

    counter = Counter({"a": 5, "b": 3, "c": 2})
    result = concentration(counter)
    assert result["total"] == 10
    assert result["distinct"] == 3
    assert result["top1_share"] == pytest.approx(0.5)
    assert result["top10_share"] == pytest.approx(1.0)


def test_split_overlap_finds_shared_keys() -> None:
    splits = {
        "train": [{"workflow_id": "a"}, {"workflow_id": "b"}],
        "heldout": [{"workflow_id": "b"}, {"workflow_id": "c"}],
        "reserve": [{"workflow_id": "z"}],
    }
    overlap = split_overlap(splits, ["workflow_id"])
    assert overlap["workflow_id"]["heldout"] == 1
    assert overlap["workflow_id"]["reserve"] == 0


def test_numeric_literal_stats_ignores_booleans_and_references() -> None:
    records = [
        {
            "gold_calls": build(
                [("a", {"x": 10, "flag": True}), ("b", {"x": ref(0), "y": 25.5})]
            )
        }
    ]
    stats = numeric_literal_stats(records)
    assert stats["n"] == 2
    assert stats["min"] == 10
    assert stats["max"] == 25.5
    assert stats["share_integer"] == pytest.approx(0.5)


def test_primitive_usage_reports_source_disagreement() -> None:
    records = [
        {
            "gold_calls": [
                {"name": "combine_amounts", "arguments": {}, "label": "$var_1", "primitive_id": "add"},
                {"name": "grow_by_rate", "arguments": {}, "label": "$var_2", "primitive_id": "subtract"},
            ]
        }
    ]
    mapping = {"combine_amounts": "add", "grow_by_rate": "increase_by_percent"}
    usage = primitive_usage(records, mapping)
    assert usage.counts == {"add": 1, "increase_by_percent": 1}
    assert usage.disagreements == 1
    assert usage.source == "surface_to_primitive"


def test_surface_to_primitive_from_tools_uses_exported_data() -> None:
    records = [
        {
            "tools": [
                {"name": "combine_amounts", "semantic_id": "add"},
                {"name": "exceeds_value", "semantic_id": "is_greater"},
            ]
        }
    ]
    derived = surface_to_primitive_from_tools(records)
    assert derived["mapping"] == {"combine_amounts": "add", "exceeds_value": "is_greater"}
    assert derived["collisions"] == {}
    assert derived["n_surfaces"] == 2


# ---------------------------------------------------------------------------
# end-to-end disagreement detection
# ---------------------------------------------------------------------------


def _write_export(tmp_path: Path, records: Sequence[Dict[str, Any]]) -> Path:
    export = tmp_path / "export"
    export.mkdir()
    with (export / "train.jsonl").open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
    return export


def _spec(out_dir: Path) -> Dict[str, Any]:
    return {
        "run_label": "unit",
        "files": {"train": "train.jsonl"},
        "train_split": "train",
        "declared_paths": {
            "call_count": "call_count",
            "structural_pattern": "pattern_family",
        },
        "node_value_kinds": {"mode": "sink_only"},
        "out_dir": str(out_dir),
        "csv_name": "per_task.csv",
        "report_prefix": "AUDIT",
    }


def _record(task_id: str, specs, declared_pattern: str, call_count: int) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "question": f"Question for {task_id} with 5 and 6?",
        "gold_answer": 11,
        "gold_calls": build(specs),
        "call_count": call_count,
        "pattern_family": declared_pattern,
    }


def test_auditor_passes_when_declarations_match_content(tmp_path: Path) -> None:
    records = [
        _record("t1", LINEAR3, "LINEAR_CHAIN", 3),
        _record("t2", DIAMOND5, "DIAMOND", 5),
    ]
    export = _write_export(tmp_path, records)
    result = audit_export(export, _spec(tmp_path / "out"))
    assert result["deficits"] == []
    assert result["INDEPENDENT_AUDIT_PASSED"] is True
    assert result["verdict"] == "PASS"
    assert result["disagreements"]["structural_pattern"]["n_disagree"] == 0
    assert (tmp_path / "out" / "per_task.csv").exists()
    assert (tmp_path / "out" / "AUDIT.json").exists()
    assert (tmp_path / "out" / "AUDIT.md").exists()


def test_auditor_fails_when_record_declares_unsatisfied_pattern(tmp_path: Path) -> None:
    records = [
        _record("t1", LINEAR3, "DIAMOND", 3),
        _record("t2", DIAMOND5, "DIAMOND", 5),
    ]
    export = _write_export(tmp_path, records)
    result = audit_export(export, _spec(tmp_path / "out"))
    assert result["INDEPENDENT_AUDIT_PASSED"] is False
    assert result["verdict"] == "FAIL"
    block = result["disagreements"]["structural_pattern"]
    assert block["n_checked"] == 2
    assert block["n_disagree"] == 1
    assert block["examples"][0]["task_id"] == "t1"
    assert block["examples"][0]["declared"] == "DIAMOND"
    assert any("declared_pattern_disagreement" in d for d in result["deficits"])


def test_auditor_fails_on_declared_call_count_mismatch(tmp_path: Path) -> None:
    records = [_record("t1", LINEAR3, "LINEAR_CHAIN", 7)]
    export = _write_export(tmp_path, records)
    result = audit_export(export, _spec(tmp_path / "out"))
    assert result["INDEPENDENT_AUDIT_PASSED"] is False
    assert result["disagreements"]["call_count"]["n_disagree"] == 1
    assert any("declared_call_count_disagreement" in d for d in result["deficits"])


def test_auditor_records_missing_field_as_deficit_without_crashing(tmp_path: Path) -> None:
    record = _record("t1", LINEAR3, "LINEAR_CHAIN", 3)
    record.pop("pattern_family")
    export = _write_export(tmp_path, [record])
    result = audit_export(export, _spec(tmp_path / "out"))
    assert any(d.startswith("missing_field:pattern_family") for d in result["deficits"])
    assert result["INDEPENDENT_AUDIT_PASSED"] is False
    assert result["disagreements"]["structural_pattern"]["n_checked"] == 0


def test_auditor_reports_expected_count_mismatch(tmp_path: Path) -> None:
    export = _write_export(tmp_path, [_record("t1", LINEAR3, "LINEAR_CHAIN", 3)])
    spec = _spec(tmp_path / "out")
    spec["expected_counts"] = {"train": 5}
    result = audit_export(export, spec)
    assert any("count_mismatch:train measured 1 vs required exactly 5" == d for d in result["deficits"])


def test_audit_package_imports_no_producer_code() -> None:
    package_dir = Path(__file__).resolve().parents[1] / "analysis" / "pilot43_independent_audit"
    forbidden = ("targeted_tool_data", "pilot42", "pilot43", "numpy", "pandas", "pydantic")
    for path in sorted(package_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            for name in forbidden:
                assert name not in stripped, f"{path.name}: {stripped}"
