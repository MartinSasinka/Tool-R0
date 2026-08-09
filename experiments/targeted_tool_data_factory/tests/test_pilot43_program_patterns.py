"""Program gates and the structural-pattern classifier.

Every program here is hand-built from real registry ops so the classifier sees
the same kind of graph the builder produces. Each pattern gets an explicit
positive *and* an explicit negative graph: a classifier that returns everything
would pass the positives alone.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Sequence, Set, Tuple

import pytest

from targeted_tool_data.pilot43 import semtypes as st
from targeted_tool_data.pilot43.ops import build_ops
from targeted_tool_data.pilot43.patterns import (features, primary_pattern,
                                                 satisfied_patterns)
from targeted_tool_data.pilot43.program import (ExecError, Node, Program,
                                                ProgramError, Ref, execute,
                                                gold_calls, replay_identical,
                                                validate_semantic_edges,
                                                validate_structure)

OPS = build_ops()
REF_RE = re.compile(r"^\$var_(\d+)\.([A-Za-z0-9_]+)\$$")


def program(nodes: Sequence[Tuple[str, str, Dict[str, Any]]], sink: str,
            sems: Dict[str, Dict[str, str]] | None = None) -> Program:
    sems = sems or {}
    return Program(nodes=[Node(nid, op, dict(args), dict(sems.get(nid, {})))
                          for nid, op, args in nodes], sink=sink)


def kinds_of(prog: Program) -> Dict[str, str]:
    values, _answer = execute(prog)
    return {nid: st.value_kind(v) for nid, v in values.items()}


# ---------------------------------------------------------------------------
# the graph zoo
# ---------------------------------------------------------------------------
LINEAR = program([("n1", "add", {"a": 2.0, "b": 3.0}),
                  ("n2", "square", {"a": Ref("n1")}),
                  ("n3", "sqrt", {"a": Ref("n2")})], "n3")

FAN_IN = program([("n1", "add", {"a": 2.0, "b": 3.0}),
                  ("n2", "multiply", {"a": 4.0, "b": 6.0}),
                  ("n3", "subtract", {"a": Ref("n1"), "b": Ref("n2")})], "n3")

FAN_IN_3 = program([("n1", "add", {"a": 2.0, "b": 3.0}),
                    ("n2", "multiply", {"a": 4.0, "b": 6.0}),
                    ("n3", "subtract", {"a": 9.0, "b": 1.0}),
                    ("n4", "sum_three", {"a": Ref("n1"), "b": Ref("n2"),
                                         "c": Ref("n3")})], "n4")

DIAMOND = program([("n1", "add", {"a": 2.0, "b": 3.0}),
                   ("n2", "square", {"a": Ref("n1")}),
                   ("n3", "sqrt", {"a": Ref("n1")}),
                   ("n4", "multiply", {"a": Ref("n2"), "b": Ref("n3")})], "n4")

LATE = program([("n1", "add", {"a": 2.0, "b": 3.0}),
                ("n2", "square", {"a": Ref("n1")}),
                ("n3", "sqrt", {"a": Ref("n2")}),
                ("n4", "round_to_int", {"a": Ref("n3")}),
                ("n5", "add", {"a": Ref("n1"), "b": Ref("n4")})], "n5")

MULTI_JOIN = program([("n1", "add", {"a": 2.0, "b": 3.0}),
                      ("n2", "multiply", {"a": 4.0, "b": 6.0}),
                      ("n3", "subtract", {"a": Ref("n1"), "b": Ref("n2")}),
                      ("n4", "square", {"a": 3.0}),
                      ("n5", "add", {"a": Ref("n3"), "b": Ref("n4")})], "n5")

TWO_STAGE = program([("n1", "sum_values", {"values": [1.0, 2.0, 3.0, 4.0]}),
                     ("n2", "sum_values", {"values": [5.0, 6.0, 7.0]}),
                     ("n3", "add", {"a": Ref("n1"), "b": Ref("n2")})], "n3")

REPEATED = program([("n1", "add", {"a": 2.0, "b": 3.0}),
                    ("n2", "add", {"a": Ref("n1"), "b": 7.0})], "n2")

TYPE_CHAIN = program([("n1", "sum_values", {"values": [1.5, 2.5, 3.5]}),
                      ("n2", "format_fixed", {"a": Ref("n1"), "places": 2}),
                      ("n3", "text_length", {"text": Ref("n2")})], "n3")

ZOO = {"LINEAR": LINEAR, "FAN_IN": FAN_IN, "FAN_IN_3": FAN_IN_3,
       "DIAMOND": DIAMOND, "LATE": LATE, "MULTI_JOIN": MULTI_JOIN,
       "TWO_STAGE": TWO_STAGE, "REPEATED": REPEATED, "TYPE_CHAIN": TYPE_CHAIN}


@pytest.mark.parametrize("name", sorted(ZOO))
def test_every_fixture_program_is_structurally_and_semantically_valid(name: str):
    prog = ZOO[name]
    validate_structure(prog)
    assert validate_semantic_edges(prog) == []


# ---------------------------------------------------------------------------
# structural validation
# ---------------------------------------------------------------------------
def test_reference_to_a_later_node_is_rejected():
    prog = program([("n1", "add", {"a": Ref("n2"), "b": 1.0}),
                    ("n2", "add", {"a": 1.0, "b": 2.0})], "n1")
    with pytest.raises(ProgramError, match="non-topological"):
        validate_structure(prog)


def test_reference_to_a_missing_node_is_rejected():
    prog = program([("n1", "add", {"a": 1.0, "b": 2.0}),
                    ("n2", "square", {"a": Ref("nowhere")})], "n2")
    with pytest.raises(ProgramError, match="nowhere"):
        validate_structure(prog)


def test_wrong_arity_is_rejected():
    prog = program([("n1", "add", {"a": 1.0})], "n1")
    with pytest.raises(ProgramError, match="args"):
        validate_structure(prog)


def test_unknown_op_is_rejected():
    prog = program([("n1", "not_a_real_op", {"a": 1.0})], "n1")
    with pytest.raises(ProgramError, match="unknown op"):
        validate_structure(prog)


def test_sink_outside_the_node_list_is_rejected():
    prog = program([("n1", "add", {"a": 1.0, "b": 2.0})], "n9")
    with pytest.raises(ProgramError, match="sink is not a node"):
        validate_structure(prog)


def test_duplicate_node_id_is_rejected():
    prog = program([("n1", "add", {"a": 1.0, "b": 2.0}),
                    ("n1", "square", {"a": 3.0})], "n1")
    with pytest.raises(ProgramError, match="duplicate node id"):
        validate_structure(prog)


def test_node_with_no_path_to_the_sink_is_rejected():
    prog = program([("n1", "add", {"a": 1.0, "b": 2.0}),
                    ("n2", "square", {"a": 3.0})], "n2")
    with pytest.raises(ProgramError, match="no path to the sink"):
        validate_structure(prog)


def test_semantic_edge_gate_rejects_money_feeding_a_percentage_parameter():
    prog = program(
        [("n1", "apply_tax", {"amount": 100.0, "tax_percent": 20.0}),
         ("n2", "apply_tax", {"amount": 500.0, "tax_percent": Ref("n1")})],
        "n2",
        sems={"n1": {"amount": st.MONEY, "tax_percent": st.PERCENTAGE},
              "n2": {"amount": st.MONEY}})
    validate_structure(prog)                    # structurally fine on purpose
    errs = validate_semantic_edges(prog)
    assert errs and "Money -> Percentage" in errs[0]


def test_semantic_edge_gate_accepts_the_same_shape_with_a_percentage_producer():
    prog = program(
        [("n1", "share_percent", {"part": 40.0, "total": 200.0}),
         ("n2", "apply_tax", {"amount": 500.0, "tax_percent": Ref("n1")})],
        "n2", sems={"n2": {"amount": st.MONEY}})
    validate_structure(prog)
    assert validate_semantic_edges(prog) == []


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(ZOO))
def test_execution_is_deterministic(name: str):
    assert replay_identical(ZOO[name], n=3)


def test_execute_returns_one_observation_per_node_plus_the_sink_answer():
    values, answer = execute(DIAMOND)
    assert sorted(values) == ["n1", "n2", "n3", "n4"]
    assert answer == values["n4"]


def test_repeated_power_overflows_into_an_exec_error():
    prog = program([("n1", "power", {"a": 900.0, "b": 40.0}),
                    ("n2", "power", {"a": Ref("n1"), "b": 40.0})], "n2")
    with pytest.raises(ExecError, match="overflow"):
        execute(prog)


def test_division_by_zero_becomes_an_exec_error():
    prog = program([("n1", "divide", {"a": 12.0, "b": 0.0})], "n1")
    with pytest.raises(ExecError):
        execute(prog)


def test_empty_list_result_is_degenerate():
    prog = program([("n1", "filter_above", {"values": [1.0, 2.0, 3.0],
                                            "threshold": 10_000.0})], "n1")
    with pytest.raises(ExecError, match="degenerate list"):
        execute(prog)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_node_values_are_rejected(value: float):
    from targeted_tool_data.pilot43.program import _check

    assert math.isnan(value) or math.isinf(value)
    with pytest.raises(ExecError, match="NaN/Inf"):
        _check(value, "n1")


# ---------------------------------------------------------------------------
# gold calls
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("track", ["A_NATIVE", "G_GENERAL_1"])
def test_gold_calls_shape_and_reference_resolution(track: str):
    prog = DIAMOND
    calls = gold_calls(prog, track)
    assert len(calls) == len(prog.nodes)
    assert [c["node_id"] for c in calls] == [nd.node_id for nd in prog.nodes]
    assert [c["call_index"] for c in calls] == [1, 2, 3, 4]
    for i, call in enumerate(calls):
        op = OPS[call["primitive_id"]]
        surf = op.surface(track)
        assert call["name"] == surf.name
        assert surf.track == track
        assert call["capability"] == op.capability
        assert sorted(call["arguments"]) == sorted(surf.param_names)
        for value in call["arguments"].values():
            match = REF_RE.match(value) if isinstance(value, str) else None
            if match is None:
                continue
            producer = int(match.group(1)) - 1
            assert 0 <= producer < i, "reference must point at an earlier call"
            assert match.group(2) == OPS[calls[producer]["primitive_id"]] \
                .surface(track).output_field


def test_gold_calls_never_borrow_a_surface_from_another_track():
    calls = gold_calls(DIAMOND, "G_GENERAL_2")
    borrowed = [c["name"] for c in calls
                if OPS[c["primitive_id"]].surface("G_GENERAL_2").track
                != "G_GENERAL_2"]
    assert borrowed == []


def _edges_from_gold_calls(calls: Sequence[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    """Producer-side mirror of the independent audit's reconstruction."""
    by_label = {c["label"]: c["node_id"] for c in calls}
    edges: Set[Tuple[str, str]] = set()
    for call in calls:
        for value in call["arguments"].values():
            if not isinstance(value, str):
                continue
            match = REF_RE.match(value)
            if match is None:
                continue
            edges.add((by_label[f"$var_{match.group(1)}"], call["node_id"]))
    return edges


@pytest.mark.parametrize("name", sorted(ZOO))
def test_edges_rebuilt_from_gold_calls_equal_the_program_edges(name: str):
    prog = ZOO[name]
    rebuilt = _edges_from_gold_calls(gold_calls(prog, "A_NATIVE"))
    assert rebuilt == set(prog.edges())


# ---------------------------------------------------------------------------
# graph features
# ---------------------------------------------------------------------------
def test_features_of_the_diamond():
    f = features(DIAMOND)
    assert f.n_nodes == 4
    assert f.n_edges == 4
    assert f.indegree == {"n1": 0, "n2": 1, "n3": 1, "n4": 2}
    assert f.outdegree == {"n1": 2, "n2": 1, "n3": 1, "n4": 0}
    assert f.roots == ["n1"]
    assert f.leaves == ["n4"]
    assert f.depth == 3
    assert f.critical_path == ["n1", "n2", "n4"]
    assert f.join_nodes == ["n4"]
    assert f.fan_out_nodes == ["n1"]
    assert f.late_edges == []
    assert f.reference_distances == [1, 2, 2, 1]
    assert f.n_parallel_branches == 1


def test_features_of_the_late_reference_graph():
    f = features(LATE)
    assert (f.n_nodes, f.n_edges, f.depth) == (5, 5, 5)
    assert f.critical_path == ["n1", "n2", "n3", "n4", "n5"]
    assert f.reference_distances == [1, 1, 1, 4, 1]
    assert f.late_edges == [("n1", "n5")]
    assert f.as_dict()["max_reference_distance"] == 4
    assert f.as_dict()["mean_reference_distance"] == 1.6


def test_features_of_the_multi_join_graph():
    f = features(MULTI_JOIN)
    assert f.join_nodes == ["n3", "n5"]
    assert sorted(f.roots) == ["n1", "n2", "n4"]
    assert f.n_parallel_branches == 3
    assert f.as_dict()["n_multi_parent_nodes"] == 2


def test_type_transitions_are_counted_from_observed_value_kinds():
    f = features(TYPE_CHAIN, kinds_of(TYPE_CHAIN))
    assert f.n_type_transitions == 2
    flat = features(LINEAR, kinds_of(LINEAR))
    assert flat.n_type_transitions == 0


# ---------------------------------------------------------------------------
# pattern invariants: one positive and one negative graph each
# ---------------------------------------------------------------------------
PATTERN_CASES = [
    ("LINEAR_CHAIN", "LINEAR", "DIAMOND"),
    ("FAN_IN_SINGLE", "FAN_IN", "LINEAR"),
    ("FAN_IN_MULTIPLE", "FAN_IN_3", "FAN_IN"),
    ("FAN_OUT", "DIAMOND", "FAN_IN"),
    ("DIAMOND", "DIAMOND", "FAN_IN"),
    ("PARALLEL_THEN_MERGE", "FAN_IN", "LINEAR"),
    ("REUSE_EARLY_OUTPUT", "DIAMOND", "FAN_IN"),
    ("LATE_REFERENCE", "LATE", "LINEAR"),
    ("TWO_STAGE_AGGREGATION", "TWO_STAGE", "LINEAR"),
    ("MULTI_JOIN", "MULTI_JOIN", "FAN_IN"),
    ("REPEATED_PRIMITIVE", "REPEATED", "LINEAR"),
    ("TYPE_TRANSITION_CHAIN", "TYPE_CHAIN", "LINEAR"),
    ("NESTED_AGGREGATION", "TWO_STAGE", "FAN_IN"),
]


@pytest.mark.parametrize("pattern,positive,negative", PATTERN_CASES)
def test_pattern_invariant(pattern: str, positive: str, negative: str):
    good, bad = ZOO[positive], ZOO[negative]
    assert pattern in satisfied_patterns(good, kinds_of(good))
    assert pattern not in satisfied_patterns(bad, kinds_of(bad))


#: the classifier is a set, so the full set is the real regression lock
EXPECTED_SETS = {
    "LINEAR": {"LINEAR_CHAIN"},
    "FAN_IN": {"FAN_IN_SINGLE", "MIXED_INDEPENDENT_DEPENDENT",
               "PARALLEL_THEN_MERGE"},
    "FAN_IN_3": {"FAN_IN_MULTIPLE", "LATE_REFERENCE",
                 "MIXED_INDEPENDENT_DEPENDENT", "PARALLEL_THEN_MERGE"},
    "DIAMOND": {"DIAMOND", "FAN_IN_SINGLE", "FAN_OUT", "REUSE_EARLY_OUTPUT"},
    "LATE": {"DIAMOND", "FAN_IN_SINGLE", "FAN_OUT", "LATE_REFERENCE",
             "REPEATED_PRIMITIVE", "REUSE_EARLY_OUTPUT",
             "TYPE_TRANSITION_CHAIN"},
    "MULTI_JOIN": {"ALTERNATING_BRANCH_CHAIN", "FAN_IN_MULTIPLE",
                   "MIXED_INDEPENDENT_DEPENDENT", "MULTI_JOIN",
                   "NESTED_AGGREGATION", "PARALLEL_THEN_MERGE",
                   "REPEATED_PRIMITIVE", "TWO_STAGE_AGGREGATION"},
    "TWO_STAGE": {"FAN_IN_SINGLE", "MIXED_INDEPENDENT_DEPENDENT",
                  "NESTED_AGGREGATION", "PARALLEL_THEN_MERGE",
                  "REPEATED_PRIMITIVE", "TWO_STAGE_AGGREGATION"},
    "REPEATED": {"LINEAR_CHAIN", "REPEATED_PRIMITIVE"},
    "TYPE_CHAIN": {"LINEAR_CHAIN", "TYPE_TRANSITION_CHAIN"},
}


@pytest.mark.parametrize("name", sorted(EXPECTED_SETS))
def test_satisfied_pattern_set_is_exact(name: str):
    prog = ZOO[name]
    assert satisfied_patterns(prog, kinds_of(prog)) == EXPECTED_SETS[name]


@pytest.mark.parametrize("name", sorted(ZOO))
def test_the_independent_audit_reaches_the_same_verdict_from_the_export(name: str):
    """The two classifiers must agree, or the export can never pass its audit.

    The audit sees only what the record carries: call names, arguments and the
    observation each call produced. It re-implements the invariants separately,
    so this is the test that catches the two definitions drifting apart.
    """
    from analysis.pilot43_independent_audit.graph_recon import reconstruct
    from analysis.pilot43_independent_audit.pattern_rules import (
        satisfied_patterns as audit_patterns)

    prog = ZOO[name]
    observations, _answer = execute(prog)
    calls = gold_calls(prog, "A_NATIVE", observations)
    graph = reconstruct(calls)
    audited = audit_patterns(graph, [st.value_kind(c["observation"]) for c in calls])
    assert audited == satisfied_patterns(prog, kinds_of(prog))


def test_primary_pattern_takes_the_most_specific_satisfied_invariant():
    assert primary_pattern(EXPECTED_SETS["MULTI_JOIN"]) == "NESTED_AGGREGATION"
    assert primary_pattern(EXPECTED_SETS["LINEAR"]) == "LINEAR_CHAIN"
    assert primary_pattern(set()) == "UNCLASSIFIED"
