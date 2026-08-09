"""The dev-200 pattern mirror must not drift away from ``patterns.py``.

Failure mode prevented: ``dev_patterns`` silently diverging from the Pilot4.3
classifier, which would put the target profile and the produced dataset back on
two different pattern vocabularies - the exact apples-to-oranges comparison the
module exists to remove.
"""
from __future__ import annotations

from typing import Any, Dict, List

from targeted_tool_data.pilot43 import blueprints as B
from targeted_tool_data.pilot43 import build as BD
from targeted_tool_data.pilot43 import dev_patterns as DP
from targeted_tool_data.pilot43 import patterns as PT
from targeted_tool_data.pilot43 import profile as PROF
from targeted_tool_data.pilot43 import semtypes as st
from targeted_tool_data.pilot43.ops import build_ops
from targeted_tool_data.pilot43.program import gold_calls, observation_types

#: TYPE_TRANSITION_CHAIN depends on executed values on one side and on declared
#: schema types on the other, so it is compared separately, not here.
_KIND_DEPENDENT = {"TYPE_TRANSITION_CHAIN"}


def _schema_tools(prog: Any, track: str) -> List[Dict[str, Any]]:
    """Dev-shaped tool schemas for a Pilot4.3 program's own ops."""
    ops = build_ops()
    out: List[Dict[str, Any]] = []
    for node in prog.nodes:
        op = ops[node.op]
        surface = op.surface(track)
        params = {
            shown: {"type": "array" if p.sem in st.COLLECTIONS else "number"}
            for p, shown in zip(op.params, surface.param_names)
        }
        collection_out = op.out_sem != "@preserve" and op.out_sem in st.COLLECTIONS
        out.append({
            "name": surface.name,
            "parameters": params,
            "output_parameters": {
                surface.output_field: {
                    "type": "array" if collection_out else "number"}},
        })
    return out


def _instances(limit_per_blueprint: int = 1):
    for blueprint in B.all_blueprints():
        for plan in blueprint.plans[:limit_per_blueprint]:
            try:
                yield BD.instantiate(blueprint, plan, 4242, track="A_NATIVE")
            except BD.BuildError:
                continue


def test_mirror_agrees_with_pilot43_classifier() -> None:
    checked = 0
    for inst in _instances():
        program = inst.program
        reference = PT.satisfied_patterns(program, observation_types(program))
        graph = DP.reconstruct(gold_calls(program, inst.track),
                               _schema_tools(program, inst.track))
        assert graph.n == len(program.nodes)
        assert len(graph.edges) == len(program.edges())
        assert reference - _KIND_DEPENDENT == DP.satisfied_patterns(
            graph) - _KIND_DEPENDENT
        checked += 1
    assert checked > 20


def test_dev_conditionals_reproduce_the_frozen_v2_call_count_distribution() -> None:
    if not PROF.DEFAULT_DEV_ROWS.is_file():
        return
    rows = PROF.read_dev_rows(PROF.DEFAULT_DEV_ROWS)
    structural = PROF.dev_conditionals(rows)
    assert structural["n_rows"] == 200
    assert set(structural["P(primary_pattern|call_count)"]) == {
        "2", "3", "4", "5", "6+"}
    for bucket, dist in structural["P(primary_pattern|call_count)"].items():
        assert abs(sum(dist.values()) - 1.0) < 1e-3, bucket
        assert set(dist) <= set(PT.PATTERN_PRIORITY) | {"UNCLASSIFIED"}
