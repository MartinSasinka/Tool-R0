"""Core semantic layer: registry, graph, executor."""
import random

import pytest

from targeted_tool_data import registry as reg
from targeted_tool_data.executor import (ExecutionError, execute,
                                         question_constants, replay_consistent)
from targeted_tool_data.graph import (argument_skeleton, build_program,
                                      graph_template_id, is_acyclic,
                                      program_family)
from targeted_tool_data.schemas import GenerationCell


def _cell(cc=3, motif="linear", track="A", ns=False):
    return GenerationCell(
        generation_cell_id=f"t_{cc}call_{motif}", track=track,
        mode="adaptation" if track == "A" else "generalization",
        call_count=cc, motif=motif, target_skill="s", target_failure="f",
        numeric_string=ns, hard_distractor_type="near_semantics")


def test_registry_has_typed_diversity():
    prims = reg.all_primitives()
    assert len(prims) >= 30
    out_types = {p.out_type for p in prims.values()}
    assert {"number", "integer", "string"} <= out_types
    param_types = {t for p in prims.values() for _n, t, _s in p.params}
    assert "array" in param_types
    assert any(t.startswith("enum:") for t in param_types)


def test_registry_surfaces_both_tracks():
    for sid, p in reg.all_primitives().items():
        assert p.surfaces_a, sid
        assert p.surfaces_g, sid
        a_names = {s.name for s in p.surfaces_a}
        g_names = {s.name for s in p.surfaces_g}
        assert not a_names & g_names, f"{sid}: A/G surface names overlap"


def test_registry_hash_stable():
    assert reg.registry_hash() == reg.registry_hash()


@pytest.mark.parametrize("motif,cc", [("linear", 2), ("linear", 4),
                                      ("fan_in", 3), ("fan_in", 5),
                                      ("branch_aggregate", 4),
                                      ("selection", 5)])
def test_build_program_exact_call_count_and_acyclic(motif, cc):
    rng = random.Random(42)
    for _ in range(20):
        prog = build_program(_cell(cc, motif), rng)
        assert len(prog.nodes) == cc
        assert is_acyclic(prog)


def test_deterministic_generation_same_seed():
    p1 = build_program(_cell(4, "fan_in"), random.Random(7))
    p2 = build_program(_cell(4, "fan_in"), random.Random(7))
    assert p1.model_dump() == p2.model_dump()


def test_execute_typed_and_replay():
    rng = random.Random(3)
    for _ in range(30):
        prog = build_program(_cell(3, "linear"), rng)
        try:
            obs, ans = execute(prog)
        except ExecutionError:
            continue
        assert len(obs) == 3
        assert replay_consistent(prog, n=3)


def test_reference_resolution_in_lists():
    rng = random.Random(11)
    prog = build_program(_cell(4, "branch_aggregate"), rng)
    obs, ans = execute(prog)
    # sink aggregates the three branch outputs
    assert obs[-1] == ans


def test_numeric_string_program():
    rng = random.Random(5)
    found = False
    for _ in range(40):
        prog = build_program(_cell(3, "linear", ns=True), rng)
        sids = [n.semantic_id for n in prog.nodes]
        if "parse_number" in sids or "format_fixed" in sids or "number_to_string" in sids:
            obs, ans = execute(prog)
            found = True
            break
    assert found


def test_no_decorative_nodes():
    rng = random.Random(9)
    for _ in range(30):
        prog = build_program(_cell(5, "fan_in"), rng)
        # every node reachable to sink is asserted inside build_program;
        # double check consumers here
        consumed = set()
        from targeted_tool_data.graph import _refs_in
        for nd in prog.nodes:
            for v in nd.inputs.values():
                consumed.update(_refs_in(v))
        non_sink = {n.node_id for n in prog.nodes} - {prog.sink}
        assert non_sink <= consumed


def test_family_and_template_ids_deterministic():
    prog = build_program(_cell(3, "linear"), random.Random(1))
    assert program_family(prog) == program_family(prog)
    assert graph_template_id(prog).startswith("gt_")
    assert argument_skeleton(prog) == argument_skeleton(prog)


def test_question_constants_extracted():
    prog = build_program(_cell(3, "linear"), random.Random(2))
    consts = question_constants(prog)
    assert consts
