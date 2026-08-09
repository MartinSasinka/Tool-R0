"""Node necessity, the V4 shortcut gate, distractors and counterfactual sets.

These four gates share one design rule: a verdict that rests on a single input
is not believed. Each test therefore builds an explicit counterfactual set and
checks that the gate uses it -- including the false-positive cases that a
single-input check would get wrong.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, Tuple

import pytest

from targeted_tool_data.pilot43 import semtypes as st
from targeted_tool_data.pilot43.blueprints import (Blueprint, Plan, Role, Step,
                                                   all_blueprints)
from targeted_tool_data.pilot43.build import BuildError, Instance, instantiate
from targeted_tool_data.pilot43.counterfactuals import (LOW_ENTROPY_N,
                                                        as_fact_pairs,
                                                        as_programs,
                                                        counterfactual_instances)
from targeted_tool_data.pilot43.distractors import (behaviourally_wrong,
                                                    build_offered_tools)
from targeted_tool_data.pilot43.necessity import (all_nodes_necessary,
                                                  necessity_summary,
                                                  node_necessity)
from targeted_tool_data.pilot43.ops import build_ops
from targeted_tool_data.pilot43.program import Node, Program, Ref
from targeted_tool_data.pilot43.v4 import V4Config, v4_gate

OPS = build_ops()
BLUEPRINTS = all_blueprints()
PAIRS: List[Tuple[Blueprint, Plan]] = [(bp, plan) for bp in BLUEPRINTS
                                       for plan in bp.plans]
TRACK = "A_NATIVE"
G = st.GENERIC

NECESSITY_KEYS = {"node_id", "removal_executable", "removal_changes_answer",
                  "target_unreachable", "alternative_binding_found", "necessary"}


def program(nodes: Sequence[Tuple[str, str, Dict[str, Any]]],
            sink: str) -> Program:
    return Program(nodes=[Node(nid, op, dict(args)) for nid, op, args in nodes],
                   sink=sink)


# ---------------------------------------------------------------------------
# node necessity
# ---------------------------------------------------------------------------
def redundant_program(amount: float) -> Program:
    """``n1`` adds zero, so ``n2`` would get the same input without it."""
    return program([("n1", "add", {"a": amount, "b": 0.0}),
                    ("n2", "multiply", {"a": Ref("n1"), "b": 3.0})], "n2")


def chained_program(amount: float) -> Program:
    return program([("n1", "add", {"a": amount, "b": 5.0}),
                    ("n2", "multiply", {"a": Ref("n1"), "b": 3.0}),
                    ("n3", "subtract", {"a": Ref("n2"), "b": 4.0})], "n3")


def boolean_program(amount: float) -> Program:
    return program([("n1", "add", {"a": amount, "b": 10.0}),
                    ("n2", "is_greater", {"a": Ref("n1"), "b": 50.0})], "n2")


REDUNDANT_CFS = [redundant_program(v) for v in (12.0, 55.0, 71.0, 90.0)]
CHAIN_CFS = [chained_program(v) for v in (2.0, 9.0, 40.0, 61.0)]
BOOLEAN_CFS = [boolean_program(v) for v in (45.0, 41.0, 300.0, 39.0)]


def test_a_node_that_adds_zero_is_reported_unnecessary():
    rows = node_necessity(redundant_program(100.0), counterfactuals=REDUNDANT_CFS)
    by_id = {r["node_id"]: r for r in rows}
    assert by_id["n1"]["necessary"] is False
    assert by_id["n1"]["removal_executable"] is True
    assert by_id["n1"]["removal_changes_answer"] is False
    assert by_id["n1"]["counterfactuals_tested"] == len(REDUNDANT_CFS)
    assert (by_id["n1"]["counterfactuals_agreeing_with_bypass"]
            == by_id["n1"]["counterfactuals_tested"])
    assert by_id["n2"]["necessary"] is True
    assert necessity_summary(rows)["unnecessary_nodes"] == ["n1"]


def test_every_node_of_a_genuine_chain_is_necessary():
    rows = node_necessity(chained_program(7.0), counterfactuals=CHAIN_CFS)
    assert all_nodes_necessary(rows)
    assert [r["node_id"] for r in rows] == ["n1", "n2", "n3"]
    assert all(r["removal_changes_answer"] for r in rows)
    assert necessity_summary(rows)["necessary_nodes"] == 3


@pytest.mark.parametrize("factory", [redundant_program, chained_program,
                                     boolean_program])
def test_every_record_carries_the_full_evidence_schema(factory):
    rows = node_necessity(factory(100.0), counterfactuals=[factory(12.0),
                                                           factory(31.0)])
    assert rows
    for row in rows:
        assert NECESSITY_KEYS <= set(row)
        assert isinstance(row["necessary"], bool)
        assert isinstance(row["target_unreachable"], bool)
        assert isinstance(row["alternative_binding_found"], bool)


def test_a_bypass_that_only_survives_the_current_input_is_not_believed():
    """On this instance dropping ``n1`` keeps the answer; on others it does not."""
    from targeted_tool_data.pilot43.necessity import _run, _without

    gold = boolean_program(100.0)
    single_input_bypass = _without(gold, "n1", 100.0)
    assert _run(single_input_bypass) == _run(gold) is True

    rows = {r["node_id"]: r for r in node_necessity(gold,
                                                    counterfactuals=BOOLEAN_CFS)}
    assert rows["n1"]["removal_executable"] is True
    assert rows["n1"]["counterfactuals_tested"] >= 2
    assert (rows["n1"]["counterfactuals_agreeing_with_bypass"]
            < rows["n1"]["counterfactuals_tested"])
    assert rows["n1"]["necessary"] is True


def test_without_counterfactuals_the_gate_stays_fail_closed():
    rows = node_necessity(redundant_program(100.0))
    assert all(r["necessary"] for r in rows)


def test_alternative_binding_is_recorded_without_making_a_node_unnecessary():
    rows = node_necessity(chained_program(7.0), counterfactuals=CHAIN_CFS,
                          allowed_ops=["add", "multiply", "subtract",
                                       "average_two", "max_two"])
    by_id = {r["node_id"]: r for r in rows}
    assert by_id["n1"]["alternative_binding_found"] is False
    assert all(r["necessary"] for r in rows)


# ---------------------------------------------------------------------------
# V4
# ---------------------------------------------------------------------------
GUARANTEE_KEYS = {"v4_executed", "answer_type_checked", "search_space",
                  "search_complete", "resolved", "has_shortcut",
                  "minimal_valid_call_count", "counterfactuals_mixed"}


def test_a_real_numeric_shortcut_is_found_and_is_shorter():
    facts = {"a": (10.0, G), "b": (5.0, G), "zero": (0.0, G)}
    cfs = [({"a": 4.0, "b": 6.0, "zero": 0.0}, 10.0),
           ({"a": 7.0, "b": 1.0, "zero": 0.0}, 8.0),
           ({"a": 20.0, "b": 3.0, "zero": 0.0}, 23.0)]
    res = v4_gate(facts, ["add", "multiply", "subtract"], 15.0, 2, cfs)
    assert res["has_shortcut"] is True
    assert res["n_confirmed"] >= 1
    assert res["minimal_valid_call_count"] < 2
    shortest = min(res["confirmed_shortcuts"], key=lambda h: h["calls"])
    assert shortest["calls"] == 1
    assert shortest["rendered"] == "add($a, $b)"
    assert res["safe_for_core_train"] is False


def test_a_task_without_a_shortcut_is_resolved_and_safe():
    facts = {"a": (3.0, G), "b": (7.0, G), "c": (5.0, G)}
    cfs = [({"a": 2.0, "b": 4.0, "c": 3.0}, 18.0),
           ({"a": 1.0, "b": 1.0, "c": 9.0}, 18.0),
           ({"a": 6.0, "b": 2.0, "c": 2.0}, 16.0)]
    res = v4_gate(facts, ["add", "multiply", "subtract", "divide"], 50.0, 2, cfs)
    assert res["has_shortcut"] is False
    assert res["resolved"] is True
    assert res["search_complete"] is True
    assert res["minimal_valid_call_count"] == 2
    assert res["search_space"]["max_depth_complete"] >= 1
    assert res["safe_for_core_train"] is True


def test_a_boolean_coincidence_is_reported_as_coincidental_not_confirmed():
    facts = {"a": (60.0, G), "b": (40.0, G), "limit": (50.0, G)}
    cfs = [({"a": 10.0, "b": 10.0, "limit": 50.0}, False),
           ({"a": 45.0, "b": 20.0, "limit": 50.0}, True),
           ({"a": 20.0, "b": 45.0, "limit": 50.0}, True)]
    res = v4_gate(facts, ["is_greater", "add"], True, 2, cfs)
    assert res["has_shortcut"] is False
    assert res["n_confirmed"] == 0
    assert res["n_coincidental"] >= 1
    rendered = {c["rendered"] for c in res["coincidental_matches"]}
    assert "is_greater($a, $b)" in rendered
    for row in res["coincidental_matches"]:
        assert row["counterfactuals_agreeing"] < row["counterfactuals_tested"]


def test_an_unmixed_counterfactual_set_leaves_the_verdict_unresolved():
    facts = {"a": (60.0, G), "b": (40.0, G)}
    cfs = [({"a": 10.0, "b": 10.0}, True), ({"a": 1.0, "b": 2.0}, True)]
    res = v4_gate(facts, ["is_greater", "add"], True, 2, cfs,
                  counterfactuals_mixed=False)
    assert res["v4_executed"] is True
    assert res["resolved"] is False
    assert res["safe_for_core_train"] is False


def _instances_by_answer_type() -> Dict[str, Tuple[Blueprint, Plan, Instance]]:
    rng = random.Random(31)
    found: Dict[str, Tuple[Blueprint, Plan, Instance]] = {}
    for bp, plan in rng.sample(PAIRS, 400):
        if plan.call_count > 4:
            continue
        try:
            inst = instantiate(bp, plan, 8123, track=TRACK)
        except BuildError:
            continue
        found.setdefault(inst.answer_type, (bp, plan, inst))
    return found


BY_ANSWER_TYPE = _instances_by_answer_type()
ANSWER_TYPES = ("boolean", "category", "float", "integer", "list", "object",
                "string")


def test_every_answer_type_is_represented_in_the_sample():
    assert set(ANSWER_TYPES) <= set(BY_ANSWER_TYPE)


@pytest.mark.parametrize("answer_type", ANSWER_TYPES)
def test_v4_runs_for_every_answer_type_and_records_it(answer_type: str):
    bp, plan, inst = BY_ANSWER_TYPE[answer_type]
    cfs, meta = counterfactual_instances(bp, plan, answer_type=answer_type,
                                         track=TRACK, seed=909)
    facts = {name: (value, plan.role(name).sem)
             for name, value in inst.role_values.items()}
    offered = sorted({nd.op for nd in inst.program.nodes})
    res = v4_gate(facts, offered, inst.answer, inst.call_count,
                  as_fact_pairs(cfs), V4Config(depth_cap=2),
                  counterfactuals_mixed=meta["mixed"])
    assert res["v4_executed"] is True, "there is no skipped state"
    assert GUARANTEE_KEYS <= set(res)
    assert res["answer_type_checked"] == st.value_kind(inst.answer)
    space = res["search_space"]
    assert {"max_depth_requested", "max_depth_complete",
            "complete_to_gold_minus_one", "guarantee"} <= set(space)
    assert space["max_depth_complete"] >= 0
    assert isinstance(res["minimal_valid_call_count"], int)
    assert res["counterfactual_instances_tested"] == len(cfs)


# ---------------------------------------------------------------------------
# distractors
# ---------------------------------------------------------------------------
def _numeric_task() -> Tuple[Blueprint, Plan, Instance, List[Instance],
                             Dict[str, Any]]:
    rng = random.Random(4242)
    for bp, plan in rng.sample(PAIRS, 300):
        if plan.call_count not in (2, 3):
            continue
        try:
            inst = instantiate(bp, plan, 606, track=TRACK)
        except BuildError:
            continue
        if inst.answer_type not in ("float", "integer"):
            continue
        cfs, meta = counterfactual_instances(bp, plan,
                                             answer_type=inst.answer_type,
                                             track=TRACK, seed=77)
        facts = {n: (v, plan.role(n).sem) for n, v in inst.role_values.items()}
        base = v4_gate(facts, sorted({nd.op for nd in inst.program.nodes}),
                       inst.answer, inst.call_count, as_fact_pairs(cfs),
                       V4Config(depth_cap=2),
                       counterfactuals_mixed=meta["mixed"])
        if not base["has_shortcut"]:
            return bp, plan, inst, cfs, {"facts": facts, "meta": meta}
    raise AssertionError("no shortcut-free numeric task in the sample")


NUM_BP, NUM_PLAN, NUM_INST, NUM_CFS, NUM_EXTRA = _numeric_task()
OFFERED = build_offered_tools(NUM_INST.program, NUM_INST.answer, track=TRACK,
                              target_count=12, seed=3,
                              counterfactuals=as_programs(NUM_CFS)[:4])
GOLD_IDS = list(dict.fromkeys(nd.op for nd in NUM_INST.program.nodes))
DISTRACTOR_IDS = OFFERED["distractor_primitive_ids"]


def test_the_offered_set_actually_contains_distractors():
    assert DISTRACTOR_IDS
    assert OFFERED["offered_tool_count"] == len(GOLD_IDS) + len(DISTRACTOR_IDS)
    names = [t["name"] for t in OFFERED["tools"]]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("pid", DISTRACTOR_IDS)
def test_every_distractor_is_schema_compatible_with_a_gold_slot(pid: str):
    """It must be substitutable: same arity, and its parameters accept the values."""
    cand = OPS[pid]
    gold_arities = {OPS[gid].arity for gid in GOLD_IDS}
    assert cand.arity in gold_arities, f"{pid} fits no gold call"
    slots = [nd for nd in NUM_INST.program.nodes
             if OPS[nd.op].arity == cand.arity]
    assert slots
    for p in cand.params:
        assert p.sem in st.ALL
        assert p.runtime in ("number", "integer", "boolean", "string", "array",
                             "object")


@pytest.mark.parametrize("pid", DISTRACTOR_IDS)
def test_every_distractor_changes_the_answer_here_and_on_counterfactuals(pid: str):
    assert behaviourally_wrong(NUM_INST.program, pid, NUM_INST.answer,
                               as_programs(NUM_CFS)[:4])


#: a two-op arithmetic gold program always has same-signature siblings, so it is
#: the reliable fixture for the hardness contract
SYNTH_PROG = program([("n1", "add", {"a": 3.0, "b": 7.0}),
                      ("n2", "multiply", {"a": Ref("n1"), "b": 5.0})], "n2")
SYNTH_CFS = [program([("n1", "add", {"a": 2.0, "b": 4.0}),
                      ("n2", "multiply", {"a": Ref("n1"), "b": 3.0})], "n2")]
SYNTH_OFFERED = build_offered_tools(SYNTH_PROG, 50.0, track=TRACK,
                                    target_count=12, seed=5,
                                    counterfactuals=SYNTH_CFS)


def test_hard_distractors_share_a_gold_runtime_signature():
    from targeted_tool_data.pilot43.distractors import _signature

    gold_signatures = {_signature(OPS[pid])
                       for pid in ("add", "multiply")}
    hard = [t["primitive_id"] for t in SYNTH_OFFERED["tools"]
            if t.get("distractor_hardness") == "hard"]
    assert len(hard) == SYNTH_OFFERED["hard_distractor_count"] >= 2
    for pid in hard:
        assert _signature(OPS[pid]) in gold_signatures
        assert OPS[pid].family in {OPS["add"].family, OPS["multiply"].family}
        assert OPS[pid].capability not in {OPS["add"].capability,
                                           OPS["multiply"].capability}


def test_distractor_hardness_counts_add_up():
    counts = sum(OFFERED[f"{k}_distractor_count"]
                 for k in ("hard", "medium", "easy"))
    assert counts == OFFERED["distractor_count"] == len(DISTRACTOR_IDS)


def test_no_distractor_opens_a_v4_shortcut():
    all_ids = [t["primitive_id"] for t in OFFERED["tools"]]
    res = v4_gate(NUM_EXTRA["facts"], all_ids, NUM_INST.answer,
                  NUM_INST.call_count, as_fact_pairs(NUM_CFS),
                  V4Config(depth_cap=2),
                  counterfactuals_mixed=NUM_EXTRA["meta"]["mixed"])
    assert res["has_shortcut"] is False, res["confirmed_shortcuts"][:2]


@pytest.mark.parametrize("track", ["A_NATIVE", "G_GENERAL_1", "G_GENERAL_2"])
def test_re_rendered_tools_still_cover_the_gold_calls_on_every_track(track: str):
    """Export re-derives tool surfaces, so gold call names must stay offered."""
    from targeted_tool_data.pilot43.distractors import rerender_tools
    from targeted_tool_data.pilot43.program import gold_calls

    tools = rerender_tools(OFFERED["tools"], track)
    offered_names = {t["name"] for t in tools}
    for call in gold_calls(NUM_INST.program, track):
        assert call["name"] in offered_names
        tool = next(t for t in tools if t["name"] == call["name"])
        assert sorted(call["arguments"]) == sorted(p["name"]
                                                   for p in tool["parameters"])
    assert {t["primitive_id"] for t in tools} == {t["primitive_id"]
                                                 for t in OFFERED["tools"]}
    assert all(t["surface_track"] == track for t in tools)


def test_re_rendering_preserves_the_distractor_hardness_labels():
    from targeted_tool_data.pilot43.distractors import rerender_tools

    before = {t["primitive_id"]: t.get("distractor_hardness")
              for t in OFFERED["tools"]}
    after = {t["primitive_id"]: t.get("distractor_hardness")
             for t in rerender_tools(OFFERED["tools"], "G_GENERAL_2")}
    assert before == after


def test_an_answer_preserving_alias_is_rejected_as_a_distractor():
    """``add`` in an ``add`` slot leaves the answer intact, so it is not offered."""
    prog = program([("n1", "add", {"a": 3.0, "b": 7.0}),
                    ("n2", "multiply", {"a": Ref("n1"), "b": 5.0})], "n2")
    assert behaviourally_wrong(prog, "add", 50.0) is False
    assert behaviourally_wrong(prog, "subtract", 50.0) is True


# ---------------------------------------------------------------------------
# counterfactual sets
# ---------------------------------------------------------------------------
def _boolean_plans(n: int) -> List[Tuple[Blueprint, Plan]]:
    rng = random.Random(7)
    out: List[Tuple[Blueprint, Plan]] = []
    for bp, plan in rng.sample(PAIRS, 300):
        try:
            inst = instantiate(bp, plan, 4242, track=TRACK)
        except BuildError:
            continue
        if inst.answer_type == "boolean":
            out.append((bp, plan))
        if len(out) == n:
            break
    return out


BOOLEAN_PLANS = _boolean_plans(8)


def test_boolean_plans_were_found():
    assert len(BOOLEAN_PLANS) == 8


@pytest.mark.parametrize("bp,plan", BOOLEAN_PLANS,
                         ids=[f"{b.workflow_id}/{p.plan_id}"
                              for b, p in BOOLEAN_PLANS])
def test_boolean_counterfactual_sets_contain_both_labels(bp: Blueprint,
                                                         plan: Plan):
    insts, meta = counterfactual_instances(bp, plan, answer_type="boolean",
                                           track=TRACK, seed=555)
    answers = {inst.answer for inst in insts}
    assert answers == {True, False}
    assert meta["built"] >= LOW_ENTROPY_N // 2
    assert meta["minority_share"] >= 0.25
    assert meta["mixed"] is True
    assert meta["weak"] is False
    assert meta["low_entropy_answer"] is True


CONSTANT_BLUEPRINT = Blueprint(
    workflow_id="test.constant_ratio", domain="test",
    natural_user_goal="check a self-ratio", target_description="always one",
    value_generator_id="test.constant", query_asset_family="test",
    plans=(Plan(plan_id="const.v1",
                roles=(Role("amount", "generic_value", "the amount"),),
                steps=(Step("n1", "arithmetic.divide", ("amount", "amount"),
                            "the amount over itself"),),
                sink="n1"),))


def test_a_plan_with_only_one_possible_answer_is_reported_weak():
    plan = CONSTANT_BLUEPRINT.plans[0]
    insts, meta = counterfactual_instances(CONSTANT_BLUEPRINT, plan,
                                           answer_type="float", track=TRACK,
                                           seed=11)
    assert {inst.answer for inst in insts} == {1.0}
    assert meta["distinct_answers"] == 1
    assert meta["mixed"] is False
    assert meta["weak"] is True
