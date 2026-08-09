"""Instantiation invariants, workflow-first provenance and value realism.

The central claim of Pilot4.3 is that the workflow plan *generates* the program
rather than labelling it, so the provenance tests below compare the built graph
against the plan step by step and edge by edge.
"""
from __future__ import annotations

import random
import re
from datetime import date
from typing import Any, Dict, List, Sequence, Tuple

import pytest

from targeted_tool_data.pilot43 import semtypes as st
from targeted_tool_data.pilot43 import values as V
from targeted_tool_data.pilot43.blueprints import Blueprint, Plan, all_blueprints
from targeted_tool_data.pilot43.build import BuildError, Instance, instantiate
from targeted_tool_data.pilot43.ops import build_ops
from targeted_tool_data.pilot43.patterns import satisfied_patterns
from targeted_tool_data.pilot43.program import Ref, execute, semantic_types

OPS = build_ops()
BLUEPRINTS = all_blueprints()
PAIRS: List[Tuple[Blueprint, Plan]] = [(bp, plan) for bp in BLUEPRINTS
                                       for plan in bp.plans]
TRACK = "A_NATIVE"
SAMPLE_SEED = 4242


def _spread_sample(n: int, seed: int) -> List[Tuple[Blueprint, Plan]]:
    """``n`` (blueprint, plan) pairs taken round-robin across workflow families."""
    rng = random.Random(seed)
    by_family: Dict[str, List[Tuple[Blueprint, Plan]]] = {}
    for bp, plan in PAIRS:
        by_family.setdefault(bp.family, []).append((bp, plan))
    for rows in by_family.values():
        rng.shuffle(rows)
    families = sorted(by_family)
    out: List[Tuple[Blueprint, Plan]] = []
    depth = 0
    while len(out) < n:
        added = False
        for family in families:
            rows = by_family[family]
            if depth < len(rows):
                out.append(rows[depth])
                added = True
                if len(out) == n:
                    break
        if not added:
            break
        depth += 1
    return out


PLAN_SAMPLE = _spread_sample(60, SAMPLE_SEED)
BIG_SAMPLE = _spread_sample(140, SAMPLE_SEED + 1)


def _ids(pairs: Sequence[Tuple[Blueprint, Plan]]) -> List[str]:
    return [f"{bp.workflow_id}/{plan.plan_id}" for bp, plan in pairs]


def _literal_args(inst: Instance) -> List[str]:
    return [repr(v) for nd in inst.program.nodes for v in nd.args.values()
            if not isinstance(v, Ref)]


def _plan_role_args(inst: Instance, plan: Plan) -> List[str]:
    return [repr(inst.role_values[arg]) for step in plan.steps
            for arg in step.args if not arg.startswith("@")]


def _declared_edges(plan: Plan) -> set[Tuple[str, str]]:
    return {(ref, step.node_id) for step in plan.steps
            for ref in dict.fromkeys(step.refs())}


# ---------------------------------------------------------------------------
# instantiation invariants
# ---------------------------------------------------------------------------
def test_the_sample_is_wide_and_mostly_buildable():
    """Guards every ``except BuildError: return`` below from being vacuous."""
    assert len({bp.family for bp, _ in PLAN_SAMPLE}) >= 20
    built = sum(1 for bp, plan in PLAN_SAMPLE
                if _safe_answer_type(bp, plan) is not None)
    assert built >= 55, f"only {built}/60 sampled plans instantiated"


@pytest.mark.parametrize("bp,plan", PLAN_SAMPLE, ids=_ids(PLAN_SAMPLE))
def test_instantiate_either_raises_build_error_or_is_self_consistent(
        bp: Blueprint, plan: Plan):
    try:
        inst = instantiate(bp, plan, SAMPLE_SEED, track=TRACK)
    except BuildError:
        return                      # a declared, non-silent failure is allowed
    assert inst.call_count == plan.call_count == len(inst.program.nodes)

    kinds = {nid: st.value_kind(v) for nid, v in inst.observations.items()}
    assert inst.value_kinds == kinds
    assert inst.actual_patterns == sorted(satisfied_patterns(inst.program, kinds))

    _observations, answer = execute(inst.program)
    assert answer == inst.answer or answer is inst.answer
    _assert_answer_type(inst.answer_type, answer)

    assert sorted(_literal_args(inst)) == sorted(_plan_role_args(inst, plan))


def _assert_answer_type(answer_type: str, answer: Any) -> None:
    kind = st.value_kind(answer)
    if answer_type == "category":
        assert kind == "string"
    elif answer_type == "boolean":
        assert kind == "boolean"
    else:
        assert kind == answer_type


@pytest.mark.parametrize("bp,plan", PLAN_SAMPLE, ids=_ids(PLAN_SAMPLE))
def test_workflow_first_provenance(bp: Blueprint, plan: Plan):
    """The plan produced the program: same nodes, same capabilities, same edges."""
    try:
        inst = instantiate(bp, plan, SAMPLE_SEED, track=TRACK)
    except BuildError:
        return
    prog = inst.program
    assert [nd.node_id for nd in prog.nodes] == [s.node_id for s in plan.steps]
    for step in plan.steps:
        node = prog.node(step.node_id)          # raises if absent or duplicated
        assert OPS[node.op].capability == step.capability
        assert inst.op_binding[step.node_id] == node.op
    assert set(prog.edges()) == _declared_edges(plan)


@pytest.mark.parametrize("bp,plan", PLAN_SAMPLE[:30], ids=_ids(PLAN_SAMPLE[:30]))
def test_same_inputs_give_a_byte_identical_instance_id(bp: Blueprint, plan: Plan):
    try:
        first = instantiate(bp, plan, 777, track=TRACK)
    except BuildError:
        return
    second = instantiate(bp, plan, 777, track=TRACK)
    assert first.workflow_instance_id() == second.workflow_instance_id()
    assert first.role_values == second.role_values
    assert first.program.program_id() == second.program.program_id()


def test_different_seeds_change_almost_every_instance():
    changed = 0
    compared = 0
    for bp, plan in PLAN_SAMPLE:
        try:
            a = instantiate(bp, plan, 100, track=TRACK)
            b = instantiate(bp, plan, 200, track=TRACK)
        except BuildError:
            continue
        compared += 1
        changed += a.workflow_instance_id() != b.workflow_instance_id()
    assert compared >= 40
    assert changed / compared >= 0.90, f"{changed}/{compared} instances differed"


@pytest.mark.parametrize("bp,plan", BIG_SAMPLE, ids=_ids(BIG_SAMPLE))
def test_no_money_value_ever_reaches_a_duration_or_rate_parameter(
        bp: Blueprint, plan: Plan):
    forbidden = {st.DUR_S, st.DUR_MIN, st.DUR_H, st.DUR_D, st.PERCENTAGE,
                 st.RATIO}
    try:
        inst = instantiate(bp, plan, SAMPLE_SEED + 5, track=TRACK)
    except BuildError:
        return
    sems = semantic_types(inst.program)
    for nd in inst.program.nodes:
        op = OPS[nd.op]
        for p in op.params:
            value = nd.args[p.name]
            incoming = (sems[value.node_id] if isinstance(value, Ref)
                        else nd.arg_sems.get(p.name, p.sem))
            if p.sem in forbidden:
                assert incoming != st.MONEY, f"{nd.node_id}.{p.name}"
            assert st.compatible(p.sem, incoming), (
                f"{nd.node_id}.{p.name}: {incoming} -> {p.sem}")


# ---------------------------------------------------------------------------
# value realism, per hint family
# ---------------------------------------------------------------------------
N_VALUE_SAMPLES = 200


def samples(hint: str, seed: int = 20260731) -> List[Any]:
    rng = random.Random(f"{hint}:{seed}")
    return [V.sample_hint(hint, rng) for _ in range(N_VALUE_SAMPLES)]


def hints_with_prefix(*prefixes: str) -> List[str]:
    return sorted(h for h in V.HINTS if h.startswith(prefixes))


def hints_with_sem(*sems: str) -> List[str]:
    return sorted(h for h in V.HINTS if V.HINTS[h][0] in sems)


@pytest.mark.parametrize("hint", hints_with_sem(st.MONEY))
def test_money_hints_are_positive_with_two_decimals(hint: str):
    for value in samples(hint):
        assert isinstance(value, float) and value > 0
        assert round(value, 2) == value, value


@pytest.mark.parametrize("hint", hints_with_sem(st.PERCENTAGE))
def test_percentage_hints_stay_in_a_sane_band(hint: str):
    for value in samples(hint):
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert 0.0 < float(value) <= 100.0, value


@pytest.mark.parametrize("hint", hints_with_sem(st.RATIO))
def test_ratio_hints_are_proper_fractions(hint: str):
    for value in samples(hint):
        assert 0.0 < float(value) < 1.0, value


DURATION_SEMS = (st.DUR_S, st.DUR_MIN, st.DUR_H, st.DUR_D)


@pytest.mark.parametrize("hint", hints_with_sem(*DURATION_SEMS))
def test_duration_hints_are_positive_and_carry_a_unit(hint: str):
    from targeted_tool_data.pilot43.queries import UNIT_WORD

    sem = V.sem_of_hint(hint)
    assert sem in DURATION_SEMS
    assert UNIT_WORD[sem] in ("seconds", "minutes", "hours", "days")
    for value in samples(hint):
        assert float(value) > 0, value


GEOMETRY_SEMS = (st.LEN_M, st.LEN_KM, st.MASS_KG, st.MASS_G, st.VOL_L,
                 st.VOL_ML, st.AREA)


@pytest.mark.parametrize("hint", hints_with_sem(*GEOMETRY_SEMS))
def test_dimension_hints_are_strictly_positive(hint: str):
    for value in samples(hint):
        assert float(value) > 0, value


TRIANGLE_PLANS = [(bp, plan) for bp in BLUEPRINTS for plan in bp.plans
                  if any(s.capability == "geometry.triangle_perimeter"
                         for s in plan.steps)]


def test_the_registry_still_contains_triangle_perimeter_plans():
    assert TRIANGLE_PLANS, "the triangle-inequality test below would be vacuous"


@pytest.mark.parametrize("bp,plan", TRIANGLE_PLANS, ids=_ids(TRIANGLE_PLANS))
def test_triangle_sides_satisfy_the_triangle_inequality(bp: Blueprint, plan: Plan):
    checked = 0
    for seed in range(20):
        try:
            inst = instantiate(bp, plan, 100 + seed, track=TRACK)
        except BuildError:
            continue
        for nd in inst.program.nodes:
            if OPS[nd.op].capability != "geometry.triangle_perimeter":
                continue
            sides = sorted(
                float(inst.observations[v.node_id] if isinstance(v, Ref) else v)
                for v in nd.args.values())
            assert all(s > 0 for s in sides)
            assert sides[0] + sides[1] > sides[2], sides
            checked += 1
    assert checked >= 10


PATH_RE = re.compile(r"^/?(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+$")
URL_RE = re.compile(r"^https?://[a-z0-9.-]+(?::\d+)?(?:/[^\s?]*)?(?:\?\S+)?$")


@pytest.mark.parametrize("hint", hints_with_sem(st.PATH))
def test_path_hints_look_like_paths(hint: str):
    for value in samples(hint):
        assert isinstance(value, str) and value
        assert " " not in value
        assert PATH_RE.match(value), value
        assert not value.strip("/.").replace("/", "").isdigit()


@pytest.mark.parametrize("hint", hints_with_sem(st.URL))
def test_url_hints_look_like_urls(hint: str):
    for value in samples(hint):
        assert isinstance(value, str)
        assert URL_RE.match(value), value


@pytest.mark.parametrize("hint", hints_with_sem(st.DATE))
def test_date_hints_are_iso_dates(hint: str):
    for value in samples(hint):
        assert date.fromisoformat(value).isoformat() == value


@pytest.mark.parametrize("hint", hints_with_sem(st.NUMBER_LIST, st.TEXT_LIST))
def test_list_hints_are_non_empty_and_not_constant(hint: str):
    for value in samples(hint):
        assert isinstance(value, list) and len(value) >= 3
        assert len({repr(v) for v in value}) >= 2, value


@pytest.mark.parametrize("hint", hints_with_sem(st.MAPPING, st.RECORD,
                                                st.RECORD_LIST))
def test_structured_hints_are_non_empty(hint: str):
    for value in samples(hint):
        assert value
        assert len(value) >= 1


@pytest.mark.parametrize("hint", hints_with_sem(st.TEXT, st.IDENTIFIER,
                                                st.CATEGORY, st.UNIT_NAME))
def test_text_hints_are_non_numeric_strings(hint: str):
    for value in samples(hint):
        assert isinstance(value, str) and value.strip()
        assert not value.strip().replace(".", "", 1).isdigit(), value


# ---------------------------------------------------------------------------
# boolean balancing
# ---------------------------------------------------------------------------
def _boolean_plans(n: int) -> List[Tuple[Blueprint, Plan]]:
    out: List[Tuple[Blueprint, Plan]] = []
    for bp, plan in _spread_sample(320, SAMPLE_SEED + 2):
        try:
            inst = instantiate(bp, plan, 4242, track=TRACK)
        except BuildError:
            continue
        if inst.answer_type == "boolean":
            out.append((bp, plan))
        if len(out) == n:
            break
    return out


BOOLEAN_PLANS = _boolean_plans(15)


def test_fifteen_boolean_plans_were_found():
    assert len(BOOLEAN_PLANS) == 15


@pytest.mark.parametrize("bp,plan", BOOLEAN_PLANS, ids=_ids(BOOLEAN_PLANS))
def test_boolean_sinks_are_balanced_across_seeds(bp: Blueprint, plan: Plan):
    answers = []
    for seed in range(40):
        try:
            answers.append(instantiate(bp, plan, 1000 + 37 * seed,
                                       track=TRACK).answer)
        except BuildError:
            continue
    assert len(answers) >= 30, "too few instances to judge balance"
    assert all(isinstance(a, bool) for a in answers)
    share = sum(answers) / len(answers)
    assert 0.30 <= share <= 0.70, f"true share {share:.2f}"


@pytest.mark.parametrize("bp,plan", BOOLEAN_PLANS, ids=_ids(BOOLEAN_PLANS))
def test_requested_boolean_label_is_delivered_or_refused(bp: Blueprint, plan: Plan):
    delivered = 0
    for want in (True, False):
        for seed in range(6):
            try:
                inst = instantiate(bp, plan, 500 + 11 * seed, track=TRACK,
                                   want_bool=want)
            except BuildError:
                continue                  # refusing is allowed, lying is not
            assert inst.answer is want
            assert inst.boolean_band == ("clear_true" if want else "clear_false")
            delivered += 1
    assert delivered >= 6


def _sink_gap(inst: Instance) -> float | None:
    """Relative distance between the two magnitudes the sink compares."""
    node = inst.program.node(inst.program.sink)
    op = OPS[node.op]
    if len(op.params) != 2:
        return None
    left, right = (node.args[p.name] for p in op.params)
    values = []
    for arg in (left, right):
        value = inst.observations[arg.node_id] if isinstance(arg, Ref) else arg
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        values.append(float(value))
    return abs(values[0] - values[1]) / max(abs(values[0]), 1.0)


COMPARISON_SINKS = {"comparison.at_least", "comparison.greater",
                    "comparison.less_than"}


def _sink_capability(bp: Blueprint, plan: Plan) -> str | None:
    try:
        inst = instantiate(bp, plan, 4242, track=TRACK)
    except BuildError:
        return None
    return OPS[inst.program.node(inst.program.sink).op].capability


BOUNDARY_PLANS = [(bp, plan) for bp, plan in BOOLEAN_PLANS
                  if _sink_capability(bp, plan) in COMPARISON_SINKS]


def test_boundary_plans_were_found():
    assert BOUNDARY_PLANS, "the near-boundary test below would be vacuous"


@pytest.mark.parametrize("bp,plan", BOUNDARY_PLANS, ids=_ids(BOUNDARY_PLANS))
def test_near_boundary_requests_sit_closer_to_the_threshold(bp: Blueprint,
                                                            plan: Plan):
    gaps: Dict[bool, List[float]] = {True: [], False: []}
    for near in (False, True):
        for seed in range(12):
            try:
                inst = instantiate(bp, plan, 700 + seed, track=TRACK,
                                   want_bool=True, near_boundary=near)
            except BuildError:
                continue
            gap = _sink_gap(inst)
            if gap is not None:
                gaps[near].append(gap)
    assert len(gaps[True]) >= 5 and len(gaps[False]) >= 5
    near_mean = sum(gaps[True]) / len(gaps[True])
    clear_mean = sum(gaps[False]) / len(gaps[False])
    assert near_mean < clear_mean, (near_mean, clear_mean)


def test_a_boolean_target_on_a_non_boolean_sink_is_refused():
    numeric = next((bp, plan) for bp, plan in PLAN_SAMPLE
                   if _safe_answer_type(bp, plan) == "float")
    with pytest.raises(BuildError):
        instantiate(numeric[0], numeric[1], 4242, track=TRACK, want_bool=True)


def _safe_answer_type(bp: Blueprint, plan: Plan) -> str | None:
    try:
        return instantiate(bp, plan, SAMPLE_SEED, track=TRACK).answer_type
    except BuildError:
        return None
