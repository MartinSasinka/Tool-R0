"""Value realism the type system cannot catch.

Both cases here came out of the smoke stage, where the second critic rejected
queries stating that handover was due nine months before the job started, and that
a permission mask had to reach 352.15. Each executes perfectly and describes a
situation that cannot exist, so each is a generator defect rather than a writing
defect.
"""
from __future__ import annotations

import random

import pytest

from targeted_tool_data.pilot43 import semtypes as st
from targeted_tool_data.pilot43.blueprints import Plan, Role, Step, all_blueprints
from targeted_tool_data.pilot43.build import (BuildError, _order_dates,
                                              instantiate)
from targeted_tool_data.pilot43.tasks import (_integer_comparison_flags,
                                              value_realism_flags)
from targeted_tool_data.pilot43.values import Band, band_for, calibrate_predicate


# ── integer quantities get integer limits ────────────────────────────────
@pytest.mark.parametrize("capability", ["comparison.at_least",
                                        "comparison.greater"])
@pytest.mark.parametrize("want", [True, False])
@pytest.mark.parametrize("near", [True, False])
def test_a_whole_quantity_is_never_compared_against_a_fraction(capability, want,
                                                               near):
    rng = random.Random(4)
    for observed in (0, 1, 3, 7, 201, 247):
        got = calibrate_predicate(capability, ["value", "minimum"],
                                  {"value": observed}, band_for(want, near), rng)
        assert got is not None
        limit = got["minimum"]
        assert isinstance(limit, int), (observed, limit)
        assert limit >= 0


@pytest.mark.parametrize("want", [True, False])
def test_an_integer_comparison_still_lands_on_the_intended_verdict(want):
    rng = random.Random(9)
    for observed in (1, 2, 3, 5, 7, 40, 201):
        got = calibrate_predicate("comparison.at_least", ["value", "minimum"],
                                  {"value": observed}, band_for(want, False), rng)
        assert (observed >= got["minimum"]) is want, (observed, got, want)


def test_a_fractional_quantity_still_gets_a_fractional_limit():
    rng = random.Random(4)
    got = calibrate_predicate("comparison.at_least", ["value", "minimum"],
                              {"value": 240.25}, band_for(True, False), rng)
    assert isinstance(got["minimum"], float)


@pytest.mark.parametrize("want", [True, False])
def test_an_integer_range_check_gets_integer_bounds(want):
    rng = random.Random(2)
    got = calibrate_predicate("validation.in_range", ["value", "lo", "hi"],
                              {"value": 12}, band_for(want, False), rng)
    assert isinstance(got["lo"], int) and isinstance(got["hi"], int)
    assert (got["lo"] <= 12 <= got["hi"]) is want


@pytest.mark.parametrize("want", [True, False])
def test_an_integer_list_limit_stays_whole(want):
    rng = random.Random(6)
    got = calibrate_predicate("validation.list_limit", ["values", "limit"],
                              {"values": [3, 9, 14]}, band_for(want, False), rng)
    assert isinstance(got["limit"], int)
    assert (max([3, 9, 14]) <= got["limit"]) is want


# ── date pairs are ordered ───────────────────────────────────────────────
def test_a_start_and_a_deadline_are_put_the_right_way_round():
    plan = Plan(
        plan_id="p", sink="n1",
        roles=(Role("job_start", "date_iso", "the day the job starts"),
               Role("job_deadline", "date_deadline", "the day it is due")),
        steps=(Step("n1", "date.difference", ("job_start", "job_deadline")),))
    values = {"job_start": "2024-07-15", "job_deadline": "2023-09-20"}
    _order_dates(plan, values)
    assert values["job_start"] == "2023-09-20"
    assert values["job_deadline"] == "2024-07-15"


def test_an_already_ordered_pair_is_left_alone():
    plan = Plan(
        plan_id="p", sink="n1",
        roles=(Role("job_start", "date_iso", ""),
               Role("job_deadline", "date_deadline", "")),
        steps=(Step("n1", "date.difference", ("job_start", "job_deadline")),))
    values = {"job_start": "2023-01-02", "job_deadline": "2023-04-05"}
    _order_dates(plan, values)
    assert values == {"job_start": "2023-01-02", "job_deadline": "2023-04-05"}


# ── the gates see it ─────────────────────────────────────────────────────
def test_the_realism_gate_reports_a_backwards_period():
    from targeted_tool_data.pilot43.tasks import _date_order_flags

    plan = Plan(
        plan_id="p", sink="n1",
        roles=(Role("job_start", "date_iso", ""),
               Role("job_deadline", "date_deadline", "")),
        steps=(Step("n1", "date.difference", ("job_start", "job_deadline")),))

    class _Inst:
        role_values = {"job_start": "2024-07-15", "job_deadline": "2023-09-20"}

    flags = _date_order_flags(_Inst(), plan)
    assert flags and "before" in flags[0]
    _Inst.role_values = {"job_start": "2023-09-20", "job_deadline": "2024-07-15"}
    assert _date_order_flags(_Inst(), plan) == []


def test_the_realism_gate_reports_a_fractional_limit_on_a_whole_quantity():
    from targeted_tool_data.pilot43.ops import build_ops
    from targeted_tool_data.pilot43.program import Node, Program, Ref

    ops = build_ops()
    op_id = next(oid for oid, op in ops.items()
                 if op.capability == "comparison.at_least")
    params = [p.name for p in ops[op_id].params]
    program = Program(nodes=[Node("n1", op_id, {params[0]: Ref("n0"),
                                               params[1]: 2.09})], sink="n1")

    class _Inst:
        observations = {"n0": 3}

    _Inst.program = program
    assert _integer_comparison_flags(_Inst())
    program.nodes[0].args[params[1]] = 2
    assert _integer_comparison_flags(_Inst()) == []


def test_a_generated_instance_has_no_realism_flag():
    """Spot-check the live registry: the gate must be satisfiable, not just strict."""
    checked = 0
    for bp in all_blueprints()[:40]:
        for plan in bp.plans[:2]:
            for seed in (1, 2, 3):
                try:
                    inst = instantiate(bp, plan, track="A_NATIVE", seed=seed)
                except BuildError:
                    continue
                checked += 1
                assert value_realism_flags(inst, plan) == [], (bp.workflow_id,
                                                              plan.plan_id, seed)
    assert checked > 20
