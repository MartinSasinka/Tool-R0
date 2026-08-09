"""Determinability: which plans need their rule stated, and what the rule says.

The smoke stage rejected a quarter of its queries on
``all_program_nodes_required``, and the cause was not the writing: a threshold
computed inside the program ("oversized" = above the mean plus the spread) or an
answer welded together from unrelated parts cannot be asked for from facts alone.
These tests pin the three outcomes -- self-evident, rule-bearing, unstatable --
and the wiring that keeps a rule-bearing task out of the fully implicit modes.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import pytest

from targeted_tool_data.pilot43 import determinability as det
from targeted_tool_data.pilot43 import queries
from targeted_tool_data.pilot43.blueprints import Blueprint, Plan, all_blueprints
from targeted_tool_data.pilot43.build import BuildError, instantiate
from targeted_tool_data.pilot43.ops import build_ops
from targeted_tool_data.pilot43.qstage import RULE_MODES, assign_modes
from targeted_tool_data.pilot43.queries import build_contract
from targeted_tool_data.pilot43.qvalidate import (check_specification,
                                                  contract_payload)

OPS = build_ops()
PAIRS: List[Tuple[Blueprint, Plan]] = [(bp, plan) for bp in all_blueprints()
                                       for plan in bp.plans]


def _instance(bp: Blueprint, plan: Plan, seed: int):
    return instantiate(bp, plan, track="A_NATIVE", seed=seed)


def _contract(bp: Blueprint, plan: Plan, seed: int = 7, mode: str = "SEMI_IMPLICIT"):
    inst = _instance(bp, plan, seed)
    return inst, build_contract(inst, bp, plan, mode=mode,
                                task_id=f"t{seed}", seed=seed)


def _first(predicate, limit: int = 400):
    """First (blueprint, plan, instance, contract) whose contract matches."""
    rng = random.Random(11)
    pairs = PAIRS[:]
    rng.shuffle(pairs)
    for bp, plan in pairs[:limit]:
        try:
            inst, contract = _contract(bp, plan)
        except (BuildError, Exception):
            continue
        if predicate(contract, plan):
            return bp, plan, inst, contract
    return None


# ── classification ───────────────────────────────────────────────────────
def test_a_plan_with_no_computed_rule_is_self_evident():
    found = _first(lambda c, p: c.determinability == det.SELF_EVIDENT)
    assert found is not None
    _bp, _plan, _inst, contract = found
    assert contract.specification == ()


def test_a_computed_threshold_makes_the_task_rule_bearing():
    found = _first(lambda c, p: any(r.startswith("a value counts")
                                    or r.startswith("a category counts")
                                    or r.startswith("a record counts")
                                    for r in c.specification))
    assert found is not None, "no plan in the registry computes a threshold"
    _bp, _plan, _inst, contract = found
    assert contract.determinability in (det.NEEDS_RULE, det.NOT_STATABLE)
    assert contract.specification


def test_the_rule_names_the_facts_it_is_built_from_not_the_tools():
    """A rule has to be readable by the user, so it may not name a capability."""
    seen = 0
    for bp, plan in PAIRS[:400]:
        try:
            _inst, contract = _contract(bp, plan)
        except Exception:
            continue
        for rule in contract.specification:
            seen += 1
            low = rule.lower()
            assert "_" not in rule, rule
            for forbidden in contract.forbidden_terms:
                if len(forbidden) >= 6:
                    assert forbidden.lower() not in low, (forbidden, rule)
    assert seen, "no rule sentences were produced at all"


def test_an_unreadable_rule_marks_the_task_unstatable_rather_than_shipping_it():
    """A rule built from a composite of composites is dropped, not rendered."""
    found = _first(lambda c, p: c.determinability == det.NOT_STATABLE)
    assert found is not None, "the registry has no deeply composite sink"
    _bp, plan, inst, contract = found
    unreadable = [r for r in contract.specification
                  if len(r) > det.MAX_RULE_CHARS or det.DEEP_FALLBACK in r]
    assert unreadable, contract.specification

    verdict = det.classify(inst.program, plan, inst.role_values,
                           target_phrase=contract.target_phrase)
    assert verdict.level == det.NOT_STATABLE
    assert any(r.startswith("rule_not_statable") for r in verdict.reasons)


@pytest.mark.parametrize("phrase", [
    "the requested figure",
    "the positions in the requested figure",
    "the same figure restated as a percentage",
    "whether the stated requirement is met",
    "the value",
    "",
])
def test_a_target_that_names_nothing_is_recognised(phrase: str):
    assert queries.target_is_vacuous(phrase) is True


@pytest.mark.parametrize("phrase", [
    "the total cost of the basket",
    "the share of readings above the limit",
    "whether the shipment clears customs in time",
])
def test_a_target_that_names_a_quantity_is_kept(phrase: str):
    assert queries.target_is_vacuous(phrase) is False


def test_a_contract_with_a_vacuous_target_is_never_rendered():
    """The writer must not be asked to build a question around a placeholder."""
    contract = _first(lambda c, p: queries.target_is_vacuous(c.target_phrase))
    if contract is None:
        pytest.skip("no registry plan produces a vacuous target")
    _bp, _plan, _inst, found = contract
    assert found.determinability == det.NOT_STATABLE


def test_plan_level_and_instance_level_screens_agree():
    """Routing runs on plans; rendering runs on instances. They must not disagree."""
    checked = 0
    for bp, plan in PAIRS[:250]:
        try:
            _inst, contract = _contract(bp, plan)
        except Exception:
            continue
        if queries.target_is_vacuous(contract.target_phrase):
            # dropped for a reason the plan screen cannot see: the sink's
            # purpose was named after its tool, so the target says nothing
            assert contract.determinability == det.NOT_STATABLE
            continue
        checked += 1
        plan_says = det.plan_needs_rule(plan, OPS)
        instance_says = contract.determinability != det.SELF_EVIDENT
        assert plan_says == instance_says, (bp.workflow_id, plan.plan_id,
                                            contract.determinability)
    assert checked > 100


# ── the validator layer ──────────────────────────────────────────────────
RULE = "a value counts as over the line when it is above the average of the readings"


def test_a_query_that_omits_the_rule_fails_the_specification_layer():
    payload = {"specification": [RULE]}
    query = "The readings are 4, 9 and 12. What share of the run was over?"
    layer = check_specification(query, payload)
    assert not layer["passed"]
    assert layer["missing_rules"][0]["rule"] == RULE


def test_a_paraphrased_rule_passes():
    payload = {"specification": [RULE]}
    query = ("The readings are 4, 9 and 12. We count a reading as over the line "
             "when it sits above the average of the readings. "
             "What share of the run was over?")
    assert check_specification(query, payload)["passed"]


def test_a_task_with_no_rule_passes_the_layer_trivially():
    assert check_specification("anything at all", {})["passed"]
    assert check_specification("anything at all", {"specification": []})["passed"]


def test_the_rule_wording_is_contract_vocabulary_not_leakage():
    found = _first(lambda c, p: c.determinability == det.NEEDS_RULE)
    assert found is not None
    _bp, _plan, inst, contract = found
    payload = contract_payload(contract, answer=inst.answer,
                               gold_capabilities=[], predicate_steps=0)
    assert payload["specification"] == list(contract.specification)
    for rule in contract.specification:
        assert rule in payload["domain_vocabulary"]


# ── routing ──────────────────────────────────────────────────────────────
def _rows(n: int, bucket: str = "3") -> List[Dict[str, Any]]:
    return [{"task_id": f"t{i}", "call_bucket": bucket,
             "workflow_id": "w", "plan_id": "p"} for i in range(n)]


def test_a_rule_bearing_task_never_lands_in_a_fully_implicit_mode():
    rows = _rows(100)
    needs = {row["task_id"]: (i % 4 == 0) for i, row in enumerate(rows)}
    targets = {"DOMAIN_GROUNDED_IMPLICIT": 0.5, "GOAL_BASED_IMPLICIT": 0.2,
               "SEMI_IMPLICIT": 0.2, "OPERATION_EXPLICIT_GRAPH_IMPLICIT": 0.08,
               "GRAPH_EXPLICIT": 0.02}
    modes = assign_modes(rows, targets, seed=3, needs_rule=needs)
    for task_id, mode in modes.items():
        if needs[task_id] and mode:
            assert mode in RULE_MODES, (task_id, mode)


def test_a_surplus_of_rule_bearing_tasks_is_dropped_not_mis_routed():
    rows = _rows(20)
    needs = {row["task_id"]: True for row in rows}      # more than any mode holds
    targets = {"DOMAIN_GROUNDED_IMPLICIT": 0.5, "GOAL_BASED_IMPLICIT": 0.2,
               "SEMI_IMPLICIT": 0.2, "OPERATION_EXPLICIT_GRAPH_IMPLICIT": 0.08,
               "GRAPH_EXPLICIT": 0.02}
    modes = assign_modes(rows, targets, seed=3, needs_rule=needs)
    routed = [m for m in modes.values() if m]
    assert all(m in RULE_MODES for m in routed)
    assert sum(1 for m in modes.values() if not m) == len(rows) - len(routed)


def test_without_a_rule_map_every_task_still_gets_a_mode():
    rows = _rows(50)
    targets = {"DOMAIN_GROUNDED_IMPLICIT": 0.6, "SEMI_IMPLICIT": 0.4}
    modes = assign_modes(rows, targets, seed=5)
    assert len(modes) == 50 and all(modes.values())
