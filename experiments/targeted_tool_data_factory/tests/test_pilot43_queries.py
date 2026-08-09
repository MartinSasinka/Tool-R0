"""Deterministic query rendering, the qvalidate gates and template diversity.

The query pool below is built once from a seeded sample of workflow plans and
reused by every test in the module, so the suite stays well under a minute.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Sequence, Tuple

import pytest

from targeted_tool_data.pilot43.blueprints import Blueprint, Plan, all_blueprints
from targeted_tool_data.pilot43.qstage import contract_seed
from targeted_tool_data.pilot43.build import BuildError, Instance, instantiate
from targeted_tool_data.pilot43.ops import build_ops
from targeted_tool_data.pilot43.queries import (build_contract,
                                                render_deterministic)
from targeted_tool_data.pilot43 import qvalidate as qv
from targeted_tool_data.pilot43.qvalidate import (classify_mode, contract_payload,
                                                  diversity_report,
                                                  exact_fingerprint, fingerprints,
                                                  intent_fingerprint,
                                                  lexical_skeleton,
                                                  skeleton_fingerprint,
                                                  validate_query)

OPS = build_ops()
PAIRS: List[Tuple[Blueprint, Plan]] = [(bp, plan) for bp in all_blueprints()
                                       for plan in bp.plans]
TRACK = "A_NATIVE"
MODES = ("GRAPH_EXPLICIT", "OPERATION_EXPLICIT_GRAPH_IMPLICIT", "SEMI_IMPLICIT",
         "DOMAIN_GROUNDED_IMPLICIT", "GOAL_BASED_IMPLICIT")
IMPLICIT_MODES = ("SEMI_IMPLICIT", "DOMAIN_GROUNDED_IMPLICIT",
                  "GOAL_BASED_IMPLICIT")


# ---------------------------------------------------------------------------
# the shared query pool
# ---------------------------------------------------------------------------
def _payload_for(inst: Instance, contract) -> Dict[str, Any]:
    caps = [OPS[nd.op].capability for nd in inst.program.nodes]
    predicates = sum(1 for nd in inst.program.nodes
                     if OPS[nd.op].out_sem == "Flag")
    return contract_payload(contract, answer=inst.answer,
                            gold_capabilities=caps, predicate_steps=predicates)


def _build_pool(n_instances: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    pool: List[Dict[str, Any]] = []
    for i, (bp, plan) in enumerate(rng.sample(PAIRS, n_instances * 3)):
        if len({row["workflow_id"] for row in pool}) and \
                len(pool) >= n_instances * len(MODES):
            break
        try:
            inst = instantiate(bp, plan, 3000 + i, track=TRACK)
        except BuildError:
            continue
        for mode in MODES:
            contract = build_contract(inst, bp, plan, mode=mode,
                                      task_id=f"t{i}", seed=7 + i)
            rendered = render_deterministic(contract, mode, seed=7 + i)
            pool.append({
                "workflow_id": bp.workflow_id,
                "plan_id": plan.plan_id,
                "mode": mode,
                "query": rendered["query"],
                "template_id": rendered["template_id"],
                "answer_type": inst.answer_type,
                "call_count": inst.call_count,
                "payload": _payload_for(inst, contract),
                "result": None,
            })
    for row in pool:
        row["result"] = validate_query(row["query"], row["payload"])
    return pool


POOL = _build_pool(40, seed=99)
IMPLICIT_POOL = [r for r in POOL if r["mode"] in IMPLICIT_MODES]


def _pool_ids(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return [f"{r['workflow_id']}/{r['plan_id']}/{r['mode']}" for r in rows]


def test_the_pool_is_large_and_covers_every_mode():
    assert len(POOL) >= 200
    assert {r["mode"] for r in POOL} == set(MODES)
    assert len({r["workflow_id"] for r in POOL}) >= 25


# ---------------------------------------------------------------------------
# deterministic rendering
# ---------------------------------------------------------------------------
def _named_plan(workflow_id: str, plan_id: str) -> Tuple[Blueprint, Plan]:
    return next((bp, plan) for bp, plan in PAIRS
                if bp.workflow_id == workflow_id and plan.plan_id == plan_id)


def _render_once(seed: int, mode: str = "DOMAIN_GROUNDED_IMPLICIT") -> str:
    bp, plan = _named_plan(POOL[0]["workflow_id"], POOL[0]["plan_id"])
    inst = instantiate(bp, plan, 3000, track=TRACK)
    contract = build_contract(inst, bp, plan, mode=mode, task_id="fixed",
                              seed=seed)
    return render_deterministic(contract, mode, seed=seed)["query"]


def test_rendering_is_reproducible_for_a_fixed_seed():
    assert _render_once(11) == _render_once(11)


def test_rendering_varies_with_the_seed():
    variants = {_render_once(seed) for seed in range(12)}
    assert len(variants) >= 8


# ---------------------------------------------------------------------------
# what a rendered query must and must not contain
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("row", POOL, ids=_pool_ids(POOL))
def test_every_stated_fact_number_survives_into_the_query(row: Dict[str, Any]):
    assert row["result"]["layers"]["facts"]["missing_numbers"] == []
    assert row["result"]["layers"]["facts"]["missing_strings"] == []


@pytest.mark.parametrize("row", IMPLICIT_POOL, ids=_pool_ids(IMPLICIT_POOL))
def test_implicit_queries_never_contain_a_tool_identifier(row: Dict[str, Any]):
    leak = row["result"]["layers"]["tool_leak"]
    assert leak["leaked_identifiers"] == []
    assert row["result"]["layers"]["var_leak"]["passed"], \
        row["result"]["layers"]["var_leak"]["hits"]


@pytest.mark.parametrize("row", IMPLICIT_POOL, ids=_pool_ids(IMPLICIT_POOL))
def test_implicit_queries_never_disclose_the_call_count(row: Dict[str, Any]):
    cls = row["result"]["classification"]
    assert cls["call_count_leakage"] is False
    assert cls["stage_label_count"] == 0


@pytest.mark.parametrize("row", IMPLICIT_POOL, ids=_pool_ids(IMPLICIT_POOL))
def test_implicit_queries_stay_inside_their_classified_mode_limits(
        row: Dict[str, Any]):
    assert row["result"]["layers"]["mode_limits"]["passed"], \
        row["result"]["layers"]["mode_limits"]["violations"]


NON_STRING_IMPLICIT = [r for r in IMPLICIT_POOL if r["answer_type"] != "string"]


@pytest.mark.parametrize("row", NON_STRING_IMPLICIT,
                         ids=_pool_ids(NON_STRING_IMPLICIT))
def test_implicit_queries_do_not_state_a_non_string_answer(row: Dict[str, Any]):
    assert row["result"]["layers"]["answer_leak"]["passed"]


@pytest.mark.parametrize("row", POOL, ids=_pool_ids(POOL))
def test_target_and_entities_are_preserved(row: Dict[str, Any]):
    layers = row["result"]["layers"]
    assert layers["target"]["passed"], layers["target"]
    assert layers["entities"]["passed"]


@pytest.mark.parametrize("row", POOL, ids=_pool_ids(POOL))
def test_no_query_invents_a_condition(row: Dict[str, Any]):
    assert row["result"]["layers"]["new_conditions"]["passed"], \
        row["result"]["layers"]["new_conditions"]["condition_cues"]


def test_no_rendered_query_trips_the_language_checks():
    bad = [(r["workflow_id"], r["mode"], r["result"]["layers"]["language"]["issues"])
           for r in POOL if not r["result"]["layers"]["language"]["passed"]]
    assert bad == []


def test_the_pool_has_no_language_defects_at_all():
    """The doubled-article defect the renderer used to emit is gone for good."""
    issues = {issue for r in POOL
              for issue in r["result"]["layers"]["language"]["issues"]}
    assert issues == set()


def test_no_rendered_query_contains_an_unexpected_number():
    bad = [(r["workflow_id"], r["result"]["layers"]["facts"]["extra_numbers"])
           for r in POOL if r["result"]["layers"]["facts"]["extra_numbers"]]
    assert bad == []


@pytest.mark.xfail(reason="build_contract scrubs forbidden phrases out of the "
                          "target phrase only, so a role description such as "
                          "'the smallest acceptable column average' reaches the "
                          "query and reproduces the surface name "
                          "'column_average'",
                   strict=True)
def test_implicit_queries_never_name_a_tool_even_as_a_phrase():
    bad = [(r["workflow_id"], r["mode"],
            r["result"]["layers"]["tool_leak"]["leaked_phrases"])
           for r in IMPLICIT_POOL
           if r["result"]["layers"]["tool_leak"]["leaked_phrases"]]
    assert bad == []


@pytest.mark.xfail(reason="queries._stage_hint joins the capability families of "
                          "a plan with ' and ', so a bitwise plan renders the "
                          "phrase 'bitwise and', which is the surface name "
                          "'bitwise_and' under qvalidate's phrase normalisation",
                   strict=True)
def test_a_bitwise_plan_does_not_render_its_tool_name_as_a_phrase():
    bp, plan = _named_plan("bitwise.capacity_register", "cap.v4")
    inst = instantiate(bp, plan, 900, track=TRACK)
    contract = build_contract(inst, bp, plan, mode="SEMI_IMPLICIT",
                              task_id="bits", seed=5)
    query = render_deterministic(contract, "SEMI_IMPLICIT", seed=5)["query"]
    layer = validate_query(query, _payload_for(inst, contract))["layers"]
    assert layer["tool_leak"]["leaked_phrases"] == []


STRING_IMPLICIT = [r for r in IMPLICIT_POOL if r["answer_type"] == "string"]


def test_the_pool_covers_string_answers():
    assert len(STRING_IMPLICIT) >= 10


@pytest.mark.parametrize("row", STRING_IMPLICIT, ids=_pool_ids(STRING_IMPLICIT))
def test_string_answers_in_the_pool_are_not_reported_as_leaked(
        row: Dict[str, Any]):
    assert row["result"]["layers"]["answer_leak"]["passed"]


@pytest.mark.parametrize("seed", [42, 500, 3010])
def test_a_normalised_string_answer_is_not_an_answer_leak(seed: int):
    bp, plan = _named_plan("text.note_cleanup", "note.v2")
    inst = instantiate(bp, plan, seed, track=TRACK)
    contract = build_contract(inst, bp, plan, mode="DOMAIN_GROUNDED_IMPLICIT",
                              task_id="leak", seed=seed)
    query = render_deterministic(contract, "DOMAIN_GROUNDED_IMPLICIT",
                                 seed=seed)["query"]
    result = validate_query(query, _payload_for(inst, contract))
    assert inst.answer_type == "string"
    assert result["layers"]["answer_leak"]["passed"]


GRAPH_POOL = [r for r in POOL if r["mode"] == "GRAPH_EXPLICIT"]


@pytest.mark.xfail(reason="check_facts whitelists the step indices that "
                          "_graph_disclosure adds, but check_answer_leak does "
                          "not, so a small integer answer equal to a step "
                          "number is reported as leaked",
                   strict=True)
def test_graph_explicit_step_numbers_are_not_read_as_a_leaked_answer():
    bad = [(r["workflow_id"], r["payload"]["answer_rendered"])
           for r in GRAPH_POOL
           if not r["result"]["layers"]["answer_leak"]["passed"]
           and r["answer_type"] == "integer"]
    assert bad == []


# ---------------------------------------------------------------------------
# independent mode classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("row", GRAPH_POOL, ids=_pool_ids(GRAPH_POOL))
def test_graph_explicit_renders_classify_as_graph_explicit(row: Dict[str, Any]):
    assert row["result"]["classification"]["actual_query_mode"] == "GRAPH_EXPLICIT"


DOMAIN_POOL = [r for r in POOL if r["mode"] == "DOMAIN_GROUNDED_IMPLICIT"]


@pytest.mark.parametrize("row", DOMAIN_POOL, ids=_pool_ids(DOMAIN_POOL))
def test_domain_grounded_renders_never_classify_as_graph_explicit(
        row: Dict[str, Any]):
    cls = row["result"]["classification"]
    assert cls["actual_query_mode"] != "GRAPH_EXPLICIT"
    assert cls["graph_edge_coverage"] < 0.5


def test_classify_mode_reports_the_actual_mode_not_the_requested_one():
    """Render graph-explicit text while the contract asks for the implicit mode."""
    bp, plan = _named_plan(POOL[0]["workflow_id"], POOL[0]["plan_id"])
    inst = instantiate(bp, plan, 3000, track=TRACK)
    contract = build_contract(inst, bp, plan, mode="DOMAIN_GROUNDED_IMPLICIT",
                              task_id="mismatch", seed=5)
    assert contract.requested_mode == "DOMAIN_GROUNDED_IMPLICIT"
    text = render_deterministic(contract, "GRAPH_EXPLICIT", seed=5)["query"]
    payload = _payload_for(inst, contract)
    cls = classify_mode(text, payload["gold_capabilities"], payload["call_count"],
                        payload["entities"], payload["domain_vocabulary"])
    assert cls["actual_query_mode"] == "GRAPH_EXPLICIT"
    assert cls["stage_label_count"] == inst.call_count


# ---------------------------------------------------------------------------
# qvalidate must reject a tampered query
# ---------------------------------------------------------------------------
CLEAN_PAYLOAD: Dict[str, Any] = {
    "mode": "DOMAIN_GROUNDED_IMPLICIT",
    "call_count": 3,
    "target_phrase": "the invoice total for the order",
    "expected_numbers": ["24.5", "12"],
    "expected_strings": [],
    "expected_units": ["EUR"],
    "entities": ["Neomark", "the Brno depot"],
    "forbidden_terms": ["n1", "n2", "arithmetic.multiply", "apply_tax_rate",
                        "amount_with_tax"],
    "gold_capabilities": ["arithmetic.multiply", "rates.apply_tax"],
    "predicate_steps": 0,
    "answer_rendered": "331.8",
    "domain_vocabulary": ["the unit price", "the order size",
                          "the invoice total for the order", "commerce",
                          "work out an invoice total"],
}
CLEAN_QUERY = ("Going through Neomark's figures for last week at the Brno depot. "
               "The unit price is 24.5 EUR and the order size is 12. "
               "What is the invoice total for the order?")


def test_the_clean_reference_query_passes_every_layer():
    result = validate_query(CLEAN_QUERY, CLEAN_PAYLOAD)
    assert result["passed"], result["failed_layers"]


def test_an_injected_extra_number_is_rejected():
    tampered = CLEAN_QUERY.replace("order size is 12.",
                                   "order size is 12. The reference code is 4471.")
    layer = validate_query(tampered, CLEAN_PAYLOAD)["layers"]["facts"]
    assert not layer["passed"]
    assert "4471" in layer["extra_numbers"]


def test_a_changed_unit_is_rejected():
    tampered = CLEAN_QUERY.replace("24.5 EUR", "24.5 GBP")
    layer = validate_query(tampered, CLEAN_PAYLOAD)["layers"]["units"]
    assert not layer["passed"]
    assert layer["missing_units"] == ["eur"] and "gbp" in layer["foreign_units"]


def test_the_contract_seed_is_the_same_in_a_freshly_salted_interpreter():
    """A per-task seed must not depend on ``PYTHONHASHSEED``.

    When it did, the contract rebuilt for validation or for the independent audit
    drew different entities and fact wording than the contract the writer was given,
    so a perfectly good query was scored against facts it had never seen.
    """
    import subprocess
    import sys

    code = ("from targeted_tool_data.pilot43.qstage import contract_seed;"
            "print(contract_seed('p43_110151bcdfc8afd7', 4242))")
    seeds = set()
    for salt in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=salt)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env, check=True)
        seeds.add(out.stdout.strip())
    assert len(seeds) == 1, seeds
    assert seeds == {str(contract_seed("p43_110151bcdfc8afd7", 4242))}


def test_a_unit_named_by_a_fact_description_is_not_a_foreign_unit():
    """A list of lengths carries no scalar unit, but its description names one."""
    payload = dict(CLEAN_PAYLOAD)
    payload["expected_numbers"] = ["2.49", "3.1"]
    payload["expected_units"] = []
    payload["domain_vocabulary"] = ["the kilometres each leg runs",
                                    "the total distance", "travel"]
    query = ("Neomark's driver logged 2.49 km on the first leg and 3.1 km on the "
             "second. What is the total distance?")
    layer = validate_query(query, payload)["layers"]["units"]
    assert layer["passed"], layer
    assert layer["allowed_by_description"] == ["km"]


def test_a_unit_no_description_mentions_is_still_rejected():
    payload = dict(CLEAN_PAYLOAD)
    payload["expected_numbers"] = ["2.49", "3.1"]
    payload["expected_units"] = []
    payload["domain_vocabulary"] = ["the kilometres each leg runs",
                                    "the total distance", "travel"]
    query = ("Neomark's driver logged 2.49 km and then 3.1 kg. "
             "What is the total distance?")
    layer = validate_query(query, payload)["layers"]["units"]
    assert not layer["passed"]
    assert "kg" in layer["foreign_units"]


def test_a_small_integer_written_as_a_word_still_counts_as_stated():
    payload = dict(CLEAN_PAYLOAD)
    payload["expected_numbers"] = ["24.5", "2"]
    query = ("Going through Neomark's figures at the Brno depot. The unit price "
             "is 24.5 EUR and it is the second line on the ledger. "
             "What is the invoice total for the order?")
    layer = validate_query(query, payload)["layers"]["facts"]
    assert layer["passed"], layer
    assert layer["missing_numbers"] == []


def test_a_large_integer_must_still_appear_as_a_digit():
    payload = dict(CLEAN_PAYLOAD)
    payload["expected_numbers"] = ["24.5", "40"]
    query = CLEAN_QUERY.replace("order size is 12", "order size is forty")
    layer = validate_query(query, payload)["layers"]["facts"]
    assert not layer["passed"]
    assert "40" in layer["missing_numbers"]


def test_only_the_scenario_grounded_mode_has_to_name_an_entity():
    bare = ("The unit price is 24.5 EUR and the order size is 12. "
            "What is the invoice total for the order?")
    grounded = validate_query(bare, CLEAN_PAYLOAD)["layers"]["entities"]
    assert not grounded["passed"] and grounded["required"] is True

    goal_based = dict(CLEAN_PAYLOAD, mode="GOAL_BASED_IMPLICIT")
    layer = validate_query(bare, goal_based)["layers"]["entities"]
    assert layer["passed"] and layer["required"] is False


def test_an_added_condition_is_rejected():
    tampered = CLEAN_QUERY + " Only if the total exceeds five hundred."
    layer = validate_query(tampered, CLEAN_PAYLOAD)["layers"]["new_conditions"]
    assert not layer["passed"]
    assert "only if" in layer["condition_cues"]


def test_a_leaked_tool_name_is_rejected():
    tampered = CLEAN_QUERY.replace("What is", "Use amount_with_tax. What is")
    layer = validate_query(tampered, CLEAN_PAYLOAD)["layers"]["tool_leak"]
    assert not layer["passed"]
    assert "amount_with_tax" in layer["leaked_identifiers"]


def test_a_leaked_node_id_is_rejected():
    tampered = CLEAN_QUERY.replace("What is", "After n2, what is")
    layer = validate_query(tampered, CLEAN_PAYLOAD)["layers"]["var_leak"]
    assert not layer["passed"]
    assert "n2" in layer["hits"]


def test_a_query_that_states_the_answer_is_rejected():
    tampered = CLEAN_QUERY.replace("order size is 12.",
                                   "order size is 12, so it is 331.8.")
    layer = validate_query(tampered, CLEAN_PAYLOAD)["layers"]["answer_leak"]
    assert not layer["passed"]
    assert layer["leaked"] is True


def test_a_query_disclosing_the_call_count_is_rejected():
    tampered = CLEAN_QUERY + " Do it in three steps."
    result = validate_query(tampered, CLEAN_PAYLOAD)
    assert result["classification"]["call_count_leakage"] is True
    layer = result["layers"]["mode_limits"]
    assert not layer["passed"]
    assert any("call count" in v for v in layer["violations"])


# ---------------------------------------------------------------------------
# template fingerprints
# ---------------------------------------------------------------------------
SUBSTITUTED = (CLEAN_QUERY
               .replace("Neomark", "Vantera")
               .replace("Brno depot", "Leipzig warehouse")
               .replace("24.5 EUR", "31.75 GBP")
               .replace("is 12.", "is 407."))
OTHER_STRUCTURE = ("The auditor rang about the winter campaign. "
                   "Confirm the invoice total for the order.")


def test_exact_fingerprint_is_a_deterministic_function_of_the_text():
    assert exact_fingerprint(CLEAN_QUERY) == exact_fingerprint(CLEAN_QUERY)
    assert exact_fingerprint(CLEAN_QUERY) == exact_fingerprint(
        "  " + CLEAN_QUERY.upper() + " ")
    assert exact_fingerprint(CLEAN_QUERY) != exact_fingerprint(SUBSTITUTED)


def test_skeleton_and_intent_survive_number_entity_and_unit_substitution():
    assert lexical_skeleton(CLEAN_QUERY) == lexical_skeleton(SUBSTITUTED)
    assert skeleton_fingerprint(CLEAN_QUERY) == skeleton_fingerprint(SUBSTITUTED)
    assert intent_fingerprint(CLEAN_QUERY) == intent_fingerprint(SUBSTITUTED)


def test_a_different_sentence_structure_gets_a_different_fingerprint():
    assert skeleton_fingerprint(CLEAN_QUERY) != skeleton_fingerprint(
        OTHER_STRUCTURE)
    assert intent_fingerprint(CLEAN_QUERY) != intent_fingerprint(OTHER_STRUCTURE)
    assert fingerprints(CLEAN_QUERY)["question_form"] == "wh_question"
    assert fingerprints(OTHER_STRUCTURE)["question_form"] == "imperative"


def test_diversity_metrics_are_computed_correctly_on_a_known_input():
    q1, q2 = "What is the invoice total for the order?", "Give me the cost."
    report = diversity_report([q1, q1, q1, q2])
    assert report["n"] == 4
    assert report["exact_duplicate_rate"] == 0.5      # two surplus copies of q1
    assert report["distinct_exact"] == 2
    assert report["distinct_skeletons"] == 2
    assert report["max_skeleton_share"] == 0.75
    assert report["top10_intent_share"] == 1.0
    assert report["question_form_distribution"] == {"wh_question": 3,
                                                    "imperative": 1}
    assert diversity_report([q1, q2])["exact_duplicate_rate"] == 0.0


def _large_pool(target: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    out: List[str] = []
    for i, (bp, plan) in enumerate(rng.sample(PAIRS, 300)):
        if len(out) >= target:
            break
        try:
            inst = instantiate(bp, plan, 12000 + i, track=TRACK)
        except BuildError:
            continue
        for k in range(2):
            for mode in MODES:
                contract = build_contract(inst, bp, plan, mode=mode,
                                          task_id=f"q{i}_{k}",
                                          seed=100 + i * 3 + k)
                out.append(render_deterministic(contract, mode,
                                                seed=100 + i * 3 + k)["query"])
    return out[:target]


LARGE_POOL = _large_pool(2000, seed=505)
LARGE_REPORT = diversity_report(LARGE_POOL)


def test_a_rate_card_is_keyed_by_data_so_its_keys_must_be_stated():
    card = {"dispatch": 7.9, "quality": 36.26}
    assert sorted(qv._mapping_strings(card)) == ["dispatch", "quality"]


def test_a_record_is_keyed_by_its_schema_so_its_values_must_be_stated():
    record = {"label": "desk lamp", "amount": 666.97, "site": "central store"}
    assert sorted(qv._mapping_strings(record)) == ["central store", "desk lamp"]


def test_a_query_that_never_recites_a_column_name_still_states_its_facts():
    """"label" is a column name; naming it would be disclosure, not preservation."""
    contract = {
        "entities": [], "expected_numbers": ["666.97"], "mode": "SEMI_IMPLICIT",
        "call_count": 3,
        "expected_strings": qv._mapping_strings(
            {"label": "desk lamp", "amount": 666.97}),
    }
    query = "Ines is processing the card for a desk lamp worth 666.97."
    assert qv.check_facts(query, contract)["passed"] is True


def test_two_thousand_generated_queries_contain_no_exact_duplicate():
    assert LARGE_REPORT["n"] == 2000
    assert LARGE_REPORT["exact_duplicate_rate"] == 0.0
    assert LARGE_REPORT["distinct_exact"] == 2000


def test_the_large_pool_reports_plausible_concentration_metrics(record_property):
    for key in ("max_skeleton_share", "max_intent_share", "top10_intent_share"):
        record_property(key, LARGE_REPORT[key])
    # scale-dependent production gates are asserted by the pipeline, not here
    assert 0.0 < LARGE_REPORT["max_skeleton_share"] <= 1.0
    assert LARGE_REPORT["max_skeleton_share"] <= LARGE_REPORT["max_intent_share"]
    assert (LARGE_REPORT["max_intent_share"]
            <= LARGE_REPORT["top10_intent_share"] <= 1.0)
    assert LARGE_REPORT["distinct_skeletons"] >= LARGE_REPORT[
        "distinct_intent_templates"]
