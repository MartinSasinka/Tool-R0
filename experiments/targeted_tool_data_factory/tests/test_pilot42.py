"""Pilot4.2 workflow-first invariants."""
from __future__ import annotations

import json

import pytest

from targeted_tool_data.pilot42.cells import build_cells, cells_summary
from targeted_tool_data.pilot42.generate import (generate_program_from_workflow,
                                                  generate_semantic_pool,
                                                  instantiate_workflow)
from targeted_tool_data.pilot42.openrouter import (load_openrouter_config,
                                                    needs_second_critic,
                                                    redact_secret)
from targeted_tool_data.pilot42.primitives_v2 import (bind_capability,
                                                       build_primitive_registry,
                                                       validate_primitive_registry)
from targeted_tool_data.pilot42.query_contract import build_query_contract
from targeted_tool_data.pilot42.query_render import render_query
from targeted_tool_data.pilot42.query_validators import validate_query
from targeted_tool_data.pilot42.select import eligible
from targeted_tool_data.pilot42.semantic_types import (SemanticType,
                                                        semantic_compatible)
from targeted_tool_data.pilot42.split import split_records
from targeted_tool_data.pilot42.subsets import (assert_nested,
                                                 nested_stratified_subsets)
from targeted_tool_data.pilot42.v4_gate import evaluate_v4
from targeted_tool_data.pilot42.validate_semantic import validate_record
from targeted_tool_data.pilot42.workflows_v2 import get_workflows


@pytest.fixture
def record():
    workflow = get_workflows()[0]
    instance = instantiate_workflow(workflow, 11)
    return generate_program_from_workflow(
        workflow, instance, workflow.allowed_structural_patterns[0], "medium", 11)


def test_workflow_registry_has_55_to_70():
    assert 55 <= len(get_workflows()) <= 70


def test_every_workflow_has_real_plan():
    assert all(w.plan_template and w.input_roles and w.target_role for w in get_workflows())


def test_workflow_first_provenance(record):
    assert record["was_generated_from_workflow"] is True
    assert record["provenance"]["semantic_plan_source"] == "workflow_blueprint"


def test_capability_binding_exact():
    assert bind_capability("arithmetic.percentage_of").primitive_id == "percent_of"


def test_unknown_capability_fails_closed():
    with pytest.raises(ValueError):
        bind_capability("not.a.capability")


def test_primitive_registry_complete():
    assert validate_primitive_registry(build_primitive_registry()) == []


def test_generic_to_money_forbidden():
    ok, reason = semantic_compatible("GenericScalar", "Money")
    assert not ok and "forbidden" in reason


def test_generic_to_duration_forbidden():
    assert not semantic_compatible("GenericScalar", "DurationDays")[0]


def test_instance_matches_workflow(record):
    assert record["workflow_instance"]["workflow_id"] == record["workflow_id"]


def test_disallowed_pattern_rejected():
    workflow = get_workflows()[0]
    with pytest.raises(ValueError):
        generate_program_from_workflow(
            workflow, instantiate_workflow(workflow, 1), "DIAMOND", "easy", 1)


def test_generated_capabilities_equal_plan(record):
    workflow = get_workflows()[0]
    assert [n["capability"] for n in record["semantic_program"]["nodes"]] == [
        n.capability for n in workflow.plan_template]


def test_semantic_replay_and_necessity(record):
    report = validate_record(record)
    assert report["passed"], report


def test_query_contract_uses_instance_facts():
    workflow = get_workflows()[0]
    instance = instantiate_workflow(workflow, 3)
    contract = build_query_contract(instance, workflow)
    assert contract["facts"][0]["value"] == instance["facts"][workflow.input_roles[0]]["value"]


def test_renderer_does_not_dump_graph(record):
    low = record["question"].lower()
    assert "stages are related" not in low
    assert "$var" not in low


def test_graph_leak_rejected(record):
    record["question"] += " The stages are related as follows: stage 1 then stage 2."
    assert not validate_query(record)["layers"]["V_QUERY_GRAPH_LEAK"]["passed"]


def test_query_hard_validators_pass(record):
    result = validate_query(record)
    assert result["passed"], result


def test_v4_equal_length_not_shortcut(record, monkeypatch):
    monkeypatch.setattr(
        "targeted_tool_data.pilot42.v4_gate.v4_minimal_path",
        lambda *a, **k: (["raw"], {"searched": True, "shortcut_depth": 2}))
    result = evaluate_v4({**record, "semantic_program_id": record["semantic_program_id"] + "x"})
    assert result["has_shortcut"] is False


def test_v4_shorter_path_rejected(record, monkeypatch):
    monkeypatch.setattr(
        "targeted_tool_data.pilot42.v4_gate.v4_minimal_path",
        lambda *a, **k: (["raw"], {"searched": True, "shortcut_depth": 1}))
    result = evaluate_v4({**record, "semantic_program_id": record["semantic_program_id"] + "y"})
    assert result["passed"] is False


def test_v4_unresolved_rejected(record, monkeypatch):
    monkeypatch.setattr(
        "targeted_tool_data.pilot42.v4_gate.v4_minimal_path",
        lambda *a, **k: ([], {"searched": True, "exhausted": True}))
    result = evaluate_v4({**record, "semantic_program_id": record["semantic_program_id"] + "z"})
    assert result["unresolved"] and not result["passed"]


def test_selection_requires_all_gates(record):
    record["semantic_validation"] = {"passed": True}
    record["query_validation"] = {"passed": True}
    record["v4_gate"] = {"passed": True, "has_shortcut": False, "unresolved": False}
    assert eligible(record)
    record["v4_gate"]["unresolved"] = True
    assert not eligible(record)


def test_nested_subsets_are_nested_not_input_prefix():
    rows = [{"task_id": str(i), "semantic_program_id": str(i),
             "workflow_id": f"w{i % 3}", "pattern_family": f"p{i % 2}",
             "requested_query_mode": f"m{i % 4}"} for i in range(40)]
    subsets = nested_stratified_subsets(rows, sizes=(10, 20, 30), seed=7)
    assert assert_nested(subsets)
    assert [r["task_id"] for r in subsets[10]] != [str(i) for i in range(10)]


def test_split_union_find_has_no_hard_key_leakage():
    rows = [{"task_id": str(i), "semantic_program_id": f"s{i // 2}",
             "program_family_id": f"p{i // 2}", "workflow_instance_id": f"i{i // 2}",
             "query_template_fingerprint": f"q{i // 2}"} for i in range(20)]
    _, audit = split_records(rows, {"train": 12, "heldout": 4, "reserve": 4}, 1)
    assert audit["leak_free"]


def test_openrouter_budget_and_models():
    cfg = load_openrouter_config()
    assert cfg["max_total_cost_usd"] == 20
    assert cfg["allow_fallbacks"] is False


def test_second_critic_required_for_six_calls():
    assert needs_second_critic({"task_id": "x", "gold_calls": [{}] * 6})


def test_secret_redaction():
    value = {"Authorization": "Bearer sk-secret", "nested": ["sk-secret"]}
    assert "sk-secret" not in json.dumps(redact_secret(value))


def test_cells_cover_core_and_enrichment():
    cells = build_cells()
    summary = cells_summary(cells)
    assert 50 <= summary["n_cells"] <= 80
    assert summary["target_total"] >= 1500
    tiers = summary["tier_targets"]
    assert tiers.get("CORE_PROFILE", 0) > 0
    assert "STRUCTURAL_ENRICHMENT" in tiers or "CAPABILITY_ENRICHMENT" in tiers
    assert all(c.min_support >= 5 for c in cells if c.tier == "CORE_PROFILE")


def test_no_random_dag_assignment_symbol():
    import targeted_tool_data.pilot42.generate as module
    assert not hasattr(module, "create_random_dag")


def test_small_pool_is_deterministic():
    a = generate_semantic_pool(build_cells()[:2], 8, 19)
    b = generate_semantic_pool(build_cells()[:2], 8, 19)
    assert [r["semantic_program_id"] for r in a] == [
        r["semantic_program_id"] for r in b]
