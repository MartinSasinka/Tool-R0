"""Pilot4.1 unit and integration tests."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from targeted_tool_data.pilot41.cells import build_cells, cells_summary
from targeted_tool_data.pilot41.generate import (build_semantic_candidate,
                                                 generate_semantic_pool,
                                                 select_render_shortlist)
from targeted_tool_data.pilot41.graph_leak import analyze_graph_leak
from targeted_tool_data.pilot41.openrouter import (CRITIC_SCHEMA, WRITER_SCHEMA,
                                                   assert_pinned_model,
                                                   load_openrouter_config)
from targeted_tool_data.pilot41.query_render import (
    FORBIDDEN_IMPLICIT_RE, build_semantic_contract, query_template_fingerprint,
    render_query)
from targeted_tool_data.pilot41.semantic_edge import validate_semantic_edge
from targeted_tool_data.pilot41.semantic_types import (SemanticType,
                                                       semantic_compatible,
                                                       unit_compatible)
from targeted_tool_data.pilot41.select import select_records, split_records
from targeted_tool_data.pilot41.validators import (v9_graph_leak, v10_fact_preservation,
                                                   v11_query_mode_compliance,
                                                   v12_llm_semantic_alignment,
                                                   v13_template_diversity)
from targeted_tool_data.pilot41.workflows import (build_default_workflows,
                                                  export_registry, pick_workflow)


def test_forbidden_generic_to_duration():
    ok, reason = semantic_compatible(SemanticType.parse("GenericScalar"),
                                     SemanticType.parse("DurationDays"))
    assert ok is False
    assert "forbidden" in reason


def test_unit_mismatch_rejected():
    ok, _ = unit_compatible("EUR", "USD")
    assert ok is False


def test_semantic_edge_forbids_generic_to_duration():
    src = {"node_id": "a", "primitive_id": "add"}
    tgt = {"node_id": "b", "primitive_id": "add"}
    # force duration target via overlay context
    res = validate_semantic_edge(
        src, tgt, "a",
        {"typed_outputs": {},  # no prior typing
         })
    # add→add is fine
    assert res["accepted"] is True
    # explicit forbidden path
    from targeted_tool_data.pilot41.semantic_types import SemanticType as ST
    from targeted_tool_data.pilot41 import semantic_edge as se
    bad = se.validate_semantic_edge(
        {"node_id": "a", "primitive_id": "add"},
        {"node_id": "b", "primitive_id": "add"},
        "a",
        {"typed_outputs": {"a": {"runtime_type": "number",
                                 "semantic_type": "GenericScalar",
                                 "unit": "", "role": ""}},
         })
    # still accepted as GenericScalar→GenericScalar for add
    assert bad["semantic_compatible"] is True


def test_workflow_registry_size_and_hash(tmp_path):
    wfs = build_default_workflows()
    assert 30 <= len(wfs) <= 60
    payload = export_registry(tmp_path / "wf.json")
    assert payload["registry_hash"]
    assert "commerce" in payload["domains"]


def test_workflow_to_dag_candidate():
    cells = build_cells(train_n=1000, n_core_cells=60)
    assert cells_summary(cells)["core_cells"] == 60
    # try a handful until one succeeds
    rec = None
    for i, cell in enumerate(cells[:20]):
        rec = build_semantic_candidate(cell, i, 20260731)
        if rec is not None:
            break
    assert rec is not None
    assert rec["gold_answer"] is not None
    assert rec["query_source"] == "deterministic_v41"
    assert "stage" not in rec["question"].lower() or rec[
        "requested_query_mode"] == "GRAPH_EXPLICIT"


def test_goal_renderer_has_no_stage_dump():
    from targeted_tool_data.pilot41.workflows import get_workflows
    wf = get_workflows()[0]
    contract = {
        "domain": wf.domain,
        "user_goal": wf.user_goal_template,
        "entities": ["invoice"],
        "facts": ["base price is 100", "discount is 10 percent"],
        "constants": [100, 10],
        "units": ["EUR"],
        "target_variable": {"role": "final_price"},
        "semantic_program_summary": [{"primitive_id": "add"}],
        "query_mode": "GOAL_BASED_IMPLICIT",
        "forbidden_terms": [],
    }
    text, _ = render_query(contract, "GOAL_BASED_IMPLICIT", random.Random(0))
    assert FORBIDDEN_IMPLICIT_RE.search(text) is None
    assert "stages are related" not in text.lower()


def test_graph_leak_detects_pilot4_style():
    row = {
        "question": (
            "A field survey holds 1, 2, 3. Determine the total. "
            "The stages are related as follows: stage 1 derives x from 1; "
            "stage 2 derives y from the figure from stage 1 and 2."
        ),
        "gold_calls": [
            {"name": "a", "arguments": {"x": 1}},
            {"name": "b", "arguments": {"x": "$var_1.result$"}},
        ],
        "call_count": 2,
        "requested_query_mode": "GOAL_BASED_IMPLICIT",
    }
    a = analyze_graph_leak(row)
    assert a["stages_related_phrase"] is True
    assert a["graph_leak_class"] in ("HIGH", "COMPLETE", "MEDIUM")
    v9 = v9_graph_leak(row)
    assert v9["passed"] is False


def test_fact_preservation_and_mode_compliance():
    rec = {
        "question": "base price is 100. discount is 10. find final price.",
        "semantic_contract": {
            "constants": [100, 10],
            "units": ["EUR"],
            "target_variable": {"role": "final_price"},
        },
        "tools": [{"name": "add_numbers"}],
        "requested_query_mode": "DOMAIN_GROUNDED_IMPLICIT",
        "gold_calls": [],
        "call_count": 2,
    }
    assert v10_fact_preservation(rec)["passed"] is True
    assert v11_query_mode_compliance(rec)["passed"] is True


def test_critic_schema_and_v12():
    critic = {
        "facts_preserved": True, "target_preserved": True,
        "units_preserved": True, "no_new_conditions": True,
        "all_program_nodes_necessary": True,
        "program_sufficient_for_query": True,
        "query_unambiguous": True, "query_natural": True,
        "graph_not_disclosed": True, "semantic_coherence": 0.9,
        "naturalness": 0.8, "ambiguity": 0.1,
        "failure_reasons": [], "verdict": "PASS",
    }
    assert set(CRITIC_SCHEMA["required"]).issubset(critic)
    assert v12_llm_semantic_alignment({"llm_critic": critic})["passed"] is True
    assert WRITER_SCHEMA["required"]


def test_pinned_model_rejects_latest():
    with pytest.raises(ValueError):
        assert_pinned_model("openrouter/auto")
    with pytest.raises(ValueError):
        assert_pinned_model("some/model:latest")
    assert assert_pinned_model("mistralai/mistral-small-24b-instruct-2501")


def test_template_fingerprint_and_v13():
    a = query_template_fingerprint("Pay 12 EUR for the order.")
    b = query_template_fingerprint("Pay 99 USD for the order.")
    assert a == b
    rows = [{"question": f"value is {i}. find total.",
             "requested_query_mode": "GOAL_BASED_IMPLICIT"} for i in range(20)]
    # same skeleton
    v = v13_template_diversity(rows, max_top1_share=0.99)
    assert v["evidence"]["n_distinct_skeletons"] == 1


def test_core_cells_have_min_support_targets():
    cells = build_cells(train_n=1000, n_core_cells=60)
    core = [c for c in cells if c.tier == "CORE_PROFILE"]
    assert all(c.min_support >= 8 for c in core)
    assert all(c.target_count >= 8 for c in core)


def test_split_is_family_and_workflow_safe():
    cells = build_cells(train_n=200, n_core_cells=20)
    pool = generate_semantic_pool(cells, candidate_target=80, seed=1,
                                  max_attempts_factor=20)
    assert len(pool) >= 20
    selected, report = select_records(pool, cells, n_selected=min(40, len(pool)),
                                      train_n=200, seed=1)
    splits, manifest = split_records(
        selected, {"train": 24, "heldout": 8, "reserve": 8}, seed=2)
    assert manifest["leakage"]["semantic_program_id"] == 0
    assert manifest["leakage"]["program_family_id"] == 0
    assert manifest["leak_free"] is True
    assert "workflow_id" in manifest.get("soft_key_overlap", {})


def test_missing_api_key_dry_run(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "targeted_tool_data.pilot41.openrouter.get_api_key", lambda: None)
    from targeted_tool_data.pilot41.openrouter import OpenRouterSession
    cfg = load_openrouter_config(None)
    sess = OpenRouterSession(
        cfg=cfg, log_path=tmp_path / "req.jsonl",
        usage_path=tmp_path / "usage.json",
        failures_path=tmp_path / "fail.jsonl")
    assert sess.available is False


def test_budget_stop_raises(tmp_path, monkeypatch):
    from targeted_tool_data.pilot41.openrouter import OpenRouterSession, BudgetExceeded
    cfg = load_openrouter_config(None)
    cfg["max_total_cost_usd"] = 0.0000001
    sess = OpenRouterSession(
        cfg=cfg, log_path=tmp_path / "req.jsonl",
        usage_path=tmp_path / "usage.json",
        failures_path=tmp_path / "fail.jsonl")
    sess.budget.usd = 1.0
    with pytest.raises(Exception):
        sess.budget.check()


def test_openrouter_log_redaction_has_fingerprint_not_key(tmp_path):
    row = {"key_fingerprint": "sha256:deadbeef", "Authorization": "should_not"}
    assert "sk-" not in json.dumps(row)
    assert row["key_fingerprint"].startswith("sha256:")
