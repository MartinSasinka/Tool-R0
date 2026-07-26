"""Exporter parity, providers, probing P0 + predicted-call execution."""
import json

from targeted_tool_data.export import to_grpo_row, to_nestful_row
from targeted_tool_data.generation import make_candidate
from targeted_tool_data.probing import (execute_predicted, p0_structural,
                                        parse_calls)
from targeted_tool_data.providers import (OpenAICompatibleLocalProvider,
                                          TemplateOnlyProvider, make_provider)
from targeted_tool_data.schemas import GenerationCell

BUCKETS_CFG = {"small": [8, 9], "medium": [10, 12], "large": [13, 18]}
CONV = {"param_styles": ["semantic"], "label_styles": ["$var{i}"]}


def _rec(cc=3, idx=0):
    cell = GenerationCell(
        generation_cell_id=f"A_{cc}call_linear_test_00", track="A",
        mode="adaptation", call_count=cc, motif="linear",
        target_skill="s", target_failure="f",
        hard_distractor_type="near_semantics", quota_weight=1.0)
    r = make_candidate(cell, idx, 42, CONV, BUCKETS_CFG, "pv", "rh", "ch")
    assert r is not None
    return r.model_dump()


def test_nestful_export_parity():
    r = _rec()
    row = to_nestful_row(r)
    assert row["input"] == r["query"]
    assert row["gold_answer"] == r["gold_answer"]
    assert [c["name"] for c in row["output"]] == \
        [c["name"] for c in r["canonical_calls"]]
    # NESTFUL flat-dict parameters shape
    t = row["tools"][0]
    assert "parameters" in t and "output_parameters" in t
    assert "properties" not in t["parameters"]
    json.dumps(row)   # JSON-serializable


def test_grpo_export_parity_and_metadata():
    r = _rec()
    row = to_grpo_row(r)
    for key in ("sample_id", "question", "tools", "gold_calls", "gold_answer",
                "observations", "num_calls", "stage", "motif_type",
                "answer_type", "provenance"):
        assert key in row, key
    assert row["gold_calls"] == [
        {"name": c["name"], "arguments": c["arguments"], "label": c["label"]}
        for c in r["canonical_calls"]]
    assert row["observations"] == r["oracle_observations"]
    # JSON-schema style tools (trainer contract)
    t = row["tools"][0]
    assert t["parameters"]["type"] == "object"
    assert "properties" in t["parameters"]
    # metadata not silently dropped
    prov = row["provenance"]
    for key in ("generation_cell_id", "track", "target_skill",
                "registry_hash", "executor_hash"):
        assert prov.get(key), key


def test_provider_fallback_and_no_remote_default():
    p = make_provider({"kind": "nonexistent_kind"})
    assert isinstance(p, TemplateOnlyProvider)
    p2 = make_provider({"kind": "openai_compatible_local",
                        "base_url": "https://api.openai.com/v1"})
    assert isinstance(p2, OpenAICompatibleLocalProvider)
    assert not p2.available()   # remote endpoints are never "available"
    p3 = make_provider({"kind": "openai_compatible_local",
                        "base_url": "http://127.0.0.1:1234/v1"}, no_llm=True)
    assert isinstance(p3, TemplateOnlyProvider)   # --no-llm forces templates


def test_p0_structural_range():
    r = _rec()
    d = p0_structural(r)
    assert 0.0 <= d <= 1.0


def test_parse_and_execute_predicted_gold_wins():
    r = _rec()
    text = json.dumps([{"name": c["name"], "arguments": c["arguments"],
                        "label": c["label"]} for c in r["canonical_calls"]])
    calls = parse_calls(text)
    assert calls is not None
    res = execute_predicted(calls, r)
    assert res["success"], res


def test_execute_predicted_wrong_tool_fails():
    r = _rec()
    calls = [{"name": r["offered_tools"][0]["name"],
              "arguments": {}, "label": "$var1"}]
    res = execute_predicted(calls, r)
    assert not res["success"]
