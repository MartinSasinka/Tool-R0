"""Offline tests for the Pilot4.3 OpenRouter layer.

Every request goes through an injected fake transport; nothing in this module
opens a socket, and a test that tried to would fail on the missing API key.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from targeted_tool_data.pilot43 import (RUN_ID, TIER_CAPABILITY, TIER_CHALLENGE,
                                        TIER_LONG_HORIZON, TIER_PROFILE_CORE)
from targeted_tool_data.pilot43 import orclient as oc
from targeted_tool_data.pilot43 import orprompts as op
from targeted_tool_data.pilot43 import orrun

CONFIG_PATH = (Path(__file__).resolve().parents[1] / "configs"
               / "pilot4_3_openrouter.yaml")


# ── fake transport ───────────────────────────────────────────────────────
class FakeTransport:
    """Serves scripted responses and records every call."""

    def __init__(self, responses: Sequence[oc.HttpResponse]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, url: str, payload: Dict[str, Any],
                 headers: Dict[str, str], timeout: float) -> oc.HttpResponse:
        self.calls.append({"url": url, "payload": payload, "timeout": timeout})
        if not self.responses:
            raise AssertionError("fake transport ran out of scripted responses")
        return self.responses.pop(0)


def ok_body(model: str, content: Dict[str, Any], *, cost: float = 0.0001,
            provider: str = "FakeProvider") -> str:
    return json.dumps({
        "id": "gen-fake-1",
        "model": model,
        "provider": provider,
        "choices": [{"message": {"content": json.dumps(content)}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": cost},
    })


def ok(model: str, content: Dict[str, Any], **kw: Any) -> oc.HttpResponse:
    return oc.HttpResponse(200, ok_body(model, content, **kw),
                           {"x-request-id": "req-1"})


def rate_limited(retry_after: str = "1") -> oc.HttpResponse:
    return oc.HttpResponse(429, '{"error":"slow down"}',
                           {"retry-after": retry_after})


WRITER_CONTENT = {
    "query": "Marta left the figures: the base is 120 EUR. What is the total?",
    "facts_stated": ["base"],
    "units_stated": ["EUR"],
    "target_sentence": "What is the total?",
    "self_check": {k: True for k in op._SELF_CHECK_KEYS},
    "notes": [],
}
REWRITE_CONTENT = {
    **WRITER_CONTENT,
    "query": "Marta sent the base of 120 EUR. What total do we report?",
    "changes_made": ["removed a limit"],
}


def critic_content(verdict: str = "PASS") -> Dict[str, Any]:
    return {
        "workflow_matches_query": verdict == "PASS",
        "sink_answers_target": True,
        "all_query_facts_used": True,
        "all_program_nodes_required": True,
        "no_extra_conditions": True,
        "units_semantically_valid": True,
        "query_unambiguous": True,
        "query_natural": True,
        "graph_not_disclosed": True,
        "node_alignment": [{"node_id": "n1", "required_by_query": True,
                            "query_evidence": "the total",
                            "semantic_role_matches": True, "aligned": True}],
        "verdict": verdict,
    }


# ── fixtures ─────────────────────────────────────────────────────────────
@pytest.fixture()
def cfg() -> oc.OpenRouterConfig:
    return oc.load_openrouter_config(CONFIG_PATH)


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    path = tmp_path / RUN_ID
    path.mkdir()
    return path


def make_client(cfg: oc.OpenRouterConfig, out_dir: Path,
                transport: Optional[Any] = None, **kw: Any) -> oc.OpenRouterClient:
    slept: List[float] = []
    client = oc.OpenRouterClient(
        cfg, out_dir, transport=transport, api_key="test-key",
        sleep=slept.append, jitter=lambda: 0.0, **kw)
    client.slept = slept                                    # type: ignore[attr-defined]
    return client


SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["query"], "properties": {"query": {"type": "string"}}}
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
META = {"sample_id": "t1", "workflow_id": "wf1", "semantic_program_id": "sp1"}


# ── configuration ────────────────────────────────────────────────────────
def test_config_pins_three_distinct_model_families(cfg: oc.OpenRouterConfig) -> None:
    writer = cfg.model_for("writer")
    critic = cfg.model_for("critic")
    second = cfg.model_for("critic2")
    assert writer.split("/")[0] != critic.split("/")[0]
    assert critic.split("/")[0] != second.split("/")[0]
    for purpose in oc.PURPOSES:
        oc.assert_pinned_model(cfg.model_for(purpose))
        assert cfg.prompt_version_for(purpose).startswith("pilot43.")
    assert cfg.run_id == RUN_ID


@pytest.mark.parametrize("slug", ["openai/gpt-4o:free", "openrouter/auto",
                                  "anthropic/claude-latest", "gpt-4o", ""])
def test_unpinned_model_slugs_are_refused(slug: str) -> None:
    with pytest.raises(ValueError):
        oc.assert_pinned_model(slug)


def test_no_model_slug_literals_in_python() -> None:
    src_dir = Path(__file__).resolve().parents[1] / "src" / "targeted_tool_data"
    for name in ("orclient.py", "orprompts.py", "orrun.py"):
        text = (src_dir / "pilot43" / name).read_text(encoding="utf-8")
        for vendor in ("openai/", "anthropic/", "google/", "mistralai/"):
            assert vendor not in text, f"{name} hard-codes a model slug"


def test_config_rejects_same_family_critic(cfg: oc.OpenRouterConfig) -> None:
    block = dict(cfg.raw)
    block["critic_model"] = block["writer_model"]
    with pytest.raises(ValueError, match="different family"):
        oc.build_config(block)


# ── run isolation ────────────────────────────────────────────────────────
def test_configured_and_actual_model_mismatch_raises(cfg, out_dir) -> None:
    other = cfg.model_for("critic")
    transport = FakeTransport([ok(other, {"query": "x"})])
    client = make_client(cfg, out_dir, transport)
    with pytest.raises(oc.RunIsolationError, match="came from"):
        client.chat("writer", MESSAGES, SCHEMA, META)


def test_prompt_version_must_carry_the_run_prefix(cfg, out_dir) -> None:
    client = make_client(cfg, out_dir, FakeTransport([]))
    meta = {**META, "prompt_version": "pilot42.writer.v1"}
    with pytest.raises(oc.RunIsolationError, match="not a pilot43 prompt"):
        client.chat("writer", MESSAGES, SCHEMA, meta)


def test_output_directory_basename_must_be_the_run_id(cfg, tmp_path) -> None:
    wrong = tmp_path / "some_other_run"
    wrong.mkdir()
    with pytest.raises(oc.RunIsolationError, match="not the run id"):
        oc.OpenRouterClient(cfg, wrong, transport=FakeTransport([]),
                            api_key="test-key")


def test_refuses_to_append_to_a_foreign_run_log(cfg, out_dir) -> None:
    log = out_dir / oc.REQUEST_LOG
    log.write_text(json.dumps({"run_id": "pilot4_2_something"}) + "\n",
                   encoding="utf-8")
    with pytest.raises(oc.RunIsolationError, match="other runs"):
        oc.OpenRouterClient(cfg, out_dir, transport=FakeTransport([]),
                            api_key="test-key")
    with pytest.raises(oc.RunIsolationError):
        oc.assert_log_isolation(out_dir)
    assert oc.count_foreign_run_records(out_dir) == 1


def test_log_isolation_accepts_own_records(cfg, out_dir) -> None:
    transport = FakeTransport([ok(cfg.model_for("writer"), {"query": "x"})])
    client = make_client(cfg, out_dir, transport)
    client.chat("writer", MESSAGES, SCHEMA, META)
    assert oc.assert_log_isolation(out_dir)["records"] == 1
    assert oc.count_foreign_run_records(out_dir) == 0


def test_request_record_has_every_required_field(cfg, out_dir) -> None:
    transport = FakeTransport([ok(cfg.model_for("writer"), {"query": "x"})])
    client = make_client(cfg, out_dir, transport)
    result = client.chat("writer", MESSAGES, SCHEMA, META)
    row = json.loads((out_dir / oc.REQUEST_LOG).read_text(
        encoding="utf-8").splitlines()[0])
    required = {"run_id", "sample_id", "workflow_id", "semantic_program_id",
                "purpose", "prompt_version", "configured_model", "actual_model",
                "provider", "request_id", "latency_ms", "usage", "cost_usd",
                "cache_hit", "http_status", "attempt"}
    assert required <= set(row)
    assert row["run_id"] == RUN_ID
    assert row["configured_model"] == row["actual_model"] == cfg.model_for("writer")
    assert row["purpose"] == "writer"
    assert row["cache_hit"] is False
    assert row["raw_response_sha256"] == result["raw_response_sha256"]
    assert "test-key" not in (out_dir / oc.REQUEST_LOG).read_text(encoding="utf-8")


# ── retries, failures, budget ────────────────────────────────────────────
def test_429_backoff_retries_and_eventually_succeeds(cfg, out_dir) -> None:
    model = cfg.model_for("writer")
    transport = FakeTransport([rate_limited("3"), rate_limited("3"),
                               ok(model, {"query": "x"})])
    client = make_client(cfg, out_dir, transport)
    result = client.chat("writer", MESSAGES, SCHEMA, META)
    assert result["attempts"] == 3
    assert result["record"]["attempt"] == 2
    # Retry-After is respected and capped by backoff_max_seconds
    assert client.slept == [3.0, 3.0]                       # type: ignore[attr-defined]
    failures = (out_dir / oc.FAILURE_LOG).read_text(encoding="utf-8").splitlines()
    assert len(failures) == 2
    assert json.loads(failures[0])["http_status"] == 429


def test_backoff_is_exponential_without_retry_after(cfg, out_dir) -> None:
    model = cfg.model_for("writer")
    transport = FakeTransport([oc.HttpResponse(503, "busy", {}),
                               oc.HttpResponse(503, "busy", {}),
                               ok(model, {"query": "x"})])
    client = make_client(cfg, out_dir, transport)
    client.chat("writer", MESSAGES, SCHEMA, META)
    base = cfg.backoff_seconds_base
    assert client.slept == [base, base * 2]                 # type: ignore[attr-defined]


def test_invalid_json_retries_then_records_a_failure(cfg, out_dir) -> None:
    model = cfg.model_for("writer")
    bad = oc.HttpResponse(200, json.dumps({
        "model": model, "provider": "p",
        "choices": [{"message": {"content": "not json at all"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0}}), {})
    transport = FakeTransport([bad] * (cfg.max_retries + 1))
    client = make_client(cfg, out_dir, transport)
    with pytest.raises(oc.StructuredOutputError):
        client.chat("writer", MESSAGES, SCHEMA, META)
    failures = (out_dir / oc.FAILURE_LOG).read_text(encoding="utf-8").splitlines()
    assert len(failures) == cfg.max_retries + 1
    assert not (out_dir / oc.REQUEST_LOG).exists()


def test_schema_violating_json_is_treated_as_invalid(cfg, out_dir) -> None:
    model = cfg.model_for("writer")
    wrong = ok(model, {"query": 7})
    transport = FakeTransport([wrong] * (cfg.max_retries + 1))
    client = make_client(cfg, out_dir, transport)
    with pytest.raises(oc.StructuredOutputError):
        client.chat("writer", MESSAGES, SCHEMA, META)


def messages(marker: str) -> List[Dict[str, str]]:
    """Distinct messages so the on-disk cache does not answer for the network."""
    return [{"role": "system", "content": "s"},
            {"role": "user", "content": marker}]


def test_budget_exceeded_raises_and_stops_further_calls(cfg, out_dir) -> None:
    model = cfg.model_for("writer")
    tight = oc.build_config({**cfg.raw, "max_total_cost_usd": 0.001,
                             "max_cost_per_task_usd": 10.0})
    transport = FakeTransport([ok(model, {"query": "x"}, cost=0.002),
                               ok(model, {"query": "y"}, cost=0.002)])
    client = make_client(tight, out_dir, transport)
    with pytest.raises(oc.BudgetExceeded) as first:
        client.chat("writer", messages("one"), SCHEMA, META)
    assert first.value.scope == "total"
    # the spend is persisted, so the next call refuses before the network
    with pytest.raises(oc.BudgetExceeded):
        client.chat("writer", messages("two"), SCHEMA, {**META, "sample_id": "t2"})
    assert len(transport.calls) == 1
    usage = json.loads((out_dir / oc.USAGE_FILE).read_text(encoding="utf-8"))
    assert usage["run_id"] == RUN_ID
    assert usage["totals"]["cost_usd"] == pytest.approx(0.002)


def test_per_task_budget_is_scoped_to_the_task(cfg, out_dir) -> None:
    model = cfg.model_for("writer")
    tight = oc.build_config({**cfg.raw, "max_cost_per_task_usd": 0.001})
    transport = FakeTransport([ok(model, {"query": "x"}, cost=0.002),
                               ok(model, {"query": "z"}, cost=0.002)])
    client = make_client(tight, out_dir, transport)
    client.chat("writer", messages("one"), SCHEMA, META)
    with pytest.raises(oc.BudgetExceeded) as exc:
        client.chat("writer", messages("two"), SCHEMA, META)
    assert exc.value.scope == "task"
    other = client.chat("writer", messages("three"), SCHEMA,
                        {**META, "sample_id": "t2"})
    assert other["cache_hit"] is False


def test_missing_api_key_degrades_instead_of_crashing(cfg, out_dir,
                                                      monkeypatch) -> None:
    monkeypatch.setattr(oc, "get_api_key", lambda: None)
    client = oc.OpenRouterClient(cfg, out_dir, transport=FakeTransport([]))
    assert client.available() is False


# ── cache and replay ─────────────────────────────────────────────────────
def test_cache_replay_returns_identical_content_without_network(cfg,
                                                                out_dir) -> None:
    model = cfg.model_for("writer")
    transport = FakeTransport([ok(model, {"query": "cached answer"})])
    client = make_client(cfg, out_dir, transport)
    first = client.chat("writer", MESSAGES, SCHEMA, META)
    assert len(transport.calls) == 1

    empty = FakeTransport([])
    replay = make_client(cfg, out_dir, empty, replay_only=True)
    assert replay.available() is True
    second = replay.chat("writer", MESSAGES, SCHEMA, META)
    assert empty.calls == []
    assert second["parsed"] == first["parsed"]
    assert second["raw_text"] == first["raw_text"]
    assert second["raw_response_sha256"] == first["raw_response_sha256"]
    assert second["cache_hit"] is True
    assert second["cost_usd"] == 0.0


def test_replay_only_raises_on_a_cache_miss(cfg, out_dir) -> None:
    client = make_client(cfg, out_dir, FakeTransport([]), replay_only=True)
    with pytest.raises(oc.ReplayMiss):
        client.chat("writer", MESSAGES, SCHEMA, META)


def test_cache_key_depends_on_namespace_model_prompt_version_and_sample() -> None:
    args = (MESSAGES, SCHEMA)
    base = oc.cache_key("pilot43", "a/b", "pilot43.writer.v1", *args)
    assert base != oc.cache_key("other", "a/b", "pilot43.writer.v1", *args)
    assert base != oc.cache_key("pilot43", "a/c", "pilot43.writer.v1", *args)
    assert base != oc.cache_key("pilot43", "a/b", "pilot43.writer.v2", *args)
    assert base == oc.cache_key("pilot43", "a/b", "pilot43.writer.v1", *args)
    # two tasks may share a contract word for word and must not share a query
    assert base != oc.cache_key("pilot43", "a/b", "pilot43.writer.v1", *args, "t1")
    assert (oc.cache_key("pilot43", "a/b", "pilot43.writer.v1", *args, "t1")
            != oc.cache_key("pilot43", "a/b", "pilot43.writer.v1", *args, "t2"))


def test_two_tasks_with_an_identical_contract_each_get_their_own_request(
        cfg, out_dir) -> None:
    body = {"query": "What is the closing balance?"}
    transport = FakeTransport([ok(cfg.model_for("writer"), body),
                               ok(cfg.model_for("writer"), body)])
    client = make_client(cfg, out_dir, transport)
    client.chat("writer", MESSAGES, SCHEMA, {**META, "sample_id": "t1"})
    client.chat("writer", MESSAGES, SCHEMA, {**META, "sample_id": "t2"})
    assert client.totals.requests == 2
    assert client.totals.cache_hits == 0


# ── prompts ──────────────────────────────────────────────────────────────
def test_writer_prompt_states_every_prohibition() -> None:
    prompt = op.writer_prompt({"target": "the total"}, "DOMAIN_GROUNDED_IMPLICIT")
    assert prompt.prompt_version == "pilot43.writer.v6"
    body = (prompt.system + prompt.user).lower()
    for phrase in ("do not add a number", "never convert", "add no constraint",
                   "never write a tool name", "never disclose how many",
                   "never state", "must appear", "unambiguous question"):
        assert phrase in body
    assert "domain_grounded_implicit" in body
    assert prompt.schema is op.WRITER_SCHEMA


def test_writer_prompt_pairs_each_value_with_its_own_meaning() -> None:
    view = {"target": "the closing balance", "answer_type": "float",
            "user_goal": "close the month", "domain": "commerce",
            "scenario_entities": {"org": "Aurea Labs", "site": "the depot"},
            "stated_facts": [
                {"name": "readings", "means": "the readings taken on the run",
                 "value": "16.8, 25.18", "unit": ""},
                {"name": "places", "means": "decimals the report carries",
                 "value": "0", "unit": ""}]}
    user = op.writer_prompt(view, "GOAL_BASED_IMPLICIT").user
    assert "the readings taken on the run = 16.8, 25.18" in user
    assert "decimals the report carries = 0" in user
    assert "Aurea Labs" in user
    # the writer must never see a bag of values detached from their meanings
    assert "expected_numbers" not in user


@pytest.mark.parametrize("mode", ["DOMAIN_GROUNDED_IMPLICIT",
                                  "GOAL_BASED_IMPLICIT", "SEMI_IMPLICIT"])
def test_writer_prompt_is_mode_specific(mode: str) -> None:
    assert f"MODE {mode}" in op.writer_prompt({}, mode).user


def test_writer_prompt_rejects_an_explicit_mode() -> None:
    with pytest.raises(ValueError):
        op.writer_prompt({}, "GRAPH_EXPLICIT")


def test_critic_schema_matches_the_specified_verdict_shape() -> None:
    props = op.CRITIC_SCHEMA["properties"]
    assert set(op.CRITIC_SCHEMA["required"]) == {
        "workflow_matches_query", "sink_answers_target", "all_query_facts_used",
        "all_program_nodes_required", "no_extra_conditions",
        "units_semantically_valid", "query_unambiguous", "query_natural",
        "graph_not_disclosed", "node_alignment", "verdict"}
    assert props["verdict"]["enum"] == ["PASS", "REWRITE", "REJECT"]
    node = props["node_alignment"]["items"]
    assert set(node["required"]) == {"node_id", "required_by_query",
                                     "query_evidence", "semantic_role_matches",
                                     "aligned"}


def test_critic_prompt_carries_program_edges_and_oracle() -> None:
    context = {"workflow_goal": "price a refit", "target": "the total",
               "program": [{"node_id": "n1", "capability": "arithmetic.add",
                            "args": {"a": 1, "b": "$var_1.output_0$"}}],
               "edges": [["n1", "n2"]], "node_purposes": {"n1": "combine"},
               "input_facts": [{"name": "base", "value": "120 EUR"}],
               "observations": {"n1": 120}, "answer": 140}
    prompt = op.critic_prompt(context, "the query", {"passed": True,
                                                     "layers": {}})
    assert prompt.prompt_version == "pilot43.critic.v6"
    assert "arithmetic.add" in prompt.user and "oracle_observations" in prompt.user
    second = op.critic_prompt(context, "the query", {"passed": True,
                                                     "layers": {}},
                              second_opinion=True)
    assert second.prompt_version == "pilot43.critic2.v6"
    assert "second, independent critic" in second.system


def test_the_critic_is_told_that_unnamed_intermediate_steps_are_by_design() -> None:
    system = op.critic_prompt({}, "q", {"passed": True, "layers": {}}).system
    assert "implicit" in system
    assert "not when the query" in system
    assert "Intermediate quantities are supposed to be unnamed" in system


def test_rewrite_prompt_repeats_the_writer_constraints() -> None:
    prompt = op.rewrite_prompt({}, "SEMI_IMPLICIT", "bad query",
                               {"verdict": "REWRITE", "no_extra_conditions": False},
                               {"passed": False,
                                "layers": {"facts": {"passed": False}}})
    assert prompt.prompt_version == "pilot43.rewrite.v6"
    assert op.WRITER_RULES in prompt.user
    assert "no_extra_conditions" in prompt.user
    assert "changes_made" in prompt.schema["required"]


def test_prompt_hashes_cover_every_version() -> None:
    assert set(op.PROMPT_HASHES) == {"pilot43.writer.v6", "pilot43.critic.v6",
                                     "pilot43.critic2.v6", "pilot43.rewrite.v6"}
    assert all(len(h) == 64 for h in op.PROMPT_HASHES.values())
    assert len(set(op.PROMPT_HASHES.values())) == 4


# ── routing, stages, gates ───────────────────────────────────────────────
def task(task_id: str, *, mode: str = "DOMAIN_GROUNDED_IMPLICIT",
         tier: str = TIER_PROFILE_CORE, call_count: int = 3,
         coding: int = 0, answer_type: str = "float") -> orrun.RenderTask:
    return orrun.RenderTask(
        task_id=task_id, requested_mode=mode, workflow_id=f"wf_{task_id}",
        semantic_program_id=f"sp_{task_id}", tier=tier, call_count=call_count,
        coding_call_count=coding, answer_type=answer_type,
        prompt_contract={"target": "the total", "task_id": task_id},
        validator_contract=CONTRACT,
        critic_context={"program": [{"node_id": f"n_{task_id}"}]})


def test_second_critic_routing_selects_the_mandatory_subsets() -> None:
    assert orrun.needs_second_critic(task("a", call_count=9),
                                     first_verdict="PASS",
                                     rewritten=False)[1] == "call_count_8plus"
    assert orrun.needs_second_critic(
        task("a2", call_count=6, coding=2), first_verdict="PASS",
        rewritten=False)[1] == "coding_6plus"
    assert orrun.needs_second_critic(
        task("a3", call_count=6, answer_type="list"), first_verdict="PASS",
        rewritten=False)[1] == "structured_answer_6plus"
    # a plain 6-call numeric PROFILE_CORE task is no longer mandatory
    assert orrun.needs_second_critic(
        task("a4", call_count=6), first_verdict="PASS",
        rewritten=False, sample_rate=0.0)[0] is False
    assert orrun.needs_second_critic(
        task("b", tier=TIER_CHALLENGE), first_verdict="PASS",
        rewritten=False)[1] == "tier:CHALLENGE"
    assert orrun.needs_second_critic(
        task("b2", tier=TIER_CAPABILITY, call_count=6), first_verdict="PASS",
        rewritten=False)[1] == "tier:CAPABILITY_ENRICHMENT_6plus"
    assert orrun.needs_second_critic(task("c"), first_verdict="PASS",
                                     rewritten=True)[1] == "rewritten"
    for verdict in ("REWRITE", "REJECT", None):
        routed, reason = orrun.needs_second_critic(
            task("d"), first_verdict=verdict, rewritten=False)
        assert routed and reason == "first_critic_not_pass"


def test_second_critic_random_sample_is_deterministic_and_near_the_rate() -> None:
    plain = [task(f"t{i}", tier=TIER_LONG_HORIZON, call_count=3)
             for i in range(2000)]
    routed = [orrun.needs_second_critic(t, first_verdict="PASS",
                                        rewritten=False) for t in plain]
    reasons = {r for ok_, r in routed if ok_}
    assert reasons == {"random_sample"}
    share = sum(1 for ok_, _ in routed if ok_) / len(routed)
    assert 0.07 <= share <= 0.13
    again = [orrun.needs_second_critic(t, first_verdict="PASS", rewritten=False)
             for t in plain]
    assert again == routed


def test_disagreement_only_fires_on_pass_versus_blocking() -> None:
    assert orrun.disagreement("PASS", "REJECT") is True
    assert orrun.disagreement("REWRITE", "PASS") is True
    assert orrun.disagreement("REWRITE", "REJECT") is False
    assert orrun.disagreement("PASS", "PASS") is False
    assert orrun.disagreement("PASS", None) is False


def test_smoke_selection_covers_every_stratum_and_size() -> None:
    pool: List[orrun.RenderTask] = []
    modes = ("DOMAIN_GROUNDED_IMPLICIT", "GOAL_BASED_IMPLICIT", "SEMI_IMPLICIT")
    answers = ("float", "boolean", "string", "list", "object")
    for i in range(400):
        pool.append(task(f"t{i}", mode=modes[i % 3],
                         call_count=2 + (i % 7), coding=i % 2,
                         answer_type=answers[i % 5]))
    picked = orrun.select_stage("smoke", pool, seed=7)
    assert len(picked) == 50
    covered = {s for t in picked for s in orrun.strata_of(t)}
    available = {s for t in pool for s in orrun.strata_of(t)}
    for stratum in orrun.REQUIRED_SMOKE_STRATA:
        if stratum in available:
            assert stratum in covered, stratum
    assert {f"mode:{m}" for m in modes} <= covered
    assert picked == orrun.select_stage("smoke", pool, seed=7)
    assert len(orrun.select_stage("pilot", pool, seed=7)) == 300
    assert len(orrun.select_stage("full", pool, seed=7)) == 400


def test_smoke_gate_arithmetic() -> None:
    def rec(structured: bool, det: bool, verdict: str) -> Dict[str, Any]:
        return {"structured_output_ok": structured,
                "validation": {"passed": det},
                "critic": {"verdict": verdict}, "second_critic": None,
                "blocked": not det, "disagreement": False}

    records = [rec(True, True, "PASS") for _ in range(90)]
    records += [rec(True, True, "REWRITE") for _ in range(5)]
    records += [rec(True, False, "PASS") for _ in range(4)]
    records += [rec(False, False, "REJECT")]
    report = orrun.gate_report("smoke", records)
    assert report["metrics"]["structured_output_pass_rate"] == 0.99
    assert report["metrics"]["deterministic_pass_rate"] == 0.95
    assert report["metrics"]["critic_pass_rate"] == 0.94
    assert report["passed"] is True
    assert report["may_advance"] is True
    assert report["next_stage"] == "pilot"
    assert orrun.advance_or_raise(report) == "pilot"


def test_gate_report_refuses_to_advance_when_a_gate_fails() -> None:
    records = [{"structured_output_ok": True, "validation": {"passed": True},
                "critic": {"verdict": "REWRITE"}, "second_critic": None,
                "blocked": True, "disagreement": False} for _ in range(20)]
    report = orrun.gate_report("smoke", records)
    assert report["metrics"]["critic_pass_rate"] == 0.0
    assert report["failed_gates"] == ["critic_pass_rate"]
    assert report["may_advance"] is False
    with pytest.raises(orrun.StageGateFailed):
        orrun.advance_or_raise(report)


def test_mixed_run_log_records_fail_the_gate() -> None:
    records = [{"structured_output_ok": True, "validation": {"passed": True},
                "critic": {"verdict": "PASS"}, "second_critic": None,
                "blocked": False, "disagreement": False} for _ in range(10)]
    report = orrun.gate_report("smoke", records, mixed_run_log_records=1)
    assert report["failed_gates"] == ["mixed_run_log_records"]
    assert report["may_advance"] is False


def test_empty_stage_never_passes() -> None:
    assert orrun.gate_report("smoke", [])["may_advance"] is False


# ── end-to-end stage over the fake transport ─────────────────────────────
CONTRACT: Dict[str, Any] = {
    "mode": "DOMAIN_GROUNDED_IMPLICIT",
    "call_count": 3,
    "target_phrase": "the total",
    "expected_numbers": ["120"],
    "expected_strings": [],
    "expected_units": ["EUR"],
    "entities": ["Marta"],
    "forbidden_terms": [],
    "gold_capabilities": ["arithmetic.add"],
    "predicate_steps": 0,
    "answer_rendered": "140",
    "domain_vocabulary": ["base", "the total"],
}


def scripted(cfg: oc.OpenRouterConfig, n_tasks: int, verdict: str = "PASS",
             cost: float = 0.0001) -> FakeTransport:
    """Writer + critic + second critic per task, in call order."""
    responses: List[oc.HttpResponse] = []
    for _ in range(n_tasks):
        responses.append(ok(cfg.model_for("writer"), WRITER_CONTENT, cost=cost))
        responses.append(ok(cfg.model_for("critic"), critic_content(verdict),
                            cost=cost))
        responses.append(ok(cfg.model_for("critic2"), critic_content(verdict),
                            cost=cost))
    return FakeTransport(responses)


def test_run_stage_writes_records_and_is_resumable(cfg, out_dir) -> None:
    tasks = [task(f"t{i}", call_count=8 + i) for i in range(3)]
    client = make_client(cfg, out_dir, scripted(cfg, 3))
    report = orrun.run_stage("smoke", tasks, client, out_dir, seed=1)
    assert report["metrics"]["n"] == 3
    assert report["metrics"]["structured_output_pass_rate"] == 1.0
    # every task has 8+ calls, so all of them are routed to the second critic
    assert report["metrics"]["second_critic_rate"] == 1.0
    lines = (out_dir / orrun.RENDER_LOG).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    row = json.loads(lines[0])
    assert row["prompt_version"] == "pilot43.writer.v6"
    assert row["model"] == cfg.model_for("writer")
    assert row["validation"]["passed"] in (True, False)
    assert (out_dir / orrun.GATE_REPORT.format(stage="smoke")).is_file()

    resumed_client = make_client(cfg, out_dir, FakeTransport([]))
    again = orrun.run_stage("smoke", tasks, resumed_client, out_dir, seed=1)
    assert again["resumed"] == 3
    assert len((out_dir / orrun.RENDER_LOG).read_text(
        encoding="utf-8").splitlines()) == 3


def test_run_stage_records_a_critic_disagreement(cfg, out_dir) -> None:
    tasks = [task("t0", call_count=9)]
    transport = FakeTransport([
        ok(cfg.model_for("writer"), WRITER_CONTENT),
        ok(cfg.model_for("critic"), critic_content("PASS")),
        ok(cfg.model_for("critic2"), critic_content("REJECT")),
    ])
    client = make_client(cfg, out_dir, transport)
    report = orrun.run_stage("smoke", tasks, client, out_dir, seed=1)
    assert report["metrics"]["disagreement_rate"] == 1.0
    assert report["blocked_task_ids"] == ["t0"]
    row = json.loads((out_dir / orrun.DISAGREEMENT_LOG).read_text(
        encoding="utf-8").splitlines()[0])
    assert row["first_verdict"] == "PASS" and row["second_verdict"] == "REJECT"


def test_a_pass_that_contradicts_its_own_evidence_is_not_a_pass() -> None:
    contradictory = {**critic_content("PASS"), "query_natural": False}
    assert orrun.effective_verdict(contradictory) == "REJECT"
    assert orrun.critic_evidence(contradictory) == ["query_natural"]


def test_a_rejection_with_no_evidence_behind_it_is_uncertain() -> None:
    bare = {**critic_content("PASS"), "verdict": "REJECT"}
    assert orrun.effective_verdict(bare) == orrun.UNCERTAIN


def test_an_unevidenced_rejection_is_settled_by_the_second_critic(cfg,
                                                                  out_dir) -> None:
    bare_reject = {**critic_content("PASS"), "verdict": "REJECT"}
    transport = FakeTransport([
        ok(cfg.model_for("writer"), WRITER_CONTENT),
        ok(cfg.model_for("critic"), bare_reject),
        ok(cfg.model_for("critic2"), critic_content("PASS")),
    ])
    client = make_client(cfg, out_dir, transport)
    record = orrun.render_one(task("t0", call_count=9), client)
    assert record["disagreement"] is False
    assert record["blocked"] is False


def test_an_unevidenced_rejection_the_second_critic_cannot_settle_blocks(
        cfg, out_dir) -> None:
    bare_reject = {**critic_content("PASS"), "verdict": "REJECT"}
    transport = FakeTransport([
        ok(cfg.model_for("writer"), WRITER_CONTENT),
        ok(cfg.model_for("critic"), bare_reject),
        ok(cfg.model_for("critic2"), bare_reject),
    ])
    client = make_client(cfg, out_dir, transport)
    record = orrun.render_one(task("t0", call_count=9), client)
    assert record["blocked_reason"] == "critic_uncertain_unsettled"


def test_a_disagreement_that_a_rewrite_resolves_no_longer_blocks(cfg,
                                                                 out_dir) -> None:
    transport = FakeTransport([
        ok(cfg.model_for("writer"), WRITER_CONTENT),
        ok(cfg.model_for("critic"), critic_content("PASS")),
        ok(cfg.model_for("critic2"), critic_content("REJECT")),
        ok(cfg.model_for("rewrite"), REWRITE_CONTENT),
        ok(cfg.model_for("critic"), critic_content("PASS")),
        ok(cfg.model_for("critic2"), critic_content("PASS")),
    ])
    client = make_client(cfg, out_dir, transport)
    record = orrun.render_one(task("t0", call_count=9), client)
    assert [h["reason"] for h in record["rewrite_history"]] == [
        "critic_disagreement"]
    assert record["query"] == REWRITE_CONTENT["query"]
    assert record["disagreement"] is False
    assert record["blocked"] is False


def test_a_disagreement_the_rewrite_cannot_resolve_still_blocks(cfg,
                                                                out_dir) -> None:
    transport = FakeTransport([
        ok(cfg.model_for("writer"), WRITER_CONTENT),
        ok(cfg.model_for("critic"), critic_content("PASS")),
        ok(cfg.model_for("critic2"), critic_content("REJECT")),
        ok(cfg.model_for("rewrite"), REWRITE_CONTENT),
        ok(cfg.model_for("critic"), critic_content("PASS")),
        ok(cfg.model_for("critic2"), critic_content("REJECT")),
    ])
    client = make_client(cfg, out_dir, transport)
    record = orrun.render_one(task("t0", call_count=9), client)
    assert record["disagreement"] is True
    assert record["blocked_reason"] == "critic_disagreement"


def test_a_repair_that_breaks_the_query_is_discarded(cfg, out_dir) -> None:
    """A rewrite that drops a stated fact to please a critic is not a repair."""
    stripped = {**REWRITE_CONTENT, "query": "What total do we report?",
                "changes_made": ["dropped the figures"]}
    transport = FakeTransport([
        ok(cfg.model_for("writer"), WRITER_CONTENT),
        ok(cfg.model_for("critic"), critic_content("PASS")),
        ok(cfg.model_for("critic2"), critic_content("REJECT")),
        ok(cfg.model_for("rewrite"), stripped),
    ])
    client = make_client(cfg, out_dir, transport)
    record = orrun.render_one(task("t0", call_count=9), client)
    assert record["query"] == WRITER_CONTENT["query"]
    assert record["rewrite_history"][0]["discarded"] is True
    assert record["validation"]["passed"] is True
    assert record["blocked_reason"] == "critic_disagreement"


def test_the_disagreement_rewrite_can_be_switched_off(cfg, out_dir) -> None:
    transport = FakeTransport([
        ok(cfg.model_for("writer"), WRITER_CONTENT),
        ok(cfg.model_for("critic"), critic_content("PASS")),
        ok(cfg.model_for("critic2"), critic_content("REJECT")),
    ])
    client = make_client(cfg, out_dir, transport)
    record = orrun.render_one(task("t0", call_count=9), client,
                              disagreement_rewrites=0)
    assert record["rewrite_history"] == []
    assert record["blocked_reason"] == "critic_disagreement"


def test_run_stage_rewrites_once_then_re_criticises(cfg, out_dir) -> None:
    tasks = [task("t0", call_count=3, tier=TIER_LONG_HORIZON)]
    transport = FakeTransport([
        ok(cfg.model_for("writer"), WRITER_CONTENT),
        ok(cfg.model_for("critic"), critic_content("REWRITE")),
        ok(cfg.model_for("rewrite"), REWRITE_CONTENT),
        ok(cfg.model_for("critic"), critic_content("PASS")),
        ok(cfg.model_for("critic2"), critic_content("PASS")),
    ])
    client = make_client(cfg, out_dir, transport)
    orrun.run_stage("smoke", tasks, client, out_dir, seed=1)
    row = json.loads((out_dir / orrun.RENDER_LOG).read_text(
        encoding="utf-8").splitlines()[0])
    assert len(row["rewrite_history"]) == 1
    assert row["rewrite_history"][0]["changes_made"] == ["removed a limit"]
    # a rewritten task is always routed to the second critic
    assert row["second_critic_reason"] == "rewritten"
    assert row["second_critic"]["verdict"] == "PASS"


def test_a_stage_records_the_gate_it_was_started_on(cfg, out_dir) -> None:
    """Advancing past a failed gate is allowed, but never silently."""
    (out_dir / f"stage_gate_{RUN_ID}_smoke.json").write_text(
        json.dumps({"stage": "smoke", "may_advance": False,
                    "failed_gates": ["critic_pass_rate"],
                    "metrics": {"critic_pass_rate": 0.8}}), encoding="utf-8")
    client = make_client(cfg, out_dir, ByModelTransport(cfg))
    report = orrun.run_stage("pilot", [task("t0", call_count=3)], client,
                             out_dir, seed=1)
    assert report["entered_against_failed_gate"] is True
    assert report["entered_on"]["failed_gates"] == ["critic_pass_rate"]


def test_a_first_stage_has_no_earlier_gate_to_fail(cfg, out_dir) -> None:
    client = make_client(cfg, out_dir, ByModelTransport(cfg))
    report = orrun.run_stage("smoke", [task("t0", call_count=3)], client,
                             out_dir, seed=1)
    assert report["entered_against_failed_gate"] is False


def test_run_stage_stops_on_total_budget(cfg, out_dir) -> None:
    tight = oc.build_config({**cfg.raw, "max_total_cost_usd": 0.001,
                             "max_cost_per_task_usd": 10.0})
    tasks = [task(f"t{i}", call_count=6 + i) for i in range(3)]
    client = make_client(tight, out_dir, scripted(tight, 3, cost=0.002))
    report = orrun.run_stage("smoke", tasks, client, out_dir, seed=1)
    assert report["stopped"].startswith("budget_exceeded")
    assert report["may_advance"] is False


def test_run_stage_refuses_a_foreign_output_directory(cfg, out_dir,
                                                      tmp_path) -> None:
    client = make_client(cfg, out_dir, FakeTransport([]))
    with pytest.raises(oc.RunIsolationError):
        orrun.run_stage("smoke", [], client, tmp_path / "elsewhere")


def test_run_stage_ignores_deterministically_rendered_modes(cfg, out_dir) -> None:
    tasks = [task("t0", mode="GRAPH_EXPLICIT"),
             task("t1", mode="OPERATION_EXPLICIT_GRAPH_IMPLICIT")]
    client = make_client(cfg, out_dir, FakeTransport([]))
    report = orrun.run_stage("smoke", tasks, client, out_dir, seed=1)
    assert report["skipped_non_writer_modes"] == 2
    assert report["metrics"]["n"] == 0


class ByModelTransport:
    """Answers according to the requested model, so concurrent calls are safe."""

    def __init__(self, cfg: oc.OpenRouterConfig, *,
                 unavailable: Sequence[str] = ()) -> None:
        self.cfg = cfg
        self.unavailable = set(unavailable)
        self.lock = threading.Lock()
        self.n = 0

    def __call__(self, url: str, payload: Dict[str, Any],
                 headers: Dict[str, str], timeout: float) -> oc.HttpResponse:
        model = payload["model"]
        with self.lock:
            self.n += 1
        if model in self.unavailable:
            return oc.HttpResponse(
                404, json.dumps({"error": {"message": f"no endpoints for {model}",
                                           "code": 404}}), {})
        if model == self.cfg.model_for("writer"):
            return ok(model, WRITER_CONTENT)
        return ok(model, critic_content("PASS"))


def test_a_routed_second_critic_that_never_answers_blocks_the_task(cfg,
                                                                  out_dir) -> None:
    transport = ByModelTransport(cfg, unavailable=[cfg.model_for("critic2")])
    client = make_client(cfg, out_dir, transport)
    record = orrun.render_one(task("t0", call_count=9), client, sample_rate=1.0)
    assert record["second_critic_reason"] == "call_count_8plus"
    assert record["second_critic"] is None
    assert record["blocked"] is True
    assert record["blocked_reason"] == "second_critic_unavailable"
    report = orrun.gate_report("smoke", [record])
    assert report["metrics"]["routed_second_critic_answered_rate"] == 0.0


def test_a_parallel_stage_writes_every_record_exactly_once(cfg, out_dir) -> None:
    tasks = [task(f"t{i}", call_count=8) for i in range(8)]
    client = make_client(cfg, out_dir, ByModelTransport(cfg))
    report = orrun.run_stage("smoke", tasks, client, out_dir, seed=1, workers=4)
    lines = (out_dir / orrun.RENDER_LOG).read_text(encoding="utf-8").splitlines()
    ids = [json.loads(line)["task_id"] for line in lines]
    assert sorted(ids) == sorted(t.task_id for t in tasks)
    assert report["metrics"]["n"] == 8
    assert report["metrics"]["blocked_rate"] == 0.0
    # writer + critic + second critic for every 8+ task, none lost to a race
    assert client.totals.requests == 24
    usage = json.loads((out_dir / oc.USAGE_FILE).read_text(encoding="utf-8"))
    assert usage["totals"]["requests"] == 24
