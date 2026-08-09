"""Selection quotas, nested subsets, gates, human audit, probe and freeze.

These tests use synthetic records rather than the generated dataset so they fail for
one reason only: the logic changed. The dataset-level numbers are checked by the
acceptance gates and the independent audit, not here.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from targeted_tool_data.pilot43 import gates, human, probe
from targeted_tool_data.pilot43 import select as sel
from targeted_tool_data.pilot43.export import _check_nesting, tv_distance


# ── fixtures ─────────────────────────────────────────────────────────────
def _task(i: int, *, calls: int = 3, workflow: str = "wf.a",
          pattern: str = "LINEAR_CHAIN", coding: bool = False,
          track: str = "A_NATIVE", answer: str = "float",
          template: str = "t0", combo: str = "arithmetic") -> sel.Task:
    row = {
        "task_id": f"p43_{i:05d}",
        "workflow_id": workflow,
        "plan_id": "plan.v1",
        "seed": i,
        "surface_track": track,
        "call_count": calls,
        "call_bucket": str(calls) if calls <= 5 else "6+",
        "answer_type": answer,
        "actual_primary_pattern": pattern,
        "normalized_capability_sequence": f"{combo}-plan",
        "capability_families": [combo],
        "coding_like": coding,
        "difficulty_band": "medium",
        "cell_id": f"cell::{workflow}",
        "semantic_program_id": "sp_x",
        "primitive_sequence": "a->b",
        "coding_call_share": 1.0 if coding else 0.0,
        "graph_features": {"n_nodes": calls, "n_join_nodes": 1 + (i % 2),
                           "n_late_edges": i % 2, "n_fan_out_nodes": i % 2,
                           "n_reused_outputs": i % 2, "depth": max(2, calls - 1)},
        "workflow_instance_id": f"inst_{i}",
        "program_fingerprint": f"fp_{i}",
        "primitives": [f"p{k}" for k in range(calls)],
        "boolean_label": None,
    }
    query = {"task_id": row["task_id"], "actual_mode": "DOMAIN_GROUNDED_IMPLICIT",
             "query": f"question {i}", "requested_mode": "DOMAIN_GROUNDED_IMPLICIT",
             "query_source": "deterministic", "passed": True,
             "fingerprints": {"intent_fingerprint": template,
                              "exact_fingerprint": f"e{i}"}}
    return sel.Task(row["task_id"], row, query, {"selectable": True})


def _pool(n: int = 400) -> list[sel.Task]:
    out = []
    for i in range(n):
        calls = (2, 3, 4, 5, 6, 7, 8)[i % 7]
        out.append(_task(i, calls=calls,
                         workflow=f"wf.{i % 20}",
                         pattern=("LINEAR_CHAIN", "MULTI_JOIN", "DIAMOND",
                                  "FAN_OUT")[i % 4],
                         coding=(i % 3 == 0),
                         track=("A_NATIVE", "G_GENERAL_1", "G_GENERAL_2")[i % 3],
                         answer=("float", "boolean", "string", "list")[i % 4],
                         template=f"t{i % 40}",
                         combo=("arithmetic", "list", "string")[i % 3]))
    return out


# ── tier quotas are hard ─────────────────────────────────────────────────
def test_tier_quota_shortfall_fails_and_never_cross_fills():
    pool = _pool(60)
    tiers, _left = sel.select_tiers(pool, {"PROFILE_CORE": 3000,
                                           "LONG_HORIZON_ENRICHMENT": 10,
                                           "CAPABILITY_ENRICHMENT": 10,
                                           "CHALLENGE": 5})
    assert tiers["PROFILE_CORE"].met is False
    assert tiers["PROFILE_CORE"].deficits
    assert len(tiers["PROFILE_CORE"].tasks) < 3000
    # a task selected into one tier is never reused by another
    chosen = [t.task_id for tier in tiers.values() for t in tier.tasks]
    assert len(chosen) == len(set(chosen))


def test_capability_tier_is_all_coding():
    pool = _pool(300)
    tiers, _left = sel.select_tiers(pool, {"PROFILE_CORE": 20,
                                           "LONG_HORIZON_ENRICHMENT": 10,
                                           "CAPABILITY_ENRICHMENT": 20,
                                           "CHALLENGE": 5})
    picked = tiers["CAPABILITY_ENRICHMENT"].tasks
    assert picked, "expected some capability-enrichment tasks"
    assert all(t.coding for t in picked)


def test_long_horizon_prefers_deep_programs():
    pool = _pool(300)
    tiers, _left = sel.select_tiers(pool, {"PROFILE_CORE": 10,
                                           "LONG_HORIZON_ENRICHMENT": 30,
                                           "CAPABILITY_ENRICHMENT": 5,
                                           "CHALLENGE": 5})
    picked = tiers["LONG_HORIZON_ENRICHMENT"].tasks
    six_plus = sum(1 for t in picked if t.call_count >= 6)
    assert six_plus / max(1, len(picked)) >= 0.6
    depths = {t.call_count for t in picked if t.call_count >= 6}
    assert len(depths) > 1, "6+ must not collapse onto a single depth"


# ── split integrity ──────────────────────────────────────────────────────
def test_heldout_keys_leave_the_training_pool():
    pool = _pool(400)
    keys = sel.plan_heldout(pool)
    parts, train = sel.cut_heldout(pool, keys)
    train_ids = {t.task_id for t in train}
    for part in parts.values():
        for task in part:
            assert task.task_id not in train_ids
    assert not ({t.workflow_id for t in train} & keys.workflows)
    assert not ({t.plan_key for t in train} & keys.plans)
    assert not ({t.template_key for t in train} & keys.templates)
    assert all(t.track != keys.track for t in train)


def test_no_single_heldout_part_eats_the_pool():
    pool = _pool(120)
    keys = sel.plan_heldout(pool)
    parts, train = sel.cut_heldout(pool, keys)
    for name, part in parts.items():
        assert len(part) <= 0.35 * len(pool), f"{name} took {len(part)}/{len(pool)}"
    assert train, "training pool must survive the holdout"


def test_split_overlap_report_flags_leakage():
    pool = _pool(200)
    keys = sel.plan_heldout(pool)
    parts, train = sel.cut_heldout(pool, keys)
    clean = sel.split_overlap_report(train, parts)
    assert clean["passed"], clean["violations"]
    leaked = sel.split_overlap_report(train + list(parts["workflow_family"]), parts)
    assert not leaked["passed"]
    assert leaked["violations"]


# ── nested subsets ───────────────────────────────────────────────────────
def test_nested_subsets_are_nested_and_exact():
    pool = _pool(400)
    master = [(t, "PROFILE_CORE") for t in pool]
    mixes = sel.nested_subsets(master, (50, 100, 200, 400))
    assert _check_nesting(mixes, (50, 100, 200))
    assert set(mixes[50]) < set(mixes[100]) < set(mixes[200])
    assert len(mixes[100]) == 100


def test_nested_subsets_are_not_the_first_n():
    pool = _pool(200)
    master = [(t, "PROFILE_CORE") for t in pool]
    mixes = sel.nested_subsets(master, (50, 200))
    assert mixes[50] != [t.task_id for t, _ in master[:50]]


def test_nested_subsets_keep_the_strata_mix():
    pool = _pool(400)
    master = [(t, "PROFILE_CORE") for t in pool]
    mixes = sel.nested_subsets(master, (100, 400))
    by_id = {t.task_id: t for t in pool}
    subset = [by_id[i] for i in mixes[100]]
    for bucket in ("2", "3", "6+"):
        whole = sum(1 for t in pool if t.bucket == bucket) / len(pool)
        part = sum(1 for t in subset if t.bucket == bucket) / len(subset)
        assert abs(whole - part) < 0.10, bucket


def test_tv_distance():
    assert tv_distance({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0.0
    assert tv_distance({"a": 1.0}, {"b": 1.0}) == 1.0


# ── acceptance gates ─────────────────────────────────────────────────────
def _record(i: int, **over) -> dict:
    calls = over.pop("calls", 3)
    rec = {
        "task_id": f"p43_{i:05d}",
        "run_id": "pilot4_3_nestful_final",
        "workflow_id": "wf.a",
        "question": f"a question {i}",
        "call_count": calls,
        "call_bucket": str(calls) if calls <= 5 else "6+",
        "answer_type": "boolean",
        "gold_answer": bool(i % 2),
        "cell_id": "cell::wf.a",
        "cell_tier": "PROFILE_CORE",
        "surface_track": "A_NATIVE",
        "query_source": "deterministic",
        "actual_query_mode": "DOMAIN_GROUNDED_IMPLICIT",
        "offered_tool_count": 8,
        "gold_calls": [{"name": f"t{k}", "primitive_id": f"p{k}",
                        "capability": "arithmetic.add",
                        "capability_family": "arithmetic",
                        "coding_like": False, "node_id": f"n{k}",
                        "arguments": {}, "label": f"$var_{k}",
                        "observation": k} for k in range(calls)],
        "tools": [{"name": f"t{k}", "description": "a tool",
                   "parameters": {"x": "number"}, "output_field": "output_0",
                   "output_type": "number", "primitive_id": f"p{k}"}
                  for k in range(calls)]
        + [{"name": f"d{j}", "description": "a distractor",
            "parameters": {"x": "number"}, "output_field": "output_0",
            "output_type": "number", "primitive_id": f"pd{j}"}
           for j in (1, 2)],
        "stated_facts": [{"role": "a", "description": "an input", "hint": "money",
                          "semantic_type": "MONEY", "value": 12.5}],
        "natural_user_goal": "do the thing",
        "query_fingerprints": {"intent_fingerprint": "t0",
                               "exact_fingerprint": f"e{i}"},
        "distractor_profile": {"distractor_count": 2, "hard": 1, "medium": 1,
                               "easy": 0, "gold_tool_count": calls},
        "declared": {"structural_pattern": "LINEAR_CHAIN",
                     "satisfied_patterns": ["LINEAR_CHAIN"],
                     "requested_structural_skill": "LINEAR_CHAIN",
                     "graph_features": {"n_nodes": calls, "n_join_nodes": 1,
                                        "n_late_edges": 1, "n_fan_out_nodes": 1,
                                        "n_reused_outputs": 1, "depth": 2}},
        "validation": {
            "v4": {"v4_executed": True, "resolved": True, "has_shortcut": False},
            "node_necessity": [{"node_id": f"n{k}", "necessary": True}
                               for k in range(calls)],
            "query_checks": {"passed": True, "classification": {}},
            "critic": {"executed": False},
            "second_critic": {"executed": False, "routed": False},
        },
        "verifier": {"minimal_valid_call_count": calls, "gold_call_count": calls,
                     "accepted_solution_classes": ["canonical"],
                     "strict_trace_required": True},
    }
    rec.update(over)
    return rec


def _write(tmp: Path, name: str, rows) -> None:
    (tmp / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                            encoding="utf-8")


def test_gates_report_specific_blocking_failures(tmp_path: Path):
    rows = [_record(i) for i in range(10)]
    _write(tmp_path, "train_master_5000.jsonl", rows)
    _write(tmp_path, "train_profile_core_3000.jsonl", rows)
    result = gates.evaluate(tmp_path)
    assert result["AUTOMATED_GATES_PASSED"] is False
    failures = {c["id"] for c in result["blocking_failures"]}
    assert "size.TRAIN_MASTER" in failures
    assert "size.PROFILE_CORE" in failures
    # the things that *are* satisfied must not be reported as failures
    assert "structure.pattern_match" not in failures
    assert "v4.shortcuts" not in failures
    assert all(c["id"] and c["requirement"] for c in result["blocking_failures"])


def test_gates_catch_a_v4_shortcut_and_an_unnecessary_node(tmp_path: Path):
    bad = _record(1)
    bad["validation"]["v4"]["has_shortcut"] = True
    bad["validation"]["node_necessity"][0]["necessary"] = False
    _write(tmp_path, "train_master_5000.jsonl", [_record(0), bad])
    result = gates.evaluate(tmp_path)
    failures = {c["id"] for c in result["blocking_failures"]}
    assert "v4.shortcuts" in failures
    assert "necessity.unnecessary_nodes" in failures


def test_gates_catch_pattern_mismatch(tmp_path: Path):
    bad = _record(1)
    bad["declared"]["requested_structural_skill"] = "MULTI_JOIN"
    _write(tmp_path, "train_master_5000.jsonl", [bad])
    failures = {c["id"] for c in gates.evaluate(tmp_path)["blocking_failures"]}
    assert "structure.pattern_match" in failures


def test_gates_catch_boolean_landslide(tmp_path: Path):
    rows = [_record(i, gold_answer=True) for i in range(30)]
    _write(tmp_path, "train_master_5000.jsonl", rows)
    failures = {c["id"] for c in gates.evaluate(tmp_path)["blocking_failures"]}
    assert "answers.boolean_balance_overall" in failures
    assert "answers.boolean_balance_per_workflow" in failures
    assert "answers.no_boolean_landslide" in failures


def _with_real_primitives(rec: dict, *primitives: str) -> dict:
    """Point the gold calls at registry ops so sibling lookup is meaningful."""
    rec["gold_calls"] = [dict(call, primitive_id=pid)
                         for call, pid in zip(rec["gold_calls"], primitives)]
    return rec


def test_a_missing_hard_distractor_only_fails_when_one_was_usable(tmp_path: Path):
    from targeted_tool_data.pilot43.distractors import HARD, candidate_distractors

    gold = "add"
    assert any(c.hardness == HARD for c in candidate_distractors([gold]))

    bad = _with_real_primitives(_record(1, calls=1), gold)
    bad["distractor_profile"] = {"distractor_count": 2, "hard": 0, "medium": 2,
                                 "easy": 0, "gold_tool_count": 1,
                                 "hard_rejected_as_alias": 0}
    _write(tmp_path, "train_master_5000.jsonl", [bad])
    failures = {c["id"] for c in gates.evaluate(tmp_path)["blocking_failures"]}
    assert "tools.hard_distractor" in failures

    # the same empty slot is fine once every sibling turned out to be an alias
    excused = dict(bad, distractor_profile=dict(bad["distractor_profile"],
                                                hard_rejected_as_alias=3))
    _write(tmp_path, "train_master_5000.jsonl", [excused])
    failures = {c["id"] for c in gates.evaluate(tmp_path)["blocking_failures"]}
    assert "tools.hard_distractor" not in failures


def test_training_ready_requires_every_status(tmp_path: Path):
    _write(tmp_path, "train_master_5000.jsonl", [_record(0)])
    ready = gates.readiness(tmp_path, {"AUTOMATED_GATES_PASSED": True})
    assert ready["statuses"]["TRAINING_READY"] is False
    assert set(ready["blockers"]) >= {"INDEPENDENT_AUDIT_PASSED", "LLM_VALIDATED",
                                      "HUMAN_VALIDATED", "GRPO_SIGNAL_READY"}
    assert ready["statuses"]["HUMAN_REVIEW_PENDING"] is True
    assert ready["statuses"]["GRPO_PROBE_PENDING"] is True


def test_readiness_all_true_only_with_every_artifact(tmp_path: Path):
    rec = _record(0)
    rec["query_source"] = "openrouter"
    rec["validation"]["critic"] = {"executed": True, "verdict": "PASS"}
    _write(tmp_path, "train_master_5000.jsonl", [rec])
    (tmp_path / "selection_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "PILOT43_INDEPENDENT_AUDIT.json").write_text(
        json.dumps({"INDEPENDENT_AUDIT_PASSED": True}), encoding="utf-8")
    (tmp_path / "human_audit_results.json").write_text(
        json.dumps({"thresholds_met": True}), encoding="utf-8")
    (tmp_path / "model_probe_report.json").write_text(
        json.dumps({"thresholds_met": True}), encoding="utf-8")
    (tmp_path / "openrouter_usage_pilot43.json").write_text(
        json.dumps({"foreign_run_records": 0}), encoding="utf-8")
    ready = gates.readiness(tmp_path, {"AUTOMATED_GATES_PASSED": True})
    assert ready["statuses"]["TRAINING_READY"] is True
    # one foreign log record is enough to withdraw LLM_VALIDATED
    (tmp_path / "openrouter_usage_pilot43.json").write_text(
        json.dumps({"foreign_run_records": 1}), encoding="utf-8")
    ready = gates.readiness(tmp_path, {"AUTOMATED_GATES_PASSED": True})
    assert ready["statuses"]["LLM_VALIDATED"] is False
    assert ready["statuses"]["TRAINING_READY"] is False


# ── human audit package ──────────────────────────────────────────────────
def test_human_sample_covers_strata_and_prefers_long_tasks(tmp_path: Path):
    rows = [_record(i, calls=(2, 3, 6, 7)[i % 4],
                    answer_type=("boolean", "float", "string", "list")[i % 4],
                    gold_answer=[True, 1.5, "x", [1]][i % 4]) for i in range(200)]
    _write(tmp_path, "train_master_5000.jsonl", rows)
    stats = human.prepare(tmp_path, size=60)
    assert stats["n"] >= 60
    assert not stats["uncovered_strata"]
    with (tmp_path / human.TEMPLATE_FILE).open(encoding="utf-8") as fh:
        template = list(csv.DictReader(fh))
    assert len(template) == 2 * stats["n"], "two reviewers per task"
    assert all(q in template[0] for q, _t in human.QUESTIONS)
    assert (tmp_path / human.GUIDE_FILE).read_text(encoding="utf-8").strip()


def test_human_validated_false_until_ratings_are_imported(tmp_path: Path):
    _write(tmp_path, "train_master_5000.jsonl", [_record(i) for i in range(30)])
    human.prepare(tmp_path, size=10)
    notice = human.pending_notice(tmp_path)
    assert notice["thresholds_met"] is False
    assert notice["n_tasks_rated"] == 0


def test_human_import_computes_agreement_and_blocks_on_defects(tmp_path: Path):
    rows = [_record(i) for i in range(20)]
    _write(tmp_path, "train_master_5000.jsonl", rows)
    human.prepare(tmp_path, size=10)
    with (tmp_path / human.TEMPLATE_FILE).open(encoding="utf-8") as fh:
        template = list(csv.DictReader(fh))
    for i, line in enumerate(template):
        for q, _t in human.QUESTIONS:
            if q in human.NEGATIVE:
                line[q] = "no"
            else:
                line[q] = "no" if (i == 0 and q == "program_solves_query") else "yes"
    path = tmp_path / "ratings.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(template[0]))
        writer.writeheader()
        writer.writerows(template)
    result = human.import_results(tmp_path, path)
    assert result["n_tasks_rated"] == len({r["task_id"] for r in template})
    assert result["thresholds_met"] is False
    assert any("query_program_alignment" in u for u in result["unmet_thresholds"])
    assert result["question_pass_rates"]["plausible_user_request"]["pass_rate"] == 1.0


def test_cohens_kappa_extremes():
    assert human.cohens_kappa([True, False, True], [True, False, True]) == 1.0
    assert human.cohens_kappa([True, True, False, False],
                              [False, False, True, True]) == -1.0
    assert human.cohens_kappa([None], [None]) is None


# ── model probe ──────────────────────────────────────────────────────────
def test_probe_group_classification():
    ok = {"parsed": True, "correct": True, "progress": True}
    wrong = {"parsed": True, "correct": False, "progress": False}
    partial = {"parsed": True, "correct": False, "progress": True}
    junk = {"parsed": False, "correct": False, "progress": False}
    assert probe.classify([ok] * 4) == probe.ALL_CORRECT
    assert probe.classify([ok, wrong, wrong, wrong]) == probe.MIXED_TERMINAL
    assert probe.classify([partial] * 4) == probe.ALL_FAIL_WITH_PROGRESS
    assert probe.classify([wrong] * 4) == probe.ALL_FAIL_NO_PROGRESS
    assert probe.classify([junk] * 4) == probe.INVALID
    assert probe.classify([]) == probe.INVALID


def test_probe_parses_only_real_call_lists():
    assert probe.parse_calls('[{"name": "a", "arguments": {"x": 1}}]')
    assert probe.parse_calls("I cannot help") is None
    assert probe.parse_calls("[]") is None
    assert probe.parse_calls('[{"arguments": {}}]') is None


def test_probe_writes_not_run_artifacts_without_a_backend(tmp_path: Path):
    _write(tmp_path, "train_master_5000.jsonl", [_record(i) for i in range(5)])
    report = probe.run(tmp_path, sampler=None, sample_size=3)
    assert report["executed"] is False
    assert report["thresholds_met"] is False
    assert report["reason"]
    assert report["next_command"]
    assert (tmp_path / probe.GROUPS_FILE).exists()
    assert (tmp_path / probe.ROLLOUTS_FILE).exists()


def test_probe_uses_more_rollouts_only_for_uncertain_groups(tmp_path: Path):
    _write(tmp_path, "train_master_5000.jsonl", [_record(i) for i in range(4)])
    calls = {"n": 0}

    def sampler(prompt: str, n: int, seed: int):
        calls["n"] += n
        return ["not a tool call"] * n          # every group is INVALID -> stop

    report = probe.run(tmp_path, sampler=sampler, sample_size=4,
                       initial_rollouts=4, max_rollouts=8)
    assert report["executed"] is True
    assert calls["n"] == 16, "invalid groups must not buy extra rollouts"
    assert report["observed"]["invalid_group_rate"] == 1.0
    assert report["thresholds_met"] is False


def test_probe_thresholds_pass_on_a_healthy_mix(tmp_path: Path):
    _write(tmp_path, "train_master_5000.jsonl", [_record(i) for i in range(10)])
    groups = [{"task_id": f"t{i}", "cell_id": "c", "group_class":
               probe.MIXED_TERMINAL, "effective": True} for i in range(10)]
    report = probe._report(groups, model="m", provider_id="p")
    assert report["thresholds_met"] is True
    assert report["observed"]["effective_group_rate"] == 1.0


# ── freeze ───────────────────────────────────────────────────────────────
def test_freeze_refuses_when_no_inputs_can_be_hashed(tmp_path: Path):
    from targeted_tool_data.pilot43 import freeze

    with pytest.raises(RuntimeError, match="input hashes"):
        freeze.build(tmp_path, repo_root=tmp_path)


def test_freeze_hashes_inputs_and_artifacts(tmp_path: Path):
    from targeted_tool_data.pilot43 import freeze

    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "pilot4_3_openrouter.yaml").write_text("openrouter: {}\n",
                                                              encoding="utf-8")
    out = tmp_path / "pilot4_3_nestful_final"
    out.mkdir()
    _write(out, "train_master_5000.jsonl", [_record(0)])
    (out / "split_manifest.json").write_text(
        json.dumps({"train": {"PROFILE_CORE": ["p43_00000"]}, "heldout": {},
                    "reserve": []}), encoding="utf-8")
    payload = freeze.build(out, repo_root=repo, seeds={"generate": 1})
    assert payload["input_hashes"], "input hashes must not be empty"
    assert payload["artifact_hashes"]
    assert payload["n_ordered_sample_ids"] == 1
    assert (out / freeze.MANIFEST).exists()
    assert (out / freeze.PATCH).exists()
    assert freeze.verify(out)["verified"] is True
    # touching an artifact after the freeze is detected
    (out / "train_master_5000.jsonl").write_text("{}\n", encoding="utf-8")
    assert freeze.verify(out)["verified"] is False


def test_the_source_snapshot_carries_files_git_has_never_seen(tmp_path: Path):
    """An untracked source file must reach the patch, or the run cannot be rerun."""
    import subprocess

    from targeted_tool_data.pilot43 import freeze

    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    tracked = repo / "pkg" / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    for args in (["init", "-q"], ["config", "user.email", "t@example.com"],
                 ["config", "user.name", "t"], ["add", "."],
                 ["commit", "-qm", "base"]):
        subprocess.run(["git", *args], cwd=repo, check=True,
                       capture_output=True)
    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    (repo / "pkg" / "brand_new.py").write_text("NEW = True\n", encoding="utf-8")
    (repo / "pkg" / "big.json").write_text(
        "x" * (freeze.MAX_PATCH_FILE_BYTES + 1), encoding="utf-8")

    out = tmp_path / "out"
    out.mkdir()
    manifest = freeze.source_snapshot(out, repo)
    patch = (out / freeze.PATCH).read_text(encoding="utf-8")

    assert manifest["working_tree_clean"] is False
    assert manifest["patch_written"] is True
    assert "brand_new.py" in patch and "NEW = True" in patch
    assert "VALUE = 2" in patch
    # the oversized data file is hashed but its bytes stay out of the patch
    assert "pkg/big.json" in manifest["changed_files"]
    assert [o["path"] for o in manifest["omitted_from_patch"]] == ["pkg/big.json"]
    assert manifest["deleted_or_missing"] == []
    # the developer's own index is untouched by all of this
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=repo,
                            capture_output=True, text=True, check=True)
    assert staged.stdout.strip() == ""
