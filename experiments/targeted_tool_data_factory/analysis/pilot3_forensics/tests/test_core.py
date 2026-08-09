"""Unit tests for Pilot3 forensics (synthetic fixtures only)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.targeted_tool_data_factory.analysis.pilot3_forensics.failure_taxonomy import (
    PRIMARY_PRIORITY,
    classify_trajectory,
)
from experiments.targeted_tool_data_factory.analysis.pilot3_forensics.graph_features import (
    build_dag,
    parse_reference,
    topology_hash,
)
from experiments.targeted_tool_data_factory.analysis.pilot3_forensics.integrity import verify_traj_pairing
from experiments.targeted_tool_data_factory.analysis.pilot3_forensics.pairing import outcome_label
from experiments.targeted_tool_data_factory.analysis.pilot3_forensics.statistics import (
    jensen_shannon,
    mcnemar_exact,
    normalize_counts,
    paired_bootstrap_delta,
    total_variation,
)
from experiments.targeted_tool_data_factory.analysis.pilot3_forensics.trajectory_features import (
    classify_divergence,
    fingerprint,
    first_divergent_turn,
)


def test_reference_parser_variants():
    assert parse_reference("$var1.output_0$")["var_num"] == "1"
    assert parse_reference("$var_1.result$")["output_key"] == "result"
    assert parse_reference("$var1$")["label_norm"] == "var1"
    assert parse_reference("not_a_ref") is None


def test_topology_hash_distinguishes_shapes_and_ignores_names():
    linear = [
        {"name": "A", "label": "$var1", "arguments": {"x": 1}},
        {"name": "B", "label": "$var2", "arguments": {"x": "$var1.result$"}},
        {"name": "C", "label": "$var3", "arguments": {"x": "$var2.result$"}},
    ]
    linear_renamed = [
        {"name": "X", "label": "$var_1", "arguments": {"x": 9}},
        {"name": "Y", "label": "$var_2", "arguments": {"x": "$var_1.output_0$"}},
        {"name": "Z", "label": "$var_3", "arguments": {"x": "$var_2.output_0$"}},
    ]
    fan_in = [
        {"name": "A", "label": "$var1", "arguments": {"x": 1}},
        {"name": "B", "label": "$var2", "arguments": {"x": 2}},
        {"name": "C", "label": "$var3", "arguments": {"a": "$var1.result$", "b": "$var2.result$"}},
    ]
    fan_out = [
        {"name": "A", "label": "$var1", "arguments": {"x": 1}},
        {"name": "B", "label": "$var2", "arguments": {"x": "$var1.result$"}},
        {"name": "C", "label": "$var3", "arguments": {"x": "$var1.result$"}},
    ]
    assert topology_hash(linear) == topology_hash(linear_renamed)
    assert topology_hash(linear) != topology_hash(fan_in)
    assert topology_hash(fan_in) != topology_hash(fan_out)
    assert len(build_dag(fan_in)["edges"]) == 2


def test_pairing_by_sample_id_and_duplicates():
    c0 = [
        {"sample_id": "a", "num_gold_calls": 2, "_traj": {"official_win": 1.0, "gold_num_turns": 2}},
        {"sample_id": "b", "num_gold_calls": 3, "_traj": {"official_win": 0.0, "gold_num_turns": 3}},
    ]
    d1 = [
        {"sample_id": "b", "num_gold_calls": 3, "_traj": {"official_win": 1.0, "gold_num_turns": 3}},
        {"sample_id": "a", "num_gold_calls": 2, "_traj": {"official_win": 1.0, "gold_num_turns": 2}},
    ]
    ok = verify_traj_pairing(c0, d1)
    assert ok["pairing_ok"]
    assert ok["wins_c0_recount"] == 1
    assert ok["wins_d1_recount"] == 2

    dup = c0 + [c0[0]]
    bad = verify_traj_pairing(dup, d1)
    assert not bad["pairing_ok"]
    assert bad["duplicate_c0"] == ["a"]


def test_mcnemar_and_bootstrap_repro():
    # b=16, c=27 style
    p = mcnemar_exact(16, 27)
    assert 0.0 < p < 1.0
    wins_a = [True] * 16 + [False] * 27 + [True] * 100 + [False] * 100
    wins_b = [False] * 16 + [True] * 27 + [True] * 100 + [False] * 100
    r1 = paired_bootstrap_delta(wins_a, wins_b, n_boot=500, seed=42)
    r2 = paired_bootstrap_delta(wins_a, wins_b, n_boot=500, seed=42)
    assert r1 == r2
    assert r1["point_pp"] == pytest.approx(100 * (27 - 16) / len(wins_a))


def test_failure_taxonomy_priority():
    row = {
        "sample_id": "x",
        "num_gold_calls": 2,
        "strict_gold_trace_pass": True,
        "final_answer_pass": True,
        "_traj": {
            "official_win": 1.0,
            "parse_valid": True,
            "executable": True,
            "turns": [{"parsed_call": {"name": "add", "arguments": {}}}],
            "num_tool_calls": 1,
        },
    }
    c = classify_trajectory(row)
    assert c["primary_failure"] == "SUCCESS_STRICT_GOLD"
    assert PRIMARY_PRIORITY.index("SUCCESS_STRICT_GOLD") < PRIMARY_PRIORITY.index("FAIL_OTHER")


def test_first_divergent_turn_and_divergence():
    def mk(texts, calls, win=False):
        turns = []
        for i, t in enumerate(texts):
            turn = {"turn_idx": i, "model_text": t}
            if i < len(calls):
                turn["parsed_call"] = calls[i]
            turns.append(turn)
        return {
            "sample_id": "s",
            "final_answer_pass": win,
            "_traj": {
                "official_win": float(win),
                "parse_valid": True,
                "executable": True,
                "turns": turns,
                "pred_answer": "1",
                "stop_reason": "final",
                "num_tool_calls": len(calls),
            },
        }

    a = mk(["t0", "t1"], [{"name": "add", "arguments": {"x": 1}}, {"name": "mul", "arguments": {"x": 2}}])
    b = mk(["t0", "DIFF"], [{"name": "add", "arguments": {"x": 1}}, {"name": "mul", "arguments": {"x": 2}}])
    fa, fb = fingerprint(a), fingerprint(b)
    assert first_divergent_turn(fa, fb) == 1
    assert classify_divergence(fa, fb) in {"IDENTICAL_CALLS_DIFFERENT_TEXT", "OTHER", "ANSWER_ONLY_DIFFERENCE"}

    c = mk(["t0"], [{"name": "sub", "arguments": {"x": 1}}])
    fc = fingerprint(c)
    assert classify_divergence(fa, fc) == "DIFFERENT_FIRST_TOOL"


def test_jsd_tv_and_outcome_label():
    p = normalize_counts(__import__("collections").Counter({"a": 2, "b": 2}))
    q = normalize_counts(__import__("collections").Counter({"a": 3, "b": 1}))
    assert 0 <= total_variation(p, q) <= 1
    assert 0 <= jensen_shannon(p, q) <= 1
    assert outcome_label(False, True) == "loss_to_win"
    assert outcome_label(True, False) == "win_to_loss"


def test_missing_optional_artifact_discovery(tmp_path: Path):
    from experiments.targeted_tool_data_factory.analysis.pilot3_forensics.discovery import discover

    # empty repo-like tree should not crash
    d = discover(tmp_path)
    assert d.get("c0_trajectories") is None
    assert "rollout_log" in d.artifacts
