"""Question ↔ gold_calls semantic alignment (v3.1)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
V3 = REPO / "experiments/nestful_synthetic_curriculum_v3"
SCRIPTS = V3 / "scripts"
sys.path.insert(0, str(SCRIPTS))

from question_templates_v3_1 import (  # noqa: E402
    describe_gold_call,
    is_incomplete_question,
    question_for_prefix,
    validate_question_trace_alignment,
)

CURR = V3 / "outputs/curriculum_v3_1"
FILTERED = CURR / "filtered"


def test_reference_reuse_self_multiply_wording():
    rng = random.Random(1)
    calls = [
        {"name": "add", "arguments": {"arg_0": 4, "arg_1": 2}, "label": "$var_1"},
        {"name": "multiply", "arguments": {"arg_0": "$var_1.result$", "arg_1": "$var_1.result$"}, "label": "$var_2"},
    ]
    q = question_for_prefix(rng, calls, 2, "stage2_2call_dependency", seed=42)
    assert "by itself" in q.lower() or "square" in q.lower()
    assert "by 2" not in q.lower() or "by itself" in q.lower()
    assert validate_question_trace_alignment(q, calls, num_calls=2) == []


def test_fan_in_add_two_refs_wording():
    rng = random.Random(2)
    calls = [
        {"name": "add", "arguments": {"arg_0": 3, "arg_1": 4}, "label": "$var_1"},
        {"name": "add", "arguments": {"arg_0": 5, "arg_1": 6}, "label": "$var_2"},
        {"name": "add", "arguments": {"arg_0": "$var_1.result$", "arg_1": "$var_2.result$"}, "label": "$var_3"},
    ]
    q = question_for_prefix(rng, calls, 3, "stage3_3call_composition", seed=7)
    assert "step 1" in q.lower() and "step 2" in q.lower()
    assert "add B" not in q
    assert validate_question_trace_alignment(q, calls, num_calls=3) == []


def test_multiply_by_literal_wording():
    rng = random.Random(3)
    calls = [
        {"name": "add", "arguments": {"arg_0": 6, "arg_1": 1}, "label": "$var_1"},
        {"name": "multiply", "arguments": {"arg_0": "$var_1.result$", "arg_1": 2}, "label": "$var_2"},
    ]
    q = question_for_prefix(rng, calls, 2, "stage2_2call_dependency", seed=11)
    assert "by 2" in q.lower()
    assert "by itself" not in q.lower()
    assert validate_question_trace_alignment(q, calls, num_calls=2) == []


def test_describe_gold_call_examples():
    assert describe_gold_call({"name": "add", "arguments": {"arg_0": 6, "arg_1": 1}}, 1) == "add 6 and 1"
    step2 = describe_gold_call(
        {"name": "multiply", "arguments": {"arg_0": "$var_1.result$", "arg_1": 2}}, 2
    )
    assert step2 == "multiply the previous result by 2"
    step_self = describe_gold_call(
        {"name": "multiply", "arguments": {"arg_0": "$var_1.result$", "arg_1": "$var_1.result$"}}, 2
    )
    assert "by itself" in step_self


def test_math_trace_gets_math_question():
    rng = random.Random(42)
    calls = [
        {"name": "add", "arguments": {"arg_0": 3, "arg_1": 5}, "label": "$var_1"},
        {"name": "multiply", "arguments": {"arg_0": "$var_1.result$", "arg_1": 2}, "label": "$var_2"},
        {"name": "multiply", "arguments": {"arg_0": "$var_2.result$", "arg_1": 3}, "label": "$var_3"},
        {"name": "multiply", "arguments": {"arg_0": "$var_3.result$", "arg_1": 2}, "label": "$var_4"},
    ]
    q = question_for_prefix(rng, calls, 4, "stage4_4to6call_persistence", seed=99)
    assert len(q) >= 25
    assert "lookup" not in q.lower()
    assert not is_incomplete_question(q)
    assert validate_question_trace_alignment(q, calls, num_calls=4) == []


def test_incomplete_questions_flagged():
    calls = [{"name": "add", "arguments": {"arg_0": 1, "arg_1": 2}}]
    assert "incomplete_or_short_question" in validate_question_trace_alignment("Separately compute", calls)
    assert "unresolved_placeholder" in validate_question_trace_alignment("add B to the result", calls)


def test_alignment_summary_passes_on_filtered_data():
    summary_path = CURR / "question_trace_alignment_summary.json"
    if not summary_path.is_file():
        pytest.skip("run validate_question_trace_alignment_v3_1 first")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data.get("question_trace_alignment_failures") == 0
    assert data.get("unresolved_placeholders") == 0
    assert data.get("constant_reference_mismatch") == 0
    assert data.get("used_tool_diversity", 0) >= 18


def test_stage4_question_diversity():
    base = FILTERED if FILTERED.is_dir() else CURR
    path = base / "stage4_4to6call_persistence.jsonl"
    if not path.is_file():
        pytest.skip("missing stage4 filtered data")
    questions = {json.loads(l)["question"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert len(questions) >= 200, f"stage4 has only {len(questions)} unique questions"
