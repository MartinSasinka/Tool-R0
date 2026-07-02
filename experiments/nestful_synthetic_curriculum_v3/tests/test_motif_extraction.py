"""Tests for motif extraction."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from motif_lib import extract_motifs, is_linear_chain  # noqa: E402


def _task(calls):
    return {"task_id": "t", "gold_calls": calls, "gold_answer": 1, "tools": []}


def test_linear_dependency():
    calls = [
        {"name": "add", "arguments": {"a": 1, "b": 2}, "label": "$var_1"},
        {"name": "mul", "arguments": {"x": "$var_1.result$"}, "label": "$var_2"},
    ]
    m = extract_motifs(_task(calls))
    assert m["linear_chain"] is True
    assert m["dependency_depth"] >= 2


def test_fan_in_detection():
    calls = [
        {"name": "a", "arguments": {}, "label": "$var_1"},
        {"name": "b", "arguments": {}, "label": "$var_2"},
        {"name": "c", "arguments": {"x": "$var_1.result$", "y": "$var_2.result$"}, "label": "$var_3"},
    ]
    m = extract_motifs(_task(calls))
    assert m["fan_in"] is True


def test_fan_out_detection():
    calls = [
        {"name": "a", "arguments": {}, "label": "$var_1"},
        {"name": "b", "arguments": {"x": "$var_1.result$"}, "label": "$var_2"},
        {"name": "c", "arguments": {"y": "$var_1.result$"}, "label": "$var_3"},
    ]
    m = extract_motifs(_task(calls))
    assert m["fan_out"] is True


def test_reference_reuse():
    calls = [
        {"name": "a", "arguments": {}, "label": "$var_1"},
        {"name": "b", "arguments": {"x": "$var_1.result$"}, "label": "$var_2"},
        {"name": "c", "arguments": {"y": "$var_1.result$"}, "label": "$var_3"},
    ]
    m = extract_motifs(_task(calls))
    assert m["reference_reuse"] is True


def test_output_type_inference():
    m = extract_motifs({"task_id": "t", "gold_calls": [], "gold_answer": [1, 2, 3], "tools": []})
    assert m["answer_type"] == "list"


def test_difficulty_in_range():
    calls = [{"name": "a", "arguments": {}, "label": "$var_1"}]
    m = extract_motifs(_task(calls))
    assert 0.0 <= m["difficulty_score"] <= 1.0


def test_is_linear_chain_helper():
    calls = [{"name": "a", "arguments": {}, "label": "$var_1"}]
    assert is_linear_chain(calls) is True
