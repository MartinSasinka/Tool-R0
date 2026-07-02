#!/usr/bin/env python3
"""Tests for crash-isolated official Win Rate in nestful_official_score."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
MINIMAL = ROOT / "experiments" / "nestful_mtgrpo_minimal"
if str(MINIMAL) not in sys.path:
    sys.path.insert(0, str(MINIMAL))

from nestful_official_score import score_items, score_items_per_sample  # noqa: E402


def _item() -> dict:
    return {
        "generated_text": json.dumps([{"name": "add", "arguments": {"a": 1, "b": 2}, "label": "$var1"}]),
        "output": json.dumps([{"name": "add", "arguments": {"a": 1, "b": 2}}]),
        "tools": json.dumps([{"name": "add", "description": "add", "parameters": {}}]),
        "gold_answer": json.dumps(3),
    }


def test_score_items_never_null_win_rate_on_empty_batch():
    out = score_items([], win_rate=True)
    assert out["win_rate"] == 0.0


def test_score_items_per_sample_isolates_numpy_ambiguous_win():
    """One bad sample must not abort the whole batch (the v1 curriculum bug)."""
    items = [_item(), _item()]

    def _boom(*_a, **_k):
        raise ValueError(
            "The truth value of an array with more than one element is ambiguous"
        )

    with patch("scorer.calculate_win_score", side_effect=_boom):
        per = score_items_per_sample(items, win_rate=True)
    assert len(per) == 2
    assert all(r["official_win"] == 0.0 for r in per)
    assert all(r.get("execution_error") for r in per)

    with patch("scorer.calculate_win_score", side_effect=_boom):
        agg = score_items(items, win_rate=True)
    assert agg["win_rate"] == 0.0
    assert agg["num_examples"] == 2


def test_score_items_win_rate_from_per_sample():
    items = [_item()]

    def _win(*_a, **_k):
        return True

    with patch("scorer.calculate_win_score", side_effect=_win):
        with patch("scorer.calculate_ans", return_value=3):
            out = score_items(items, win_rate=True)
    assert out["win_rate"] == 1.0
