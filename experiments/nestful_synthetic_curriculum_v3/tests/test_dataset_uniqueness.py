"""Dataset uniqueness analyzer tests (v3.1)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
V3 = REPO / "experiments/nestful_synthetic_curriculum_v3"
SCRIPTS = V3 / "scripts"
sys.path.insert(0, str(SCRIPTS))

from uniqueness_utils_v3_1 import analyze_stage_samples, compute_signatures  # noqa: E402

CURR = V3 / "outputs/curriculum_v3_1"


def test_compute_signatures_distinguishes_trace_and_question():
    base = {
        "question": "Add 1 and 2. Return result.",
        "gold_calls": [{"name": "add", "arguments": {"arg_0": 1, "arg_1": 2}}],
        "gold_answer": 3,
    }
    other_q = dict(base, question="Compute the sum of 1 and 2.")
    s1 = compute_signatures(base)
    s2 = compute_signatures(other_q)
    assert s1["trace"] == s2["trace"]
    assert s1["exact"] != s2["exact"]


def test_uniqueness_summary_on_filtered_data():
    summary_path = CURR / "dataset_uniqueness_summary.json"
    if not summary_path.is_file():
        pytest.skip("run analyze_dataset_uniqueness_v3_1 first")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data.get("exact_duplicate_count") == 0
    for stage, sd in data.get("per_stage", {}).items():
        assert sd.get("exact_duplicate_count") == 0
        assert sd.get("unique_question_ratio", 0) >= 0.40
