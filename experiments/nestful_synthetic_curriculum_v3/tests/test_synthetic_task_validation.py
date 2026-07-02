"""Tests for synthetic task validation."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = ROOT
REPO = ROOT.parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from motif_lib import build_dependency_graph, validate_references  # noqa: E402
from validate_synthetic_tasks import validate_task  # noqa: E402


def _valid_task():
    calls = [
        {"name": "add", "arguments": {"arg_0": 1, "arg_1": 2}, "label": "$var_1"},
    ]
    tools = [{"name": "add", "description": "d", "parameters": {}, "output_parameters": {"result": {}}}]
    return {
        "task_id": "synthetic_v3_test_001",
        "question": "q",
        "tools": tools,
        "gold_calls": calls,
        "gold_answer": 3,
        "num_calls": 1,
        "motif_type": "linear_dependency",
        "dependency_graph": build_dependency_graph(calls),
        "reference_pattern": {"num_references": 0, "fan_in_count": 0, "fan_out_count": 0,
                              "reuse_count": 0, "nested_reference_depth": 0},
        "output_type": "scalar",
        "answer_type": "scalar",
        "difficulty_score": 0.2,
        "generation_seed": 1,
    }


def test_valid_task_passes():
    assert validate_task(_valid_task(), set()) == []


def test_missing_field_fails():
    t = _valid_task()
    del t["motif_type"]
    assert any("missing:motif_type" in e for e in validate_task(t, set()))


def test_invalid_reference_fails():
    calls = [
        {"name": "add", "arguments": {"arg_0": "$var_2.result$"}, "label": "$var_1"},
    ]
    errs = validate_references(calls)
    assert any("forward ref" in e for e in errs)


def test_unknown_motif_type_fails():
    t = _valid_task()
    t["motif_type"] = "not_a_real_motif"
    assert any("unknown_motif_type" in e for e in validate_task(t, set()))


def test_duplicate_id_detected_by_script(tmp_path):
    t = _valid_task()
    p = tmp_path / "tasks.jsonl"
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(t) + "\n")
        fh.write(json.dumps(t) + "\n")
    rc = subprocess.call([
        sys.executable, str(SCRIPTS / "validate_synthetic_tasks.py"),
        "--input", str(p),
        "--out_dir", str(tmp_path / "out"),
    ], cwd=str(REPO))
    assert rc != 0
