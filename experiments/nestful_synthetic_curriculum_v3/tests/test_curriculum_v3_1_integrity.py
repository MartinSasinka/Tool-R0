"""Curriculum v3.1 integrity tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
V3 = REPO / "experiments/nestful_synthetic_curriculum_v3"
CURR = V3 / "outputs/curriculum_v3_1"
FILTERED = CURR / "filtered"

STAGE_RULES = {
    "stage1_1call_atomic.jsonl": lambda n: n == 1,
    "stage2_2call_dependency.jsonl": lambda n: n == 2,
    "stage3_3call_composition.jsonl": lambda n: n == 3,
    "stage4_4to6call_persistence.jsonl": lambda n: 4 <= n <= 6,
}


def _load_jsonl(path: Path):
    if not path.is_file():
        pytest.skip(f"missing {path} — run build pipeline first")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.mark.parametrize("fname,rule", list(STAGE_RULES.items()))
def test_exact_num_calls(fname, rule):
    base = FILTERED if FILTERED.is_dir() else CURR
    samples = _load_jsonl(base / fname)
    assert len(samples) >= 800, f"{fname} has {len(samples)} samples"
    for s in samples:
        assert rule(s["num_calls"]), f"{s['sample_id']} has {s['num_calls']} calls"


def test_no_duplicate_sample_ids():
    base = FILTERED if FILTERED.is_dir() else CURR
    ids = set()
    for fname in STAGE_RULES:
        for s in _load_jsonl(base / fname):
            sid = s["sample_id"]
            assert sid not in ids, sid
            ids.add(sid)


def test_integrity_summary_passes():
    path = CURR / "curriculum_integrity_summary.json"
    if not path.is_file():
        pytest.skip("run validate_curriculum_integrity_v3_1 first")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("exact_num_calls_integrity") is True
    assert data.get("status") == "PASS"
