"""Contract tests for the current Pilot 4.3 synthetic executor."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from executor import ToolExecutor, matches_gold  # noqa: E402
from synthetic_tool_registry import get_synthetic_registry  # noqa: E402

_EXPERIMENTS = Path(__file__).resolve().parents[2]
_DATASET = (_EXPERIMENTS / "targeted_tool_data_factory" / "outputs" /
            "pilot4_3_nestful_profile_1000" /
            "train_nestful_profile_1000.jsonl")


def _first_task() -> dict:
    with _DATASET.open(encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


def test_registry_loads_current_p43_adapter():
    registry = get_synthetic_registry()
    assert registry.available, registry.load_error
    assert len(registry.tool_names()) >= 500
    assert registry.version and "pilot43" in registry.version
    assert registry.registry_hash()


def test_gold_trace_executes_against_current_adapter():
    task = _first_task()
    executor = ToolExecutor(task, registry=None, mode="synthetic")
    final = None
    for call in task["gold_calls"]:
        result = executor.execute(call)
        assert result.error is None, result.error
        final = result.observation
    assert matches_gold(final, task["gold_answer"])


def test_unknown_argument_is_rejected():
    task = _first_task()
    call = dict(task["gold_calls"][0])
    call["arguments"] = {**call["arguments"], "definitely_unknown": 1}
    result = ToolExecutor(task, registry=None, mode="synthetic").execute(call)
    assert result.error is not None
    assert "unknown" in result.error.lower()
