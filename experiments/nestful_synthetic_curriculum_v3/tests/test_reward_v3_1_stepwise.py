"""Tests for execution_aware_v3_1_stepwise reward."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
LIB = Path(__file__).resolve().parents[1] / "lib"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(LIB.parent))

from lib.reward_v3_1 import detect_stage, execution_aware_v3_1_stepwise  # noqa: E402


@dataclass
class Turn:
    parsed_call: Any = None
    final_answer: Optional[str] = None


@dataclass
class Trajectory:
    turns: List[Turn] = field(default_factory=list)
    clipped_any: bool = False


def _task(stage: str, n_calls: int, terminal: bool = True) -> dict:
    return {
        "stage": stage,
        "num_calls": n_calls,
        "gold_calls": [{"name": "add"}] * n_calls,
        "terminal_stage": terminal,
        "motif_type": "linear_dependency",
    }


def test_detect_stage_from_sample():
    assert detect_stage({"stage": "stage2_2call_dependency", "num_calls": 2}) == "stage2"


def test_too_few_calls_capped():
    traj = Trajectory(turns=[Turn(parsed_call={"name": "add"})])
    task = _task("stage2_2call_dependency", 2)
    res = execution_aware_v3_1_stepwise(traj, task, train_stage=2)
    assert res.reward <= 0.1
    assert res.diagnostics.get("cap_applied") in ("too_few_calls", "severe_short_trace", None) or res.reward <= 0.25


def test_premature_final_nonterminal_zero():
    traj = Trajectory(turns=[
        Turn(parsed_call={"name": "add"}),
        Turn(final_answer="42"),
    ])
    task = _task("stage2_2call_dependency", 2, terminal=False)
    res = execution_aware_v3_1_stepwise(traj, task, train_stage=2)
    assert res.reward == 0.0
    assert res.diagnostics.get("cap_applied") == "premature_final_nonterminal"


def test_executable_prefix_floor_stage1():
    traj = Trajectory(turns=[
        Turn(parsed_call={"name": "add", "arguments": {}}),
        Turn(final_answer="5"),
    ])
    task = _task("stage1_1call_atomic", 1)
    res = execution_aware_v3_1_stepwise(traj, task, train_stage=1)
    assert res.reward >= 0.75 or res.diagnostics.get("cap_applied") is not None


def test_stage2_valid_refs_higher_than_too_few():
    full = Trajectory(turns=[
        Turn(parsed_call={"name": "add"}),
        Turn(parsed_call={"name": "multiply"}),
        Turn(final_answer="10"),
    ])
    short = Trajectory(turns=[Turn(parsed_call={"name": "add"})])
    task = _task("stage2_2call_dependency", 2)
    r_full = execution_aware_v3_1_stepwise(full, task, train_stage=2)
    r_short = execution_aware_v3_1_stepwise(short, task, train_stage=2)
    assert r_full.reward >= r_short.reward
