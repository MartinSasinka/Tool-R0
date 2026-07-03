"""Tests for prefix decomposition (v3.1)."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_prefix_curriculum_from_trajectories import decompose_trajectory  # noqa: E402
from generate_full_motif_trajectories_v3_1 import generate_one  # noqa: E402
import random  # noqa: E402


def test_long_chain_decomposes_to_prefix_stages():
    rng = random.Random(42)
    traj = generate_one(rng, 42, "long_chain__too_few_calls", num_calls=7)
    counter = [0]
    stages = decompose_trajectory(traj, counter, rng)
    assert "stage1_1call_atomic" in stages
    assert "stage2_2call_dependency" in stages
    assert "stage3_3call_composition" in stages
    assert all(s["num_calls"] == 1 for s in stages["stage1_1call_atomic"])
    assert all(s["num_calls"] == 2 for s in stages["stage2_2call_dependency"])
    assert all(s["num_calls"] == 3 for s in stages["stage3_3call_composition"])
    for s in stages["stage4_4to6call_persistence"]:
        assert 4 <= s["num_calls"] <= 6


def test_no_future_stage_leakage():
    rng = random.Random(99)
    traj = generate_one(rng, 99, "linear_dependency__too_few_calls", num_calls=3)
    counter = [0]
    stages = decompose_trajectory(traj, counter, rng)
    assert max(s["num_calls"] for s in stages["stage1_1call_atomic"]) == 1
    assert max(s["num_calls"] for s in stages["stage2_2call_dependency"]) == 2
    assert all(s["prefix_of_motif"] for s in stages["stage1_1call_atomic"])


def test_prefix_length_matches_stage():
    rng = random.Random(7)
    traj = generate_one(rng, 7, "fan_in__wrong_argument", num_calls=5)
    counter = [0]
    stages = decompose_trajectory(traj, counter, rng)
    for stage, samples in stages.items():
        for s in samples:
            assert s["prefix_length"] == s["num_calls"]
            assert s["gold_answer"] is not None
            assert "long_chain" not in s["question"].lower()
            assert "cluster=" not in s["question"].lower()
            assert "[prefix" not in s["question"].lower()
