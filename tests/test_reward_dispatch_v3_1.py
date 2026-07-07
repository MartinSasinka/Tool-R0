"""Unit tests for the post-audit v3.1 training-stack fixes.

Covers the seven required tests:
  1. reward resolver knows both intended policies (never strict for them)
  2. v3.1 reward produces fractional (more-than-binary) values
  3. unknown policy hard-fails unless ALLOW_STRICT_REWARD_FALLBACK=1
  4. per-position advantage logic kills the turn-position artifact
  5. zero-step checkpoints are never eligible for best
  6. replay_ratio=0.20 gives a 20/80 mix, not 50/50
  7. normalize_task preserves stage/motif metadata
"""
from __future__ import annotations

import json
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MINIMAL = os.path.join(_REPO, "experiments", "nestful_mtgrpo_minimal")
_V3 = os.path.join(_REPO, "experiments", "nestful_synthetic_curriculum_v3")
if _MINIMAL not in sys.path:
    sys.path.insert(0, _MINIMAL)
# APPEND (not insert) the v3 dir: it also contains run.py etc., and putting it
# in front of the minimal experiment would shadow those modules for other tests.
if _V3 not in sys.path:
    sys.path.append(_V3)

import vllm_dp_pool  # noqa: E402
from group_stats import compute_group_stats  # noqa: E402
from checkpoint_eligibility import evaluate_eligibility  # noqa: E402
from data import normalize_task, load_tasks_mixed  # noqa: E402
from nestful_core.rollout import Trajectory, Turn  # noqa: E402


def _cfg(policy):
    return {"reward": {"train_policy": policy}}


# ─── Test 1: resolver knows both intended policies ───────────────────────────

def test_resolver_v3_1_stepwise_not_strict():
    fn, info = vllm_dp_pool.resolve_reward_info(_cfg("execution_aware_v3_1_stepwise"))
    assert info["fallback_used"] is False
    assert info["resolved_policy"] == "execution_aware_v3_1_stepwise"
    assert info["reward_fn_module"] != "reward"
    assert "reward_v3_1" in info["reward_fn_module"]


def test_resolver_v2_1_motif_not_strict():
    fn, info = vllm_dp_pool.resolve_reward_info(_cfg("execution_aware_v2_1_motif"))
    assert info["fallback_used"] is False
    assert info["resolved_policy"] == "execution_aware_v2_1_motif"
    assert info["reward_fn_module"] != "reward"
    assert "reward_motif" in info["reward_fn_module"]


def test_resolver_strict_only_when_requested():
    fn, info = vllm_dp_pool.resolve_reward_info(_cfg("strict"))
    assert info["reward_fn_module"] == "reward"
    assert info["fallback_used"] is False


# ─── Test 3: unknown policy hard-fails unless fallback explicitly allowed ────

def test_unknown_policy_raises(monkeypatch):
    monkeypatch.delenv("ALLOW_STRICT_REWARD_FALLBACK", raising=False)
    with pytest.raises(ValueError):
        vllm_dp_pool.resolve_reward_info(_cfg("totally_made_up_policy"))


def test_unknown_policy_fallback_only_with_env(monkeypatch):
    monkeypatch.setenv("ALLOW_STRICT_REWARD_FALLBACK", "1")
    fn, info = vllm_dp_pool.resolve_reward_info(_cfg("totally_made_up_policy"))
    assert info["fallback_used"] is True
    assert info["reward_fn_module"] == "reward"


# ─── Test 2: v3.1 reward is graded, not binary ───────────────────────────────

def _mk_task(stage="stage1", gold=None, ans=3, terminal=True):
    gold = gold if gold is not None else [
        {"name": "add", "arguments": {"arg_0": 1, "arg_1": 2}, "label": "$var1"}]
    return {
        "task_id": "t0",
        "question": "add 1 and 2",
        "tools": [],
        "gold_calls": gold,
        "gold_answer": ans,
        "num_calls": len(gold),
        "stage": stage,
        "terminal_stage": terminal,
    }


def _mk_traj(task, calls, final_obs=None, terminal=True, fail_first=None):
    traj = Trajectory(task["task_id"], task["num_calls"], task["num_calls"])
    for i, call in enumerate(calls):
        t = Turn(i, model_text=json.dumps(call) if call else "")
        if fail_first and i == 0:
            t.fail_reason = fail_first
        else:
            t.parsed_call = dict(call)
            t.observation = final_obs
        traj.turns.append(t)
    if terminal:
        t = Turn(len(calls), model_text="[]")
        t.is_terminal = True
        traj.turns.append(t)
    traj.final_observation = final_obs
    traj.stop_reason = "terminal"
    return traj


def test_v3_1_reward_fractional():
    from lib import reward_v3_1

    task = _mk_task()
    gold_call = task["gold_calls"][0]
    variants = {
        "correct": _mk_traj(task, [gold_call], final_obs=3),
        "wrong_final": _mk_traj(task, [gold_call], final_obs=999),
        "wrong_tool": _mk_traj(
            task, [{"name": "subtract", "arguments": {"arg_0": 1, "arg_1": 2},
                    "label": "$var1"}], final_obs=-1),
        "wrong_args": _mk_traj(
            task, [{"name": "add", "arguments": {"arg_0": 7, "arg_1": 8},
                    "label": "$var1"}], final_obs=15),
        "parse_error": _mk_traj(task, [gold_call], terminal=False,
                                fail_first="parse:invalid_json"),
    }
    rewards = {}
    for name, traj in variants.items():
        out = reward_v3_1.episode_turn_reward_seq(traj, task)
        rewards[name] = out["episode_reward"]

    assert len(set(rewards.values())) > 2, f"reward not graded: {rewards}"
    assert any(0.0 < r < 1.0 for r in rewards.values()), \
        f"no fractional values: {rewards}"
    assert rewards["parse_error"] == 0.0
    assert rewards["correct"] >= 0.90
    assert rewards["wrong_tool"] <= 0.35 + 1e-9
    assert rewards["wrong_args"] <= 0.60 + 1e-9
    assert rewards["correct"] > rewards["wrong_final"] > rewards["wrong_args"] \
        > rewards["wrong_tool"] > rewards["parse_error"]


def test_v2_1_motif_reward_fractional():
    from lib import reward_motif

    task = _mk_task()
    gold_call = task["gold_calls"][0]
    correct = reward_motif.episode_turn_reward_seq(
        _mk_traj(task, [gold_call], final_obs=3), task)
    wrong_tool = reward_motif.episode_turn_reward_seq(
        _mk_traj(task, [{"name": "subtract",
                         "arguments": {"arg_0": 1, "arg_1": 2},
                         "label": "$var1"}], final_obs=-1), task)
    vals = {correct["episode_reward"], wrong_tool["episode_reward"]}
    assert correct["episode_reward"] > wrong_tool["episode_reward"]
    assert any(v not in (0.0, 1.0) for v in vals), f"motif reward binary: {vals}"


def test_v3_1_reward_requires_stage(monkeypatch):
    from lib import reward_v3_1

    monkeypatch.delenv("TRAIN_STAGE", raising=False)
    task = _mk_task()
    traj = _mk_traj(task, [task["gold_calls"][0]], final_obs=3)
    # detect_stage may fall back to num_calls/gold_calls; strip EVERYTHING
    # a stage could be inferred from — then it must hard-fail (audit Bug 7).
    del task["stage"]
    task["num_calls"] = 0
    task["gold_calls"] = []
    with pytest.raises(reward_v3_1.RewardError):
        reward_v3_1.detect_stage(task)
    with pytest.raises(reward_v3_1.RewardError):
        reward_v3_1.execution_aware_v3_1_stepwise(traj, task)


# ─── Test 4: turn-position advantage artifact ────────────────────────────────

def test_identical_two_turn_episodes_are_dead():
    # 4 identical 2-turn episodes. Returns G_t sum future rewards, so turn 0
    # return (1.0) > turn 1 return (0.5): flattened std is NONZERO (old logic
    # said "alive"), but between-completion std at every position is 0.
    ep_returns = [[1.0, 0.5]] * 4
    stats = compute_group_stats(ep_returns, [0.5] * 4)
    assert stats.dead_flattened is False        # old buggy logic
    assert stats.dead_corrected is True         # corrected logic
    assert stats.position_artifact_detected is True
    assert all(a == 0.0 for row in stats.advantages for a in row)


def test_between_completion_variance_is_alive():
    ep_returns = [[1.0, 0.5], [1.0, 0.5], [0.2, 0.1], [0.2, 0.1]]
    stats = compute_group_stats(ep_returns, [0.5, 0.5, 0.1, 0.1])
    assert stats.dead_corrected is False
    assert stats.position_artifact_detected is False
    # Better completions get positive advantage at BOTH positions.
    assert stats.advantages[0][0] > 0 and stats.advantages[0][1] > 0
    assert stats.advantages[2][0] < 0 and stats.advantages[2][1] < 0


# ─── Test 5: checkpoint eligibility ──────────────────────────────────────────

def test_zero_step_checkpoint_not_eligible():
    summary = {
        "steps": 0, "contributing_turns_total": 0, "dead_group_rate": 1.0,
        "reward_policy_configured": "execution_aware_v3_1_stepwise",
        "reward_policy_resolved": "execution_aware_v3_1_stepwise",
    }
    eligible, reason, meta = evaluate_eligibility(summary, react_win=0.9)
    assert eligible is False
    assert meta["trained"] is False
    assert meta["eligible_for_best"] is False


def test_trained_checkpoint_eligible():
    summary = {
        "steps": 42, "contributing_turns_total": 100, "dead_group_rate": 0.3,
        "reward_policy_configured": "execution_aware_v3_1_stepwise",
        "reward_policy_resolved": "execution_aware_v3_1_stepwise",
    }
    eligible, reason, meta = evaluate_eligibility(
        summary, react_win=0.5, global_best=0.4, baseline_win=0.45)
    assert eligible is True


def test_high_dead_rate_not_eligible():
    summary = {
        "steps": 5, "contributing_turns_total": 10, "dead_group_rate": 0.99,
        "reward_policy_configured": "x", "reward_policy_resolved": "x",
    }
    eligible, _, _ = evaluate_eligibility(summary, react_win=0.9)
    assert eligible is False


# ─── Test 6: replay ratio semantics ──────────────────────────────────────────

def _write_stage_file(tmp_path, name, n, tool):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "sample_id": f"{name}_{i}",
                "input": f"q {name} {i}",
                "tools": [],
                "output": [{"name": tool, "arguments": {"a": i}, "label": "$var1"}],
                "gold_answer": i,
            }) + "\n")
    return str(p)


def test_replay_ratio_20_80(tmp_path):
    f1 = _write_stage_file(tmp_path, "stage1.jsonl", 100, "add")
    f2 = _write_stage_file(tmp_path, "stage2.jsonl", 100, "mul")
    out = load_tasks_mixed([f1, f2], replay_ratio=0.20, seed=1)
    eff = out["effective_mix"]
    assert abs(eff[0] - 0.20) <= 0.01, f"stage1 share {eff[0]} != 0.20"
    assert abs(eff[1] - 0.80) <= 0.01, f"stage2 share {eff[1]} != 0.80"


def test_scalar_weight_multi_file_rejected(tmp_path):
    f1 = _write_stage_file(tmp_path, "s1.jsonl", 10, "add")
    f2 = _write_stage_file(tmp_path, "s2.jsonl", 10, "mul")
    with pytest.raises(ValueError):
        load_tasks_mixed([f1, f2], weights=[0.2])


# ─── Test 7: metadata preservation ───────────────────────────────────────────

def test_normalize_task_preserves_metadata():
    row = {
        "sample_id": "s1",
        "input": "q",
        "tools": [],
        "output": [{"name": "add", "arguments": {"a": 1}, "label": "$var1"}],
        "gold_answer": 1,
        "stage": "stage2",
        "terminal_stage": False,
        "motif_type": "chain",
        "prefix_of_motif": "m1",
        "target_full_motif": "m1_full",
        "source_failure_cluster": "c3",
        "trajectory_id": "traj_9",
    }
    task = normalize_task(row)
    assert task["stage"] == "stage2"
    assert task["terminal_stage"] is False
    assert task["motif_type"] == "chain"
    assert task["prefix_of_motif"] == "m1"
    assert task["target_full_motif"] == "m1_full"
    assert task["source_failure_cluster"] == "c3"
    assert task["trajectory_id"] == "traj_9"
    assert task["num_calls"] == 1
