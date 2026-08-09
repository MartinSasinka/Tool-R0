"""Sanity / property tests for execution_aware_v2_p43 (Variant A)."""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "experiments"))
sys.path.insert(0, os.path.join(_REPO, "experiments", "nestful_mtgrpo_partial"))
sys.path.insert(0, os.path.join(_REPO, "experiments", "nestful_mtgrpo_minimal"))

from nestful_core import rewards as R  # noqa: E402
from nestful_core.rollout import Trajectory, Turn  # noqa: E402


GOLD = [
    {"name": "add", "arguments": {"arg_0": 1, "arg_1": 2}, "label": "$var1"},
    {"name": "multiply", "arguments": {"arg_0": "$var1.result$", "arg_1": 3}, "label": "$var2"},
]
GOLD_ANSWER = 9

# Independent A/B then dependent C (for ordering tests)
GOLD_DAG = [
    {"name": "fetch_a", "arguments": {"q": "a"}, "label": "$var1"},
    {"name": "fetch_b", "arguments": {"q": "b"}, "label": "$var2"},
    {"name": "join", "arguments": {"a": "$var1.x$", "b": "$var2.x$"}, "label": "$var3"},
]


def _task(gold=None, answer=None, declared=None) -> Dict[str, Any]:
    g = list(gold if gold is not None else GOLD)
    return {
        "task_id": "t",
        "gold_calls": g,
        "gold_answer": GOLD_ANSWER if answer is None else answer,
        "num_calls": len(g),
        "tools": [],
        "declared": declared or {},
    }


def _traj(turns_spec, final_obs, stop_reason="max_turns", *, clipped=False) -> Trajectory:
    tr = Trajectory(task_id="t", stage=2, gold_num_turns=2, executor_mode="full")
    tr.clipped_any = clipped
    tr.stop_reason = stop_reason
    tr.final_observation = final_obs
    for i, (call, fail, obs) in enumerate(turns_spec):
        t = Turn(turn_idx=i, model_text="")
        t.parsed_call = call
        t.fail_reason = fail
        t.observation = obs
        if call is None and fail is None and obs is None:
            t.is_terminal = True
        if fail == "clipped_completion":
            t.clipped_completion = True
        tr.turns.append(t)
    return tr


def _r(traj, task=None):
    return R.execution_aware_v2_p43(traj, task or _task())


def test_variant_a_no_episode_gold_sequence():
    res = _r(_traj([(GOLD[0], None, 3), (GOLD[1], None, 9)], 9))
    assert res.diagnostics["reward_variant"] == "A"
    assert res.diagnostics["reward_gold_sequence_alignment"] is None
    assert res.diagnostics["solution_equivalence_status"] == "UNVERIFIED"
    assert res.diagnostics["solution_equivalent_verified"] is False


def test_A_canonical_highest_band():
    r = _r(_traj([(GOLD[0], None, 3), (GOLD[1], None, 9)], 9)).reward
    assert r >= 0.95


def test_B_independent_order_same_tools_high():
    """Same gold tools, swapped independent order — high without runtime verifier."""
    tk = _task(GOLD_DAG, answer={"ok": 1})
    a, b, c = GOLD_DAG
    canon = _r(_traj([(a, None, {"x": 1}), (b, None, {"x": 2}), (c, None, {"ok": 1})],
                     {"ok": 1}), tk).reward
    swapped = _r(_traj([(b, None, {"x": 2}), (a, None, {"x": 1}), (c, None, {"ok": 1})],
                       {"ok": 1}), tk).reward
    assert canon >= 0.90
    assert swapped >= 0.90
    assert abs(canon - swapped) < 0.05


def test_C_correct_final_invalid_trajectory_lower():
    alt = [
        {"name": "sum", "arguments": {"arg_0": 1, "arg_1": 2}, "label": "$var1"},
        {"name": "prod", "arguments": {"arg_0": "$var1.result$", "arg_1": 3}, "label": "$var2"},
    ]
    a = _r(_traj([(GOLD[0], None, 3), (GOLD[1], None, 9)], 9)).reward
    c = _r(_traj([(alt[0], None, 3), (alt[1], None, 9)], 9)).reward
    assert c < a - 0.25
    assert c < 0.55


def test_D_wrong_final_near_complete_partial():
    r = _r(_traj([(GOLD[0], None, 3), (GOLD[1], None, 99)], 99)).reward
    assert 0.15 < r <= 0.45 + 1e-9


def test_E_first_subgoal_then_fail():
    r = _r(_traj([(GOLD[0], None, 3), (GOLD[1], "exec:boom", None)], 3)).reward
    assert 0.05 < r < 0.40


def test_F_arg_progress_ladder():
    wrong_tool = {"name": "subtract", "arguments": {"arg_0": 1, "arg_1": 2}, "label": "$var1"}
    wrong_keys = {"name": "add", "arguments": {"x": 1, "y": 2}, "label": "$var1"}
    partial_keys = {"name": "add", "arguments": {"arg_0": 1}, "label": "$var1"}
    right = GOLD[0]
    r_wrong = _r(_traj([(wrong_tool, None, -1)], -1)).reward
    r_wkeys = _r(_traj([(wrong_keys, None, 3)], 3)).reward
    r_partial = _r(_traj([(partial_keys, None, 1)], 1)).reward
    r_right = _r(_traj([(right, None, 3)], 3)).reward
    assert r_wrong < r_wkeys <= r_partial < r_right


def test_G_invalid_var_reference():
    bad = {"name": "multiply", "arguments": {"arg_0": "$var5.result$", "arg_1": 3},
           "label": "$var2"}
    good = _r(_traj([(GOLD[0], None, 3), (GOLD[1], None, 9)], 9))
    bad_r = _r(_traj([(GOLD[0], None, 3), (bad, "exec:unresolved_variable:var5", None)], 3))
    assert bad_r.diagnostics["reward_reference_correctness"] < 1.0
    assert bad_r.reward < good.reward


def test_H_random_executable_low():
    junk = [
        {"name": "noise1", "arguments": {"z": 1}, "label": "$var1"},
        {"name": "noise2", "arguments": {"z": 2}, "label": "$var2"},
    ]
    r = _r(_traj([(junk[0], None, 1), (junk[1], None, 2)], 2)).reward
    assert r < 0.25


def test_I_no_parseable_action_zero():
    assert _r(_traj([(None, "parse:invalid_json", None)], None, "parse_fail")).reward == 0.0
    assert _r(_traj([(None, None, None)], None, "terminal")).reward == 0.0


def test_J_shorter_path_no_raw_missing_penalty_without_verifier():
    """Without runtime equivalence, shorter non-gold path is NOT awarded missing=0."""
    short = [{"name": "magic", "arguments": {}, "label": "$var1"}]
    res = _r(_traj([(short[0], None, 9)], 9))
    assert res.diagnostics["solution_equivalent_verified"] is False
    assert res.diagnostics["missing_required_semantic_subgoals"] == 2
    # Must not get full outcome via irrelevant tool
    assert res.diagnostics["reward_final_outcome"] == 0.0


def test_K_longer_same_tools_no_length_only_penalty():
    """Extra gold-named calls that aren't 'unnecessary' (name in gold) → no length pen."""
    # Third call repeats a gold tool name — counted as unnecessary only if name NOT in gold.
    # Using gold name "add" again: name is in gold → not unnecessary under our rule.
    extra = {"name": "add", "arguments": {"arg_0": 0, "arg_1": 0}, "label": "$var3"}
    base = _r(_traj([(GOLD[0], None, 3), (GOLD[1], None, 9)], 9)).reward
    longer = _r(_traj([(GOLD[0], None, 3), (GOLD[1], None, 9), (extra, None, 0)], 9))
    assert longer.diagnostics["unnecessary_extra_calls"] == 0
    assert longer.diagnostics["reward_efficiency_penalty"] == 0.0
    assert longer.reward >= base - 1e-6


def test_L_redundant_unnecessary_penalty():
    junk = {"name": "identity", "arguments": {"x": 1}, "label": "$var3"}
    base = _r(_traj([(GOLD[0], None, 3), (GOLD[1], None, 9)], 9)).reward
    with_junk = _r(_traj([(GOLD[0], None, 3), (GOLD[1], None, 9), (junk, None, 9)], 9))
    assert with_junk.diagnostics["unnecessary_extra_calls"] == 1
    assert with_junk.diagnostics["reward_efficiency_penalty"] > 0
    assert with_junk.reward < base


def test_M_gold_order_vs_independent_perm():
    tk = _task(GOLD_DAG, answer={"ok": 1})
    a, b, c = GOLD_DAG
    gold_order = _r(_traj([(a, None, {"x": 1}), (b, None, {"x": 2}), (c, None, {"ok": 1})],
                          {"ok": 1}), tk).reward
    perm = _r(_traj([(b, None, {"x": 2}), (a, None, {"x": 1}), (c, None, {"ok": 1})],
                    {"ok": 1}), tk).reward
    assert gold_order >= 0.9 and perm >= 0.9
    # Variant A: episode may be equal; turns carry sequence (not asserted here).


def test_N_dependency_invalid_order():
    tk = _task(GOLD_DAG, answer={"ok": 1})
    a, b, c = GOLD_DAG
    # C before A/B → unresolved refs on join
    bad = _r(_traj([
        (c, "exec:unresolved_variable:var1", None),
        (a, None, {"x": 1}),
        (b, None, {"x": 2}),
    ], None, "executor_error"), tk)
    good = _r(_traj([(a, None, {"x": 1}), (b, None, {"x": 2}), (c, None, {"ok": 1})],
                    {"ok": 1}), tk)
    assert bad.reward < good.reward - 0.3
    assert bad.diagnostics["reward_semantic_progress"] < good.diagnostics["reward_semantic_progress"]


def test_ordering_invariant():
    a = _r(_traj([(GOLD[0], None, 3), (GOLD[1], None, 9)], 9)).reward
    # "near complete wrong final"
    near = _r(_traj([(GOLD[0], None, 3), (GOLD[1], None, 99)], 99)).reward
    partial = _r(_traj([(GOLD[0], None, 3)], 3)).reward
    random = _r(_traj([({"name": "noise", "arguments": {"z": 1}, "label": "$v"}, None, 1)],
                      1)).reward
    malformed = _r(_traj([(None, "parse:x", None)], None, "parse_fail")).reward
    assert a > near > partial > random >= malformed
    assert a > near  # verified-equivalent unavailable; canonical still beats wrong-final


def test_variance_ladder_monotone():
    ladder = [
        _r(_traj([(None, None, None)], None, "terminal")).reward,
        _r(_traj([(GOLD[0], None, 3)], 3)).reward,
        _r(_traj([(GOLD[0], None, 3),
                  ({"name": "multiply", "arguments": {"arg_0": 1}, "label": "$var2"}, None, 1)],
                 1)).reward,
        _r(_traj([(GOLD[0], None, 3), (GOLD[1], None, 99)], 99)).reward,
        _r(_traj([(GOLD[0], None, 3), (GOLD[1], None, 9)], 9)).reward,
    ]
    # Non-decreasing with soft tolerance for equal mid bands
    for i in range(len(ladder) - 1):
        assert ladder[i] <= ladder[i + 1] + 1e-9, ladder
    assert ladder[-1] > ladder[0]
    # Not a flat collision mid-band
    assert len({round(x, 4) for x in ladder[1:-1]}) >= 2


def test_v2_unchanged_perfect():
    """Original v2 semantics must remain available and high on canonical."""
    r = R.execution_aware_v2(
        _traj([(GOLD[0], None, 3), (GOLD[1], None, 9)], 9), _task()).reward
    assert r >= 0.85


def test_dispatch_p43():
    from vllm_dp_pool import resolve_reward_info
    cfg = {
        "reward": {
            "train_policy": "execution_aware_v2_p43",
            "p43_reward_variant": "A",
            "dispatch": {"require_exact_policy": True, "allow_fallback": False},
            "weights": {
                "final_outcome": 0.40,
                "executability": 0.15,
                "semantic_completeness": 0.20,
                "reference_correctness": 0.10,
                "semantic_progress": 0.15,
            },
        }
    }
    fn, info = resolve_reward_info(cfg)
    assert info["resolved_policy"] == "execution_aware_v2_p43"
    assert info["configured_policy"] == "execution_aware_v2_p43"
    assert getattr(fn, "reward_policy") == "execution_aware_v2_p43"


def test_dispatch_requires_variant():
    from vllm_dp_pool import resolve_reward_info
    with pytest.raises(ValueError, match="p43_reward_variant"):
        resolve_reward_info({"reward": {"train_policy": "execution_aware_v2_p43"}})


def test_turn_seq_still_has_gold_part():
    seq = R.execution_aware_v2_p43_seq(
        _traj([(GOLD[0], None, 3), (GOLD[1], None, 9)], 9), _task())
    assert len(seq["r_seq"]) == 2
    assert all(x > 0.4 for x in seq["r_seq"])
    assert seq["diagnostics"]["reward_gold_sequence_alignment"] is None


def test_no_episode_mean_st_as_progress():
    traj = _traj([(GOLD[0], None, 3), (GOLD[1], None, 9)], 9)
    res = _r(traj)
    scores = res.diagnostics.get("turn_rewards") or []
    mean_st_proxy = sum(scores) / len(scores) if scores else 0.0
    # Episode progress is subgoal ladder, not mean of positional gold scores alone.
    assert abs(res.diagnostics["reward_semantic_progress"] - mean_st_proxy) > 1e-12 or \
        res.diagnostics["reward_semantic_progress"] == 1.0
