"""Unit tests for lib/reward_v3_2_dense.py (execution_aware_v3_2_dense).

Builds REAL Trajectory objects via rollout.run_episode with scripted
generate_fn callables (no model, no GPU), on a real task from the canonical
stage-2 curriculum, and checks:

  * every required failure class is detected,
  * class bands are respected and monotone (too-few can never outscore a
    complete executable correct trace),
  * within-band credit is dense (different completions in the same class get
    different rewards),
  * v3.2 produces at least as many distinct reward values as frozen v3.1 on
    the same set of trajectories,
  * dispatch via vllm_dp_pool.resolve_reward_info resolves the new policy.

Run:  python experiments/nestful_synthetic_curriculum_v3/tests/test_reward_v3_2_dense.py
  or: pytest -q experiments/nestful_synthetic_curriculum_v3/tests/test_reward_v3_2_dense.py
"""
from __future__ import annotations

import copy
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
V3_ROOT = os.path.dirname(_HERE)
REPO_ROOT = os.path.dirname(os.path.dirname(V3_ROOT))
MINIMAL = os.path.join(REPO_ROOT, "experiments", "nestful_mtgrpo_minimal")
for p in (V3_ROOT, MINIMAL):
    if p not in sys.path:
        sys.path.insert(0, p)

from data import load_tasks  # noqa: E402
from rollout import run_episode  # noqa: E402

from lib import reward_v3_1, reward_v3_2_dense  # noqa: E402

STAGE2 = os.path.join(V3_ROOT, "outputs", "curriculum_v3_1", "filtered",
                      "stage2_2call_dependency.jsonl")

CONFIG = {
    "generation": {"temperature": 0.0, "top_p": 1.0, "max_extra_turns_eval": 1},
    "executor": {"mode": "gold_replay"},
    "token_budget": {},
}


def _fmt(call) -> str:
    payload = {"name": call.get("name"), "arguments": call.get("arguments") or {}}
    if call.get("label"):
        # labels are how later $var_N$ references resolve — real completions
        # emit them (the prompt format asks for them)
        payload["label"] = call["label"]
    return ("<tool_call_answer>[" + json.dumps(payload, ensure_ascii=False)
            + "]</tool_call_answer>")


TERMINAL = "<tool_call_answer>[]</tool_call_answer>"


def make_gen(texts):
    """Scripted generate_fn: returns texts[i] on turn i (last one repeats)."""
    state = {"i": 0}

    def gen(messages, max_new_tokens):  # noqa: ARG001
        i = min(state["i"], len(texts) - 1)
        state["i"] += 1
        return {"text": texts[i], "prompt_tokens": 5, "completion_tokens": 5,
                "clipped": False, "prompt_overflow": False}
    return gen


def _base_task():
    """A terminal 2-call task from the canonical stage-2 file."""
    tasks = load_tasks(STAGE2, max_tasks=None, seed=42)
    task = copy.deepcopy(tasks[0])
    task["terminal_stage"] = True
    return task


def _roll(task, texts, mode="train"):
    return run_episode(None, None, task, CONFIG, registry=None, mode=mode,
                       generate_fn=make_gen(texts))


def _score(task, traj):
    return reward_v3_2_dense.episode_turn_reward_seq(traj, task, None)


def _build_scenarios():
    """Return {name: (task, trajectory)} for all required failure classes."""
    t = _base_task()
    g = t["gold_calls"]
    wrong_tool_call = {"name": "definitely_not_a_tool", "arguments": dict(g[0]["arguments"])}
    wrong_args_calls = [
        {"name": c["name"], "arguments": {k: "totally_wrong" for k in (c.get("arguments") or {})}}
        for c in g
    ]
    bad_ref_call = {"name": g[0]["name"],
                    "arguments": {k: "$var_99$" for k in (g[0].get("arguments") or {})}}

    scenarios = {}
    scenarios["parse_error"] = (t, _roll(t, ["<tool_call_answer>{broken json"]))
    scenarios["no_tool_call"] = (t, _roll(t, [TERMINAL]))
    scenarios["wrong_tool"] = (t, _roll(t, [_fmt(wrong_tool_call), TERMINAL]))
    scenarios["too_few_correct_first"] = (t, _roll(t, [_fmt(g[0]), TERMINAL]))
    scenarios["too_few_wrong_args_first"] = (t, _roll(t, [_fmt(wrong_args_calls[0]), TERMINAL]))
    scenarios["wrong_args"] = (t, _roll(t, [_fmt(c) for c in wrong_args_calls] + [TERMINAL],
                                        mode="eval"))
    scenarios["fully_correct"] = (t, _roll(t, [_fmt(c) for c in g] + [TERMINAL], mode="eval"))
    scenarios["invalid_reference"] = (t, _roll(t, [_fmt(bad_ref_call), TERMINAL]))
    scenarios["too_many_calls"] = (t, _roll(t, [_fmt(g[0]), _fmt(g[1]), _fmt(g[0])],
                                            mode="eval"))

    # executable full trace but WRONG final answer: same rollout as
    # fully_correct, scored against a task whose gold_answer was changed.
    t_wrong_final = copy.deepcopy(t)
    t_wrong_final["gold_answer"] = "___not_the_answer___"
    scenarios["executable_wrong_final"] = (
        t_wrong_final, _roll(t, [_fmt(c) for c in g] + [TERMINAL], mode="eval"))

    # premature final on a NON-terminal (prefix) task
    t_prefix = copy.deepcopy(t)
    t_prefix["terminal_stage"] = False
    scenarios["premature_final"] = (t_prefix, _roll(t_prefix, [_fmt(g[0]), TERMINAL],
                                                    mode="eval"))
    return scenarios


def test_classes_and_bands():
    scenarios = _build_scenarios()
    results = {name: _score(task, traj) for name, (task, traj) in scenarios.items()}
    r = {name: res["episode_reward"] for name, res in results.items()}
    cls = {name: res["diagnostics"]["reward_class"] for name, res in results.items()}

    assert cls["parse_error"] == "parse_error" and r["parse_error"] == 0.0
    assert cls["no_tool_call"] == "no_tool_call" and abs(r["no_tool_call"] - 0.02) < 1e-9
    assert cls["premature_final"] == "premature_final_nonterminal"
    assert abs(r["premature_final"] - 0.04) < 1e-9
    assert cls["invalid_reference"] == "invalid_reference"
    assert 0.05 <= r["invalid_reference"] <= 0.15
    assert cls["wrong_tool"] in ("wrong_tool", "too_few_calls")  # 1 call emitted < gold 2
    assert r["wrong_tool"] <= 0.45
    assert cls["too_few_correct_first"] == "too_few_calls"
    assert 0.10 <= r["too_few_correct_first"] <= 0.45
    assert cls["too_few_wrong_args_first"] == "too_few_calls"
    assert cls["wrong_args"] == "correct_tool_wrong_args"
    assert 0.35 <= r["wrong_args"] <= 0.60
    assert cls["too_many_calls"] == "too_many_calls"
    assert 0.45 <= r["too_many_calls"] <= 0.70
    assert cls["executable_wrong_final"] == "executable_wrong_final"
    assert 0.60 <= r["executable_wrong_final"] <= 0.80
    assert cls["fully_correct"] == "fully_correct"
    assert r["fully_correct"] >= 0.90

    # required diagnostics present
    d = results["too_few_correct_first"]["diagnostics"]
    for key in ("format_score", "call_count_progress", "per_call_tool_score",
                "per_call_argument_score", "reference_score", "execution_score",
                "final_answer_score", "too_few_penalty", "too_many_penalty",
                "cap_reason", "reward_policy"):
        assert key in d, f"missing diagnostic {key}"
    assert d["reward_policy"] == "execution_aware_v3_2_dense"

    print("[test] classes:", {k: (cls[k], round(v, 4)) for k, v in r.items()})
    return r


def test_monotonicity_and_density():
    scenarios = _build_scenarios()
    r = {name: _score(task, traj)["episode_reward"]
         for name, (task, traj) in scenarios.items()}

    # too-few must NEVER outscore a complete executable correct trace
    too_few_max = max(r["too_few_correct_first"], r["too_few_wrong_args_first"],
                      r["wrong_tool"])
    assert too_few_max < r["fully_correct"], (too_few_max, r["fully_correct"])
    # dense within-band credit: correct-first-then-stop beats wrong-args-first-then-stop
    assert r["too_few_correct_first"] > r["too_few_wrong_args_first"], r
    # every reward stays in [0, 1]
    assert all(0.0 <= v <= 1.0 for v in r.values()), r


def test_denser_than_v3_1():
    """v3.2 must produce at least as many distinct rewards as frozen v3.1."""
    os.environ["TRAIN_STAGE"] = "2"
    scenarios = _build_scenarios()
    v31 = {n: reward_v3_1.episode_turn_reward_seq(traj, task, None)["episode_reward"]
           for n, (task, traj) in scenarios.items()}
    v32 = {n: reward_v3_2_dense.episode_turn_reward_seq(traj, task, None)["episode_reward"]
           for n, (task, traj) in scenarios.items()}
    u31 = len({round(v, 6) for v in v31.values()})
    u32 = len({round(v, 6) for v in v32.values()})
    print(f"[test] distinct rewards: v3.1={u31} v3.2={u32}")
    print(f"[test] v3.1: { {k: round(v, 4) for k, v in sorted(v31.items())} }")
    print(f"[test] v3.2: { {k: round(v, 4) for k, v in sorted(v32.items())} }")
    assert u32 >= u31, (u31, u32)


def test_dispatch_resolves():
    from vllm_dp_pool import resolve_reward_info
    cfg = {"reward": {"train_policy": "execution_aware_v3_2_dense"}}
    fn, info = resolve_reward_info(cfg)
    assert info["resolved_policy"] == "execution_aware_v3_2_dense", info
    assert not info["fallback_used"]
    assert fn is reward_v3_2_dense.episode_turn_reward_seq


if __name__ == "__main__":
    test_classes_and_bands()
    test_monotonicity_and_density()
    test_denser_than_v3_1()
    test_dispatch_resolves()
    print("[test_reward_v3_2_dense] ALL TESTS PASSED")
