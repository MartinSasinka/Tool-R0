"""Turn-level MT-GRPO building blocks: per-turn rewards, returns, token budget."""
from reward import (
    strict_gold_turn_rewards,
    episode_turn_reward_seq,
    strict_gold_trace_episode_reward,
    strict_gold_trace_reward,
)
from rollout import get_stage_token_budget
from grpo_train import _turn_returns
from _helpers import make_trajectory


def _task():
    return {
        "task_id": "t1",
        "tools": [
            {"name": "add", "parameters": {"properties": {"arg_0": {}, "arg_1": {}}}},
            {"name": "multiply", "parameters": {"properties": {"arg_0": {}, "arg_1": {}}}},
        ],
        "gold_calls": [
            {"name": "add", "arguments": {"arg_0": 1, "arg_1": 2}},
            {"name": "multiply", "arguments": {"arg_0": 3, "arg_1": 10}},
        ],
        "gold_answer": 30,
        "num_calls": 2,
    }


def test_episode_reward_alias_is_strict():
    assert strict_gold_trace_episode_reward is strict_gold_trace_reward


def test_turn_rewards_all_one_on_gold():
    task = _task()
    gold_obs = [3, 30]
    traj = make_trajectory("t1", task["gold_calls"], [3, 30],
                           gold_num_turns=2, final_observation=30)
    r = strict_gold_turn_rewards(traj, task, gold_obs)
    assert r == [1.0, 1.0]


def test_turn_rewards_first_turn_wrong():
    task = _task()
    gold_obs = [3, 30]
    bad = [
        {"name": "multiply", "arguments": {"arg_0": 1, "arg_1": 2}},  # wrong name @0
        task["gold_calls"][1],
    ]
    traj = make_trajectory("t1", bad, [2, 30], gold_num_turns=2, final_observation=30)
    r = strict_gold_turn_rewards(traj, task, gold_obs)
    assert r[0] == 0.0


def test_episode_turn_reward_seq_aligned_and_gold_only():
    task = _task()
    gold_obs = [3, 30]
    traj = make_trajectory("t1", task["gold_calls"], [3, 30],
                           gold_num_turns=2, final_observation=30)
    info = episode_turn_reward_seq(traj, task, gold_obs)
    assert info["r_seq"] == [1.0, 1.0]
    assert info["episode_reward"] == 1.0


def test_turn_returns_formula():
    # gamma=1, lambda=1: G_t = sum_{k>=t} r_k + R_episode
    r_seq = [1.0, 1.0]
    G = _turn_returns(r_seq, episode_reward=1.0, gamma=1.0, lambda_episode=1.0)
    # G_0 = (1+1) + 1 = 3 ; G_1 = (1) + 1 = 2
    assert G == [3.0, 2.0]


def test_turn_returns_discount():
    r_seq = [0.0, 1.0]
    G = _turn_returns(r_seq, episode_reward=0.0, gamma=0.5, lambda_episode=0.0)
    # G_0 = 0 + 0.5*1 = 0.5 ; G_1 = 1
    assert abs(G[0] - 0.5) < 1e-9
    assert abs(G[1] - 1.0) < 1e-9


def test_stage_token_budget_smoke_smaller():
    config = {
        "token_budget": {"stage_defaults": {
            "3": {"max_prompt_tokens": 4096, "max_new_tokens": 2048, "vllm_max_model_length": 6144},
            "4": {"max_prompt_tokens": 4096, "max_new_tokens": 2560, "vllm_max_model_length": 6144},
        }},
        "generation": {"max_new_tokens_smoke": 1024, "max_new_tokens_train": 2048,
                       "max_new_tokens_eval": 2560},
    }
    smoke = get_stage_token_budget(config, 3, "smoke")
    train = get_stage_token_budget(config, 3, "train")
    eval4 = get_stage_token_budget(config, 4, "eval")
    assert smoke["max_new_tokens"] == 1024
    assert smoke["max_prompt_tokens"] == 4096
    assert train["max_new_tokens"] == 2048
    assert eval4["max_new_tokens"] == 2560
    assert eval4["vllm_max_model_length"] == 6144
