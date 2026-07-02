"""Manual integration check (no model). Run from repo root:

    C:\\anaconda\\python experiments/nestful_mtgrpo_minimal/tests/_manual_integration.py

Validates: data loader + IBM executor (full mode) + rollout + strict reward +
final_eval metrics, using a stub generator that replays the gold calls.
"""
import json
import os
import sys

_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _FOLDER)

from data import load_tasks
from executor import detect_ibm_functions_dir, IBMFunctionRegistry
from prompt import SYSTEM_PROMPT
from rollout import run_episode
from reward import strict_gold_trace_reward, compute_gold_observations
from metrics import compute_nestful_official_metrics, compute_paper_metrics

REPO_ROOT = os.path.abspath(os.path.join(_FOLDER, "..", ".."))
TRAIN = os.path.join(REPO_ROOT, "curricullum", "data",
                     "filtered_toolr0_synthetic", "epoch_3_3call.jsonl")

config = {
    "generation": {"max_new_tokens": 512, "max_extra_turns_eval": 1,
                   "temperature": 0.0, "top_p": 1.0},
    "executor": {"mode": "auto", "ibm_call_timeout": 10.0},
}


def make_stub_generator(task):
    """Emit gold calls one per turn, then terminal []."""
    gold = task["gold_calls"]
    state = {"i": 0}

    def gen(messages, max_new_tokens):
        i = state["i"]
        state["i"] += 1
        if i < len(gold):
            c = gold[i]
            payload = json.dumps(
                [{"name": c["name"], "arguments": c.get("arguments", {})}],
                ensure_ascii=False,
            )
            text = f"<tool_call_answer>{payload}</tool_call_answer>"
        else:
            text = "<tool_call_answer>[]</tool_call_answer>"
        return {"text": text, "prompt_tokens": 10, "completion_tokens": 5,
                "clipped": False, "prompt_overflow": False}

    return gen


def main():
    funcs_dir = detect_ibm_functions_dir(repo_root=REPO_ROOT)
    print(f"IBM functions dir: {funcs_dir}")
    registry = IBMFunctionRegistry(funcs_dir) if funcs_dir else None
    mode = "full" if (registry and registry.available) else "gold_replay"
    print(f"executor mode: {mode}")

    tasks = load_tasks(TRAIN, stage=3, max_tasks=5, seed=42)
    print(f"loaded {len(tasks)} tasks (stage 3)")

    n_reward_1 = 0
    for task in tasks:
        gen = make_stub_generator(task)
        traj = run_episode(task=task, model=None, tokenizer=None, config=config,
                           registry=registry, is_eval=True, generate_fn=gen)
        gold_obs = compute_gold_observations(task, registry)
        rr = strict_gold_trace_reward(traj, task, gold_obs)
        official = compute_nestful_official_metrics(
            traj.predicted_calls, task["gold_calls"], traj, task)
        paper = compute_paper_metrics(traj, task, rr, official)
        n_reward_1 += int(rr.reward == 1.0)
        print(f"  {task['task_id']}: reward={rr.reward} "
              f"full_seq={official['full_sequence_accuracy']} "
              f"win_rate={official['win_rate']} "
              f"solution_equiv={paper['solution_equivalent_pass']} "
              f"stop={traj.stop_reason} mode={traj.executor_mode}")

    print(f"\ngold-replay sanity: {n_reward_1}/{len(tasks)} got strict reward 1")
    assert n_reward_1 == len(tasks), "gold trace replay should yield reward 1"
    print("OK")


if __name__ == "__main__":
    main()
