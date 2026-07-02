#!/usr/bin/env python3
import json
from collections import defaultdict

profiles = {
    "baseline": "curricullum/evaluation/results/curriculum_baseline_multiturn_predictions.jsonl",
    "stage1": "curricullum/evaluation/results/curriculum_stage1_1call_multiturn_predictions.jsonl",
    "stage2": "curricullum/evaluation/results/curriculum_stage2_2call_multiturn_predictions.jsonl",
    "stage3": "curricullum/evaluation/results/curriculum_stage3_3call_multiturn_predictions.jsonl",
}


def load(path):
    by_task = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_task[r["task_id"]].append(r)
    return by_task


data = {k: load(v) for k, v in profiles.items()}

print("=== Rollout-level pass rate (8 rollouts / task) ===")
for name in profiles:
    with open(f"curricullum/evaluation/results/curriculum_{name if name!='baseline' else 'baseline'}_multiturn_summary.json".replace("stage1","stage1_1call").replace("stage2","stage2_2call").replace("stage3","stage3_3call")) as f:
        pass
# use known from summaries
summaries = {"baseline": 68.32, "stage1": 68.01, "stage2": 68.44, "stage3": 67.91}
for k, v in summaries.items():
    d = v - summaries["baseline"]
    print(f"  {k:8} {v:6.2f}%  ({d:+.2f} pp vs baseline)")

print("\n=== Task-level (1861 tasks) ===")
header = f"{'profile':8} {'pass@1':>8} {'pass@8':>8} {'mean_roll':>10} {'no_tool%':>10} {'avg_calls':>10}"
print(header)
for name, by_task in data.items():
    n = len(by_task)
    pass1 = pass8 = mean_roll = no_tool = avg_calls = 0
    for rolls in by_task.values():
        scores = [r.get("score", 0) or 0 for r in rolls]
        pass1 += 1 if scores[0] >= 1 else 0
        pass8 += 1 if max(scores) >= 1 else 0
        mean_roll += sum(scores) / len(scores)
        if all(r.get("num_tool_calls", 0) == 0 for r in rolls):
            no_tool += 1
        avg_calls += sum(r.get("num_tool_calls", 0) for r in rolls) / len(rolls)
    print(
        f"{name:8} {100*pass1/n:7.2f}% {100*pass8/n:7.2f}% "
        f"{100*mean_roll/n:9.2f}% {100*no_tool/n:9.1f}% {avg_calls/n:10.2f}"
    )

print("\n=== Task pass@8 vs baseline ===")
base8 = {
    tid: max(r.get("score", 0) or 0 for r in rolls) >= 1
    for tid, rolls in data["baseline"].items()
}
for stage in ["stage1", "stage2", "stage3"]:
    improved = regressed = 0
    for tid, rolls in data[stage].items():
        s8 = max(r.get("score", 0) or 0 for r in rolls) >= 1
        b = base8[tid]
        if s8 and not b:
            improved += 1
        elif b and not s8:
            regressed += 1
    print(f"  {stage}: +{improved} improved, -{regressed} regressed, net {improved-regressed:+d} tasks")

print("\n=== Execution breakdown (rollout %) ===")
for name in profiles:
    path = profiles[name]
    total = no_calls = exec_ok = fail_cat = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            total += 1
            if r.get("num_tool_calls", 0) == 0:
                no_calls += 1
            if r.get("status") == "completed":
                exec_ok += 1
            if r.get("stopped") == "no_more_calls" and r.get("num_tool_calls", 0) == 0:
                fail_cat += 1
    print(
        f"  {name:8} completed {100*exec_ok/total:5.1f}%  "
        f"no_tools {100*no_calls/total:5.1f}%  "
        f"mental_ans {100*fail_cat/total:5.1f}%"
    )
