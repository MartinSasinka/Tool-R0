import json
from collections import Counter

path = r"c:\Users\Šunka\Downloads\train_log.jsonl"
groups = []
other = []
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "unique_episode_rewards" in rec:
            groups.append(rec)
        else:
            other.append(rec)

print("=== OVERVIEW ===")
print("total lines:", len(groups) + len(other))
print("group records:", len(groups))
print("other records:", len(other))

if groups:
    g0 = groups[0]
    print("reward_policy_resolved:", g0.get("reward_policy_resolved"))
    print("epoch range:", min(g.get("epoch", 0) for g in groups), "-", max(g.get("epoch", 0) for g in groups))
    gold = Counter(g.get("gold_num_calls") for g in groups)
    print("gold_num_calls dist:", dict(gold))

dead = sum(1 for g in groups if g.get("dead_group_corrected") or g.get("dead_group"))
dead_old = sum(1 for g in groups if g.get("dead_group_old_flattened"))
artifact = sum(1 for g in groups if g.get("position_artifact_detected"))
all_one = sum(1 for g in groups if g.get("group_all_one"))
all_zero = sum(1 for g in groups if g.get("group_all_zero"))
mixed = sum(1 for g in groups if g.get("group_mixed"))
alive = len(groups) - dead
print()
print("=== GROUP SIGNAL ===")
print(f"dead_corrected: {dead}/{len(groups)} = {dead/len(groups):.3f}")
print(f"alive: {alive}/{len(groups)} = {alive/len(groups):.3f}")
print(f"dead_old_flattened: {dead_old}/{len(groups)} = {dead_old/len(groups):.3f}")
print(f"position_artifact: {artifact}/{len(groups)} = {artifact/len(groups):.3f}")
print(f"all_one={all_one} all_zero={all_zero} mixed={mixed}")

rv = Counter()
for g in groups:
    for v in g.get("unique_episode_rewards", []):
        rv[round(v, 4)] += 1
print()
print("=== REWARD VALUE FREQ ===")
for k, v in sorted(rv.items()):
    print(f"  {k}: {v}")

keys = [
    "no_tool_call_count", "wrong_tool_count", "wrong_arg_count", "parse_error_count",
    "too_few_calls_count", "premature_final_count", "invalid_ref_count",
]
print()
print("=== FAILURE COUNTS (sum over groups) ===")
for k in keys:
    print(f"  {k}: {sum(g.get(k, 0) for g in groups)}")

# predicted calls vs gold
pred1 = sum(1 for g in groups if all(p == 1 for p in g.get("predicted_num_calls", [])))
pred2 = sum(1 for g in groups if all(p == 2 for p in g.get("predicted_num_calls", [])))
print()
print("=== PREDICTED CALL PATTERN ===")
print(f"all 8 rollouts = 1 call: {pred1} groups")
print(f"all 8 rollouts = 2 calls: {pred2} groups")

steps = [r for r in other if r.get("update") == "optimizer_step"]
print()
print("=== TRAINING PROGRESS ===")
print("optimizer_steps:", len(steps))
if steps:
    print("last global_step:", steps[-1].get("global_step"))
    print("grad_norms:", [round(s.get("grad_norm", 0), 2) for s in steps[:10]], "...")

contrib_alive = [g.get("contributing_turns", 0) for g in groups if not (g.get("dead_group_corrected") or g.get("dead_group"))]
if contrib_alive:
    print(f"contributing_turns alive: mean={sum(contrib_alive)/len(contrib_alive):.1f} total={sum(contrib_alive)}")

within = Counter(len(g.get("unique_episode_rewards", [])) for g in groups)
print()
print("=== WITHIN-GROUP UNIQUE REWARDS ===")
for k, v in sorted(within.items()):
    print(f"  {k} values: {v} groups")

print()
print("=== FIRST 50 vs ALL ===")
for label, chunk in [("first50", groups[:50]), ("all", groups)]:
    d = sum(1 for g in chunk if g.get("dead_group_corrected") or g.get("dead_group"))
    print(f"{label}: dead={d/len(chunk):.3f} n={len(chunk)}")

print()
print("=== 100-GROUP BUCKETS ===")
for i in range(0, len(groups), 100):
    chunk = groups[i : i + 100]
    mr = sum(g.get("mean_reward", 0) for g in chunk) / len(chunk)
    dr = sum(1 for g in chunk if g.get("dead_group_corrected") or g.get("dead_group")) / len(chunk)
    ar = sum(1 for g in chunk if g.get("position_artifact_detected")) / len(chunk)
    print(f"  {i+1}-{i+len(chunk)}: mean_r={mr:.3f} dead={dr:.3f} artifact={ar:.3f}")

# stage1 replay in mix?
s1 = sum(1 for g in groups if "stage1_1call" in (g.get("task_id") or ""))
s2 = sum(1 for g in groups if "stage2_2call" in (g.get("task_id") or ""))
print()
print("=== TASK MIX ===")
print(f"stage1 replay tasks: {s1}")
print(f"stage2 tasks: {s2}")

# too_few dominant pattern
tf_groups = [g for g in groups if g.get("too_few_calls_count", 0) >= 6]
print(f"groups with too_few>=6/8: {len(tf_groups)}/{len(groups)} = {len(tf_groups)/len(groups):.3f}")
