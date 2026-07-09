import json, os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
p = os.path.join(REPO, "experiments", "nestful_synthetic_curriculum_v3", "outputs", "runs",
                 "20260708_212347_v3_1", "stage_3", "epoch_2", "train_log.jsonl")
rows = []
with open(p, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
grp = [r for r in rows if "episode_rewards" in r]
print("total lines:", len(rows), "group rows:", len(grp))
dead = sum(1 for r in grp if r.get("dead_group"))
mixed = sum(1 for r in grp if r.get("group_mixed"))
both = sum(1 for r in grp if r.get("dead_group") and r.get("group_mixed"))
print("dead:", dead, "mixed:", mixed, "dead&mixed:", both)
# a dead+mixed example
for r in grp:
    if r.get("dead_group") and r.get("group_mixed"):
        print("example dead&mixed: episode_rewards=", r["episode_rewards"],
              "n_unique=", r.get("n_unique_episode_rewards"),
              "between_std=", r.get("reward_std_between_completion"),
              "n_unique_completions=", r.get("n_unique_completion_hashes"))
        break
