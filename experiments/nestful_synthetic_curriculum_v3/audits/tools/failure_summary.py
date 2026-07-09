import json, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(HERE, "RUN_AUDIT.json"), encoding="utf-8"))
for run in d["train_runs"]:
    for st in run["stages"]:
        for ep in st["epochs"]:
            tl = ep["train_log_analysis"]
            if tl.get("groups", 0) == 0:
                print(f"{run['run_id']} {st['stage']} {ep['epoch']}: no group rows")
                continue
            print(f"{run['run_id']} {st['stage']} {ep['epoch']}: groups={tl['groups']} "
                  f"dead={tl['dead_group_rate']} deadOld={tl['dead_group_rate_old_flattened']} "
                  f"mixed={tl['mixed_group_rate']} allZero={tl['all_zero_group_rate']} allOne={tl['all_one_group_rate']} "
                  f"posArt={tl['position_artifact_rate']} uniqR={tl['avg_unique_episode_rewards_per_group']} "
                  f"uniqC={tl['avg_unique_completions_per_group']} entropy={tl['reward_value_entropy_bits']} "
                  f"meanR={tl['mean_reward_overall']}")
            print(f"    tooFew={tl['too_few_calls_rate']} noTool={tl['no_tool_call_rate']} parseErr={tl['parse_error_rate']} "
                  f"wrongTool={tl['wrong_tool_rate']} wrongArg={tl['wrong_arg_rate']} invRef={tl['invalid_ref_rate']} "
                  f"premature={tl['premature_final_rate']} avgCalls={tl['avg_predicted_calls']} optSteps={tl['optimizer_steps']}")
            ms = tl.get("motif_stats", {})
            worst = sorted(ms.items(), key=lambda kv: -kv[1]["dead_rate"])[:5]
            print(f"    worst motifs by dead rate: {[(k, v['dead_rate'], v['mean_reward']) for k, v in worst]}")
            print(f"    reward hist top: {tl['reward_value_hist_top']}")
