# Baseline Failure Mining

Dev split: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_mtgrpo_minimal\data\splits\nestful_dev.jsonl` (200 tasks)
Trajectories: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_mtgrpo_partial\outputs\final_eval_v2\baseline\final_eval_trajectories.jsonl` (found)

Failure clusters found: 4

- **linear_dependency__too_few_calls**: n=43, recipe=Generate linear_dependency tasks with too_few_calls failure mode; target ~2 calls; include distractors and reference chains.
- **long_chain__too_few_calls**: n=29, recipe=Generate long_chain tasks with too_few_calls failure mode; target ~7 calls; include distractors and reference chains.
- **fan_in__too_few_calls**: n=12, recipe=Generate fan_in tasks with too_few_calls failure mode; target ~4 calls; include distractors and reference chains.
- **independent_calls__too_few_calls**: n=3, recipe=Generate independent_calls tasks with too_few_calls failure mode; target ~3 calls; include distractors and reference chains.

## Leakage policy
Only abstract recipes exported. Original dev tasks are NOT written to synthetic training data.
