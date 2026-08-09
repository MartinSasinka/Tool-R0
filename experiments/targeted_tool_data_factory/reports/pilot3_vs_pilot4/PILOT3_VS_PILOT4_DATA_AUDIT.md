# PILOT3_VS_PILOT4_DATA_AUDIT

Offline dataset statistics only. No model was run, so nothing in this report predicts NESTFUL accuracy.

- baseline: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\reports\pilot3_provenance\_git_revisions\train_grpo_pilot3@e83f57de.jsonl` (n=600)
- candidate: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\pilot4_profile_safe\canonical.jsonl` (n=1000, train split n=600)
- profile reference: dev-200 (n=200), aggregates only

## Distribution distance to the profile

| distribution | pilot3 TV | pilot4 TV |
|---|---|---|
| call count | 0.0817 | 0.0217 |
| query mode | 0.8084 | 0.09 |

## Metrics

| metric | pilot3 | pilot4 train-600 | dev-200 target | direction | verdict |
|---|---|---|---|---|---|
| `mean_call_count` | 4.0917 | 3.8783 | 4.195 | closer_to_target_is_better | further_from_target |
| `join_rate` | 0.5133 | 0.5167 | 0.43 | closer_to_target_is_better | further_from_target |
| `multi_join_rate` | 0.0 | 0.2133 | 0.125 | higher_is_better | improved |
| `fan_out_rate` | 0.0 | 0.27 | 0.0 | coverage_required_above_profile | coverage_added |
| `reuse_rate` | 0.0 | 0.27 | 0.0 | coverage_required_above_profile | coverage_added |
| `late_reference_rate` | 0.5133 | 0.5167 | 0.435 | closer_to_target_is_better | further_from_target |
| `mean_reference_distance` | 1.2788 | 1.323 | 1.2378 | closer_to_target_is_better | further_from_target |
| `mean_depth` | 2.3067 | 2.18 |  | closer_to_target_is_better | descriptive |
| `mean_type_transitions` | 0.1433 | 0.3433 | 0.085 | higher_is_better | improved |
| `plan_leak_rate` | 0.6417 | 0.0967 | 0.035 | lower_is_better | improved |
| `goal_based_share` | 0.01 | 0.6533 | 0.7 | closer_to_target_is_better | closer_to_target |
| `mean_operation_explicitness` | 0.8378 | 0.235 | 0.1645 | lower_is_better | improved |
| `mean_sequence_leakage` | 0.7922 | 0.2141 | 0.152 | lower_is_better | improved |
| `mean_procedural_cue_count` | 6.6833 | 1.3467 | 0.645 | lower_is_better | improved |
| `n_distinct_tool_names` | 132 | 190 |  | higher_is_better | improved |
| `n_distinct_output_keys` | 1 | 8 |  | higher_is_better | improved |
| `output_key_entropy` | 0.0 | 0.7972 |  | higher_is_better | improved |
| `n_distinct_tool_combinations` | 582 | 599 |  | higher_is_better | improved |
| `tool_combination_entropy` | 0.9973 | 0.9999 |  | higher_is_better | improved |
| `n_distinct_primitives` | 0 | 80 |  | higher_is_better | improved |
| `primitive_entropy` | 0.0 | 0.9547 |  | higher_is_better | improved |
| `n_capability_families` | 0 | 25 |  | higher_is_better | improved |
| `top1_query_skeleton_share` | 0.0083 | 0.0017 |  | lower_is_better | improved |
| `top1_program_family_share` | 0.01 | 0.005 |  | lower_is_better | improved |
| `mean_offered_tool_count` | 11.3967 | 12.5133 | 10.99 | closer_to_target_is_better | further_from_target |
| `mean_schema_complexity` | 2.3602 | 2.5537 | 1.9173 | closer_to_target_is_better | further_from_target |
| `mean_parameter_count` | 7.1183 | 6.905 | 8.01 | closer_to_target_is_better | further_from_target |
| `mean_nested_schema_depth` | 4.15 | 4.3183 | 3.18 | closer_to_target_is_better | further_from_target |
| `schema_compatible_distractor_share` | 0.0 | 0.95 |  | higher_is_better | improved |
| `mean_hard_distractor_count` | 0.0 | 4.1283 |  | higher_is_better | improved |
| `bucket[2].n_distinct_topologies` | 1 | 1 | 1 | higher_is_better | structurally_capped |
| `bucket[2].top1_topology_share` | 1.0 | 1.0 | 1.0 | lower_is_better | structurally_capped |
| `bucket[2].normalized_entropy` | 0.0 | 0.0 | 0.0 | higher_is_better | structurally_capped |
| `bucket[2].join_rate` | 0.0 | 0.0 | 0.0 | closer_to_target_is_better | unchanged |
| `bucket[2].multi_join_rate` | 0.0 | 0.0 | 0.0 | higher_is_better | unchanged |
| `bucket[2].fan_out_rate` | 0.0 | 0.0 | 0.0 | coverage_required_above_profile | unchanged |
| `bucket[2].reuse_rate` | 0.0 | 0.0 | 0.0 | coverage_required_above_profile | unchanged |
| `bucket[2].late_reference_rate` | 0.0 | 0.0 | 0.0 | closer_to_target_is_better | unchanged |
| `bucket[2].mean_reference_distance` | 1.0 | 1.0 | 1.0 | closer_to_target_is_better | unchanged |
| `bucket[3].n_distinct_topologies` | 2 | 3 | 4 | higher_is_better | structurally_capped |
| `bucket[3].top1_topology_share` | 0.6262 | 0.7206 | 0.6818 | lower_is_better | structurally_capped |
| `bucket[3].normalized_entropy` | 0.9536 | 0.7141 | 0.6254 | higher_is_better | structurally_capped |
| `bucket[3].join_rate` | 0.6262 | 0.2794 | 0.2273 | closer_to_target_is_better | closer_to_target |
| `bucket[3].multi_join_rate` | 0.0 | 0.0 | 0.0 | higher_is_better | unchanged |
| `bucket[3].fan_out_rate` | 0.0 | 0.1544 | 0.0 | coverage_required_above_profile | coverage_added |
| `bucket[3].reuse_rate` | 0.0 | 0.1544 | 0.0 | coverage_required_above_profile | coverage_added |
| `bucket[3].late_reference_rate` | 0.6262 | 0.2794 | 0.25 | closer_to_target_is_better | closer_to_target |
| `bucket[3].mean_reference_distance` | 1.3131 | 1.114 | 1.1364 | closer_to_target_is_better | closer_to_target |
| `bucket[4].n_distinct_topologies` | 3 | 7 | 2 | higher_is_better | improved |
| `bucket[4].top1_topology_share` | 0.5761 | 0.2759 | 0.6296 | lower_is_better | improved |
| `bucket[4].normalized_entropy` | 0.8867 | 0.9158 | 0.951 | higher_is_better | improved |
| `bucket[4].join_rate` | 0.7717 | 1.0 | 0.6296 | closer_to_target_is_better | further_from_target |
| `bucket[4].multi_join_rate` | 0.0 | 0.1264 | 0.0 | higher_is_better | improved |
| `bucket[4].fan_out_rate` | 0.0 | 0.4713 | 0.0 | coverage_required_above_profile | coverage_added |
| `bucket[4].reuse_rate` | 0.0 | 0.4713 | 0.0 | coverage_required_above_profile | coverage_added |
| `bucket[4].late_reference_rate` | 0.7717 | 1.0 | 0.6296 | closer_to_target_is_better | further_from_target |
| `bucket[4].mean_reference_distance` | 1.4275 | 1.5659 | 1.2592 | closer_to_target_is_better | further_from_target |
| `bucket[5].n_distinct_topologies` | 4 | 18 | 4 | higher_is_better | improved |
| `bucket[5].top1_topology_share` | 0.5484 | 0.15 | 0.3158 | lower_is_better | improved |
| `bucket[5].normalized_entropy` | 0.8139 | 0.9273 | 0.9892 | higher_is_better | improved |
| `bucket[5].join_rate` | 0.7419 | 1.0 | 0.7895 | closer_to_target_is_better | further_from_target |
| `bucket[5].multi_join_rate` | 0.0 | 0.55 | 0.2632 | higher_is_better | improved |
| `bucket[5].fan_out_rate` | 0.0 | 0.6333 | 0.0 | coverage_required_above_profile | coverage_added |
| `bucket[5].reuse_rate` | 0.0 | 0.6333 | 0.0 | coverage_required_above_profile | coverage_added |
| `bucket[5].late_reference_rate` | 0.7419 | 1.0 | 0.7895 | closer_to_target_is_better | further_from_target |
| `bucket[5].mean_reference_distance` | 1.3669 | 1.7419 | 1.3816 | closer_to_target_is_better | further_from_target |
| `bucket[6+].n_distinct_topologies` | 16 | 62 | 26 | higher_is_better | improved |
| `bucket[6+].top1_topology_share` | 0.1939 | 0.0465 | 0.1364 | lower_is_better | improved |
| `bucket[6+].normalized_entropy` | 0.917 | 0.9582 | 0.9263 | higher_is_better | improved |
| `bucket[6+].join_rate` | 0.7515 | 0.969 | 1.0 | closer_to_target_is_better | closer_to_target |
| `bucket[6+].multi_join_rate` | 0.0 | 0.6512 | 0.4545 | higher_is_better | improved |
| `bucket[6+].fan_out_rate` | 0.0 | 0.4806 | 0.0 | coverage_required_above_profile | coverage_added |
| `bucket[6+].reuse_rate` | 0.0 | 0.4806 | 0.0 | coverage_required_above_profile | coverage_added |
| `bucket[6+].late_reference_rate` | 0.7515 | 0.969 | 1.0 | closer_to_target_is_better | closer_to_target |
| `bucket[6+].mean_reference_distance` | 1.4345 | 1.6553 | 1.6208 | closer_to_target_is_better | closer_to_target |

## Caveats

- structural only; matching the profile is the goal, not maximisation
- measured over the whole set, not per bucket
- pilot3 weakness; higher is only better up to the profile
- dev-200 contains no fan-out, so this is deliberate coverage beyond the measured profile and a known distribution-mismatch risk
- dev-200 contains no output reuse, so this is deliberate coverage beyond the measured profile and a known distribution-mismatch risk
- profile-relative
- long references are harder to track but also rarer in the benchmark
- descriptive
- type transitions exercise conversion capabilities
- rule-based classifier, not a human judgement of realism
- target taken from the dev-200 profile
- lexicon-based; a low value is not proof the task is implicit
- only defined when at least two operations are cued
- counts surface cues, not reasoning difficulty
- raw surface vocabulary size
- affects reference-format diversity
- normalised entropy, comparable across sizes
- absolute count scales with n
- normalised, size-robust
- capability breadth proxy
- pilot3 rows predate the taxonomy, so this is 0 for pilot3 by construction
- template concentration
- program-family concentration
- environment size, profile-relative
- pilot3 has no distractor metadata, so this is 0 for pilot3 by construction
- hardness labels differ between pilots; not directly comparable
- all 1 dead-call-free topologies that exist at 2 calls are covered, so the remaining spread is fixed by the join rate the profile asks for; pilot3 scores lower on the share only because its join rate overshot that target
- conditional on the call bucket
- all 3 dead-call-free topologies that exist at 3 calls are covered, so the remaining spread is fixed by the join rate the profile asks for; pilot3 scores lower on the share only because its join rate overshot that target
- distinct-topology counts scale with the number of rows in the bucket; compare the entropy row too

- diagnostic-500 was not used anywhere in this comparison.
- pilot3 rows predate several pilot4 fields; those metrics are marked as zero by construction rather than as a regression.
