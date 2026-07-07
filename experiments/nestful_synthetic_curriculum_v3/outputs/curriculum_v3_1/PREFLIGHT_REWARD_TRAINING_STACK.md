# Preflight: reward + training stack (v3.1, post-audit)

- generated: 2026-07-07 11:30:04
- status: **PASS**
- reward policy: `execution_aware_v3_1_stepwise` -> `lib.reward_v3_1.episode_turn_reward_seq`

| check | result |
|---|---|
| dataset_hard_gates_pass | True |
| reward_dispatch_ok | True |
| fractional_rewards_present | True |
| dead_group_proxy_rate_stage1 | 0.0 |
| reward_component_logging_ok | True |
| replay_ratio_ok | True |
| train_stage_metadata_visible | True |
| regression_guard_enabled | True |
| stage_advancement_gates_enabled | True |
| strict_fallback_disallowed | True |
| checkpoint_guard_enabled | True |

## Dataset gates

| gate | pass | detail |
|---|---|---|
| stage1_count_ge_800 | True | 800 samples |
| stage1_call_counts_1_1 | True | 0 out-of-range rows |
| stage2_count_ge_800 | True | 800 samples |
| stage2_call_counts_2_2 | True | 0 out-of-range rows |
| stage3_count_ge_800 | True | 800 samples |
| stage3_call_counts_3_3 | True | 0 out-of-range rows |
| stage4_count_ge_800 | True | 800 samples |
| stage4_call_counts_4_6 | True | 0 out-of-range rows |
| gold_answer_null_eq_0 | True | 0 nulls |
| duplicate_sample_ids_eq_0 | True | 0 duplicates |
| exact_duplicates_eq_0 | True | 0 duplicates |
| audit_invalid_reference_count | True | invalid_reference_count=0 |
| audit_question_trace_alignment_failures | True | question_trace_alignment_failures=0 |
| audit_gold_replay_success_rate | True | gold_replay_success_rate=1.0 |
| audit_metadata_leakage_count | True | metadata_leakage_count=0 |
| audit_unresolved_placeholder_count | True | unresolved_placeholder_count=0 |
