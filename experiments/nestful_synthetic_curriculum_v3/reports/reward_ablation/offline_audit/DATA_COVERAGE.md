# Data coverage

| field | status |
|---|---|
| raw_model_outputs | MISSING |
| parsed_calls | MISSING |
| actual_observations | MISSING |
| actual_executor_outcomes | MISSING |
| terminal_class_per_rollout | MISSING |
| failure_class_per_rollout | MISSING |
| episode_rewards | AVAILABLE_EXACT |
| turn_rewards | AVAILABLE_EXACT |
| returns_G_t | RECONSTRUCTABLE |
| normalized_advantages | RECONSTRUCTABLE |
| token_masks | MISSING |
| group_task_id | AVAILABLE_EXACT |
| rollout_index | RECONSTRUCTABLE |
| optimizer_step | AGGREGATE_ONLY |
| gradient_norm | MISSING |
| kl | AVAILABLE_EXACT |
| clipping_statistics | AVAILABLE_EXACT |
| learning_rate | AVAILABLE_EXACT |
| checkpoint_adapter_state | AVAILABLE_EXACT |
| completion_hashes | AVAILABLE_EXACT |
| strict_gold_trace_pass | AGGREGATE_ONLY |
| exec_failure_counts | AVAILABLE_EXACT |
| predicted_num_calls | AVAILABLE_EXACT |

## Per arm group counts
- **A0_R0_CURRENT**: 160 groups
- **A1_OUTCOME_ONLY**: 160 groups
- **A2_R3_OUTCOME_FIRST**: 160 groups
- **A3_VERIFIABLE_PROCESS**: 160 groups
- **A4_GATED_VERIFIABLE**: 160 groups