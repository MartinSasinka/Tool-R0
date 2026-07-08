# Stage2 Continuation SFT — Dataset Summary

**Stage2 continuation SFT serialization (derived view, NOT a new dataset)**

- source (existing GRPO Stage2 curriculum file, NOT regenerated): `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\outputs\curriculum_v3_1\filtered\stage2_2call_dependency.jsonl`
- source sha256: `0e7c01aaa78fb8bd7ff8d4562076d5fa2e7fbb6efd011c1683dcb9c2814cb611`
- manifest expected row count: 800  (source row count: 800)
- target_type: `continuation`
- seed: 42, train_frac: 0.8
- valid records: 800  (train: 640, val: 160)
- all records have exactly 2 gold calls: True (hard-gated)

## Skipped examples

None.

## Length statistics

| split | avg input chars | avg target chars | avg input tokens | avg target tokens |
|---|---:|---:|---:|---:|
| train | 8599.4 | 165.3 | 2222.3 | 52.6 |
| val | 8572.9 | 164.3 | 2215.7 | 52.4 |

_Token lengths computed with tokenizer: `Qwen/Qwen3-4B-Instruct-2507`_

## Tool / motif distribution (train + val combined)

### Tool names used (gold calls)

| value | count |
|---|---:|
| add | 356 |
| multiply | 139 |
| filter_greater_than | 128 |
| greater_than | 128 |
| less_than | 128 |
| sort_list | 128 |
| concat | 116 |
| get_field | 72 |
| make_object | 69 |
| uppercase | 68 |
| lowercase | 48 |
| divide_safe | 47 |
| subtract | 36 |
| lookup_by_key | 3 |

### Tool names offered (prompt tool menu, incl. distractors)

| value | count |
|---|---:|
| add | 356 |
| divide_safe | 356 |
| max_list | 356 |
| mean_list | 356 |
| min_list | 356 |
| multiply | 356 |
| subtract | 356 |
| sum_list | 356 |
| and_bool | 128 |
| contains | 128 |
| count_items | 128 |
| equals | 128 |
| filter_equals | 128 |
| filter_greater_than | 128 |
| get_item | 128 |
| greater_than | 128 |
| join_list | 128 |
| less_than | 128 |
| list_length | 128 |
| or_bool | 128 |
| sort_list | 128 |
| concat | 116 |
| contains_substring | 116 |
| extract_prefix | 116 |
| extract_suffix | 116 |
| lowercase | 116 |
| string_length | 116 |
| uppercase | 116 |
| get_field | 72 |
| make_object | 72 |
| merge_objects | 72 |
| nested_get | 72 |
| update_field | 72 |
| aggregate_field | 3 |
| count_records | 3 |
| lookup_by_key | 3 |
| select_record | 3 |

### target_full_motif

| value | count |
|---|---:|
| object_or_list_output | 197 |
| independent_calls | 175 |
| reference_reuse | 167 |
| long_chain | 118 |
| fan_in | 81 |
| linear_dependency | 56 |
| distractor_tools | 6 |

### source_failure_cluster

| value | count |
|---|---:|
| synthetic_gap_boolean_output__wrong_condition | 128 |
| synthetic_gap_list_output__wrong_field | 128 |
| long_chain__too_few_calls | 118 |
| synthetic_gap_string_output__wrong_answer | 116 |
| synthetic_gap_fan_in__wrong_argument | 81 |
| synthetic_gap_object_list__wrong_extraction | 69 |
| linear_dependency__too_few_calls | 56 |
| synthetic_gap_reference_reuse__invalid_reference | 48 |
| synthetic_gap_independent_calls__premature_final | 33 |
| independent_calls__premature_final | 14 |
| synthetic_gap_distractor_tools__wrong_tool | 6 |
| lookup_query__wrong_field | 3 |

### answer_type

| value | count |
|---|---:|
| scalar | 359 |
| string | 185 |
| boolean | 128 |
| list | 128 |

## Validation requirements checked

- [PASS] every_example_has_exactly_2_gold_calls
- [PASS] every_example_has_first_call_and_first_observation
- [PASS] every_example_has_second_gold_call
- [PASS] no_null_gold_answer
- [PASS] no_duplicate_sample_ids
- [PASS] no_train_val_overlap
- [PASS] source_row_count_matches_manifest

## Design notes

- Target text omits <think>...</think> reasoning, matching the existing GRPO teacher-forced-prefix convention (rollout._format_forced_call_text) — forced/gold turns are injected as bare <tool_call_answer> tags, not model-style reasoning traces.
- gold_calls[1]['label'] uses the curriculum's own '$var_1'/'$var_2' underscore convention, which differs from the SYSTEM_PROMPT's own worked example ('$var1'/'$var2', no underscore). This is a pre-existing property of the curriculum data, not introduced by this script — flagged here since it may teach an inconsistent label format; do not silently 'fix' it without re-running the GRPO gold-replay checks, since replay validated the underscore form.
- messages[] is the FULL 7-turn conversation (including the real gold observation for call 2) so a trainer can correctly mask loss to ONLY the two generation targets (indices in loss_target_message_indices) while still giving the model the real call-2 observation as context for the turn-3 stop decision.

