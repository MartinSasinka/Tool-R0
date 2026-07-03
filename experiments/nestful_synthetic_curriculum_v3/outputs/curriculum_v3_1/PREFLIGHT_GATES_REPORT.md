# Preflight Gates Report (v3.1)

Status: **PASS_PILOT_READY**

- gold_replay_success_rate: 1.0
- process_filter_pass_rate: 1.0
- exact_num_calls_integrity: True
- tool_family_realism: pilot_ready
- question_trace_alignment_failures: 0
- unresolved_placeholders: 0
- constant_reference_mismatch: 0
- ambiguous_question_count: 0
- incomplete_question_count: 0
- used_tool_diversity: 25
- stage4_unique_questions: 800
- final_dataset_audit: WARN
- pilot_decision: READY_FOR_POD_DRY_RUN

## Hard failures
- (none)

## Soft warnings
- stage1_1call_atomic question_template_duplicate_ratio > 0.30
- stage1_1call_atomic max_tool_sequence_share=0.165 > 0.15
- stage1_1call_atomic used_tool_count=15 < 20
- stage2_2call_dependency question_template_duplicate_ratio > 0.30
- stage2_2call_dependency max_tool_sequence_share=0.1737 > 0.15
- stage2_2call_dependency used_tool_count=14 < 20
- stage3_3call_composition question_template_duplicate_ratio > 0.30
- stage3_3call_composition max_tool_sequence_share=0.1537 > 0.15
- stage3_3call_composition used_tool_count=18 < 20
- stage4_4to6call_persistence question_template_duplicate_ratio > 0.30
- stage4_4to6call_persistence max_tool_sequence_share=0.1562 > 0.15
- stage3 non_scalar_output_share=0.1575 < 0.25
- final_dataset_audit status=WARN
