# Weak-model audit — discovery

**Generated:** 2026-07-23T07:53:26.885443+00:00
**Run ID:** pure_stage3_2ep_20260719_221918

## Provenance (verified against artifacts)

- C0 win rate: 0.5442504515352198
- E1 win rate: 0.5364238410596026
- E2 win rate: 0.5328115593016255
- E2 vs C0 paired: {'gained': 74, 'lost': 93, 'mcnemar': {'n_discordant': 167, 'b01': 74, 'b10': 93, 'chi2': 1.9401197604790419, 'p_value': 0.16365553128872506}, 'bootstrap': {'mean': -0.011438892233594221, 'ci95': [-0.026490066225165563, 0.003612281757977122], 'iters': 10000, 'seed': 20260715}}
- parity_ok: True
- adapter E1: `7419b731256376dc4d9f6e4f7b15f0e958ae78abd3d8644a4d66f80ada250510`
- adapter E2: `92aca741689a4fa629fbd250e893becdc13faad2c7ba62a32b5609b24ad57691`

## Input files (SHA-256)

| Artifact | exists | sha256 | n |
|----------|--------|--------|--:|
| nestful_test | True | `917ce6ec8686c97f…` | 1661 |
| eval_C0 | True | `ca7f49a535d24cc7…` | 1661 |
| eval_E1 | True | `c4a211fd10a0bedc…` | 1661 |
| eval_E2 | True | `6a3e525fefc3997a…` | 1661 |
| analysis_c0_e1_e2 | True | `75db0aa12682481d…` |  |
| discordant_audit | True | `7aa03c2682dc60bd…` |  |
| task_level_analysis | True | `fb12c958ffbed3ff…` |  |
| run_manifest | True | `e95a9321baa4aa27…` |  |

## Field mapping

- **question:** nestful_test.jsonl: question | input | prompt
- **offered_tools:** nestful_test.jsonl: tools (JSON list)
- **gold_calls:** nestful_test.jsonl: gold_calls | output | gold_output
- **expected_outcome:** nestful_test.jsonl: gold_answer | answer | final_answer
- **predicted_calls:** eval _traj.turns[].parsed_call
- **observations:** eval _traj.turns[].observation (when fail_reason is null)
- **final_answer:** eval _traj.pred_answer
- **official_win:** eval row via _traj.official_win (also scorer on test)
- **failure_taxonomy:** derived: scripts.analysis.two_phase_root_cause_analysis.classify_failure
- **reward_R0:** computed offline: lib.reward_v3_2_dense on saved trajectory
- **reward_train_strict:** eval _traj.reward_train_strict (strict gold trace, NOT training R0)
- **first_divergence:** computed: compare C0 vs E2 predicted_calls
- **tool_call_count:** eval _traj.num_tool_calls

## Limitations

- Training reward R0 is recomputed offline; eval reward_train_strict is a different policy.
- Pass-B anonymization hides checkpoint identity from the annotator model.
- Token estimates use chars/4 heuristic unless tiktoken is installed.
- Weak-model annotations are not ground truth; agreement measures annotator stability only.

## Eval trajectory schema (sample C0)

```json
{
  "row_keys": [
    "_traj",
    "alternative_valid_solution_pass",
    "correct_answer_but_unsupported_trace",
    "final_answer_pass",
    "internal_f1_func",
    "internal_f1_param",
    "internal_full_sequence_accuracy",
    "internal_partial_sequence_accuracy",
    "internal_win_rate",
    "num_eval_rollouts",
    "num_gold_calls",
    "sample_id",
    "solution_equivalent_pass",
    "strict_fail_but_solution_equivalent_pass",
    "strict_gold_trace_pass"
  ],
  "traj_keys": [
    "clipped_any",
    "executable",
    "execution_error",
    "executor_mode",
    "gold_num_turns",
    "internal",
    "internal_diagnostic_only",
    "mismatch",
    "mismatch_reason",
    "num_tool_calls",
    "official_full_match",
    "official_partial_match",
    "official_win",
    "paper",
    "parse_valid",
    "pred_answer",
    "prompt_overflow",
    "reward_train_strict",
    "stage",
    "stop_reason",
    "task_id",
    "turns"
  ],
  "turn_keys": [
    "clipped_completion",
    "completion_tokens",
    "fail_reason",
    "is_terminal",
    "model_text",
    "observation",
    "parsed_call",
    "prompt_tokens",
    "teacher_forced",
    "turn_idx"
  ],
  "official_win_row": null,
  "official_win_traj": 0.0,
  "pred_answer": 9.0,
  "reward_train_strict": 0.0
}
```