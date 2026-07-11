# Dataset quality report

Dataset: `experiments\nestful_synthetic_curriculum_v3\data\curriculum_v4_nestful_like_agentic_openrouter` | rows: 280 | status: **partial** | generated 2026-07-10T21:18:12.865329+00:00

## Completeness

| stage | rows | target | status |
|---|---|---|---|
| stage2_2call_agentic_openrouter | 280 | 800 | partial |

## Verdict

- dataset status: **partial** (partial = below target; still valid and scoreable)
- technically_acceptable: **True**
- training_candidate: **False** (needs: complete targets, distribution closer than v3.1, positive solver gap, strong_exact_win_rate >= 0.95, weak-score dominance <= 0.40, failure-type diversity, probe signal available and better)
- actually_useful: **undetermined** — only training + same-batch official NESTFUL eval can decide this. Do not claim the dataset is good before that.

## Validity

- n_rows: 280
- gold_replay_pass_rate: 1.0
- schema_pass_rate: 1.0
- null_answer_rate: 0.0
- unresolved_var_rate: 0.0
- duplicate_question_rate: 0.0
- duplicate_trace_rate: 0.0
- duplicate_sample_id_rate: 0.0
- hard_gates_pass: True

## Contamination

- question_hash_overlap: 0
- trace_hash_overlap: 0
- sample_id_overlap: 0
- hard_gates_pass: True

## Distribution similarity

| dimension | candidate | v3_1 |
|---|---|---|
| call_count_dist | 0.6728 | 0.3045 |
| offered_tools_dist | 0.6577 | 0.8918 |
| tool_arity_dist | 0.2266 | 0.1137 |
| arg_type_dist | 0.1404 | 0.1455 |
| answer_type_dist | 0.1385 | 0.2313 |

Mean distance to NESTFUL: candidate=0.3672, v3_1=0.3374
Candidate closer than v3.1 on 3/5 dimensions.

## Solver gap

- available: True
- n: 280
- weak_fail_strong_pass_rate: 1.0
- avg_weak_score: 0.4684
- avg_strong_score: 1.0
- avg_gap: 0.5316
- positive: True
- weak_score_histogram: {'0.00': 2, '0.10': 9, '0.15': 1, '0.20': 1, '0.30': 5, '0.40': 26, '0.50': 236}
- weak_score_entropy: 0.8984
- weak_score_bucket_dominance: 0.8429
- failure_type_histogram: {'partial_prefix': 232, 'wrong_answer': 26, 'execution_error': 9, 'under_call': 9, 'parse_error': 2, 'invalid_reference': 1, 'wrong_args': 1}
- failure_type_entropy: 0.971
- failure_type_dominance: 0.8286
- accepted_failure_type_diversity: {'stage2_2call_agentic_openrouter': 7}
- strong_exact_win_rate: 1.0
- diversity_pass: False
- diversity_gates: {'max_weak_score_bucket_dominance': 0.4, 'max_failure_type_dominance': 0.4, 'min_failure_types_per_stage': 4, 'min_strong_exact_win_rate': 0.95}

## GRPO signal (stage probe)

- available: False
- note: no stage-probe report found — run scripts/probe/probe_stage.sh on the pod (this scorer never launches it)
