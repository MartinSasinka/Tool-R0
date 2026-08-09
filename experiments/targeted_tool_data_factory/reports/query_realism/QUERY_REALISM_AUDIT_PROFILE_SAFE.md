# QUERY_REALISM_AUDIT_PROFILE_SAFE

Lexicon version: `ttdf.operation_lexicon.v1` — schema `ttdf.query_realism.v1`

Synthetic pilot3 sets keep their text; the NESTFUL dev profile stores aggregates and hashed ids only.

| dataset | n | mean lexical cov | mean seq leak | plan-leak rate | procedural cues |
|---|---:|---:|---:|---:|---:|
| d1_train_subset_300 | 300 | 0.852 | 0.813 | 0.653 | 6.52 |
| pilot3_train_600_as_trained | 600 | 0.838 | 0.792 | 0.642 | 6.68 |
| pilot3_train_600_worktree | 600 | 0.845 | 0.812 | 0.672 | 6.40 |
| pilot3_heldout_200 | 200 | 0.828 | 0.812 | 0.605 | 6.49 |
| pilot3_reserve_200 | 200 | 0.814 | 0.754 | 0.615 | 6.27 |
| nestful_dev_200 | 200 | 0.165 | 0.152 | 0.045 | 0.65 |

## Query-mode distribution

| dataset | PROCEDURAL_EXPLICIT | PROCEDURAL_PARTIAL | SEMI_IMPLICIT | GOAL_BASED_IMPLICIT | UNCLASSIFIED |
|---|---:|---:|---:|---:|---:|
| d1_train_subset_300 | 0.653 | 0.330 | 0.007 | 0.007 | 0.003 |
| pilot3_train_600_as_trained | 0.642 | 0.327 | 0.013 | 0.010 | 0.008 |
| pilot3_train_600_worktree | 0.670 | 0.302 | 0.013 | 0.010 | 0.005 |
| pilot3_heldout_200 | 0.610 | 0.355 | 0.025 | 0.005 | 0.005 |
| pilot3_reserve_200 | 0.635 | 0.320 | 0.015 | 0.020 | 0.010 |
| nestful_dev_200 | 0.035 | 0.125 | 0.115 | 0.700 | 0.025 |

## Reading

- `lexical_operation_coverage` = share of gold operations the question
  names exactly or by an unambiguous paraphrase.
- `sequence_leakage` combines ordered coverage, LCS ratio and Kendall
  agreement between cue order and gold call order.
- `plan_leak_rate` = share of tasks with coverage >= 0.8 AND leakage >= 0.5.

