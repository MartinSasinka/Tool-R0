# QUERY_REALISM_AUDIT_DIAGNOSTIC_EXPLORATORY

Lexicon version: `ttdf.operation_lexicon.v1` — schema `ttdf.query_realism.v1`

EXPLORATORY ONLY. Never a generation target.
Aggregates and hashed ids only; no benchmark text.
Source: `nestful_diagnostic_500.jsonl` (500 rows).

| dataset | n | mean lexical cov | mean seq leak | plan-leak rate | procedural cues |
|---|---:|---:|---:|---:|---:|
| diagnostic_500 | 500 | 0.152 | 0.151 | 0.016 | 0.51 |

## Query-mode distribution

| dataset | PROCEDURAL_EXPLICIT | PROCEDURAL_PARTIAL | SEMI_IMPLICIT | GOAL_BASED_IMPLICIT | UNCLASSIFIED |
|---|---:|---:|---:|---:|---:|
| diagnostic_500 | 0.010 | 0.124 | 0.154 | 0.698 | 0.014 |

## Reading

- `lexical_operation_coverage` = share of gold operations the question
  names exactly or by an unambiguous paraphrase.
- `sequence_leakage` combines ordered coverage, LCS ratio and Kendall
  agreement between cue order and gold call order.
- `plan_leak_rate` = share of tasks with coverage >= 0.8 AND leakage >= 0.5.

