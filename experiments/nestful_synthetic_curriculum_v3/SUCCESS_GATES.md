# Success Gates

## v3.1 pre-training gates

| gate | threshold | current |
|------|-----------|---------|
| final dataset audit | no hard FAIL | **WARN** (soft only) |
| exact_num_calls integrity | PASS | **PASS** |
| gold replay | 100% | **100%** |
| process filter | 100% | **100%** |
| stage1/2/3/4 count | ≥800 | **800 each** |
| tool realism | pilot_ready | **pilot_ready** |
| preflight | PASS_PILOT_READY | **PASS_PILOT_READY** |
| exact duplicates | 0 | **0** |
| question-trace alignment | 0 failures | **0** |
| unique question ratio | ≥0.95 soft | **1.0** |
| trace duplicate ratio | ≤0.05 soft | **0.0003** |

Reports: `FINAL_DATASET_AUDIT.md`, `FINAL_PILOT_READINESS_REPORT.md`

### Uniqueness hard gates (v3.1)

- `exact_duplicate_count = 0`
- `duplicate sample_id = 0`
- `unique_question_ratio ≥ 0.40` per stage (hard)
- stage count ≥ 800 per stage
- question-trace alignment failures = 0
- gold replay = 1.0

### Uniqueness soft warns

- trace duplicate ratio > 0.15
- question template duplicate ratio > 0.30 (same skill, different literals — allowed)
- single trajectory_id > 6 samples/stage
- single tool sequence > 15% of stage
- per-stage used tools < 20

Skill repetition (motif/tool sequence) is intentional; exact duplicates and excessive trace clones are not.

## Four-layer coverage (v3 legacy)

Structural motif coverage alone is **not sufficient** for final readiness.

## Pre-training gates

| gate | threshold | current |
|------|-----------|---------|
| validation failures | 0 | **0** |
| gold replay | 100% | **100%** |
| motif coverage | ≥80% (relax with `--prototype-only`) | **80%** |
| baseline-failure motif coverage | ≥80% | **100%** |
| tool realism | not blocking pilot if partial | **partial_tool_realism** |

Preflight: **PASS_PROTOTYPE_ONLY** → train with `ALLOW_PROTOTYPE_TRAINING=1`.

## Post-pilot gates (dev)

- dev Win ≥ baseline − 0.005
- no severe trace drift
- dead_group_rate ≤ 0.80

## Final test eval

Only if dev gates pass — see `POST_PILOT_EVAL_PLAN.md`.

**Training started: NO**
