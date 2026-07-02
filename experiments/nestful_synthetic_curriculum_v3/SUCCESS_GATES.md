# Success Gates

## Four-layer coverage (see NESTFUL_COVERAGE_DEFINITION.md)

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
