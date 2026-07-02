# Generation Scale-Up Report

Date: 2026-07-02

## Metrics: 50-task prototype vs 500-task version

| metric | 50-task prototype | 500-task version | delta |
|---|---:|---:|---:|
| total synthetic tasks | 50 | 500 | +450 |
| validation failures | 0 | 0 | 0 |
| duplicate task ids | 0 | 0 | 0 |
| invalid references | 0 | 0 | 0 |
| motif coverage | 40.0 % | **100.0 %** | +60.0 pp |
| baseline-failure motif coverage | 75.0 % | **100.0 %** | +25.0 pp |
| number of stage1 tasks | 10 | 248 | +238 |
| number of stage2 tasks | 10 | 10 | 0 |
| number of stage3 tasks | 10 | 87 | +77 |
| number of stage4 tasks | 20 | 155 | +135 |
| motif KL (nestful‖v3) | 1.5117 | 0.1385 | −1.3732 |

## Did coverage increase?

**Yes.** Motif coverage rose from 40% to **100%** (5/5 NESTFUL motif types at ≥50% of NESTFUL share).

## Is coverage near 80%?

**Yes — exceeds threshold.** Coverage is 100%; baseline-failure motif coverage is also 100%.

## Root cause of low 50-task coverage

**Not sample size alone — allocation strategy.**

The 50-task run used equal 5-per-family split (~10% each NESTFUL-mapped type). NESTFUL requires ~51% linear_dependency and ~32% long_chain; 10% each failed the 50%-of-NESTFUL-share gate for 3/5 types. Missing `independent_calls` generator also blocked the 4th type.

The 500-task run uses **weighted NESTFUL sampling** (55%), baseline-failure recipes (25%), easy preservation (20%), plus 50 v3-only curriculum tasks.

## Motif types still missing after scale-up

**None** among NESTFUL classifier types. All five present:

| motif_type | v3 count | v3 share |
|---|---:|---:|
| linear_dependency | 199 | 39.8% |
| long_chain | 107 | 21.4% |
| fan_in | 66 | 13.2% |
| independent_calls | 77 | 15.4% |
| fan_out | 1 | 0.2% |

v3-only training motifs (reference_reuse, distractor_tools, etc.) are present for curriculum breadth but are not NESTFUL taxonomy labels.

## Ready for stage1–2 prototype pilot?

**Yes, with prototype caveat.**

- Validation: PASS (0 failures)
- Gold replay: PASS (100%)
- Motif coverage: PASS (100%)
- Tool-family realism: **prototype_only** (math registry only)
- Preflight: **PASS_PROTOTYPE_ONLY**

Stage1 has 248 tasks; stage2 has 10 reference_reuse tasks (v3-only min allocation). Stage1 alone is sufficient for initial linear/independent pilot; stage2 is thin but non-empty.

Training requires `ALLOW_PROTOTYPE_TRAINING=1` on pod.
