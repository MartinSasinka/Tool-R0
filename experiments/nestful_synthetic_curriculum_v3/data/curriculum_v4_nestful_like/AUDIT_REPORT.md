# curriculum_v4_nestful_like — audit & distribution report

Generated 2026-07-09T17:19:42.890771+00:00 | generator v4.0 | seed 42 | 800 examples/stage

## Contamination gate

- NESTFUL questions / gold traces / tool schemas copied: **NONE** (tool library written from scratch; only aggregate statistics used — see `manifest.json:extra.provenance`).
- Overlap with NESTFUL dev/test/full (question hash, trace hash, sample_id): **0** across 3200 rows.
- Gold replay pass rate: **1.0** (independent re-execution).

## Validation gates (all passed)

- expected call count per stage; no null answers; no unresolved `$var$` in answers; no metadata leakage in questions; no duplicate sample_id/question/trace.

## Distribution distance to NESTFUL (total variation, lower = closer)

| dimension | v4 -> NESTFUL | v3.1 -> NESTFUL | v4 closer? |
|---|---|---|---|
| call_count_dist | 0.232 | 0.3045 | YES |
| offered_tools_dist | 0.6387 | 0.8918 | YES |
| tool_arity_dist | 0.2244 | 0.1137 | no |
| arg_type_dist | 0.0331 | 0.1455 | YES |
| answer_type_dist | 0.1236 | 0.2313 | YES |

**v4 closer to NESTFUL on 4/5 dimensions** (mean distance 0.2504 vs v3.1's 0.3374). Technically acceptable: **True**.

## Interpretation limits

Passing these gates does NOT make v4 'good'. It is a better candidate than v3.1 only if, additionally: the stage probe shows better GRPO signal on v4, AND a same-batch official NESTFUL eval improves after training on it. Neither has been run yet.

## Corpus summary

- v4 rows: 3200 (v4_stage1_2call: 800, v4_stage2_3call: 800, v4_stage3_4call: 800, v4_stage4_5to6call: 800)
- mean question length (words): v4=60.2 v3.1=32.1 nestful=33.3
