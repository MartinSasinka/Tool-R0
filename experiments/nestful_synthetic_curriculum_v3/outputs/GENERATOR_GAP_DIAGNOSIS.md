# Generator Gap Diagnosis

Date: 2026-07-02 (after 500-task weighted regeneration)

## Summary

Initial 500-task run **without** weighted sampling would have stayed at ~40% coverage (same equal-split bug as 50-task). After generator fixes, coverage reached **100%**.

## Motif families: declared vs produced (50-task prototype)

| family | declared in config | produced (50-task) | issue |
|---|---|---|---|
| linear_dependency | yes | yes (~10%) | under quota vs NESTFUL 51% |
| reference_reuse | yes | yes | v3-only label, not NESTFUL type |
| fan_in | yes | yes | OK share |
| fan_out | yes | yes | over-represented |
| object_or_list_output | yes | yes | v3-only |
| argument_transformation | yes | yes | v3-only |
| distractor_tools | yes | yes (buggy task_id) | v3-only |
| long_chain | yes | yes (~10%) | under vs NESTFUL 32% |
| alternative_valid_traces | yes | yes | v3-only |
| baseline_failure_inspired | yes | yes | overwrote motif_type → hurt coverage |
| independent_calls | **no template** | **absent** | missing generator |

## Motif families: 500-task weighted version

| family | produced | mechanism |
|---|---|---|
| linear_dependency | 199 | weighted NESTFUL + baseline + easy |
| long_chain | 107 | weighted + baseline cluster |
| fan_in | 66 | weighted + baseline cluster |
| independent_calls | 77 | **new** `_gen_independent_calls` |
| fan_out | 1 | weighted (NESTFUL rare) |
| reference_reuse | 10 | v3_only_min_tasks |
| object_or_list_output | 10 | v3_only_min_tasks |
| argument_transformation | 10 | v3_only_min_tasks |
| distractor_tools | 10 | v3_only_min_tasks |
| alternative_valid_traces | 10 | v3_only_min_tasks |

## Why families did not appear (50-task root causes)

| cause | affected families |
|---|---|
| Equal per-family split ignored NESTFUL proportions | linear_dependency, long_chain |
| Missing template | independent_calls |
| `baseline_failure_inspired` overwrote motif_type | all cluster motifs |
| `distractor_tools` reused linear task_id | distractor_tools |
| `fan_out` gold_answer ≠ last-call replay output | fan_out (1 task gold replay fail) |
| build_curriculum mis-assign | none critical after fix |
| motif classifier reclassify | none (uses declared motif_type) |
| validation rejects | none |

## Changes applied for coverage ≥ 80%

1. Weighted sampling from `nestful_motif_distribution.json` (55% of core pool)
2. Added `_gen_independent_calls` template
3. Baseline-failure tasks keep cluster `motif_type` (not overwritten)
4. Fixed `fan_out` gold_answer = last branch output
5. Reserved `v3_only_min_tasks: 50` for curriculum breadth (stage2–4 diversity)
6. Gold replay gate catches answer/tool bugs before training

## Current status

Coverage **100%** — no further generator changes required for motif gate. Tool-family realism remains **prototype_only** until IBM-tool registry pipeline is implemented.
