# Stage2 Balancing Report

Date: 2026-07-02

## Stage counts

| stage | before_count | after_count | main motifs | validation_failures | gold_replay |
|-------|-------------:|------------:|-------------|--------------------:|------------:|
| stage1_linear_simple | 248 | **417** | linear_dependency, independent_calls | 0 | 100% |
| stage2_reference_reuse | **10** | **223** | reference_reuse, simple_fan_in, object_or_list_output, argument_transformation | 0 | 100% |
| stage3_structural_motifs | 87 | **119** | fan_in, fan_out, argument_transformation | 0 | 100% |
| stage4_nestful_like_mixed | 155 | **271** | long_chain, baseline_failure_inspired, distractor_tools | 0 | 100% |

**Total tasks:** 500 → **1030**

## Changes applied

1. `stage_minimums` in config (stage2 target **200** generated tasks)
2. Dedicated stage2 generators: reference_reuse, simple_fan_in, string/list/object tools, independent aggregate
3. Curriculum stage2 expanded: `simple_fan_in` (keeps 3-call fan-in out of stage3)
4. `baseline_failure_inspired` stays in stage4 (no motif hijack to stage1)
5. Nestful motif boost for structural coverage (linear, long_chain, fan_in)

## Stage2 motif breakdown (223 tasks)

| motif | count |
|-------|------:|
| object_or_list_output | 68 |
| reference_reuse | 47 |
| simple_fan_in | 46 |
| argument_transformation | 40 |
| linear_dependency | 22 |

## Conclusion

**Stage1–2 pilot ready: YES**

- Stage2 **223 tasks** (target 150–250) ✓
- Stage1 **417 tasks** ✓
- Validation 0 failures ✓
- Gold replay 100% ✓
- Motif coverage 80% ✓
- Preflight **PASS_PROTOTYPE_ONLY** ✓

Not final-experiment-ready due to `partial_tool_realism` (low bigram overlap vs NESTFUL).
