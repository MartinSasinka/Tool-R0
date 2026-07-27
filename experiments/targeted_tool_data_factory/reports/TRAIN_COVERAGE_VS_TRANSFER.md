# Train-300 coverage vs NESTFUL-500 transfer

## Setup

- Train: `first 300 rows of experiments\targeted_tool_data_factory\outputs\selected\export_pilot3\train_grpo_pilot3.jsonl (matches run_train_nestful500_4gpu.sh)`
- Eval: paired C0 vs D1 on diagnostic-500
- Claim level: **associative**, not causal

## Call-count coverage

| calls | train300 | share | eval C0 | eval D1 | Δ pp |
|---|---:|---:|---:|---:|---:|
| 2 | 89 | 29.7% | 23.0% | 47.0% | +24.0 |
| 3 | 58 | 19.3% | 59.0% | 62.0% | +3.0 |
| 4 | 43 | 14.3% | 57.0% | 65.0% | +8.0 |
| 5 | 34 | 11.3% | 55.0% | 60.0% | +5.0 |
| 6 | 44 | 14.7% | 48.3% | 58.6% | +10.3 |
| 7 | 18 | 6.0% | 50.0% | 50.0% | +0.0 |
| 8 | 14 | 4.7% | 42.1% | 36.8% | -5.3 |

## Motifs / answers / tracks / failure-target cells

- Motifs: `{'branch_aggregate': 22, 'fan_in': 131, 'linear': 147}`
- Answers: `{'float': 203, 'string': 20, 'bool': 16, 'int': 23, 'numeric_string': 9, 'list': 29}`
- Tracks: `{'G': 139, 'A': 161}`
- Failure targets: `{'distractor_confusion': 77, 'premature_stop': 70, 'too_few_calls': 76, 'wrong_second_tool_after_correct_prefix': 68, 'numeric_string_confusion': 9}`
- Skills: `{'tool_catalog_search': 77, 'long_horizon_planning': 70, 'variable_planning': 76, 'continuation_after_observation': 68, 'argument_typing': 9}`
- Distinct cells: 38
- Top cells: `[('A_6pcall_linear_long_horizon_00', 20), ('A_4call_fan_in_variable_pla_00', 16), ('G_3call_fan_in_variable_pla_00', 13), ('G_4call_fan_in_variable_pla_00', 12), ('A_2call_linear_continuation_00', 12), ('G_6pcall_fan_in_long_horizon_00', 11), ('G_2call_linear_continuation_01', 11), ('A_3call_fan_in_tool_catalog_01', 11), ('A_2call_linear_continuation_03', 11), ('G_5call_fan_in_long_horizon_00', 10)]`
- Gold tool single-word share: 0.11872909698996656
- Mean ref density: 0.4130678444649032

## Reading guide

If a bucket is rare in train-300 and still improves, that is evidence of **broader generalization** (or backend confound). If only high-coverage buckets improve, transfer may be **coverage-aligned** rather than deep.
