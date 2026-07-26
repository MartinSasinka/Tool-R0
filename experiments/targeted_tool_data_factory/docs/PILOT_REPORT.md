# PILOT REPORT — targeted_tool_data_factory pilot2

## Verdict: **CONDITIONAL**

- profile-match: new dataset closer than Stage-3 on 7/8 metrics
- WARN: student probe NOT_RUN_LOCAL (structural P0 only)

## Counts
- generated candidates: 2495
- validated: 2175
- rejected: 320
- selected (frozen pilot): 320
- track mix: A=191, G=129 (A share 59.7 %)
- tasks with >=1 hard distractor: 80.3 % (threshold >= 50.0 %)
- deterministic replay rate: 100.0 %
- contamination hits in candidate pool (all rejected; selected pool = 0): 4
- dedup drops: 4
- split leakage collisions: 0
- split sizes: {'heldout': 80, 'reserve': 80, 'train': 160}

## Rejection taxonomy
| reason | count |
|---|---|
| V4:shorter valid path found | 182 |
| V4:single offered tool solves the whole task | 125 |
| V3:oracle answer '[2, 4, 5, 7, 8, 10]' appears in query | 4 |
| V5:gold tool-call skeleton overlap with target | 4 |
| V3:oracle answer '[13, 24, 33, 44, 53, 66]' appears in query | 1 |
| V3:oracle answer '[1421, 2518, 3574, 4670, 5685, 7107]' appears | 1 |
| V3:oracle answer '[181, 320, 454, 593, 722, 903]' appears in qu | 1 |
| V3:oracle answer '[58, 102, 145, 190, 231, 289]' appears in que | 1 |
| V3:oracle answer '[59, 105, 149, 194, 237, 296]' appears in que | 1 |
| V3:oracle answer '[8, 15, 21, 28, 34, 42]' appears in query | 1 |
| V3:oracle answer '[88, 157, 222, 291, 354, 442]' appears in que | 1 |
| V3:oracle answer '[9, 22, 52, 68, 81]' appears in query | 1 |
| V5:normalized duplicate | 1 |

## Generation cells (requested / generated / validated / rejected / selected)
| cell | req | gen | valid | rej | sel |
|---|---|---|---|---|---|
| A_2call_linear_continuation_00 | 80 | 80 | 49 | 31 | 10 |
| A_2call_linear_continuation_01 | 80 | 80 | 80 | 0 | 10 |
| A_2call_linear_continuation_03 | 80 | 80 | 77 | 3 | 10 |
| A_2call_linear_continuation_04 | 80 | 80 | 76 | 4 | 10 |
| A_2call_linear_continuation_06 | 80 | 80 | 75 | 5 | 10 |
| A_2call_linear_tool_catalog_02 | 80 | 80 | 77 | 3 | 10 |
| A_2call_linear_tool_catalog_05 | 80 | 80 | 73 | 7 | 10 |
| A_3call_fan_in_tool_catalog_01 | 84 | 84 | 57 | 27 | 11 |
| A_3call_fan_in_variable_pla_00 | 84 | 84 | 41 | 43 | 11 |
| A_3call_linear_variable_pla_00 | 94 | 94 | 50 | 44 | 12 |
| A_4call_branch_aggregate_tool_catalog_00 | 30 | 30 | 30 | 0 | 4 |
| A_4call_fan_in_tool_catalog_01 | 59 | 59 | 57 | 2 | 8 |
| A_4call_fan_in_variable_pla_00 | 59 | 59 | 58 | 1 | 8 |
| A_4call_linear_variable_pla_00 | 55 | 55 | 49 | 6 | 7 |
| A_5call_branch_aggregate_tool_catalog_00 | 21 | 21 | 21 | 0 | 3 |
| A_5call_fan_in_long_horizon_00 | 85 | 85 | 85 | 0 | 11 |
| A_5call_linear_long_horizon_00 | 36 | 36 | 35 | 1 | 5 |
| A_6pcall_branch_aggregate_tool_catalog_00 | 49 | 49 | 48 | 1 | 6 |
| A_6pcall_fan_in_long_horizon_00 | 66 | 66 | 65 | 1 | 8 |
| A_6pcall_fan_in_tool_catalog_01 | 66 | 66 | 65 | 1 | 8 |
| A_6pcall_fan_in_tool_catalog_02 | 66 | 66 | 66 | 0 | 8 |
| A_6pcall_linear_long_horizon_00 | 82 | 82 | 58 | 24 | 11 |
| G_2call_linear_continuation_00 | 75 | 75 | 74 | 1 | 10 |
| G_2call_linear_continuation_01 | 75 | 75 | 71 | 4 | 10 |
| G_2call_linear_continuation_03 | 75 | 75 | 68 | 7 | 10 |
| G_2call_linear_continuation_04 | 75 | 75 | 69 | 6 | 10 |
| G_2call_linear_tool_catalog_02 | 75 | 75 | 69 | 6 | 10 |
| G_3call_fan_in_variable_pla_00 | 112 | 112 | 55 | 57 | 14 |
| G_3call_linear_variable_pla_00 | 63 | 63 | 54 | 9 | 8 |
| G_4call_branch_aggregate_tool_catalog_00 | 20 | 20 | 19 | 1 | 3 |
| G_4call_fan_in_variable_pla_00 | 78 | 78 | 77 | 1 | 10 |
| G_4call_linear_variable_pla_00 | 36 | 36 | 29 | 7 | 5 |
| G_5call_branch_aggregate_tool_catalog_00 | 14 | 14 | 14 | 0 | 2 |
| G_5call_fan_in_long_horizon_00 | 57 | 57 | 57 | 0 | 7 |
| G_5call_linear_long_horizon_00 | 24 | 24 | 16 | 8 | 3 |
| G_6pcall_branch_aggregate_tool_catalog_00 | 33 | 33 | 32 | 1 | 4 |
| G_6pcall_fan_in_long_horizon_00 | 66 | 66 | 64 | 2 | 8 |
| G_6pcall_fan_in_tool_catalog_01 | 66 | 66 | 65 | 1 | 8 |
| G_6pcall_linear_long_horizon_00 | 55 | 55 | 50 | 5 | 7 |

## Profile match vs NESTFUL dev (lower = closer; AUC 0.5 = indistinguishable)
| dataset | JSD call | JSD motif | JSD args | JSD answer | W tools | W qlen | AUC |
|---|---|---|---|---|---|---|---|
| new_selected | 0.0030 | 0.0457 | 0.0253 | 0.0026 | 0.69 | 29.7 | 0.525 |
| stage3_old | 0.5847 | 0.1222 | 0.0091 | 0.1517 | 1.10 | 31.4 | 0.728 |

## Selected distributions
### Call counts
| value | count | share |
|---|---|---|
| 2 | 120 | 37.5 % |
| 3 | 56 | 17.5 % |
| 4 | 45 | 14.1 % |
| 5 | 31 | 9.7 % |
| 6 | 40 | 12.5 % |
| 7 | 10 | 3.1 % |
| 8 | 18 | 5.6 % |

### Motifs
| value | count | share |
|---|---|---|
| branch_aggregate | 22 | 6.9 % |
| fan_in | 120 | 37.5 % |
| linear | 178 | 55.6 % |

### Answer types
| value | count | share |
|---|---|---|
| bool | 10 | 3.1 % |
| float | 237 | 74.1 % |
| int | 20 | 6.2 % |
| list | 22 | 6.9 % |
| numeric_string | 10 | 3.1 % |
| string | 21 | 6.6 % |

### Templates (max share 4.1 %, cap 5.0 %)
top: `[('sequence_v1', 13), ('indirect_v4', 13), ('imperative_v2', 13), ('goal_first_v2', 13), ('indirect_v3', 12)]`

### Cells (max share 4.4 %, cap 10.0 %)

### Distribution warnings
- none

## Student probe
status: `{'NOT_RUN_LOCAL': 320}`

## Hashes / files
```json
{
  "analysis_csv": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot2\\analysis_pilot2.csv",
    "sha256": "72c7edd8c317d4b00cfee830e22cb6a88f9525681f4d090301e6ced5961941dc"
  },
  "canonical": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot2\\canonical_pilot2.jsonl",
    "sha256": "e066f6becf2bc3c4e30352d6ffb959cd03e7328dad28d35a1dea098eada5ef00"
  },
  "grpo_train_ready": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot2\\grpo_train_ready_pilot2.jsonl",
    "sha256": "2232f844da08ba2ad65ea79497d2d3db6eea433caeab1eff1db9dcb5a0d66ffe"
  },
  "heldout_grpo": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot2\\heldout_grpo_pilot2.jsonl",
    "sha256": "8d77581eeaa152936de5d443394893815140265e19be9792760cc61a969254b2"
  },
  "heldout_nestful": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot2\\heldout_nestful_pilot2.jsonl",
    "sha256": "4da21e55921970149ebadcc9b8534a9d165e5cc17c5ceb61042fe7d82f86573b"
  },
  "nestful_compat": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot2\\nestful_compat_pilot2.jsonl",
    "sha256": "005245a1392c80111f9741f4b071b6afd673f8a46e275c77b71c4ff4e52970b2"
  },
  "reserve_grpo": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot2\\reserve_grpo_pilot2.jsonl",
    "sha256": "ee40fac8860ad91c3c8d7f1168a45e227195787264813d923afd4b114f9a7b7b"
  },
  "reserve_nestful": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot2\\reserve_nestful_pilot2.jsonl",
    "sha256": "ee2e72bc9f4b6251b8a7fc5ee8f5a6377050feaf93b4534e7e0dbb757026ab24"
  },
  "train_grpo": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot2\\train_grpo_pilot2.jsonl",
    "sha256": "51481495a3f27deb4ea20e9ab4a4c8313f22667413758eeff1c7501e11879bbb"
  },
  "train_nestful": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot2\\train_nestful_pilot2.jsonl",
    "sha256": "32a9c99e292ae1fe571f4966a007e60103b4acb88aaab3ebdb608ef994e448a4"
  }
}
```
- config_hash: `6350f406671881f3`
- profile_hash: `54524a0fd967bcc4df1adfaa5208f7bc32a64610e41697485d1bf3aebb9b8c16`
- registry_hash: `f5210beb12d4a33af42de6c8bf1c2b8c3acb3cdd8cef7f0b030e09ee7d2ac562`
- executor_hash: `42d2696bf584cdd1f5c2c95ce0b281dc9de0a1598741256dc7bbe7a531a176e4`
- generator_version: `ttdf-0.1.0`

## Representative tasks (max 10)
### ttdf_1000941721cb (A, A_2call_linear_continuation_04)
- query: A school tracks these figures for its yearbook. Step 1: compute the ratio of 409 to 83. Step 2: decrease that result by 51 percent.
- calls: `[{"name": "ratio_of", "arguments": {"denominator": 83, "numerator": 409}}, {"name": "decrease_by_percent", "arguments": {"base": "$var_1.output_0$", "percent": 51}}]`
- answer: `2.414578` | offered 10 (hard distractors 2)

### ttdf_01441b6413d8 (A, A_4call_fan_in_tool_catalog_01)
- query: Find the remainder of 445 divided by 50. Next, calculate how many whole times 56 fits into the remainder. Then, decrease 1164 by 39 percent. Average the result of the step 2 and that value. Only report the final number.
- calls: `[{"name": "modulo_of", "arguments": {"dividend": 445, "divisor": 50}}, {"name": "floor_divide", "arguments": {"arg_0": "$var_1.output_0$", "arg_1": 56}}, {"name": "decrease_by_percent", "arguments": {"base": 1164, "percent": 39}}, {"name": "average_of_two", "arguments": {"arg_0": "$var_2.output_0$", "arg_1": "$var_3.output_0$"}}]`
- answer: `355.02` | offered 13 (hard distractors 3)

### ttdf_4720ea0a3695 (G, G_4call_branch_aggregate_tool_catalog_00)
- query: I need the final value after the following steps: round 32.95 down to a whole number; round 698.5903 to 3 decimal places; multiply 222 by 29; find the spread between the largest and smallest of the result of step 1, the result of step 2 and that result. (For context only: the meeting lasted 90 minutes.)
- calls: `[{"name": "round_down_whole", "arguments": {"value": 32.95}}, {"name": "trim_precision", "arguments": {"digits": 3, "raw_value": 698.5903}}, {"name": "pairwise_product", "arguments": {"value_one": 222, "value_two": 29}}, {"name": "dispersion_of_three", "arguments": {"value_one": "$var1.output_0$", "value_three": "$var3.output_0$", "value_two": "$var2.output_0$"}}]`
- answer: `6406.0` | offered 18 (hard distractors 4)

### ttdf_25c7a9dcff69 (A, A_6pcall_branch_aggregate_tool_catalog_00)
- query: First, determine 25 percent of 1874, then find the average of 515 and that result. Next, multiply 302 by 11, then calculate that result as a percentage of 560. After that, raise 417 by 39 percent. Lastly, obtain the spread between the largest and smallest of the result of step 2, the result of step 4, and that result.
- calls: `[{"name": "percent_of", "arguments": {"percent": 25, "whole": 1874}}, {"name": "average_of_two", "arguments": {"arg_0": 515, "arg_1": "$var1.output_0$"}}, {"name": "multiply", "arguments": {"arg_0": 302, "arg_1": 11}}, {"name": "percent_of", "arguments": {"percent": "$var3.output_0$", "whole": 560}}, {"name": "increase_by_percent", "arguments": {"base": 417, "percent": 39}}, {"name": "range_of_three", "arguments": {"arg_0": "$var2.output_0$", "arg_1": "$var4.output_0$", "arg_2": "$var5.output_0$"}}]`
- answer: `18111.45` | offered 13 (hard distractors 2)

### ttdf_1383fff20ea7 (G, G_2call_linear_continuation_03)
- query: A school tracks these figures for its yearbook. Step 1: average 469 and 56. Step 2: find the remainder of that result divided by 6. What is the final result?
- calls: `[{"name": "midpoint_value", "arguments": {"value_one": 469, "value_two": 56}}, {"name": "leftover_after_grouping", "arguments": {"group_size": 6, "total_items": "$var1.output_0$"}}]`
- answer: `4.5` | offered 8 (hard distractors 0)

### ttdf_4c2bafc00fb5 (A, A_3call_fan_in_variable_pla_00)
- query: A school tracks these figures for its yearbook. Step 1: negate 779. Step 2: multiply 123 by 2. Step 3: subtract that result from the result of step 1. Report only the final value.
- calls: `[{"name": "negate", "arguments": {"arg_0": 779}}, {"name": "multiply", "arguments": {"arg_0": 123, "arg_1": 2}}, {"name": "subtract", "arguments": {"arg_0": "$var_1.output_0$", "arg_1": "$var_2.output_0$"}}]`
- answer: `-1025.0` | offered 11 (hard distractors 2)

### ttdf_4bcc0bcb441d (A, A_5call_fan_in_long_horizon_00)
- query: Calculate the absolute difference between 175 and 77. Afterwards, 1843 by 57 percent and then square that value. Then, determine the result of step 1 percent of the squared value. Lastly, format that result with the unit 'boxes' and return that value.
- calls: `[{"name": "absolute_difference", "arguments": {"arg_0": 175, "arg_1": 77}}, {"name": "increase_by_percent", "arguments": {"base": 1843, "percent": 57}}, {"name": "square_value", "arguments": {"arg_0": "$var2.output_0$"}}, {"name": "percent_of", "arguments": {"percent": "$var1.output_0$", "whole": "$var3.output_0$"}}, {"name": "label_with_unit", "arguments": {"unit": "boxes", "value": "$var4.output_0$"}}]`
- answer: `8204952.117698 boxes` | offered 10 (hard distractors 0)

### ttdf_07480376bda3 (G, G_3call_fan_in_variable_pla_00)
- query: Step one: round 183.67 down to a whole number. After that, multiply 60 by 26. To finish, compute the ratio of the result of step 1 to that result.
- calls: `[{"name": "round_down_whole", "arguments": {"value": 183.67}}, {"name": "scale_quantity", "arguments": {"factor": 26, "quantity": 60}}, {"name": "proportion_between", "arguments": {"part_value": "$var1.output_0$", "whole_value": "$var2.output_0$"}}]`
- answer: `0.117308` | offered 10 (hard distractors 0)

### ttdf_3799eac59203 (G, G_5call_branch_aggregate_tool_catalog_00)
- query: A school tracks these figures for its yearbook. Step 1: round 336.52 down to a whole number. Step 2: subtract 30 from that result. Step 3: multiply 104 by 16. Step 4: divide 5765 by 36. Step 5: average the result of step 2, the result of step 3 and that result. What is the final result?
- calls: `[{"name": "round_down_whole", "arguments": {"value": 336.52}}, {"name": "reduce_amount", "arguments": {"base_value": "$var1.output_0$", "reduction": 30}}, {"name": "scale_quantity", "arguments": {"factor": 16, "quantity": 104}}, {"name": "per_unit_value", "arguments": {"units": 36, "whole": 5765}}, {"name": "balance_of_three", "arguments": {"reading_one": "$var2.output_0$", "reading_three": "$var4.output_0$", "reading_two": "$var3.output_0$"}}]`
- answer: `710.046296` | offered 14 (hard distractors 3)

### ttdf_39fadad53c92 (A, A_3call_linear_variable_pla_00)
- query: A lab notebook lists these measurements. Step 1: round 888.74 to the nearest whole number. Step 2: subtract 60 from that result. Step 3: scale every item of [21, 22, 40] by that result.
- calls: `[{"name": "nearest_integer", "arguments": {"arg_0": 888.74}}, {"name": "subtract", "arguments": {"arg_0": "$var1.output_0$", "arg_1": 60}}, {"name": "scale_values", "arguments": {"factor": "$var2.output_0$", "values": [21, 22, 40]}}]`
- answer: `[17409.0, 18238.0, 33160.0]` | offered 9 (hard distractors 2)

