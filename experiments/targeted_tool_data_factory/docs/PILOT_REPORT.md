# PILOT REPORT — targeted_tool_data_factory pilot3

## Verdict: **CONDITIONAL**

- profile-match: new dataset closer than Stage-3 on 6/8 metrics
- WARN: student probe NOT_RUN_LOCAL (structural P0 only)

## Counts
- generated candidates: 2213
- validated: 2210
- rejected: 3
- selected (frozen pilot): 1000
- track mix: A=563, G=437 (A share 56.3 %)
- tasks with >=1 hard distractor: 81.0 % (threshold >= 50.0 %)
- deterministic replay rate: 100.0 %
- contamination hits in candidate pool (all rejected; selected pool = 0): 0
- dedup drops: 0
- split leakage collisions: 0
- split sizes: {'heldout': 200, 'reserve': 200, 'train': 600}

## Rejection taxonomy
| reason | count |
|---|---|
| V4:shorter valid path found | 2 |
| V4:single offered tool solves the whole task | 1 |

## Generation cells (requested / generated / validated / rejected / selected)
| cell | req | gen | valid | rej | sel |
|---|---|---|---|---|---|
| A_2call_linear_continuation_00 | 1 | 1 | 50 | 0 | 30 |
| A_2call_linear_continuation_01 | 1 | 1 | 81 | 0 | 30 |
| A_2call_linear_continuation_03 | 1 | 1 | 78 | 0 | 29 |
| A_2call_linear_continuation_04 | 1 | 1 | 77 | 0 | 29 |
| A_2call_linear_tool_catalog_02 | 1 | 1 | 78 | 0 | 29 |
| A_2call_linear_tool_catalog_05 | 1 | 1 | 74 | 0 | 29 |
| A_3call_fan_in_tool_catalog_01 | 1 | 1 | 58 | 0 | 31 |
| A_3call_fan_in_variable_pla_00 | 1 | 1 | 42 | 0 | 31 |
| A_3call_linear_variable_pla_00 | 1 | 1 | 51 | 0 | 35 |
| A_4call_branch_aggregate_tool_catalog_00 | 1 | 1 | 31 | 0 | 12 |
| A_4call_fan_in_variable_pla_00 | 1 | 1 | 58 | 1 | 44 |
| A_4call_linear_variable_pla_00 | 1 | 1 | 50 | 0 | 21 |
| A_5call_branch_aggregate_tool_catalog_00 | 1 | 1 | 22 | 0 | 11 |
| A_5call_fan_in_long_horizon_00 | 1 | 1 | 86 | 0 | 40 |
| A_5call_linear_long_horizon_00 | 1 | 1 | 36 | 0 | 17 |
| A_6pcall_branch_aggregate_tool_catalog_00 | 1 | 1 | 49 | 0 | 22 |
| A_6pcall_fan_in_long_horizon_00 | 1 | 1 | 66 | 0 | 29 |
| A_6pcall_fan_in_tool_catalog_01 | 1 | 1 | 66 | 0 | 29 |
| A_6pcall_fan_in_tool_catalog_02 | 1 | 1 | 67 | 0 | 29 |
| A_6pcall_linear_long_horizon_00 | 1 | 1 | 59 | 0 | 36 |
| G_2call_linear_continuation_00 | 1 | 1 | 75 | 0 | 29 |
| G_2call_linear_continuation_01 | 1 | 1 | 72 | 0 | 29 |
| G_2call_linear_continuation_03 | 1 | 1 | 69 | 0 | 29 |
| G_2call_linear_continuation_04 | 1 | 1 | 70 | 0 | 29 |
| G_2call_linear_tool_catalog_02 | 1 | 1 | 70 | 0 | 29 |
| G_3call_fan_in_tool_catalog_01 | 1 | 1 | 1 | 0 | 1 |
| G_3call_fan_in_variable_pla_00 | 1 | 1 | 55 | 1 | 26 |
| G_3call_linear_variable_pla_00 | 1 | 1 | 54 | 1 | 29 |
| G_4call_branch_aggregate_tool_catalog_00 | 1 | 1 | 20 | 0 | 10 |
| G_4call_fan_in_variable_pla_00 | 1 | 1 | 78 | 0 | 36 |
| G_4call_linear_variable_pla_00 | 1 | 1 | 30 | 0 | 17 |
| G_5call_branch_aggregate_tool_catalog_00 | 1 | 1 | 15 | 0 | 9 |
| G_5call_fan_in_long_horizon_00 | 1 | 1 | 58 | 0 | 33 |
| G_5call_linear_long_horizon_00 | 1 | 1 | 17 | 0 | 14 |
| G_6pcall_branch_aggregate_tool_catalog_00 | 1 | 1 | 33 | 0 | 18 |
| G_6pcall_fan_in_long_horizon_00 | 1 | 1 | 65 | 0 | 35 |
| G_6pcall_fan_in_tool_catalog_01 | 1 | 1 | 66 | 0 | 35 |
| G_6pcall_linear_long_horizon_00 | 1 | 1 | 51 | 0 | 29 |

## Profile match vs NESTFUL dev (lower = closer; AUC 0.5 = indistinguishable)
| dataset | JSD call | JSD motif | JSD args | JSD answer | W tools | W qlen | AUC |
|---|---|---|---|---|---|---|---|
| new_selected | 0.0072 | 0.0517 | 0.0234 | 0.0020 | 0.67 | 45.2 | 0.549 |
| stage3_old | 0.5847 | 0.1222 | 0.0091 | 0.1517 | 1.10 | 31.4 | 0.728 |

## Selected distributions
### Call counts
| value | count | share |
|---|---|---|
| 2 | 321 | 32.1 % |
| 3 | 153 | 15.3 % |
| 4 | 140 | 14.0 % |
| 5 | 124 | 12.4 % |
| 6 | 134 | 13.4 % |
| 7 | 73 | 7.3 % |
| 8 | 55 | 5.5 % |

### Motifs
| value | count | share |
|---|---|---|
| branch_aggregate | 82 | 8.2 % |
| fan_in | 399 | 39.9 % |
| linear | 519 | 51.9 % |

### Answer types
| value | count | share |
|---|---|---|
| bool | 31 | 3.1 % |
| float | 740 | 74.0 % |
| int | 59 | 5.9 % |
| list | 66 | 6.6 % |
| numeric_string | 28 | 2.8 % |
| string | 76 | 7.6 % |

### Templates (max share 3.8 %, cap 5.0 %)
top: `[('goal_first_v1', 38), ('indirect_v6', 38), ('sequence_v1', 38), ('imperative_v1', 37), ('sequence_v3', 37)]`

### Cells (max share 4.4 %, cap 8.0 %)

### Distribution warnings
- none

## Student probe
status: `{'NOT_RUN_LOCAL': 1000}`

## Hashes / files
```json
{
  "analysis_csv": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\analysis_pilot3.csv",
    "sha256": "6e2a827166717b6328a0ebe26d71028cd5d86fd6c6ef6828f1d6237d482cb241"
  },
  "canonical": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\canonical_pilot3.jsonl",
    "sha256": "0f01f8d124f6429735ee684115aa8f10f6b5fee859939a803998eacba632e5c9"
  },
  "grpo_train_ready": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\grpo_train_ready_pilot3.jsonl",
    "sha256": "dd7bac86b02cc2a0bcf5381733f094ee39e3f5ab8476d60151d555f05fde4e1f"
  },
  "heldout_grpo": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\heldout_grpo_pilot3.jsonl",
    "sha256": "99c6a1cf2e8d50c5bcb2dfc6548f43f089d2833840fba1346dc1ae53afacfb3d"
  },
  "heldout_nestful": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\heldout_nestful_pilot3.jsonl",
    "sha256": "27115cd8822aa52558070c449b62213a02e8eb8185419b9a2d6749f0cd6d23ee"
  },
  "nestful_compat": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\nestful_compat_pilot3.jsonl",
    "sha256": "badcce161e37dd3365f4f0aa2c80b23c56e05d80f2ef5e6afe49d176aaf1717a"
  },
  "reserve_grpo": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\reserve_grpo_pilot3.jsonl",
    "sha256": "cdb63ad453d049873c870127f0a7b13003a3d1e5db4efcc0b0f83f8f4a591fe5"
  },
  "reserve_nestful": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\reserve_nestful_pilot3.jsonl",
    "sha256": "662c932550b685a4abebe9727743d997478030c1d7a0e8be67a6918dd88932e4"
  },
  "train_grpo": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\train_grpo_pilot3.jsonl",
    "sha256": "b1bf1d7e24e71521fa6fe34540f757237c113c598d461ed14777afff758f4d6e"
  },
  "train_nestful": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\train_nestful_pilot3.jsonl",
    "sha256": "00178b0279d27cfd5d640cf646cec3a59438e749cfa197b994770977b4651d11"
  }
}
```
- config_hash: `52e0fbb4a67a46f9`
- profile_hash: `54524a0fd967bcc4df1adfaa5208f7bc32a64610e41697485d1bf3aebb9b8c16`
- registry_hash: `f5210beb12d4a33af42de6c8bf1c2b8c3acb3cdd8cef7f0b030e09ee7d2ac562`
- executor_hash: `42d2696bf584cdd1f5c2c95ce0b281dc9de0a1598741256dc7bbe7a531a176e4`
- generator_version: `ttdf-0.1.0`

## Representative tasks (max 10)
### ttdf_0982b2624c98 (G, G_3call_linear_variable_pla_00)
- query: Calculate the average of 23, 47, 1, 42, 31. Then, increase 670 by that result percent. Finally, express the increase as a percent of 549 and report the result.
- calls: `[{"name": "central_value", "arguments": {"number_list": [23, 47, 1, 42, 31]}}, {"name": "grow_by_rate", "arguments": {"growth_rate": "$var1.output_0$", "starting_value": 670}}, {"name": "portion_by_rate", "arguments": {"rate": "$var2.output_0$", "reference_value": 549}}]`
- answer: `4737.6504` | offered 8 (hard distractors 2)

### ttdf_a1f4522870c7 (G, G_4call_fan_in_variable_pla_00)
- query: First, round 746.71 up to a whole number. Then negate 231. Then compute the result of step 1 percent of that result. Finally, build an identifier from the prefix 'run' and that result.
- calls: `[{"name": "round_up_whole", "arguments": {"value": 746.71}}, {"name": "flip_sign", "arguments": {"value": 231}}, {"name": "portion_by_rate", "arguments": {"rate": "$var1.output_0$", "reference_value": "$var2.output_0$"}}, {"name": "compose_reference_code", "arguments": {"code_number": "$var3.output_0$", "code_prefix": "run"}}]`
- answer: `run--1725.57` | offered 12 (hard distractors 2)

### ttdf_88d02f070bad (G, G_2call_linear_continuation_01)
- query: First, round 395.09 down to the nearest whole number. Then, use that result to find the remainder of 2247 divided by it.
- calls: `[{"name": "round_down_whole", "arguments": {"value": 395.09}}, {"name": "leftover_after_grouping", "arguments": {"group_size": "$var1.output_0$", "total_items": 2247}}]`
- answer: `272.0` | offered 10 (hard distractors 2)

### ttdf_acf64105ab0f (G, G_5call_linear_long_horizon_00)
- query: A school tracks these figures for its yearbook. Step 1: increase 820 by 39 percent. Step 2: average 459 and that result. Step 3: increase 484 by that result percent. Step 4: average 725 and that result. Step 5: find the remainder of that result divided by 44.
- calls: `[{"name": "grow_by_rate", "arguments": {"growth_rate": 39, "starting_value": 820}}, {"name": "midpoint_value", "arguments": {"value_one": 459, "value_two": "$var1.output_0$"}}, {"name": "grow_by_rate", "arguments": {"growth_rate": "$var2.output_0$", "starting_value": 484}}, {"name": "midpoint_value", "arguments": {"value_one": 725, "value_two": "$var3.output_0$"}}, {"name": "leftover_after_grouping", "arguments": {"group_size": 44, "total_items": "$var4.output_0$"}}]`
- answer: `31.048` | offered 9 (hard distractors 2)

### ttdf_52e02b0074e9 (A, A_2call_linear_continuation_04)
- query: First, determine how many times 10 fits into 424 as a whole number. Then, report the square of that result.
- calls: `[{"name": "whole_quotient", "arguments": {"dividend": 424, "divisor": 10}}, {"name": "square_value", "arguments": {"arg_0": "$var_1.output_0$"}}]`
- answer: `1764.0` | offered 11 (hard distractors 3)

### ttdf_28600db1173f (A, A_6pcall_fan_in_long_horizon_00)
- query: Tell me what comes out when I compute how many whole times 12 fits into 80; take the square root of that result; negate 191; subtract 83 from that result; average that result and 27; round that result up to a whole number; increase that result by 57 percent; multiply the result of step 2 by that result.
- calls: `[{"name": "floor_divide", "arguments": {"arg_0": 80, "arg_1": 12}}, {"name": "sqrt", "arguments": {"arg_0": "$var1.output_0$"}}, {"name": "negate", "arguments": {"arg_0": 191}}, {"name": "difference_of_numbers", "arguments": {"minuend": "$var3.output_0$", "subtrahend": 83}}, {"name": "average_of_two", "arguments": {"arg_0": "$var4.output_0$", "arg_1": 27}}, {"name": "ceiling", "arguments": {"arg_0": "$var5.output_0$"}}, {"name": "increase_by_percent", "arguments": {"base": "$var6.output_0$", "percent": 57}}, {"name": "multiply", "arguments": {"arg_0": "$var2.output_0$", "arg_1": "$var7.output_0$"}}]`
- answer: `-473.021014` | offered 11 (hard distractors 2)

### ttdf_4bcc0bcb441d (A, A_5call_fan_in_long_horizon_00)
- query: Calculate the absolute difference between 175 and 77. Afterwards, 1843 by 57 percent and then square that value. Then, determine the result of step 1 percent of the squared value. Lastly, format that result with the unit 'boxes' and return that value.
- calls: `[{"name": "absolute_difference", "arguments": {"arg_0": 175, "arg_1": 77}}, {"name": "increase_by_percent", "arguments": {"base": 1843, "percent": 57}}, {"name": "square_value", "arguments": {"arg_0": "$var2.output_0$"}}, {"name": "percent_of", "arguments": {"percent": "$var1.output_0$", "whole": "$var3.output_0$"}}, {"name": "label_with_unit", "arguments": {"unit": "boxes", "value": "$var4.output_0$"}}]`
- answer: `8204952.117698 boxes` | offered 10 (hard distractors 0)

### ttdf_1897e88c19ac (A, A_6pcall_fan_in_tool_catalog_02)
- query: Please find the remainder of 1204 divided by 38. then decrease 524 by that result percent. then divide that result by 28. then take the square root of that result. then convert 3 hours to minutes. and finally compute the ratio of the result of step 4 to that result. (For context only: the meeting lasted 90 minutes.)
- calls: `[{"name": "reminder", "arguments": {"arg_0": 1204, "arg_1": 38}}, {"name": "decrease_by_percent", "arguments": {"base": 524, "percent": "$var1.output_0$"}}, {"name": "quotient_of", "arguments": {"denominator": 28, "numerator": "$var2.output_0$"}}, {"name": "sqrt", "arguments": {"arg_0": "$var3.output_0$"}}, {"name": "hours_to_minutes", "arguments": {"hours": 3}}, {"name": "ratio_of", "arguments": {"denominator": "$var5.output_0$", "numerator": "$var4.output_0$"}}]`
- answer: `0.020674` | offered 9 (hard distractors 2)

### ttdf_a180adf5caad (A, A_3call_linear_variable_pla_00)
- query: Begin by taking 639 and finding the remainder when divided by 20. Then, raise the result by 58 percent. Finally, add that result to the list [15, 24, 46, 64, 81]
- calls: `[{"name": "modulo_of", "arguments": {"dividend": 639, "divisor": 20}}, {"name": "increase_by_percent", "arguments": {"base": "$var1.output_0$", "percent": 58}}, {"name": "append_to_values", "arguments": {"value": "$var2.output_0$", "values": [15, 24, 46, 64, 81]}}]`
- answer: `[15.0, 24.0, 46.0, 64.0, 81.0, 30.02]` | offered 8 (hard distractors 2)

### ttdf_6c5b125b6094 (A, A_6pcall_linear_long_horizon_00)
- query: First, work out the range of [10, 13, 11] by finding the difference between the largest and smallest. Then, see how many whole times that number fits into 3351. Next, calculate 79 percent of that result. Then, multiply 325 by that result. Afterwards, take the square root of that result. Finally, round that result to the nearest integer.
- calls: `[{"name": "range_of_values", "arguments": {"values": [10, 13, 11]}}, {"name": "whole_quotient", "arguments": {"dividend": 3351, "divisor": "$var_1.output_0$"}}, {"name": "percent_of", "arguments": {"percent": 79, "whole": "$var_2.output_0$"}}, {"name": "multiply", "arguments": {"arg_0": 325, "arg_1": "$var_3.output_0$"}}, {"name": "sqrt", "arguments": {"arg_0": "$var_4.output_0$"}}, {"name": "nearest_integer", "arguments": {"arg_0": "$var_5.output_0$"}}]`
- answer: `536` | offered 9 (hard distractors 2)

