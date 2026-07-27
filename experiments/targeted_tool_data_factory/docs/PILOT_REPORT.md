# PILOT REPORT — targeted_tool_data_factory pilot3

## Verdict: **CONDITIONAL**

- profile-match: new dataset closer than Stage-3 on 6/8 metrics
- WARN: student probe NOT_RUN_LOCAL (structural P0 only)

## Counts
- generated candidates: 5671
- validated: 5243
- rejected: 428
- selected (frozen pilot): 1000
- track mix: A=552, G=448 (A share 55.2 %)
- tasks with >=1 hard distractor: 79.0 % (threshold >= 50.0 %)
- deterministic replay rate: 100.0 %
- contamination hits in candidate pool (all rejected; selected pool = 0): 7
- dedup drops: 16
- split leakage collisions: 0
- split sizes: {'heldout': 200, 'reserve': 200, 'train': 600}

## Rejection taxonomy
| reason | count |
|---|---|
| V4:shorter valid path found | 241 |
| V4:single offered tool solves the whole task | 160 |
| V3:oracle answer '[2, 4, 5, 7, 8, 10]' appears in query | 10 |
| V5:gold tool-call skeleton overlap with target | 7 |
| V3:oracle answer '[105, 185, 263, 343, 418, 523]' appears in qu | 1 |
| V3:oracle answer '[127, 225, 319, 417, 508, 635]' appears in qu | 1 |
| V3:oracle answer '[1796, 3182, 4517, 5902, 7186, 8982]' appears | 1 |
| V3:oracle answer '[200, 354, 502, 656, 799, 998]' appears in qu | 1 |
| V3:oracle answer '[22, 39, 55, 72, 88, 110]' appears in query | 1 |
| V3:oracle answer '[30, 53, 75, 98, 119, 149]' appears in query | 1 |
| V3:oracle answer '[357, 633, 898, 1174, 1429, 1787]' appears in | 1 |
| V3:oracle answer '[4, 6, 9, 12, 14, 18]' appears in query | 1 |
| V3:oracle answer '[60, 106, 150, 197, 239, 299]' appears in que | 1 |
| V3:oracle answer '[692, 1226, 1741, 2275, 2769, 3462]' appears  | 1 |
| V3:oracle answer '[731487, 1295777, 1839168, 2403458, 2925949,  | 1 |
| V3:oracle answer '[817, 1447, 2054, 2685, 3268, 4085]' appears  | 1 |
| V3:oracle answer '[94, 167, 238, 310, 378, 472]' appears in que | 1 |
| V3:oracle answer '[99, 175, 248, 324, 395, 494]' appears in que | 1 |
| V5:normalized duplicate | 1 |

## Generation cells (requested / generated / validated / rejected / selected)
| cell | req | gen | valid | rej | sel |
|---|---|---|---|---|---|
| A_2call_linear_continuation_00 | 101 | 101 | 129 | 21 | 29 |
| A_2call_linear_continuation_01 | 101 | 101 | 176 | 5 | 29 |
| A_2call_linear_continuation_03 | 101 | 101 | 176 | 2 | 29 |
| A_2call_linear_continuation_04 | 101 | 101 | 172 | 5 | 29 |
| A_2call_linear_tool_catalog_02 | 101 | 101 | 173 | 5 | 29 |
| A_2call_linear_tool_catalog_05 | 101 | 101 | 164 | 10 | 29 |
| A_3call_fan_in_tool_catalog_01 | 108 | 108 | 130 | 35 | 31 |
| A_3call_fan_in_variable_pla_00 | 108 | 108 | 97 | 52 | 31 |
| A_3call_linear_variable_pla_00 | 121 | 121 | 168 | 3 | 35 |
| A_4call_branch_aggregate_tool_catalog_00 | 39 | 39 | 66 | 3 | 11 |
| A_4call_fan_in_variable_pla_00 | 151 | 151 | 156 | 53 | 43 |
| A_4call_linear_variable_pla_00 | 70 | 70 | 116 | 3 | 20 |
| A_5call_branch_aggregate_tool_catalog_00 | 35 | 35 | 54 | 2 | 10 |
| A_5call_fan_in_long_horizon_00 | 139 | 139 | 224 | 0 | 40 |
| A_5call_linear_long_horizon_00 | 58 | 58 | 90 | 3 | 17 |
| A_6pcall_branch_aggregate_tool_catalog_00 | 74 | 74 | 122 | 0 | 21 |
| A_6pcall_fan_in_long_horizon_00 | 98 | 98 | 162 | 1 | 28 |
| A_6pcall_fan_in_tool_catalog_01 | 98 | 98 | 163 | 0 | 28 |
| A_6pcall_fan_in_tool_catalog_02 | 98 | 98 | 161 | 3 | 28 |
| A_6pcall_linear_long_horizon_00 | 123 | 123 | 150 | 31 | 35 |
| G_2call_linear_continuation_00 | 99 | 99 | 168 | 5 | 28 |
| G_2call_linear_continuation_01 | 99 | 99 | 162 | 8 | 28 |
| G_2call_linear_continuation_03 | 99 | 99 | 163 | 4 | 28 |
| G_2call_linear_continuation_04 | 99 | 99 | 157 | 11 | 28 |
| G_2call_linear_tool_catalog_02 | 99 | 99 | 162 | 6 | 28 |
| G_3call_fan_in_tool_catalog_01 | 88 | 88 | 61 | 27 | 25 |
| G_3call_fan_in_variable_pla_00 | 88 | 88 | 93 | 50 | 25 |
| G_3call_linear_variable_pla_00 | 99 | 99 | 127 | 26 | 28 |
| G_4call_branch_aggregate_tool_catalog_00 | 32 | 32 | 51 | 0 | 9 |
| G_4call_fan_in_variable_pla_00 | 123 | 123 | 172 | 28 | 35 |
| G_4call_linear_variable_pla_00 | 57 | 57 | 81 | 5 | 16 |
| G_5call_branch_aggregate_tool_catalog_00 | 28 | 28 | 42 | 0 | 8 |
| G_5call_fan_in_long_horizon_00 | 113 | 113 | 167 | 3 | 32 |
| G_5call_linear_long_horizon_00 | 47 | 47 | 61 | 2 | 14 |
| G_6pcall_branch_aggregate_tool_catalog_00 | 60 | 60 | 92 | 0 | 17 |
| G_6pcall_fan_in_long_horizon_00 | 120 | 120 | 183 | 1 | 35 |
| G_6pcall_fan_in_tool_catalog_01 | 120 | 120 | 185 | 0 | 35 |
| G_6pcall_linear_long_horizon_00 | 100 | 100 | 135 | 15 | 29 |

## Profile match vs NESTFUL dev (lower = closer; AUC 0.5 = indistinguishable)
| dataset | JSD call | JSD motif | JSD args | JSD answer | W tools | W qlen | AUC |
|---|---|---|---|---|---|---|---|
| new_selected | 0.0041 | 0.0487 | 0.0231 | 0.0057 | 0.76 | 42.0 | 0.545 |
| stage3_old | 0.5847 | 0.1222 | 0.0091 | 0.1517 | 1.10 | 31.4 | 0.728 |

## Selected distributions
### Call counts
| value | count | share |
|---|---|---|
| 2 | 314 | 31.4 % |
| 3 | 175 | 17.5 % |
| 4 | 134 | 13.4 % |
| 5 | 121 | 12.1 % |
| 6 | 133 | 13.3 % |
| 7 | 55 | 5.5 % |
| 8 | 68 | 6.8 % |

### Motifs
| value | count | share |
|---|---|---|
| branch_aggregate | 76 | 7.6 % |
| fan_in | 416 | 41.6 % |
| linear | 508 | 50.8 % |

### Answer types
| value | count | share |
|---|---|---|
| bool | 42 | 4.2 % |
| float | 716 | 71.6 % |
| int | 62 | 6.2 % |
| list | 68 | 6.8 % |
| numeric_string | 35 | 3.5 % |
| string | 77 | 7.7 % |

### Templates (max share 4.0 %, cap 5.0 %)
top: `[('imperative_v2', 40), ('word_problem_report', 38), ('goal_first_v1', 37), ('indirect_v4', 37), ('sequence_v1', 37)]`

### Cells (max share 4.3 %, cap 8.0 %)

### Distribution warnings
- none

## Student probe
status: `{'NOT_RUN_LOCAL': 1000}`

## Hashes / files
```json
{
  "analysis_csv": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\analysis_pilot3.csv",
    "sha256": "6e81936836bfd7ccaa45c0777e200a819f4388e594a640db87b56e4bf91a142d"
  },
  "canonical": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\canonical_pilot3.jsonl",
    "sha256": "5deb55092d27e16209126c0f0144bd115e7e4a97f83901392c5c3a17dce397b0"
  },
  "grpo_train_ready": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\grpo_train_ready_pilot3.jsonl",
    "sha256": "ae7042b23dfb1b50ee998ea08cbf9cd3ff70217a9c0327c95bf4449c4bb0ffec"
  },
  "heldout_grpo": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\heldout_grpo_pilot3.jsonl",
    "sha256": "dce7a610153f347e1590241a4f41aaacaedaf14a92498f2cee78df3ddf36640a"
  },
  "heldout_nestful": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\heldout_nestful_pilot3.jsonl",
    "sha256": "c0f1c3ae0d9a5bfaac2efdb5fc36b60859ef2e1ee7fcddc15506b71b4dab85dd"
  },
  "nestful_compat": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\nestful_compat_pilot3.jsonl",
    "sha256": "12a97bdf22ef47568e09611150275820029120544b7ef1ebd841c32f5daf672f"
  },
  "reserve_grpo": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\reserve_grpo_pilot3.jsonl",
    "sha256": "711d7093114255b156bf8362471fb09b1188c17c511ea6d3fa491d26da6712d8"
  },
  "reserve_nestful": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\reserve_nestful_pilot3.jsonl",
    "sha256": "6f78e4e8a2a4ac4f6ca15af656e893e5160365ff3c711b8cb45a168f035640ba"
  },
  "train_grpo": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\train_grpo_pilot3.jsonl",
    "sha256": "77d7e2bf51acd9a998d4cd202f6b7add7e841a41eb31cc74af249745f765e538"
  },
  "train_nestful": {
    "path": "C:\\Users\\\u0160unka\\Documents\\GitHub\\Tool-R0\\experiments\\targeted_tool_data_factory\\outputs\\selected\\export_pilot3\\train_nestful_pilot3.jsonl",
    "sha256": "d8b3960e7460218d39b6b8db99bf773b7ddc25274697cd5c996855dbbb906972"
  }
}
```
- config_hash: `52e0fbb4a67a46f9`
- profile_hash: `54524a0fd967bcc4df1adfaa5208f7bc32a64610e41697485d1bf3aebb9b8c16`
- registry_hash: `f5210beb12d4a33af42de6c8bf1c2b8c3acb3cdd8cef7f0b030e09ee7d2ac562`
- executor_hash: `42d2696bf584cdd1f5c2c95ce0b281dc9de0a1598741256dc7bbe7a531a176e4`
- generator_version: `ttdf-0.1.0`

## Representative tasks (max 10)
### ttdf_2134f49bd3ad (G, G_5call_branch_aggregate_tool_catalog_00)
- query: An engineer checks a report. Step 1: subtract 11 from 230. Step 2: compute how many whole times 31 fits into that result. Step 3: round 151.78 up to a whole number. Step 4: negate 863. Step 5: add up the result of step 2, the result of step 3 and that result. What is the final result?
- calls: `[{"name": "reduce_amount", "arguments": {"base_value": 230, "reduction": 11}}, {"name": "full_groups_of", "arguments": {"group_size": 31, "total_items": "$var1.output_0$"}}, {"name": "round_up_whole", "arguments": {"value": 151.78}}, {"name": "flip_sign", "arguments": {"value": 863}}, {"name": "total_of_three_parts", "arguments": {"part_one": "$var2.output_0$", "part_three": "$var4.output_0$", "part_two": "$var3.output_0$"}}]`
- answer: `-704.0` | offered 14 (hard distractors 3)

### ttdf_a59ae8939796 (G, G_5call_fan_in_long_horizon_00)
- query: You will need to find the ratio of 524 to 17, then square that result, then increase the result by 9 percent, then increase 1422 by 5 percent first, and lastly, for the final goal, find the percent of the result of step 3 of the value those steps produce.
- calls: `[{"name": "proportion_between", "arguments": {"part_value": 524, "whole_value": 17}}, {"name": "self_product", "arguments": {"value": "$var1.output_0$"}}, {"name": "grow_by_rate", "arguments": {"growth_rate": 9, "starting_value": "$var2.output_0$"}}, {"name": "grow_by_rate", "arguments": {"growth_rate": 5, "starting_value": 1422}}, {"name": "portion_by_rate", "arguments": {"rate": "$var3.output_0$", "reference_value": "$var4.output_0$"}}]`
- answer: `15462.514261` | offered 11 (hard distractors 2)

### ttdf_562e14e68ec4 (A, A_2call_linear_tool_catalog_02)
- query: Determine the outcome of this procedure: round 562.55 to the nearest whole number; subtract 30 from that result. (Note: the team has 7 members, which is not needed here.)
- calls: `[{"name": "nearest_integer", "arguments": {"arg_0": 562.55}}, {"name": "subtract", "arguments": {"arg_0": "$var1.output_0$", "arg_1": 30}}]`
- answer: `533.0` | offered 15 (hard distractors 4)

### ttdf_4753fd527f32 (G, G_4call_fan_in_variable_pla_00)
- query: Find the remainder of 2051 divided by 30, then convert 34 degrees Fahrenheit to Celsius, then compute the ratio of the result of step 1 to that result, and finally build an identifier from the prefix 'order' and that result.
- calls: `[{"name": "leftover_after_grouping", "arguments": {"group_size": 30, "total_items": 2051}}, {"name": "celsius_from_fahrenheit", "arguments": {"temp_f": 34}}, {"name": "proportion_between", "arguments": {"part_value": "$var1.output_0$", "whole_value": "$var2.output_0$"}}, {"name": "compose_reference_code", "arguments": {"code_number": "$var3.output_0$", "code_prefix": "order"}}]`
- answer: `order-9.900001` | offered 11 (hard distractors 2)

### ttdf_a67dd02aea22 (A, A_3call_fan_in_variable_pla_00)
- query: First, take the negative of 459. Then, increase the value of 581 by 28 percent, that result, and finally, divide the result of step 1 by the previous value.
- calls: `[{"name": "negate", "arguments": {"arg_0": 459}}, {"name": "increase_by_percent", "arguments": {"base": 581, "percent": 28}}, {"name": "divide", "arguments": {"arg_0": "$var1.output_0$", "arg_1": "$var2.output_0$"}}]`
- answer: `-0.617201` | offered 12 (hard distractors 3)

### ttdf_2930e4ea8e12 (G, G_6pcall_linear_long_horizon_00)
- query: First, determine the ratio of 657 to 11, then round the previous value down to the nearest whole number. Next, take the ratio of that result to 81, then divide that result by 28. After that, work out that result as a percentage of 3554, then subtract the result from 731, and finally divide that result by 9.
- calls: `[{"name": "proportion_between", "arguments": {"part_value": 657, "whole_value": 11}}, {"name": "round_down_whole", "arguments": {"value": "$var1.output_0$"}}, {"name": "proportion_between", "arguments": {"part_value": "$var2.output_0$", "whole_value": 81}}, {"name": "per_unit_value", "arguments": {"units": 28, "whole": "$var3.output_0$"}}, {"name": "portion_by_rate", "arguments": {"rate": "$var4.output_0$", "reference_value": 3554}}, {"name": "remaining_after", "arguments": {"from_value": 731, "take_away": "$var5.output_0$"}}, {"name": "per_unit_value", "arguments": {"units": 9, "whole": "$var6.output_0$"}}]`
- answer: `81.119496` | offered 8 (hard distractors 2)

### ttdf_5142b25ba487 (A, A_6pcall_fan_in_tool_catalog_01)
- query: An engineer checks a report. Step 1: add 650 and 78. Step 2: multiply 197 by that result. Step 3: compute 87 percent of that result. Step 4: subtract 76 from that result. Step 5: divide that result by 12. Step 6: take the square root of that result. Step 7: find how many whole minutes fit into 10472 seconds. Step 8: compute the ratio of the result of step 6 to that result. What is the final result?
- calls: `[{"name": "sum_two_numbers", "arguments": {"first_number": 650, "second_number": 78}}, {"name": "product_of_numbers", "arguments": {"first_factor": 197, "second_factor": "$var1.output_0$"}}, {"name": "percent_of", "arguments": {"percent": 87, "whole": "$var2.output_0$"}}, {"name": "subtract", "arguments": {"arg_0": "$var3.output_0$", "arg_1": 76}}, {"name": "quotient_of", "arguments": {"denominator": 12, "numerator": "$var4.output_0$"}}, {"name": "sqrt", "arguments": {"arg_0": "$var5.output_0$"}}, {"name": "seconds_to_full_minutes", "arguments": {"seconds": 10472}}, {"name": "ratio_of", "arguments": {"denominator": "$var7.output_0$", "numerator": "$var6.output_0$"}}]`
- answer: `0.58585` | offered 16 (hard distractors 2)

### ttdf_532d50f0732d (G, G_2call_linear_continuation_00)
- query: First, take the square root of 196, then take the opposite of that result.
- calls: `[{"name": "root_extract", "arguments": {"input_value": 196}}, {"name": "flip_sign", "arguments": {"value": "$var1.output_0$"}}]`
- answer: `-14.0` | offered 8 (hard distractors 0)

### ttdf_206eda2ac240 (G, G_6pcall_fan_in_long_horizon_00)
- query: Start by transforming 25 degrees Celsius to Fahrenheit. Lower that result by 34 percent. Negate that result. Compute the ratio of 843 to the result. Find the remainder of the result when divided by 17. Decrease that result by 55 percent. Convert 42 degrees Celsius to Fahrenheit. Subtract that result from the result of the 6th operation.
- calls: `[{"name": "fahrenheit_from_celsius", "arguments": {"temp_c": 25}}, {"name": "shrink_by_rate", "arguments": {"discount_rate": 34, "starting_value": "$var1.output_0$"}}, {"name": "flip_sign", "arguments": {"value": "$var2.output_0$"}}, {"name": "proportion_between", "arguments": {"part_value": 843, "whole_value": "$var3.output_0$"}}, {"name": "leftover_after_grouping", "arguments": {"group_size": 17, "total_items": "$var4.output_0$"}}, {"name": "shrink_by_rate", "arguments": {"discount_rate": 55, "starting_value": "$var5.output_0$"}}, {"name": "fahrenheit_from_celsius", "arguments": {"temp_c": 42}}, {"name": "reduce_amount", "arguments": {"base_value": "$var6.output_0$", "reduction": "$var7.output_0$"}}]`
- answer: `-107.414581` | offered 10 (hard distractors 0)

### ttdf_4372e82cfaea (G, G_3call_fan_in_tool_catalog_01)
- query: A warehouse audit produced these numbers. Step 1: compute 52 percent of 2097. Step 2: find how many whole minutes fit into 47103 seconds. Step 3: add the result of step 1 and that result. (Unrelated: the office moved in 2019.)
- calls: `[{"name": "portion_by_rate", "arguments": {"rate": 52, "reference_value": 2097}}, {"name": "whole_minutes_from_seconds", "arguments": {"second_count": 47103}}, {"name": "total_of_pair", "arguments": {"left": "$var1.output_0$", "right": "$var2.output_0$"}}]`
- answer: `1875.44` | offered 14 (hard distractors 0)

