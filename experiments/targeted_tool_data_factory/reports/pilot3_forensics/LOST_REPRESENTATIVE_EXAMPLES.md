# LOST_REPRESENTATIVE_EXAMPLES

Selected deterministically by most common failure transitions.

## 04bd68f8-3e77-4cda-b43f-c42616d95d33
- call_bucket: `3`
- divergence: `DIFFERENT_FIRST_TOOL` turn=0
- failure: `SUCCESS_ALTERNATIVE_VALID -> FAIL_WRONG_FIRST_TOOL`
- tools C0: `["divide"]`
- tools D1: `["multiply", "multiply", "add", "divide"]`
- coverage exact: `1.0` ood=`0.0`

## 35adf4ad-dada-47dd-8c92-1af9100d6a8a
- call_bucket: `5`
- divergence: `DIFFERENT_FIRST_TOOL` turn=0
- failure: `SUCCESS_ALTERNATIVE_VALID -> FAIL_WRONG_FIRST_TOOL`
- tools C0: `["power", "add", "divide", "subtract"]`
- tools D1: `["multiply", "add", "divide"]`
- coverage exact: `1.0` ood=`0.0`

## 4b54ac08-7d24-4d16-b63d-c4d07aaccd80
- call_bucket: `4`
- divergence: `DIFFERENT_FIRST_TOOL` turn=0
- failure: `SUCCESS_ALTERNATIVE_VALID -> FAIL_WRONG_FIRST_TOOL`
- tools C0: `["multiply", "multiply", "divide"]`
- tools D1: `["subtract"]`
- coverage exact: `1.0` ood=`0.0`

## 6c668a48-30d8-46ba-a9ac-8e0c58f9d9f0
- call_bucket: `6+`
- divergence: `DIFFERENT_FIRST_TOOL` turn=0
- failure: `SUCCESS_ALTERNATIVE_VALID -> FAIL_WRONG_FIRST_TOOL`
- tools C0: `["multiply", "multiply", "divide"]`
- tools D1: `["divide", "multiply"]`
- coverage exact: `1.0` ood=`0.2`

## 7bdd0e4f-6c99-4f49-b9e8-2afc9546891b
- call_bucket: `3`
- divergence: `TOOL_COUNT_DIFFERENCE` turn=0
- failure: `SUCCESS_ALTERNATIVE_VALID -> FAIL_WRONG_FIRST_TOOL`
- tools C0: `["divide", "subtract", "multiply"]`
- tools D1: `["divide"]`
- coverage exact: `1.0` ood=`0.0`

## e1a005a4-216d-4486-a15b-d53ccb574e78
- call_bucket: `6+`
- divergence: `DIFFERENT_FIRST_TOOL` turn=0
- failure: `SUCCESS_ALTERNATIVE_VALID -> FAIL_WRONG_FIRST_TOOL`
- tools C0: `["multiply", "multiply", "divide"]`
- tools D1: `["combine"]`
- coverage exact: `1.0` ood=`0.2`

## b370d25b-a241-486b-9630-7dbd523312c2
- call_bucket: `6+`
- divergence: `DIFFERENT_LATER_TOOL` turn=2
- failure: `SUCCESS_ALTERNATIVE_VALID -> FAIL_WRONG_TOOL_SEQUENCE`
- tools C0: `["multiply", "multiply", "subtract", "divide"]`
- tools D1: `["multiply", "multiply", "add", "divide"]`
- coverage exact: `1.0` ood=`0.2`

## c28f3807-67fa-4513-8fdc-7802409d49d9
- call_bucket: `4`
- divergence: `TOOL_COUNT_DIFFERENCE` turn=1
- failure: `SUCCESS_ALTERNATIVE_VALID -> FAIL_EXECUTOR_ERROR`
- tools C0: `["divide", "multiply", "subtract", "multiply"]`
- tools D1: `["divide", "multiply"]`
- coverage exact: `1.0` ood=`0.0`

## e5eb37ac-bd11-4a84-82a3-588c93ba7712
- call_bucket: `2`
- divergence: `DIFFERENT_LATER_TOOL` turn=0
- failure: `SUCCESS_ALTERNATIVE_VALID -> FAIL_EXECUTOR_ERROR`
- tools C0: `["multiply", "divide"]`
- tools D1: `["multiply", "multiply"]`
- coverage exact: `1.0` ood=`0.0`

## fe72229a-2b0d-4eb4-8ba9-1e5179ff5029
- call_bucket: `5`
- divergence: `TOOL_COUNT_DIFFERENCE` turn=0
- failure: `SUCCESS_ALTERNATIVE_VALID -> FAIL_WRONG_TOOL_SEQUENCE`
- tools C0: `["multiply", "multiply", "divide", "divide"]`
- tools D1: `["multiply", "add", "add", "add", "add", "add"]`
- coverage exact: `1.0` ood=`0.0`

## 0219f9a1-ced9-4e6d-9b99-1570e1178b66
- call_bucket: `2`
- divergence: `REFERENCE_DIFFERENCE` turn=0
- failure: `SUCCESS_STRICT_GOLD -> FAIL_EXECUTOR_ERROR`
- tools C0: `["round_to_ceiling", "multiply_without_star"]`
- tools D1: `["round_to_ceiling", "multiply_without_star"]`
- coverage exact: `0.0` ood=`0.45`

## 52373c4f-e3e9-49cc-ada0-260f70efa0d3
- call_bucket: `2`
- divergence: `TOOL_COUNT_DIFFERENCE` turn=2
- failure: `SUCCESS_ALTERNATIVE_VALID -> FAIL_EXECUTABLE_WRONG_ANSWER`
- tools C0: `["bitwise_left_shift", "power_without_pow", "power_without_pow"]`
- tools D1: `["bitwise_left_shift", "power_without_pow"]`
- coverage exact: `0.0` ood=`0.625`

## 6a42f278-ce39-4b47-ae02-60f80782c2c6
- call_bucket: `5`
- divergence: `DIFFERENT_FIRST_TOOL` turn=0
- failure: `SUCCESS_OTHER_OFFICIAL -> FAIL_WRONG_FIRST_TOOL`
- tools C0: `["multiply", "divide", "multiply", "multiply", "subtract"]`
- tools D1: `["sum_even_numbers"]`
- coverage exact: `1.0` ood=`0.0`

## 6bcc90ed-8253-46d4-86a6-ebc1c8550f7a
- call_bucket: `5`
- divergence: `TOOL_COUNT_DIFFERENCE` turn=0
- failure: `SUCCESS_STRICT_GOLD -> FAIL_NO_TOOL_CALL`
- tools C0: `["multiply", "multiply", "multiply", "add", "subtract"]`
- tools D1: `[]`
- coverage exact: `1.0` ood=`0.2`

## 766a4d29-1adf-4c1b-ae0b-bda50257dfb3
- call_bucket: `2`
- divergence: `TOOL_COUNT_DIFFERENCE` turn=0
- failure: `SUCCESS_ALTERNATIVE_VALID -> FAIL_NO_TOOL_CALL`
- tools C0: `["multiply"]`
- tools D1: `[]`
- coverage exact: `1.0` ood=`0.0`
