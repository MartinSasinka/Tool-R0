# GAINED_REPRESENTATIVE_EXAMPLES

Selected deterministically by most common failure transitions.

## 24a1101f-819c-4ca7-a4e4-e7dee741c4f6
- call_bucket: `4`
- divergence: `TOOL_COUNT_DIFFERENCE` turn=0
- failure: `FAIL_EXECUTOR_ERROR -> SUCCESS_ALTERNATIVE_VALID`
- tools C0: `["multiply", "add", "divide"]`
- tools D1: `["multiply", "add", "subtract", "add", "divide"]`
- coverage exact: `1.0` ood=`0.0`

## 8ee1d3b2-4632-4455-b3b1-7a165383c55b
- call_bucket: `5`
- divergence: `DIFFERENT_FIRST_TOOL` turn=0
- failure: `FAIL_EXECUTOR_ERROR -> SUCCESS_ALTERNATIVE_VALID`
- tools C0: `["add"]`
- tools D1: `["multiply", "subtract"]`
- coverage exact: `1.0` ood=`0.0`

## b3e268c9-2465-41d5-ba04-adfe860d791f
- call_bucket: `5`
- divergence: `DIFFERENT_FIRST_TOOL` turn=0
- failure: `FAIL_EXECUTOR_ERROR -> SUCCESS_ALTERNATIVE_VALID`
- tools C0: `["add", "multiply"]`
- tools D1: `["divide", "multiply", "inverse", "add", "inverse"]`
- coverage exact: `1.0` ood=`0.0`

## db8d72af-7a1a-4201-a376-4b827b0162df
- call_bucket: `5`
- divergence: `TOOL_COUNT_DIFFERENCE` turn=0
- failure: `FAIL_EXECUTOR_ERROR -> SUCCESS_ALTERNATIVE_VALID`
- tools C0: `["lcm", "divide"]`
- tools D1: `["lcm", "divide", "divide", "add"]`
- coverage exact: `0.6` ood=`0.32`

## e70bd2c9-99af-410a-bc6d-61a65e00f7d8
- call_bucket: `5`
- divergence: `DIFFERENT_FIRST_TOOL` turn=0
- failure: `FAIL_EXECUTOR_ERROR -> SUCCESS_ALTERNATIVE_VALID`
- tools C0: `["multiply", "divide", "divide"]`
- tools D1: `["divide", "subtract", "divide"]`
- coverage exact: `1.0` ood=`0.0`

## e8605b92-ee9a-4265-9c3c-aa4bc3a691e9
- call_bucket: `6+`
- divergence: `DIFFERENT_FIRST_TOOL` turn=0
- failure: `FAIL_EXECUTOR_ERROR -> SUCCESS_ALTERNATIVE_VALID`
- tools C0: `["add", "divide"]`
- tools D1: `["multiply", "divide"]`
- coverage exact: `0.6667` ood=`0.4667`

## 174a02ed-7ae5-4508-9754-d9408a886790
- call_bucket: `5`
- divergence: `DIFFERENT_FIRST_TOOL` turn=0
- failure: `FAIL_WRONG_FIRST_TOOL -> SUCCESS_ALTERNATIVE_VALID`
- tools C0: `["power"]`
- tools D1: `["multiply", "subtract", "multiply"]`
- coverage exact: `1.0` ood=`0.0`

## 1ddeac74-83ee-4119-9100-83133001b0f5
- call_bucket: `5`
- divergence: `DIFFERENT_FIRST_TOOL` turn=0
- failure: `FAIL_WRONG_FIRST_TOOL -> SUCCESS_ALTERNATIVE_VALID`
- tools C0: `["divide", "divide", "multiply", "multiply", "multiply", "multiply"]`
- tools D1: `["multiply", "divide", "multiply", "divide", "subtract"]`
- coverage exact: `1.0` ood=`0.0`

## 603b4a8b-a4dc-4b16-bb35-ab167c6c45c7
- call_bucket: `6+`
- divergence: `TOOL_COUNT_DIFFERENCE` turn=2
- failure: `FAIL_WRONG_FIRST_TOOL -> SUCCESS_ALTERNATIVE_VALID`
- tools C0: `["divide", "multiply", "divide", "multiply", "subtract", "subtract", "add"]`
- tools D1: `["divide", "multiply", "subtract", "divide", "multiply", "add"]`
- coverage exact: `1.0` ood=`0.0`

## 6c0d717a-af97-47db-ad9d-45255f829e0d
- call_bucket: `3`
- divergence: `DIFFERENT_FIRST_TOOL` turn=0
- failure: `FAIL_WRONG_FIRST_TOOL -> SUCCESS_ALTERNATIVE_VALID`
- tools C0: `["subtract", "divide"]`
- tools D1: `["divide", "multiply"]`
- coverage exact: `1.0` ood=`0.0`

## d5aec711-273b-4a98-9789-02d7ac1a45be
- call_bucket: `2`
- divergence: `DIFFERENT_LATER_TOOL` turn=0
- failure: `FAIL_WRONG_FIRST_TOOL -> SUCCESS_ALTERNATIVE_VALID`
- tools C0: `["add", "add", "subtract"]`
- tools D1: `["add", "add", "multiply"]`
- coverage exact: `1.0` ood=`0.0`

## 38e7c22f-c209-4f49-8bbd-08bc3cb08203
- call_bucket: `3`
- divergence: `TOOL_COUNT_DIFFERENCE` turn=0
- failure: `FAIL_NO_TOOL_CALL -> SUCCESS_ALTERNATIVE_VALID`
- tools C0: `[]`
- tools D1: `["subtract", "divide"]`
- coverage exact: `1.0` ood=`0.0`

## 42143b4c-1a3d-4f30-a3a6-c5e10facc1fe
- call_bucket: `5`
- divergence: `TOOL_COUNT_DIFFERENCE` turn=0
- failure: `FAIL_NO_TOOL_CALL -> SUCCESS_ALTERNATIVE_VALID`
- tools C0: `[]`
- tools D1: `["divide"]`
- coverage exact: `1.0` ood=`0.0`

## 58f0da42-f029-4773-86a1-dc7548859054
- call_bucket: `3`
- divergence: `TOOL_COUNT_DIFFERENCE` turn=0
- failure: `FAIL_NO_TOOL_CALL -> SUCCESS_ALTERNATIVE_VALID`
- tools C0: `[]`
- tools D1: `["divide"]`
- coverage exact: `1.0` ood=`0.0`

## 9855f520-4ad7-4658-85fc-b66003e186bb
- call_bucket: `5`
- divergence: `TOOL_COUNT_DIFFERENCE` turn=0
- failure: `FAIL_WRONG_FIRST_TOOL -> SUCCESS_OTHER_OFFICIAL`
- tools C0: `["multiply", "divide"]`
- tools D1: `["multiply", "multiply", "multiply", "divide"]`
- coverage exact: `1.0` ood=`0.0`
