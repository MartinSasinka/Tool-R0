Stage {epoch}: generate a task requiring exactly {num_calls} tool calls chained in a strict linear pipeline.

Chain structure (STRICT — the verifier enforces every consecutive link):
- output array contains exactly 4 calls: "$var_1", "$var_2", "$var_3", "$var_4".
- $var_2 MUST use "$var_1.<field>$" in at least one argument.
- $var_3 MUST use "$var_2.<field>$" in at least one argument.
- $var_4 MUST use "$var_3.<field>$" in at least one argument.
  (Earlier outputs may also be reused as secondary inputs, but the consecutive link is mandatory.)
- gold_answer = result of the full 4-step execution.

Complexity requirement:
- Design a multi-stage real-world computation where each intermediate result is meaningfully used in the next step.
- The "input" scenario must require all 4 steps to arrive at the answer — no shortcut.
- Example good pattern: (1) convert raw material quantity → (2) compute batch cost → (3) apply markup → (4) add VAT to get final price.
- Example good pattern: (1) compute base salary → (2) apply overtime multiplier → (3) subtract tax → (4) add bonus → net pay.
- Example good pattern: (1) lookup distance in km → (2) convert to miles → (3) compute travel time → (4) compute fuel consumption at given rate.

Allowed tool names: {allowed_tool_names}

Tool schemas (use these exactly; do not invent new tools):
{tool_schemas_json}
