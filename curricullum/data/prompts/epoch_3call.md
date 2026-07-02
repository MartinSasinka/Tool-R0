Stage {epoch}: generate a task requiring exactly {num_calls} tool calls chained in a strict linear pipeline.

Chain structure (STRICT — the verifier enforces every link):
- output array contains exactly 3 calls: "$var_1", "$var_2", "$var_3".
- $var_2 MUST reference $var_1 output: use "$var_1.<field>$" in at least one argument.
- $var_3 MUST reference $var_2 output: use "$var_2.<field>$" in at least one argument.
  ($var_3 may additionally reference $var_1 output as a second input, but must reference $var_2.)
- gold_answer = result of executing all 3 calls in order.

Complexity requirement:
- Each step must genuinely depend on the previous step's numeric/string result.
- The input scenario should require exactly this 3-step reasoning chain to answer.
- Example good pattern: (1) calculate gross weight → (2) subtract tare → (3) compute shipping cost on net weight.
- Example good pattern: (1) convert speed → (2) compute distance over time → (3) compute fuel cost at given rate.

Allowed tool names: {allowed_tool_names}

Tool schemas (use these exactly; do not invent new tools):
{tool_schemas_json}
