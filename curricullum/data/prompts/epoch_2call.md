Stage {epoch}: generate a task requiring exactly {num_calls} tool calls chained in sequence.

Chain structure (STRICT — the verifier enforces this):
- output array contains exactly 2 calls: "$var_1" then "$var_2".
- Call $var_2 MUST reference the output of $var_1 in at least one argument.
  Use "$var_1.<field>$" where <field> is a key listed in $var_1's tool output_parameters (e.g. "result", "output_0").
- gold_answer must equal the result of executing both calls in order.

Complexity requirement:
- The two-step chain must be genuinely necessary: step 1 produces an intermediate value that step 2 consumes.
- Example good pattern: (1) convert currency → (2) apply tax to converted amount.
- Example good pattern: (1) compute area → (2) multiply by price per unit area.
- Avoid trivially constant second arguments (e.g. do not just add 0 or multiply by 1).

Allowed tool names: {allowed_tool_names}

Tool schemas (use these exactly; do not invent new tools):
{tool_schemas_json}
