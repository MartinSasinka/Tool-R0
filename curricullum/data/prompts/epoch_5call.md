Stage {epoch}: generate a task requiring exactly {num_calls} tool calls chained in a strict linear pipeline.

Chain structure (STRICT — the verifier enforces every consecutive link):
- output array contains exactly 5 calls: "$var_1", "$var_2", "$var_3", "$var_4", "$var_5".
- Each call i (for i=2..5) MUST reference call (i-1) output in at least one argument, e.g. "$var_2" uses "$var_1.<field>$", "$var_3" uses "$var_2.<field>$", etc.
  In other words: var_2→var_1, var_3→var_2, var_4→var_3, var_5→var_4.
  (A call may also reference any earlier var as a secondary input, but the direct predecessor link is mandatory.)
- gold_answer = result of the full 5-step execution.

Complexity requirement:
- Model a multi-stage business, scientific, or logistical calculation where intermediate results feed forward.
- The input question must be realistic, domain-specific, and require all 5 steps.
- Keep each individual tool call simple (basic arithmetic/conversion/lookup), but the composition must be non-trivial.
- Example good pattern: (1) compute raw volume → (2) convert units → (3) apply density to get mass → (4) deduct waste fraction → (5) compute cost at price-per-kg.
- Example good pattern: (1) compute loan principal → (2) apply monthly rate → (3) compute interest for period → (4) add origination fee → (5) compute monthly payment.

Allowed tool names: {allowed_tool_names}

Tool schemas (use these exactly; do not invent new tools):
{tool_schemas_json}
