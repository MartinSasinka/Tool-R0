Stage {epoch}: generate a task requiring exactly {num_calls} tool calls forming a computation DAG.

Chain structure (DAG mode — verifier checks that EVERY call has at least one dependency on a prior call):
- output array contains exactly 6 calls: "$var_1" through "$var_6".
- $var_1 has no dependencies (uses only literal argument values).
- Each of $var_2 through $var_6 MUST reference at least one earlier "$var_K.<field>$" (K < current index).
- You may use fan-in patterns (a later call combines outputs of two earlier calls) or a linear pipeline — both are valid.
- Recommended structure: at least one call combines two earlier outputs (fan-in), ensuring the DAG has depth ≥ 4.
- gold_answer = result of the final call ($var_6) after full execution.

Complexity requirement:
- Design a scenario with two parallel sub-chains that eventually merge into a final computation.
- Example good pattern:
    (1) compute material cost A → (2) compute labor cost A → (3) total cost A [fan-in: 1+2]
    (4) compute material cost B → (5) compute overhead [uses var_4] → (6) combined total [fan-in: 3+5]
- Example good pattern:
    (1) gross revenue → (2) cost of goods sold → (3) gross profit [1-2]
    (4) operating expenses → (5) depreciation [uses var_4] → (6) net income [3-5]
- The "input" should read as a plausible finance/logistics/engineering question requiring multi-branch reasoning.

Allowed tool names: {allowed_tool_names}

Tool schemas (use these exactly; do not invent new tools):
{tool_schemas_json}
