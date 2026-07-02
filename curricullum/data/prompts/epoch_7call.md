Stage {epoch}: generate a task requiring exactly {num_calls} tool calls forming a computation DAG.

Chain structure (DAG mode — verifier checks that EVERY call has at least one dependency on a prior call):
- output array contains exactly 7 calls: "$var_1" through "$var_7".
- $var_1 has no dependencies (uses only literal argument values).
- Each of $var_2 through $var_7 MUST reference at least one earlier "$var_K.<field>$" (K < current index).
- The DAG must have at least two fan-in points (calls that combine results of two or more earlier calls).
- Minimum required depth: 4 levels (i.e. at least one execution path of length 4 from $var_1 to $var_7).
- gold_answer = result of the final call ($var_7) after full execution.

Complexity requirement:
- Design a scenario with three sub-computations that progressively combine into a final answer.
- Each sub-chain should represent a meaningful independent quantity (e.g. three cost components, three revenue streams).
- Example good structure:
    Branch A: (1) raw units → (2) unit price applied → subtotal_A
    Branch B: (3) hours worked → (4) hourly rate applied → subtotal_B
    Branch C: (5) overhead rate → (6) overhead total [uses var_3 or var_5]
    Merge: (7) grand total [fan-in: var_2 + var_4 + var_6]
- Example good structure:
    (1) base conversion → (2) step 2 on A → (3) step 3 on A
    (4) independent quantity B → (5) transform B
    (6) combine A-result and B-result → (7) final adjustment on (6)
- The "input" should be a realistic multi-component business or engineering problem requiring all branches.

Allowed tool names: {allowed_tool_names}

Tool schemas (use these exactly; do not invent new tools):
{tool_schemas_json}
