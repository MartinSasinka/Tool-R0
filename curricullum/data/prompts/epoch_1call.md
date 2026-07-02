Stage {epoch}: generate a task requiring exactly {num_calls} tool call.

Chain structure:
- output array contains exactly 1 call with label "$var_1".
- gold_answer must equal the result of executing that single call.

Complexity requirement:
- Pick a tool that accepts at least one numeric argument and produces a numeric/string result.
- The "input" must be a scenario question where the answer is NOT obvious without running the tool (e.g. a unit conversion, a lookup with non-round numbers, a mathematical operation on domain-specific quantities).
- Avoid trivial inputs like "what is 2 + 2" or single-digit arithmetic.

Allowed tool names: {allowed_tool_names}

Tool schemas (use these exactly; do not invent new tools):
{tool_schemas_json}
