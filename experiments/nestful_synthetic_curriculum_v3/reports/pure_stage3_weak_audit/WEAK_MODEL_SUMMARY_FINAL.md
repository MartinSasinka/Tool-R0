# Weak model summary final

- Pass A: 248, Pass B: 248, both: 248

## Root causes (Pass A)

- valid_shorter_path: 82
- initial_tool_selection: 47
- argument_values: 43
- premature_stop: 23
- wrong_final_answer: 19
- later_tool_selection: 14
- wrong_output_field: 7
- reward_mismatch: 6
- executable_wrong_global_plan: 3
- observation_ignored: 2

## Limitations

- Weak annotations are hypotheses, not ground truth.
- first_divergence_turn is relatively more stable than root_cause.