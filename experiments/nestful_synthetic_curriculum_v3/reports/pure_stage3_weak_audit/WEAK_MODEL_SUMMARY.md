# Weak model summary

## 1. Deterministic facts (from eval artifacts)

- Selected tasks: 248
- C0 win -> E2 loss cohort: 93

## 2. Majority weak-model annotations (Pass A)

- valid_shorter_path: 79
- initial_tool_selection: 47
- argument_values: 40
- premature_stop: 23
- wrong_final_answer: 18
- later_tool_selection: 13
- wrong_output_field: 7
- reward_mismatch: 6
- executable_wrong_global_plan: 3
- unclear: 1

## 3. Unstable / unclear (Pass A vs B disagreement)

- Root cause changed: 0.49361702127659574
- Reward ordering changed: 0.30638297872340425

## 4. Limitations

- Weak-model labels are not verified root causes.
- Pass B anonymization may shift categorical answers.
- Missing annotations treated as absent in aggregates.