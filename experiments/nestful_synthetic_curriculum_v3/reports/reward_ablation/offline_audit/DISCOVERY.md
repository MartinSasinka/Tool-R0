# Discovery — Round 1 offline audit

- Runs root: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\outputs\runs\_local_round1_analysis`
- Seed: `20260724`
- Arms found: **5** / 5

## Validation issues
- A1_OUTCOME_ONLY: REWARD DISPATCH MISMATCH — declared 'reward_ablation_A1_OUTCOME_ONLY' but train log shows ['execution_aware_v3_2_dense']
- A2_R3_OUTCOME_FIRST: REWARD DISPATCH MISMATCH — declared 'reward_ablation_A2_R3_OUTCOME_FIRST' but train log shows ['execution_aware_v3_2_dense']
- A3_VERIFIABLE_PROCESS: REWARD DISPATCH MISMATCH — declared 'reward_ablation_A3_VERIFIABLE_PROCESS' but train log shows ['execution_aware_v3_2_dense']
- A4_GATED_VERIFIABLE: REWARD DISPATCH MISMATCH — declared 'reward_ablation_A4_GATED_VERIFIABLE' but train log shows ['execution_aware_v3_2_dense']

## Per arm
### A0_R0_CURRENT
- Run dir: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\outputs\runs\_local_round1_analysis\reward_ablation_r1_A0_R0_CURRENT_seed20260724\reward_ablation_r1_A0_R0_CURRENT_seed20260724`
- Optimizer steps: 29
- Train log: True
- Final checkpoint: True
- Eval 500: True
- Reward dispatch OK: True (expected `execution_aware_v3_2_dense`, logged ['execution_aware_v3_2_dense'])

### A1_OUTCOME_ONLY
- Run dir: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\outputs\runs\_local_round1_analysis\reward_ablation_r1_A1_OUTCOME_ONLY_seed20260724\reward_ablation_r1_A1_OUTCOME_ONLY_seed20260724`
- Optimizer steps: 32
- Train log: True
- Final checkpoint: True
- Eval 500: True
- Reward dispatch OK: False (expected `reward_ablation_A1_OUTCOME_ONLY`, logged ['execution_aware_v3_2_dense'])

### A2_R3_OUTCOME_FIRST
- Run dir: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\outputs\runs\_local_round1_analysis\reward_ablation_r1_A2_R3_OUTCOME_FIRST_seed20260724\reward_ablation_r1_A2_R3_OUTCOME_FIRST_seed20260724`
- Optimizer steps: 30
- Train log: True
- Final checkpoint: True
- Eval 500: True
- Reward dispatch OK: False (expected `reward_ablation_A2_R3_OUTCOME_FIRST`, logged ['execution_aware_v3_2_dense'])

### A3_VERIFIABLE_PROCESS
- Run dir: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\outputs\runs\_local_round1_analysis\reward_ablation_r1_A3_VERIFIABLE_PROCESS_seed20260724\reward_ablation_r1_A3_VERIFIABLE_PROCESS_seed20260724`
- Optimizer steps: 31
- Train log: True
- Final checkpoint: True
- Eval 500: True
- Reward dispatch OK: False (expected `reward_ablation_A3_VERIFIABLE_PROCESS`, logged ['execution_aware_v3_2_dense'])

### A4_GATED_VERIFIABLE
- Run dir: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\outputs\runs\_local_round1_analysis\reward_ablation_r1_A4_GATED_VERIFIABLE_seed20260724\reward_ablation_r1_A4_GATED_VERIFIABLE_seed20260724`
- Optimizer steps: 28
- Train log: True
- Final checkpoint: True
- Eval 500: True
- Reward dispatch OK: False (expected `reward_ablation_A4_GATED_VERIFIABLE`, logged ['execution_aware_v3_2_dense'])
