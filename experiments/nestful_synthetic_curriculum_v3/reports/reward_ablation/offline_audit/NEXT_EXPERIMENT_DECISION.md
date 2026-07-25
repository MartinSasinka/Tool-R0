# Next experiment decision

**Verdict:** `REWARD_DISPATCH_BUG`

## Reasons
- Declared reward policy did not run:
- A1_OUTCOME_ONLY: declared 'reward_ablation_A1_OUTCOME_ONLY', train log ran ['execution_aware_v3_2_dense']
- A2_R3_OUTCOME_FIRST: declared 'reward_ablation_A2_R3_OUTCOME_FIRST', train log ran ['execution_aware_v3_2_dense']
- A3_VERIFIABLE_PROCESS: declared 'reward_ablation_A3_VERIFIABLE_PROCESS', train log ran ['execution_aware_v3_2_dense']
- A4_GATED_VERIFIABLE: declared 'reward_ablation_A4_GATED_VERIFIABLE', train log ran ['execution_aware_v3_2_dense']

**Recommended:** Fix reward dispatch (config policy must win over REWARD_POLICY env default), then re-run the ablation arms.

## Do not run now
- Any cross-arm reward conclusion from this round