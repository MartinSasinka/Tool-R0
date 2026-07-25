# On-policy metrics by arm

`synthetic_terminal_success_rate` is a reward-threshold PROXY (episode_reward >= 0.90 == v3_2_dense `fully_correct`, i.e. gold-trace match + final-answer pass). It is NOT the path-invariant terminal success check (`tool_final_answer_pass`).

## A0_R0_CURRENT
- reward-threshold success proxy (rollout): **0.2594**
- dead groups: **0.2812**
- mixed groups: **0.9250**
- success w/ negative advantage: **0.0**
- executable-wrong w/ positive advantage: **0.2630098452883263**

## A1_OUTCOME_ONLY
- reward-threshold success proxy (rollout): **0.2711**
- dead groups: **0.2062**
- mixed groups: **0.9313**
- success w/ negative advantage: **0.0**
- executable-wrong w/ positive advantage: **0.27886056971514245**

## A2_R3_OUTCOME_FIRST
- reward-threshold success proxy (rollout): **0.2617**
- dead groups: **0.2500**
- mixed groups: **0.9437**
- success w/ negative advantage: **0.0**
- executable-wrong w/ positive advantage: **0.28885832187070154**

## A3_VERIFIABLE_PROCESS
- reward-threshold success proxy (rollout): **0.2602**
- dead groups: **0.2250**
- mixed groups: **0.9437**
- success w/ negative advantage: **0.0**
- executable-wrong w/ positive advantage: **0.2786885245901639**

## A4_GATED_VERIFIABLE
- reward-threshold success proxy (rollout): **0.2828**
- dead groups: **0.3125**
- mixed groups: **0.9062**
- success w/ negative advantage: **0.0**
- executable-wrong w/ positive advantage: **0.27596439169139464**
