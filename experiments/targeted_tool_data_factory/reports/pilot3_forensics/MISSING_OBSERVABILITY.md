# MISSING_OBSERVABILITY

Pilot3 D1 training did not leave locally recoverable per-rollout reward groups.

## Consequences

- dead-group rate can be verified only as a train-log aggregate
- cannot determine all-success vs all-fail share for D1
- cannot identify problematic generation cells for D1 reward groups
- cannot evaluate reward ranking alignment on D1 rollouts
- this is critical missing observability for reward-bottleneck claims

## Future logging requirements (do not run now)

- Persist each GRPO group: prompt_id, sample_id, generation_cell, rollout_id, step
- Persist terminal reward T, process reward P, total R, epsilon
- Persist parse_valid / executable / official_win per rollout
- Persist unique reward count and group dead flag at write time
- Write `train_rollouts.jsonl` under the run directory before checkpointing

