# SAMPLER_SIMULATION

- response model: `synthetic_difficulty_model`
- caveat: synthetic difficulty model: exercises the sampler only, it is not evidence about the trained policy
- dataset: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\targeted_tool_data_factory\outputs\pilot4_profile_safe\train.jsonl` (600 prompts)
- steps: 200, group size: 8, seed: 0

No rollout, no model and no GPU were used to produce this table.

| sampler | dead-group rate before filter | effective-group rate after filter | rollout utilisation | refill rounds | entropy |
|---|---|---|---|---|---|
| `uniform` | 0.0361 | 0.9639 | 0.9639 | 1.005 | 1.0 |
| `dynamic_effective_group` | 0.0372 | 0.9628 | 0.9628 | 1.01 | 0.998805 |
| `history_adaptive` | 0.0322 | 0.9678 | 0.9678 | 1.01 | 0.994196 |
| `cell_curriculum` | 0.0302 | 0.9698 | 0.9698 | 1.01 | 0.995081 |

## Curriculum states at the end of the simulation

- `cell_curriculum`: {'ACTIVE': 113, 'PROBING': 43}
