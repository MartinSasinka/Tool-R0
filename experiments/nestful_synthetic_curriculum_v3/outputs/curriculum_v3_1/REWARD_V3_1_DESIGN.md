# Reward v3.1 Stepwise Design

## Why we change the reward

The v3 pilot used `execution_aware_v2_1_motif`, which weights `tool_final_answer_pass` at 30%. This design:

- Rewards correct final answers even when the trace is incomplete
- Does not penalize premature final answers in non-terminal prefix contexts
- Failed to reduce `too_few_calls` (101 → 102 on dev)
- Left `long_chain too_few_calls` unchanged (55 vs 55)

Stage2 dead_group_rate ~69% suggests the reward did not provide strong continuation signal for 2-call dependency learning.

## v3.1 step-wise process curriculum

v3.1 trains on **prefix samples** derived from full NESTFUL failure motifs. Each sample has:

- `stage` — exact call-count stage
- `terminal_stage` — whether the prefix is complete for that stage
- `prefix_of_motif` — derived from a longer failure trajectory

The reward must evaluate the **correct next action** in the prefix context, not dominate on final answer for non-terminal prefixes.

## Stage-specific weighting

| Stage | Focus |
|---|---|
| stage1 | Atomic tool invocation, argument validity, tool selection |
| stage2 | Reference passing, continuation after first observation |
| stage3 | Multi-step composition, valid references, motif consistency |
| stage4 | Longer persistence, tool use completeness, final answer |

## Hard caps

| Cap | Value | Rationale |
|---|---|---|
| `premature_final_nonterminal` | 0.0 | Non-terminal prefix must not emit final answer |
| `too_few_calls` | 0.1 | Hard penalty for stopping early |
| `no_tool_call` | 0.0 | Must use tools |
| `invalid_reference` | 0.1 | Broken dependency chain |

## Floors

- `executable_complete_prefix`: 0.75 — reward completing the prefix correctly
- `executable_complete_prefix_with_valid_refs`: 0.85 — stage2+ with valid references

## Runtime wiring

- Default for `CURRICULUM_VERSION=v3_1`: `execution_aware_v3_1_stepwise`
- Fallback: `REWARD_NAME=execution_aware_v2_1_motif` or config override
- Stage detection: `sample.stage` → `train_stage` metadata → epoch mapping

Config source: [`configs/reward_v3_1_stepwise.yaml`](../../configs/reward_v3_1_stepwise.yaml)
