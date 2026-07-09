# REWARD

Full analysis: `audits/REWARD_AUDIT.md`. Policy modules live in `lib/` and are selected by
the `REWARD_POLICY` config; `lib/reward_v3_1.py` is the frozen audited baseline — new
policies go in new modules (e.g. `reward_v3_2_dense.py`, planned P1) so A/B stays possible.

## Active policy: `execution_aware_v3_1_stepwise`

Episode reward composed from stepwise execution-aware turn scores with band caps/floors:

| condition | reward |
|---|---|
| fully correct trace + final answer | 1.0 |
| executable, right answer, non-gold trace | ≤ ~0.85 band |
| wrong argument binding (right tools) | capped at 0.60–0.65 |
| too few calls / premature answer | capped at 0.25–0.30 |
| parse error / no tool call | floor ~0.05–0.10 |
| invalid reference syntax | low band |

Fractional stepwise credit exists *within* turns (17 unique reward values observed live in
the Stage-3 run), but the dominant behavior modes still collapse into a few flat bands.

## The core problem (why GRPO starves)

GRPO's advantage is group-relative: no reward **difference within a group of 8 completions**
means zero gradient from that group. Audited dead-group rates: Stage 1 ≈ 1.0 (fully
saturated), Stage 2 0.65–0.88, Stage 3 0.65–0.71. Typical group has 1.1–1.3 unique reward
values out of 8. Root cause is the band structure: behaviorally different completions (e.g.
different wrong arguments; stopping after call 1 vs emitting no calls) receive identical
rewards. Additionally 0.35–0.41 of alive Stage-3 groups get variance only from turn position
(position artifact), training on noise.

## Historical policies (for reading old runs)

| policy | used by | character |
|---|---|---|
| `strict_gold_trace` | July-2/3 v3-era runs | binary 0/1 exact-trace — extreme sparsity |
| `partial_gold_trace` | mtgrpo_partial experiments | graded gold-trace prefix credit |
| `execution_aware_v2_1_motif` | v3 motif pilots | motif consistency; its `r_seq` is all zeros (turn credit never fired) |
| `execution_aware_v3_1_stepwise` | all v3.1 runs (July 7–8) | current, described above |

## Planned: `execution_aware_v3_2_dense` (experiment E1, not implemented)

Densify within-band credit without changing band ceilings: per-correct-call fraction inside
the too-few band, per-correct-argument fraction inside the wrong-args band, a nonzero step
between "no calls" and "one correct call". Ordering guarantee kept: an incomplete trace can
never outscore a complete correct one. Gate before any GRPO time: stage-probe dead-group
rate < 0.5 and ≥ 2 unique episode rewards per group on average (RESEARCH_FIX_PLAN E1).

## Rules for reward changes

1. New module + new policy name; never edit `reward_v3_1.py` in place.
2. Unit tests on crafted trajectories covering every band boundary.
3. Probe before train: reward-band histogram + predicted dead-group rate on ≤100 tasks.
4. `grpo_train._verify_reward_dispatch` must pass (the `.reward_policy` attribute check).
5. Reward values are a training signal, not an evaluation — improvements are claimed only
   from same-batch official win (docs/EVALUATION.md).
