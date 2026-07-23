# Reward variant audit (deterministic, eval trajectories)

**Generated:** 2026-07-23T11:16:02.029184+00:00
**Run:** `pure_stage3_2ep_20260719_221918`
**Tasks:** 1661 (pseudo-group = C0/E1/E2 per task, n=1661)
**Policy:** R0=`execution_aware_v3_2_dense`, R2 ε=0.05, R3 ε=0.04

No LLM. Official Nestful win is authority for outcome variants.

## 1. Official wins recognized by reward

| Variant | Official wins | Recognized | Rate | Mislabeled | Too-few on win (R0 only) |
|---------|--------------:|-----------:|-----:|-----------:|-------------------------:|
| R0 | 2680 | 1584 | 59.1% | 1096 | 1584 |
| R1 | 2680 | 2680 | 100.0% | 0 | 0 |
| R2 | 2680 | 2680 | 100.0% | 0 | 0 |
| R3 | 2680 | 2680 | 100.0% | 0 | 0 |

## 2. C0-win / E2-loss ordering (reward prefers loser?)

| Variant | Cohort n | Wrong (E2>C0) | Correct | Tie | Fixed vs R0 |
|---------|--------:|--------------:|--------:|----:|------------:|
| R0 | 93 | 22 | 64 | 7 | 0 |
| R1 | 93 | 0 | 93 | 0 | 22 |
| R2 | 93 | 0 | 93 | 0 | 22 |
| R3 | 93 | 0 | 93 | 0 | 22 |

## 3. Pseudo-group dead rate & advantages (C0/E1/E2)

| Variant | Dead group rate | Δ vs R0 | Mean |adv| E2 | E2 adv sign flip vs R0 |
|---------|----------------:|--------:|---------------:|------------------------:|
| R0 | 63.0% | +0.0% | 0.348 | — |
| R1 | 79.0% | +16.0% | 0.198 | 198 |
| R2 | 62.6% | -0.4% | 0.351 | 110 |
| R3 | 62.6% | -0.4% | 0.352 | 117 |

## 4. Executable-wrong high reward (trajectory-level)

| Variant | Count (reward too high for wrong executable outcome) |
|---------|-----------------------------------------------------:|
| R0 | 367 |
| R1 | 1182 |
| R2 | 1182 |
| R3 | 0 |

## Interpretation (deterministic only)

- **R1** fixes C0/E2 ordering and official-win labeling but **increases dead groups** sharply (outcome bands collapse many pseudo-groups).
- **R2/R3** keep dead-group rate near R0 while fixing C0-win/E2-loss wrong ordering.
- **R3** additionally separates executable-wrong from official-success (see section 4).
- Advantage sign flips vs R0 measure how much GRPO credit would shift on E2 under each variant.

## Variant definitions

- **R0**: Current (execution_aware_v3_2_dense)
- **R1**: Outcome-only
- **R2**: Outcome-first + ε process (0.05)
- **R3**: Outcome-first + wider executable_wrong gap (0.04 ε)