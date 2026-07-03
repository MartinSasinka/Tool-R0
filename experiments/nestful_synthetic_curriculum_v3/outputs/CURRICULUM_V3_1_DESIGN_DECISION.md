# Curriculum v3.1 Design Decision

Generated: 2026-07-03

## 1. Why we change v3 to v3.1

The v3 offline pipeline achieved strong structural quality:

- Motif coverage: 100%
- Baseline-failure motif coverage: 100%
- Gold replay: 100%
- Validation failures: 0
- Invalid references: 0

The pod pilot (`20260702_112150`) showed a small positive signal:

- Best checkpoint: **s1_e2**
- Dev Win: **0.575** vs baseline **0.555** (Δ +0.020)
- Net gain: +4 wins on dev subset (n=200)

However, the dominant failure mode **did not improve**:

- `too_few_calls`: baseline 101 vs model 102
- `long_chain too_few_calls`: 55 vs 55 (unchanged)
- Stage 2 dead_group_rate ~69% — stage2 checkpoints should not be deployed
- Tool-family realism remains **prototype_only** (math-heavy, low NESTFUL overlap)

v3 assigned tasks by motif + **max_calls** (e.g. stage1 allowed up to 2 calls), not by **exact call count**. Long-chain motifs could appear in early stages without prefix decomposition. Stage2 GRPO signal was weak because mixed replay and motif complexity collapsed learnable groups.

v3.1 refines the methodology without abandoning curriculum learning.

## 2. Why long-chain tasks must not go into stage1/2

The original experiment rests on **curriculum learning** — easy-to-hard progression by tool-call depth:

- Stage 1 teaches **atomic tool use** (exactly 1 call)
- Stage 2 teaches **one dependency** (exactly 2 calls)
- Stage 3 teaches **composition** (exactly 3 calls)
- Stage 4 teaches **persistence** (4–6 calls)

Putting 7–9 call long-chain tasks into stage1 or stage2 would:

- Destroy the easy-to-hard structure
- Teach the model to stop early on long tasks (premature final answer)
- Mix difficulty levels within a stage, breaking GRPO group comparability
- Fail to address `too_few_calls` on long chains — the model never learns intermediate continuation steps

The correct strategy: generate full NESTFUL-like trajectories from failure motifs, then **decompose into prefix subtasks** at each exact call count.

## 3. Prefix/motif-aware call-count curriculum

> Each stage preserves the exact number of tool calls, but tasks within the stage are generated as prefixes of NESTFUL failure motifs.

Example for `long_chain / too_few_calls / 7-call trajectory`:

| Stage | Sample |
|---|---|
| stage1 | 1-call atomic first-step task |
| stage2 | 2-call prefix with reference passing |
| stage3 | 3-call prefix |
| stage4 | 4–6 call shortened long-chain task |
| stage5 (gated) | full 7–9 call task only after stage4 gates pass |

Process filtering validates each prefix step. Training progresses stage-by-stage with dev gates between stages.

Inspired by step-wise / process-based approaches (e.g. SWiRL): the model learns the **correct next action** in each prefix context, not only the final answer of a full trajectory.

## 4. How v2, v3, and v3.1 differ

| version | idea | limitation | fix |
|---|---|---|---|
| v2 | N-call curriculum | too coarse, weak transfer | motif analysis |
| v3 | motif-aligned prototype | math-only, stage2 weak, no prefix decomposition | v3.1 |
| v3.1 | prefix/motif-aware call-count curriculum | still prototype until tool realism improves | stage-wise pilot |

## 5. Implementation scope

- All artifacts under `experiments/nestful_synthetic_curriculum_v3/`
- v3 dataset and scripts preserved (not destructively edited)
- Training: **NO** (build + preflight + dry-run only)
- GPU eval: **NO**
- Test split eval: **NO**
