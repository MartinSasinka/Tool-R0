# Next Dataset Improvement Plan (v3.1)

## A. Motif priorities

| priority | motif/failure cluster | evidence | proposed generation change | expected effect |
|---:|---|---|---|---|
| 1 | long_chain / too_few_calls | baseline cluster n=29; eval-stage3 strict ~0.17 | +200 long_chain tasks (7–9 calls); stage2 oversample | improve 3-call NESTFUL Win |
| 2 | linear_dependency / too_few_calls | cluster n=43; s1 dev Win +2pp | keep stage1 linear 2–3 call; add IBM-like tool names | preserve s1 gain |
| 3 | fan_in / wrong_argument | cluster n=12; stage3 not run | fan_in with numeric refs + distractors | reduce fan_in regression |
| 4 | reference_reuse / invalid_reference | stage2 synthetic focus | cross-stage ref chains with validation | cut invalid_reference failures |
| 5 | object/list output | stage2 motif 68 tasks | nested field extraction answers | output type transfer |
| 6 | independent_calls | missing in v3 | new generator template | cover 1.5% NESTFUL share |
| 7 | distractor_tools / wrong_tool | stage4 not run | IBM tool pool sampling | tool selection robustness |

## B. Stage balance (proposed counts)

| stage | current | proposed v3.1 | rationale |
|---|---:|---:|---|
| stage1 linear | 417 | 400 | OK; slight trim |
| stage2 reference | 223 | **350** | thicker + more long_chain/ref reuse |
| stage3 structural | 119 | **250** | fan_in/fan_out — hold until s1–2 beat baseline consistently |
| stage4 mixed | 271 | 300 | distractor + baseline_failure_inspired |

**Gate:** Do not train stage 3/4 until best dev Win ≥ baseline + 0.5pp for 2 consecutive epochs.

## C. Tool/output realism

- Add non-math tool family stubs mirroring IBM name patterns (string ops, list ops).
- Target 40% non-scalar outputs in stage2+.
- Increase tool bigram overlap with NESTFUL (mine from nestful_tool_sequence_motifs.csv).
- Build `nestful_like_tool_registry.json` — map 20 IBM families to synthetic implementations.

## D. Reward changes

- Raise cap penalty for `too_few_calls_without_final` (0.20 → 0.10 max reward).
- Increase `tool_use_completeness` weight 0.10 → 0.15.
- Add explicit `w_final` floor only when num_calls ≥ gold_n - 1.
- Log all reward components to W&B each step.

## E. Sampling

- Oversample baseline-failure motifs (linear/long_chain too_few_calls) 2× in stage1–2.
- Undersample independent_calls until generator exists.
- Curriculum replay stage2: reduce stage1 weight 0.35 → 0.25 after baseline beat.

## F. Evaluation

- Mandatory per-sample dev trajectories saved each val_eval.
- Run motif_level_eval.py automatically post-training.
- Track overlap CSV vs baseline every epoch.
- Report dead_group_rate gate (<50%) before stage advance.
