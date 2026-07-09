# FAILURE-MODE AUDIT — GRPO signal quality per stage

Date: 2026-07-09 · Read-only audit. Numbers computed from `train_log.jsonl` group rows by
`audits/tools/run_audit.py` (full per-epoch detail in `RUN_AUDIT.json` under
`train_log_analysis`).

Group-row semantics (from the trainer's own logging): `dead_group` = zero between-completion
reward std (corrected definition; a group can be "mixed" across turn positions yet dead);
`group_mixed` = more than one distinct reward value appears anywhere in the group;
`dead_group_old_flattened` = the pre-fix flattened-turn definition. `n_gen` = 4 (July 2–3
runs) or 8 (July 7–8 runs).

## 1. Signal-quality table

| condition | run / stage / epoch | groups | dead rate | mixed rate | all-zero | all-one | pos-artifact | uniq rewards/group | uniq completions/group | reward entropy (bits) | mean reward |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Stage 1 (old data, motif reward) | 112150 s1 e1 | 417 | 0.389 | 0.000 | 0.42 | 0.58 | 0 | 1.0* | n/a* | 0.98 | 0.578 |
| Stage 2 (old data, motif reward) | 112150 s2 e1 | 640 | 0.686 | 0.003 | 0.73 | 0.27 | 0 | 1.0* | n/a* | 0.84 | 0.269 |
| Stage 1 (v3.1 data, stepwise reward — but binary era) | 0260703 s1 e1 | 800 | **1.000** | 0.000 | 0.765 | 0.235 | 0 | 1.0* | n/a* | 0.79 | 0.235 |
| Stage 2 (v3.1, binary era) | 0260703 s2 e1 | 1600 | 0.778 | 0.001 | 0.69 | 0.31 | 0 | 1.0* | n/a* | 0.89 | 0.310 |
| **Stage 2 + mixed replay 0.2** | 103035 s2 e1 | 1600 | 0.876 | 0.906 | 0.00 | 0.094 | 0.276 | 1.09 | 4.36 | 2.16 | 0.525 |
| **Stage 2 no replay** | 152750 s2 e1 | 800 | 0.855 | 0.939 | 0.00 | 0.061 | 0.344 | 1.10 | 4.57 | 1.67 | 0.469 |
| **Stage 2 teacher-forced (prefix=1)** | 183801 s2 e1 | 800 | 0.844 | 0.779 | 0.00 | 0.221 | 0.048 | 1.11 | 3.50 | 1.75 | 0.541 |
| **Stage 3 no replay** | 212347 s3 e1 | 800 | 0.709 | 0.958 | 0.00 | 0.043 | 0.413 | 1.26 | 4.52 | 2.01 | 0.487 |
| **Stage 3 no replay** | 212347 s3 e2 | 800 | 0.646 | 0.970 | 0.00 | 0.030 | 0.351 | 1.33 | 7.06 | 1.99 | 0.490 |

\* July 2–3 logs predate the richer schema; rewards there are strictly binary {0,1} per
episode and per-group diagnostics (unique rewards, completion hashes) were not logged.

Fine-grained error rates (per episode, July 7–8 runs only):

| condition | too_few_calls | no_tool_call | parse_error | wrong_tool | wrong_arg | invalid_ref | premature_final | avg pred calls (gold) |
|---|---|---|---|---|---|---|---|---|
| s2 replay | 0.438 | 0.002 | 0.002 | 0.436 | 0.287 | 0.000 | 0.000 | 1.36 (2) |
| s2 no replay | 0.558 | 0.003 | 0.004 | 0.555 | 0.344 | 0.000 | 0.000 | 1.44 (2) |
| s2 teacher-forced | 0.555 | 0.001 | 0.001 | 0.557 | 0.168 | 0.000 | 0.000 | 1.44 (2) |
| s3 e1 | 0.418 | 0.032 | 0.032 | 0.387 | 0.479 | 0.000 | 0.001 | 2.18 (3) |
| s3 e2 | 0.414 | 0.025 | 0.026 | 0.390 | 0.482 | 0.000 | 0.001 | 2.20 (3) |

Motif-level dead-group hotspots (dead rate, mean reward):

- **string_output**: dead 0.75–1.00 with mean reward stuck at ~0.24–0.34 across all
  conditions — the model consistently produces the same wrong string handling; reward gives
  every completion the identical partial band.
- **boolean_output** (s3): dead ~0.98–1.00 at mean 0.25 — same collapse.
- **reference_reuse**: dead 0.85–1.00 at mean ~0.59–0.65 — model reliably does the first hop
  and reliably misses reuse, all 8 completions land in the same band.
- Healthiest motifs: distractor_tools, argument_transformation, linear_dependency (dead
  0.45–0.78 with genuinely mixed groups).

## 2. Interpretation against the candidate hypotheses

| hypothesis | verdict | evidence |
|---|---|---|
| Too little reward variance (dead groups) | **YES — primary** | 65–88% of groups have zero between-completion reward std even with 8 generations at T=1.0. GRPO advantage is zero there; effective batch is 12–35% of nominal. Note the paradox: `mixed_rate` is 0.78–0.97 (turn positions get different rewards) yet `dead` stays high — the *episode-level* rewards within a group are identical because different completions land in the same band. |
| Too little action diversity | **PARTLY** | 3.5–7.1 unique completions per 8-sample group (so text diversity exists), but unique *episode rewards* per group is only 1.09–1.33. Diverse text maps to identical reward bands → reward quantization, not sampling, is the bottleneck. |
| Premature final answer | NO | premature_final_rate ≤ 0.001 everywhere. |
| Invalid reference syntax | NO | invalid_ref_rate = 0.000 in all logged runs. |
| Wrong tool selection | **YES — dominant error** | wrong_tool 0.39–0.56 of episodes; correlates with too_few_calls (model answers directly instead of calling the second/third tool — `too_few_calls` and `wrong_tool` co-occur: stopping early scores as both). |
| Wrong argument binding | **YES — second error** | wrong_arg 0.17–0.48; grows with call depth (s3 > s2). Teacher forcing the first call halves wrong_arg (0.34→0.17) but does not raise dev win. |
| Synthetic ↔ NESTFUL mismatch | **YES — transfer gap** | Stage-2 training reward climbs (mean 0.47–0.54) and synthetic-stage metrics improve, but dev win never moves. Synthetic tools are ~20 simple math/string functions with `arg_0/arg_1` schemas; NESTFUL has hundreds of heterogeneous APIs. Skill learned ("call add then subtract with $var refs") is already at ceiling for the base model on NESTFUL-easy and doesn't touch NESTFUL-hard failure modes. |
| Metric/scorer mismatch | YES (reporting, not training) | internal win overstates official win by 6–7 pp systematically (see METRIC_AUDIT). Does not affect gradient, but inflated internal numbers repeatedly suggested progress that the official scorer does not confirm. |
| Decoding instability | MINOR | temp0 rerun lowered all cells ~1 pp uniformly; ranking mostly preserved. Sampled-vs-temp0 batches were nevertheless mixed in earlier conclusions. |
| Replay contamination | NO (but replay ≠ fix) | replay run's train set is 20% earlier-stage tasks by design; no id leakage into eval. Replay ablation shows nearly identical dev outcome (.535 vs .545 vs .515 official) — replay is not the differentiator. |
| Stage progression issue | **YES** | Stage 1 was fully saturated in `0260703` (dead rate 1.0, 0 optimizer steps — the model already solves 1-call tasks or fails them deterministically), i.e. Stage 1 contributes nothing. Stage 3 run failed its own position-artifact gate (0.35–0.41 of groups show reward differences only across turn positions, not between completions). |

## 3. Root-cause summary (ranked)

1. **Reward quantization + high dead-group rate**: the v3.1 stepwise reward maps most
   completions of a task into one of 2–3 coarse bands (0.25/0.3, 0.6/0.65, 1.0 dominate the
   histograms), so even diverse completions get identical episode rewards → zero GRPO
   advantage on 65–88% of groups.
2. **Curriculum-to-NESTFUL transfer gap**: what the reward does teach (synthetic 2–3-call
   motifs over a toy tool registry) is disjoint from where NESTFUL loses wins (long chains,
   heterogeneous real APIs, argument formats). Zero data overlap is good for contamination
   but the task distributions are also disjoint in tools, phrasing and answer types.
3. **Dominant behavioral failure is under-calling** (too_few_calls 0.41–0.56, avg predicted
   calls ~1.4 of 2 / ~2.2 of 3): the base model prefers answering directly; the partial-band
   reward still pays ~0.3–0.6 for that, so the gradient toward "make the second call" is weak
   relative to band width.
4. Secondary hygiene: internal-vs-official metric inflation, missing same-batch baselines,
   stage-1 saturation, position-artifact groups in stage 3.
