# C0 / C1 / C2 Root Cause Analysis

## Executive summary

Two-phase GRPO moved official **Win Rate** from **53.52% → 53.70% (C1) → 53.94% (C2)** on nestful_test (n=1661). Net paired gain C2 vs C0 = **+7 tasks**; bootstrap 95% CI for C1−C0 includes zero → **not statistically significant** (McNemar p≈0.88).

**Root cause:** Phase 1 had **78% dead GRPO groups** (identical rewards across 8 rollouts) on 429 Stage-2 tasks — learning signal was sparse. Phase 2 improved reward contrast (31% dead) and lifted **long_chain** (+5.3pp) and **6+ call** buckets (+5.3pp), but **4–5 call** buckets regressed and Function F1 dropped −1.0pp. Training optimizes dense synthetic reward (≥0.99 = success); NESTFUL win requires IBM re-execution — gap visible in final_answer_pass (59%) vs official win (54%).

Generated: 2026-07-19T13:10:39.815347+00:00
Run dir: `C:\Users\Šunka\Documents\GitHub\Tool-R0\experiments\nestful_synthetic_curriculum_v3\outputs\runs\two_phase_20260718_192902\two_phase_20260718_192902`

## 1. Experiment verification

- Git: `0bc680e527d2684815d452d6dd976d0da37938a6` (dirty=True)
- Model: `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554`
- Registry: v5.0.2 `f945b18ccdc2…`
- C1/C2 adapter hashes differ: **True** (C1 `c42a7240596d…`, C2 `80e4c1966a90…`)
- Optimizer continuous: global_step 0→24→105, same optimizer_id, unchanged=True
- Executor: `synthetic`; reward `execution_aware_v3_2_dense`; gold_replay absent
- Eval parity (test): all arms use same eval_set + decoding per eval_manifest.json

**Gaps:** eval/C1_phase1/metrics_official.json, two_phase_state: eval_C1/eval_C2 steps not recorded (manual test evals)

## 2. C0 / C1 / C2 summary (nestful_test, n=1661)

| Metric | C0 | C1 | C2 | C1−C0 | C2−C1 | C2−C0 |
|---|---:|---:|---:|---:|---:|---:|
| Win Rate | 0.5352 | 0.5370 | 0.5394 | +0.0018 | +0.0024 | +0.0042 |
| Function F1 | 0.8980 | 0.8890 | 0.8880 | -0.0090 | -0.0010 | -0.0100 |
| Parameter F1 | 0.4390 | 0.4420 | 0.4420 | +0.0030 | +0.0000 | +0.0030 |
| Partial seq acc | 0.1820 | 0.1820 | 0.1850 | +0.0000 | +0.0030 | +0.0030 |
| Full seq acc | 0.0200 | 0.0240 | 0.0250 | +0.0040 | +0.0010 | +0.0050 |
| Executability | 0.8013 | 0.7923 | 0.7911 | -0.0090 | -0.0012 | -0.0102 |
| Under-calling | 0.6081 | 0.6117 | 0.6033 | +0.0036 | -0.0084 | -0.0048 |
| Over-calling | 0.0608 | 0.0590 | 0.0560 | -0.0018 | -0.0030 | -0.0048 |
| Final answer pass | 0.5828 | 0.5846 | 0.5900 | +0.0018 | +0.0054 | +0.0072 |
| Unsupported trace | 0.0572 | 0.0536 | 0.0602 | -0.0036 | +0.0066 | +0.0030 |
| Avg pred calls | 2.2505 | 2.2438 | 2.2529 | — | — | — |

### Win rate by expected gold calls

| Bucket | n | C0 | C1 | C2 | C2−C0 |
|---|---:|---:|---:|---:|---:|
| 2 | 543 | 0.4622 | 0.4641 | 0.4549 | -0.0074 |
| 3 | 363 | 0.5895 | 0.5868 | 0.5950 | +0.0055 |
| 4 | 223 | 0.6233 | 0.6099 | 0.5874 | -0.0359 |
| 5 | 154 | 0.5974 | 0.5974 | 0.5779 | -0.0195 |
| 6+ | 378 | 0.5106 | 0.5265 | 0.5635 | +0.0529 |

### Win rate by motif (NESTFUL gold-trace structure)

| Motif | n | C0 | C1 | C2 | C2−C0 |
|---|---:|---:|---:|---:|---:|
| linear_dependency | 850 | 0.5141 | 0.5082 | 0.5047 | -0.0094 |
| long_chain | 532 | 0.5357 | 0.5470 | 0.5677 | +0.0320 |
| fan_in | 255 | 0.6275 | 0.6353 | 0.6235 | -0.0039 |
| independent_calls | 23 | 0.2609 | 0.2609 | 0.2609 | +0.0000 |
| fan_out | 1 | 1.0000 | 1.0000 | 0.0000 | -1.0000 |

## 3. Paired task analysis (identical 1661 IDs)

**C1 vs C0:** gained 84, lost 81, net 3; Δwin=0.0018 (95% CI -0.0132 .. 0.0169); McNemar p=0.8763 (discordant 165)

**C2 vs C0:** gained 88, lost 81, net 7

**C1 gained → C2 lost:** 42 tasks
**C1 lost → C2 gained:** 83 tasks

**C2 vs C1:** gained 83, lost 79, net 4; McNemar p=0.8137

Phase 2 mostly recovers C1 regressions (83 tasks) but undoes part of C1 gains (42). Exemplars: `C0_C1_C2_exemplars.json`.

## 4. Failure taxonomy

| Failure | C0 | C1 | C2 | C2−C0 |
|---|---:|---:|---:|---:|
| correct keys, wrong argument values | 157 | 152 | 141 | -16 |
| correct tool, wrong argument keys | 41 | 42 | 42 | +1 |
| correct trajectory, wrong final answer | 41 | 38 | 36 | -5 |
| executable trajectory ending wrong result | 172 | 179 | 187 | +15 |
| no tool call | 108 | 113 | 112 | +4 |
| parse/format error | 74 | 68 | 63 | -11 |
| success | 885 | 889 | 894 | +9 |
| too few calls | 18 | 19 | 13 | -5 |
| too many calls | 2 | 1 | 1 | -1 |
| wrong tool | 163 | 160 | 172 | +9 |

Full CSV: `C0_C1_C2_failure_taxonomy.csv`. Net: +9 success, −16 wrong-arg-value, +15 executable-wrong-result.

## 5. Official scorer semantics

official_win=1 iff the extracted predicted call sequence re-executes through IBM executable_functions and the executed result equals gold_answer (strict decimal-aware equality). Alternative valid sequences CAN win while official_full_match=0.

Reference tests: `experiments/nestful_mtgrpo_minimal/tests/test_nestful_official.py`.

## 6. Reward alignment (training logs)

- **phase1:** dead_group_rate=0.7832, mixed_groups=0.2168, GRPO ordering violations=0
- **phase2:** dead_group_rate=0.3090, mixed_groups=0.6910, GRPO ordering violations=0

Training win uses reward ≥ 0.99 (`grpo_train._WIN_REWARD_THRESHOLD`). No within-group violations where success reward ≤ failure reward.

## 7. Train-to-eval transfer

- C0 dev win (200 tasks, in-run): 0.5400
- C0 test win: 0.5352 → consistent ~53.5%
- Phase1 mean training win_rate: 0.0312 (synthetic stage2)
- Phase2 stage3 mean_reward 0.625 vs stage2 replay 0.500 (from two_phase_state stage_split_metrics)

## 8. Dataset coverage vs NESTFUL test

Phase1: 429×2-call stage2 synthetic. Phase2: 326×3-call stage3 + 140×2-call replay. NESTFUL test: 543×2-call, 363×3-call, 755×4+ call tasks — long-tail underrepresented in training.

## 9. C1 vs C2

- Stage 2 (Phase1): tiny test win +0.18pp but 2-call bucket flat; high dead groups.
- Stage 3 (Phase2): 3-call +0.55pp, 6+ +5.3pp; 2-call −0.73pp vs C0 → partial forgetting.
- C2 under-calling vs C0: slightly down on average calls but 6+ bucket calls up.
- Stage3 training groups: dead 17% vs stage2 replay dead 63% — replay still starved of GRPO signal.

## 10. Decision tree

**Branch A:** Training signal weak in Phase 1 (dead groups ~78%); check reward variance / worker sync — but ordering violations=0.

## Proposed follow-up experiments (max 5)

### 1. Reduce Phase1 dead groups via adaptive task filtering
- Change: Drop Stage-2 tasks whose 8 rollouts share identical reward (dead_group) before GRPO step; backfill from stage2 pool.
- Control: Same 429 tasks but unfiltered — this run.
- Expect: dead_group_rate phase1 < 0.4; nestful_test win CI lower bound > +1pp.
- Predict: If dead groups cause null gradients, filtered run beats C2 win rate by ≥1pp with same data count.
- Stop if: dead_group_rate stays > 0.65 after filtering OR win delta ≤ this run (+0.42pp).

### 2. Increase Stage2 replay fraction in Phase 2
- Change: Phase2 mix 326 stage3 + 280 stage2 replay (vs 140) matched for steps.
- Control: Current 140 replay — C2.
- Expect: 2-call bucket win recovers vs C2 (-0.9pp regression); C1→C2 forgetting tasks shrink.
- Predict: C2_replay win on 2-call bucket ≥ C1 and net paired gain vs C0 ≥ +10 tasks.
- Stop if: 2-call win still below C0 after replay doubling.

### 3. Terminal-outcome reward ablation (dense kept but final band widened)
- Change: Map fully_correct band to [0.95,1.0] and executable_wrong_final to [0.40,0.55] — no other change.
- Control: execution_aware_v3_2_dense — this run.
- Expect: official win +1–2pp with flat f1_func; final_answer_pass closer to official_win.
- Predict: Paired gained tasks include >30% prior executable_wrong_final failures.
- Stop if: GRPO ordering violations > 0 OR dead_group_rate increases >5pp.

### 4. Stage 4/5 prefix curriculum pilot (200 tasks)
- Change: Add 200 synthetic 4–5 call linear_chain tasks to phase2 (keep total steps).
- Control: Current phase2 only stage3+replay.
- Expect: 6+ call bucket win +3pp; avg_predicted_calls on NESTFUL closer to gold on 4+ tasks.
- Predict: Motif long_chain win delta ≥ +0.03 with n≥120 tasks in bucket.
- Stop if: 6+ win unchanged AND 4-call bucket regresses >1pp.

### 5. On-policy NESTFUL dev mini-loop (50 tasks, no test leakage)
- Change: After phase1, 1 epoch GRPO on 50 held-out NESTFUL dev tasks (not in train manifest) with IBM executor.
- Control: Synthetic-only C1.
- Expect: dev official win +2pp; test win unchanged or +0.5pp (transfer probe).
- Predict: If schema gap dominates, dev improves while test flat; if pure synthetic mismatch, both flat.
- Stop if: dev win +≥2pp but test win <-0.5pp → stop (overfit dev).
