# FULL PIPELINE AUDIT — NESTFUL / MT-GRPO

**Goal:** determine why strict/minimal and partial-reward training did not improve the baseline, and localize the cause across reward, data, training, parser, executor, evaluation, prompt, checkpoint selection and training dynamics.

**Scope / constraints honored:** no new long training run; no automatic code fixes; no large/expensive re-evaluations (only an offline n=150 gold-replay sanity check, parser/reward unit cases, and re-aggregation of *existing* logs/trajectories). Every claim links to a file/function/log/metric/example.

**Audit commit:** `fd222c7` (upstream "initial code added"; the `experiments/` tree is untracked working state — see Limitations). Full machine-readable state in `audit_manifest.json`.

**Headline result (official NESTFUL scorer, ReAct, n=1861):**

| model | ReAct Win | ReAct F1 Func | Direct Win |
|---|---|---|---|
| **baseline (Qwen3-4B, no LoRA)** | **0.544** | 0.894 | 0.292 |
| partial s1_e4 (best ckpt) | 0.543 | 0.926 | 0.274 |
| partial s2_e2 | 0.533 | 0.808 | 0.272 |
| partial s3_e2 | 0.479 | 0.472 | 0.268 |
| partial s4_e1 | 0.450 | 0.350 | 0.274 |
| minimal (strict) s4e2 | 0.325 | 0.153 | 0.243 |

> No trained checkpoint beats the untrained baseline on ReAct Win, and performance **degrades monotonically as the curriculum advances**. Source: `final_eval_all.csv`, `diagnostics.csv`, per-run `metrics_official.json`.

---

## Pipeline map

```
raw synthetic (Tool-R0)  +  raw NESTFUL (eval)
  → data.py load_tasks (normalize_task; filter num_calls==stage)
  → prompt.py build_messages (SYSTEM_PROMPT)               [TRAIN prompt]
  → grpo_train._rollout_episode_for_train (vLLM/HF gen)    [rollout generation]
  → parser.parse_tool_call(lenient=False)                  [TRAIN parser, strict]
  → executor.ToolExecutor.execute (mode=full)              [executor observations]
  → reward: strict_gold_trace_reward / partial_gold_trace_reward / execution_aware_reward
            via episode_turn_reward_seq                     [reward computation]
  → grpo_train.train: turn-level returns G_t, group-relative advantage, KL(k3) [GRPO update]
  → model.save_pretrained adapter_epoch_N                   [checkpoint]
  → run_curriculum.sh carry best-by-strict_pass to next stage
  → prompt.build_messages(eval_hardening=True)             [EVAL prompt + _EVAL_HARDENING]
  → rollout.run_episode (parser lenient=True, executor full)[ReAct/Direct eval loop]
  → nestful_official_score (data/NESTFUL-main/src/scorer.py)[official scorer]
  → metrics_official.json (F1 Func/Param, Part/Full Acc, Win)[final metrics]
```

| step | file:function | input → output |
|---|---|---|
| data prep / stage split | `data.py:load_tasks` (filter `num_calls==stage`) | JSONL → normalized tasks |
| curriculum orchestration | `run_curriculum.sh` (per-stage, multi-epoch loop) | stage files → checkpoints + summaries |
| train config | `config.yaml` (minimal/partial) | hyperparameters |
| rollout (train) | `grpo_train.py:_rollout_episode_for_train` | task → Episode + TurnTokens |
| parser | `parser.py:parse_tool_call` (strict train / lenient eval) | text → ParseResult |
| executor | `executor.py:ToolExecutor.execute` | call → observation/error |
| reward | `reward.py` / `partial_reward.py` / `execution_reward.py` | trajectory+gold → R, r_seq |
| GRPO update | `grpo_train.py:train` (`_turn_returns`, advantage, KL) | rewards → optimizer step |
| checkpoint | `grpo_train.py` `save_pretrained` + `_write_checkpoint_sidecars` | adapter_epoch_N |
| best-ckpt decision | `run_curriculum.sh:762-773,955-957` (best strict_pass; carry forward) | per-stage best |
| eval loop | `rollout.py:run_episode`, `direct_eval.py` | task → Trajectory |
| official scorer | `nestful_official_score.py` → `scorer.py:calculate_win_score/calculate_ans` | items → paper metrics |
| metric aggregation | `metrics.py`, comparison CSVs | metrics_official.json, final_eval_all.csv |

Metrics origin: per-epoch `eval/metrics.json` (`strict_gold_trace_pass`, `clipped_completion_rate`); per-task `train_log.jsonl` (mean_reward, dead/mixed group, etc.); final `metrics_official.json` (canonical paper metrics). Checkpoints saved under `outputs/curriculum/stage_N/checkpoints/adapter_epoch_E`.

---

## Data audit  (see `data_audit_summary.csv`)

- Synthetic train: stages 1–6 = 400/400/400/400/400/268 = **2268 tasks**, each file is exactly N gold calls (no contamination). 0 empty gold, 0 malformed.
- NESTFUL eval: **1861 tasks**, num_calls 2–53 (2-call=609, 3-call=407 → 55% are 2–3 calls; **no 1-call tasks**). All gold tool names + arg keys valid vs schema.
- **Gold-trace replay sanity (KEY): Win=1.000, executable=1.000, full_match=1.000, parse_valid=1.000 on n=150** (`_audit_gold_replay.py`). → Data + executor + official scorer are internally consistent; the benchmark is sound. **PASS.**
- **Train/eval domain mismatch (WARNING):** 0/1861 question overlap; train = synthetic Tool-R0, eval = real NESTFUL. Train depth ≤6 calls; 45% of eval is >3 calls. No leakage, but a genuine distribution shift.

## Prompt audit  (see `prompt_mismatch_audit_summary.csv`)

- No gold-trace / gold-answer leak: user content = question + tool schemas only (`prompt.py:build_user_content`); ReAct history = the model's own turns + real executor observations (`rollout.py`).
- `[]` terminal is gated ("only when the most recent observation IS the final answer", "never on the first turn") → no trivial no-tool shortcut.
- **Train vs eval mismatch (WARNING, format-compatible):** train = `SYSTEM_PROMPT` + strict parser + `max_turns=gold_n`; eval = `SYSTEM_PROMPT + _EVAL_HARDENING` + lenient parser + `gold_n+1` turns. Baseline and checkpoints get the *same* eval prompt, so cross-model comparison is fair; but the policy is optimized under slightly different conditions than it is scored.

## Parser audit  (see `parser_executor_audit_summary.csv`)

All 10 cases behave correctly: strict gate rejects invalid JSON / missing name / non-dict args / multiple tags; `[]` → terminal; lenient recovers mangled close tags and bare arrays; round-trip parsed→serialized→parsed holds for valid calls. **PASS.** The only flag is the deliberate train(strict)/eval(lenient) asymmetry (also in the prompt finding).

## Executor and ReAct loop audit  (see `parser_executor_audit_summary.csv`)

- Observation is computed from the **predicted** call (`rollout.py:300-303 exec_res = executor.execute(call)`), appended verbatim to history; gold observations are never injected in eval. **PASS.**
- Final eval runs in **full** (real IBM functions) mode; gold replay → Win 1.0 confirms execution correctness. Executor errors set `fail_reason=exec:*` and stop the episode (penalized). Clipped episodes → reward 0 and masked from updates.
- The official scorer used is the real NESTFUL `scorer.py` (`calculate_win_score` / `calculate_ans`), via the Windows SIGALRM shim.
- **Not run (per constraints):** no-tool ablation and shuffled-observation ablation → `NEEDS_MANUAL_REVIEW` (recommended in `audit_recommendations.md`). Indirect evidence that ReAct depends on real execution: gold-replay Win=1.0, final_answer_pass≈Win (94.9%), and Direct (single-shot) Win (~0.27) is much lower than ReAct (~0.54), i.e. the multi-turn observed-feedback loop is doing real work.

## Reward audit  (see `reward_audit_summary.csv`, `reward_audit_cases.csv`)

13 unit cases run against the **real** reward modules. Key results:
- `perfect_gold_trace` → strict 1.00 / partial 1.00 / exec 1.00. **PASS.**
- `correct_answer_alt_path` (right answer, different tools) → strict **0.00** / partial **0.30** / exec **0.90**. This is the crux: strict/partial *punish* a winning alternative path; execution rewards it.
- `clipped_rollout`, `no_tool_call` → all three 0 (execution hard-caps). **PASS.**
- **WARNINGS:** partial does **not** zero on a `parse_error` (keeps prefix credit 0.35); `extra_calls` beyond gold are **not** penalized (`length_penalty=0`); strict/partial do **not** validate argument values/references (they rely on the executor erroring) — only `execution_aware` scores `valid_references`.
- Aggregate alignment with official Win (n=7444): **execution pearson 0.876 > partial 0.636 > strict 0.387** (`execution_reward_correlation_summary.csv`).
- The strict/partial rewards match their canvas description (gold-trace fidelity, [0,1], perfect=1.0); confirmed.

## Training process audit  (see `training_process_audit_summary.csv`)

Aggregated from `outputs/curriculum/stage_*/epoch_*/train_log.jsonl` (partial run):

- Reward is computed from **rollouts** via `episode_turn_reward_seq` (not gold outputs) — `grpo_train.py:439`. GRPO advantage = group-relative over turn-level returns, normalized; dead groups (std 0) skipped; clipped masked. Mechanics look correct.
- **Stage 1 (1-call):** reward rises 0.789→0.862; many groups all-correct → dead → few updates (86→40). Eval strict_pass plateaus ~0.40.
- **Stages 2–4:** dense signal (~380 updates/epoch) but **train reward DECLINES every epoch** — s2 0.78→0.56, s3 0.62→0.34, s4 0.39→0.26. Across the whole curriculum the partial train reward falls from **0.86 → 0.26**. Eval strict_pass on NESTFUL collapses 0.40→0.01.
- `group_all_one` collapses 0.75→0.00; `group_mixed` rises 0.30→0.95; `zero_tool_calls` rises slightly (0.05→0.077); clipping negligible (<0.01).

**Interpretation:** hundreds of gradient updates per epoch while *both* the training reward and the eval metric move **down** → this is not a sparse-signal problem (signal is dense) but **reward mismatch + curriculum forgetting / policy drift**: optimizing gold-trace fidelity on synthetic multi-call data pushes the policy away from the behavior that wins on NESTFUL. KL / entropy / grad-norm are **not logged**, so drift magnitude is unquantified (WARNING).

## Checkpoint lineage audit

| checkpoint | on disk | adapter_config | sidecars | final-eval Win (ReAct) | status |
|---|---|---|---|---|---|
| stage_1 e2/e3/e4 | yes (e1 only on /mnt/raid) | yes | **none** | s1_e4 = 0.543 | WARNING |
| stage_2 e1–e4 | yes | yes | none | s2_e2 = 0.533 | WARNING |
| stage_3 e2/e3/e4 | yes | yes | none | s3_e2 = 0.479 | WARNING |
| stage_4 e1 | yes | yes | none | s4_e1 = 0.450 | WARNING |

- Final eval evaluated real, existing adapters (s1_e4/s2_e2/s3_e2/s4_e1 all present) → naming is correct, not a different run. **PASS on identity.**
- **WARNING:** `trainer_state.json`/`config_used.json` sidecars are **absent on disk** (glob = 0), so reward-version/init-checkpoint per checkpoint can't be confirmed from artifacts. Stage-1 checkpoint paths point to `/mnt/raid/...` while stages 2–4 point to `/workspace/...` (multi-pod run); continuity is plausible but not provable from sidecars.
- **WARNING (selection):** `run_curriculum.sh` carries forward the **best-by-`strict_gold_trace_pass`** checkpoint per stage and `advance_threshold=0.50` is never reached (max ≈0.40 at stage 1), so the curriculum always advances by plateau/max-epochs and the *final* checkpoint is the last (worst) stage. The best Win checkpoint is the **earliest** (s1_e4).

## Evaluation consistency audit  (see `eval_consistency_audit_summary.csv`)

Same eval prompt / parser / executor / scorer / n=1861 / 0 parse errors for baseline and all checkpoints → fair comparison (**PASS**). Direct vs ReAct not swapped (ReAct Win > Direct Win; Direct Full Acc ~0.16 vs ReAct ~0). Internal `final_answer_pass` equals official Win on 94.9% of samples (`_audit_alignment.py`) → the live ReAct success signal is execution-grounded and trustworthy. The only FAILs here are the *results* themselves (no improvement; stage degradation; F1-Func collapse).

## Offline reward alignment  (see `reward_audit_summary.csv`, `execution_reward_correlation_summary.csv`)

| reward | pearson vs Win | mean@Win | mean@Loss | FP(win0,r>0.7) | FN(win1,r<0.3) |
|---|---|---|---|---|---|
| strict | 0.387 | 0.30 | 0.007 | 0.002 | — |
| partial | 0.636 | 0.62 | 0.19 | 0.013 | — |
| **execution** | **0.876** | **0.87** | **0.20** | **0.007** | **0.000** |
| final_answer_pass | 0.93 | — | — | — | — |

- Execution-aware reward is far better aligned with Win than strict/partial, with **<1% false positives and ~0% false negatives** → low reward-hacking risk at current caps. `final_answer_pass` dominates (it ≈ Win), which is acceptable because it is execution-grounded (the model must actually produce the gold answer via tool calls).
- Per-checkpoint correlation is stable across s1–s4 (execution pearson 0.856–0.896).

## Manual failure analysis  (see `manual_failure_samples.csv`)

77 samples from the best checkpoint (`partial_s1_e4_react`) across 5 categories (exec-high-but-win0, win1-but-exec-low, strict-fail, no_tool_call, too_few_calls). Dominant pattern: **strict fails are overwhelmingly "alternative correct path / partial trace" rather than parser/executor/data/scorer bugs** — consistent with `correct_answer_alt_path` (strict 0, exec 0.9) in the reward cases and the 94.9% FAP≈Win agreement. Failure attribution: **reward mismatch** (primary) and **model behavior / curriculum forgetting** (secondary); not parser/executor/data/scorer.

---

## Cross-pass consistency findings

| # | check across passes | result | status |
|---|---|---|---|
| 1 | Reward audit says no_tool_call/parse penalized AND training shows only a small rise in zero_tool_calls (0.05→0.08) | consistent; shortcut is NOT the main driver | PASS |
| 2 | Executor audit says invalid refs fail in full mode, but reward audit shows strict/partial don't validate refs | reconciled: validity is executor-gated, not reward-gated | WARNING |
| 3 | Eval-consistency says "same prompt" but prompt audit finds train≠eval prompt/parser | reconciled: *eval-side* is identical across models; the mismatch is train-vs-eval | WARNING |
| 4 | Checkpoint lineage (best-by-strict, carry last stage) vs results (best ckpt is earliest s1_e4) | the selection rule actively picks worse checkpoints | FAIL (selection) |
| 5 | Reward correlation says execution≫strict/partial AND training shows partial reward fights Win | consistent: training target is misaligned with eval target | PASS (coherent) |
| 6 | Gold replay Win=1.0 vs trained Win≤0.544 | consistent: pipeline correct, the policy (not the harness) is the limiter | PASS |

No internal contradictions that undermine the audit; the WARNINGs are reconciled above.

---

## Root cause ranking

| rank | cause | evidence | confidence | recommended action |
|---|---|---|---|---|
| 1 | **Reward mismatch** (strict/partial optimize gold-trace fidelity, not execution Win) | F15; `correct_answer_alt_path` strict 0/exec 0.9; corr strict 0.39/partial 0.64 vs exec 0.88; train reward falls as Win falls | high | switch training target to execution-aware reward |
| 2 | **Curriculum forgetting / policy drift** (degradation grows with stage) | F08/F09/F10/F11; Win 0.543→0.450; F1Func 0.926→0.350; train reward 0.86→0.26 with ~380 updates/epoch | high | mixed replay, lower LR, higher KL, per-epoch validation-Win early-stop; log KL/entropy/grad-norm |
| 3 | **Checkpoint selection by strict_pass + carry-last-stage** | F19; advance_threshold 0.50 never reached; best Win = earliest ckpt | high | select/carry by validation **ReAct Win** (best_react_win_adapter) |
| 4 | **Train/eval domain shift** (synthetic train vs NESTFUL eval) | F01; 0 overlap; depth gap | medium | add NESTFUL-style tasks to training mix / validate on NESTFUL each epoch (already wired in stabilized profile) |
| 5 | **Train/eval prompt+parser+turn-budget mismatch** | F21; prompt_mismatch_audit_summary.csv | medium | unify train and eval prompt/parser/turn budget |
| 6 | Reward edge cases (parse not zeroed in partial; extra calls unpenalized; refs unchecked) | F16/F17/F18 | medium | tighten partial (zero on parse fail), enable small length_penalty, keep execution caps |
| 7 | No-tool / too-few-call shortcut | F12; zero_tool_calls only 0.05→0.08 | low | monitor; not the main driver |
| 8 | Parser/executor bug | F02/F04/F22; gold replay Win=1.0 | low (ruled out) | none |
| 9 | Data quality issue | F02/F03 | low (ruled out) | none |
| 10 | Official scorer mismatch | F05/F06 | low (ruled out) | none |

---

## Final recommendation

1. **Is the pipeline technically trustworthy?** **Yes.** Data, parser, executor, official scorer, and eval-consistency all PASS (gold replay Win=1.0; FAP≈Win 94.9%; identical eval for baseline+ckpts). The harness is sound — the limiter is the policy/training, not bugs.
2. **Is the most likely problem reward mismatch?** **Yes (primary), compounded by curriculum forgetting and strict-pass checkpoint selection.** Strict/partial rewards optimize gold-trace fidelity; the Win metric rewards execution success via any valid path. Training therefore pushes the policy *away* from Win (train reward and Win both fall together), and selection keeps the worst (last-stage) checkpoint.
3. **Fix code/data before next run?** Small, targeted changes only (no harness rewrite): (a) log KL/entropy/grad-norm; (b) select & carry checkpoints by validation ReAct Win; (c) optionally tighten partial reward edge cases. These are reported, not auto-applied.
4. **Run an execution-aware pilot?** **Yes** — it is the best-aligned target (pearson 0.876, FP<1%, FN~0%) and is already implemented/unit-tested. Run a *small* pilot (stages 1–2, mixed replay, lower LR / higher KL, validation-Win early stopping), not a full overnight curriculum.
5. **Success gates for the pilot:** validation ReAct Win ≥ baseline 0.544 by end of stage 1 and **non-decreasing** across stages; train reward and validation Win move **together** (no decoupling); F1 Func stays ≥0.85 (no degeneration); zero_tool_calls < 0.10; clipped < 0.05. Abort/early-stop if validation Win drops > 0.005 for one eval.
6. **Usable for the ITAT paper:** the gold-replay validation (Win=1.0), the reward↔Win correlation table (0.876/0.636/0.387), the FAP≈Win 94.9% evidence, and the curriculum-degradation curves (Win & F1-Func vs stage) are clean, reproducible evidence that (a) the eval is honest and (b) gold-trace rewards are misaligned with execution Win — motivating the execution-aware reward.
7. **Remaining limitations:** no-tool / shuffled-observation ablations not run; KL/entropy not logged in the audited run; checkpoint sidecars absent on disk; `experiments/` tree untracked in git; gold-replay sanity is a 150-sample subset (not all 1861).

### Final PASS / WARNING / FAIL table

| area | status | evidence | action |
|---|---|---|---|
| data | PASS (1 WARNING) | gold replay Win=1.0; schema valid; WARNING: synthetic↔NESTFUL domain shift | add NESTFUL-style train mix |
| reward | FAIL | strict/partial corr 0.39/0.64 vs Win; punish alt-path wins; train reward falls as Win falls | switch to execution-aware reward |
| parser | PASS | 10/10 cases incl round-trip | unify train/eval parser (minor) |
| executor | PASS | predicted-call obs; gold replay Win=1.0 | run no-tool/shuffled ablations |
| train process | FAIL | reward 0.86→0.26; ~380 updates/epoch but Win↓ | lower LR / higher KL / mixed replay; log KL/entropy |
| checkpoint lineage | WARNING | real ckpts evaluated; no sidecars; best=earliest | select/carry by validation Win |
| eval consistency | PASS | identical prompt/parser/scorer/n; FAP≈Win 94.9% | none |
| prompt mismatch | WARNING | train≠eval prompt/parser/turn budget (eval fair across models) | unify before next run |
| reward alignment | PASS | execution pearson 0.876, FP<1%, FN~0% | use execution reward; keep caps |
