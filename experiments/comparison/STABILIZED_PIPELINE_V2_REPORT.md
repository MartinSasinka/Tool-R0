# Stabilized v2 NESTFUL / MT-GRPO pipeline — master report

This report documents the v2 rebuild of the `nestful_mtgrpo_minimal` /
`nestful_mtgrpo_partial` solution. All offline / CPU checks below were executed
locally and pass; GRPO training, the pilot, the full run and the final eval are
delivered as ready-to-run pod scripts (no training was executed locally). Legacy
strict/partial behaviour is preserved bit-for-bit via `*_legacy` policies.

---

## 1. What was wrong (audit recap)

| # | Problem | Evidence |
|---|---------|----------|
| 1 | **Reward/Win mismatch.** strict/partial reward measure gold-trace fidelity, but the eval metric is execution Win. The policy drifts to alternate-but-correct execution paths the reward penalises. | F15, F07/F08 |
| 2 | **Reward never beat baseline.** best checkpoint partial_s1_e4 ReAct Win 0.543 vs baseline 0.544; longer curriculum monotonically degrades (0.543→0.450). | F07/F08/F10 |
| 3 | **partial reward edge cases.** does not zero on parse error (gives 0.35 prefix credit), does not penalise extra calls, does not validate references. | F16/F17/F18 |
| 4 | **Checkpoint selection by the wrong metric** (strict_gold_trace_pass, not Win). | F19 |
| 5 | **Train/eval prompt + parser mismatch.** train: SYSTEM_PROMPT + strict parser + `max_turns=gold_n`; eval: +EVAL_HARDENING + lenient parser + `gold_n+1`. | F21, parser_diagnostics (23.5% recovery gap on partial_s4e1) |
| 6 | **No KL / entropy / grad-norm logging**, so policy drift was unquantified. | F23 |
| 7 | **Unverified per-sample Win.** overlap/taxonomy were built from heterogeneous trajectory fields, never asserted against the aggregate. | meeting_analysis (legacy) |
| 8 | **Malformed report CSVs** (unescaped commas) broke `pandas.read_csv`. | audit_findings.csv / parser_executor_audit_summary.csv |

## 2. What we fixed (architecture)

- **Single shared core** `experiments/nestful_core/` (parser, executor, rollout,
  prompt, rewards, scoring, data, eval_loop, logging_utils). `minimal` and
  `partial` both import the canonical implementations; legacy modules stay frozen
  and `*_legacy` policies delegate to them. See `docs/PIPELINE_V2_STRUCTURE.md`.
- **CSV hygiene writer** (`logging_utils.write_csv`, `csv.DictWriter` + JSON for
  complex cells) + `tests/test_reports_loadable.py` asserting every
  `experiments/comparison/*.csv` loads via pandas. The two malformed audit CSVs
  were re-emitted through it (`_rebuild_audit_csvs.py`).

## 3. Reward v2 definition

`nestful_core/rewards.py` adds explicit trajectory predicates (single source of
truth: `has_parse_error`, `has_no_tool_call`, `terminal_before_first_successful_tool`,
`num_successful_calls`, `has_invalid_reference`, `has_executor_error`,
`is_executable_trajectory`, `tool_final_answer_pass`, `tool_use_completeness`,
`gold_trace_progress`, `num_extra_calls`) and two new policies.

**`execution_aware_v2`** (new primary training reward):

```
R = 0.55*final + 0.20*executable + 0.10*completeness + 0.10*valid_refs + 0.05*gold_progress
hard caps  -> 0:  parse_error | clipped | no_tool_call | terminal_before_first_successful_tool
soft caps:  executor_error<=0.25, invalid_reference<=0.30, not_executable<=0.25,
            too_few_calls&!final<=0.25, !final<=0.35
floor:      executable & final  -> 0.85
extra calls: mild penalty (0.02/call, cap 0.10) ONLY when final answer correct; capped otherwise
```

**`partial_gold_trace_v2`** (fixed graded baseline): parse/clipped/no_tool/terminal → 0;
invalid_reference & executor_error penalised; `!final → cap 0.60`;
extra-calls-without-correct-answer capped. Both return full per-component
breakdown dicts + `cap_applied`.

Unit tests `tests/test_execution_reward_v2.py` + `tests/test_rewards_v2.py` (shared
case table incl. invalid_reference, too_few≤0.25, executable+wrong≤0.35,
executor_error≤0.25) — all pass. Synthetic audit cases →
`experiments/comparison/reward_v2_audit_cases.csv`.

## 4. Data

- All 1861 `nestful_data.jsonl` stay the **clean test set** (never trained on).
- Fixed stratified-by-num_calls **synthetic** train/val splits built from
  `clean_curriculum` → `data/splits/synthetic_{train,val}.jsonl` (+ per-stage
  variants) with `splits_manifest.json` recording sizes + sha256.
- **Deliberate limitation:** validation Win is a synthetic proxy (the original
  "60% real NESTFUL in train" idea was dropped to avoid train-on-test
  contamination). Recorded as a tradeoff.

## 5. Prompt + parser unification

- `nestful_core/prompt.py` = `SYSTEM_PROMPT_BASE + REACT_TOOL_FORMAT_RULES +
  REACT_STOP_RULES + {TRAIN|EVAL}_HARDENING`; train and eval now carry identical
  format/stop rules and differ only in a thin hardening tail. Versions
  (`v2.0-train` / `v2.0-eval`) saved per run via `prompt_versions()`.
- `nestful_core/parser.py` `parse_canonical` exposes `strict_ok` / `lenient_ok` /
  `call` / `is_terminal` / `reason`; eval logs strict/lenient/recovery rates via
  `parser_diagnostics.py` → `parser_diagnostics.csv`.
- **Turn budget unified:** `max_turns = gold_n + 1` (cap `gold_n + 4`) for BOTH
  train and eval (`eval_loop.max_turns_for`; `max_extra_turns_train` wired through
  `vllm_dp_pool` / `grpo_train`).

## 6. Selection change

`run_curriculum.sh` (`stabilized_curriculum`) selection retargeted to the
**synthetic val split**, choosing `best_react_win_adapter` by validation ReAct
Win (NOT training reward / strict_pass / F1 / epoch), with `checkpoint_lineage.json`
+ `best_react_win_metrics.json` + early stopping on validation Win.

## 7. Trustworthy scoring (verified)

- `gold_replay_full.py`: full 1861 gold replay → **GATE: PASS** (Win/full/parse/
  executable ≈ 1.0). `gold_replay_full_report.md`.
- `recompute_per_sample_official.py`: per-sample official Win with
  `mean(per_sample) == aggregate` assertion (rounding-aware tol 0.0012) →
  **OVERALL: PASS** for partial_s1_e4_react / partial_s4_e1_react / baseline_react.
  Runs without preserved trajectories (minimal_s4e2) are flagged
  WARNING_MISSING_TRAJECTORIES and excluded from per-sample analysis.
- `meeting_analysis.py` (canonical default) consumes `per_sample_official_win.csv`,
  **refuses** to emit overlap/taxonomy unless consistency passes
  (`--assert-consistency`), and writes `win_loss_overlap_verified.csv`,
  `failure_taxonomy_verified.csv`, `MEETING_BRIEF_VERIFIED.md`. Legacy heuristic
  kept behind `--legacy`.

Verified win/loss overlap (per-task official Win, n=1861):

| comparison | both win | A win / B fail | A fail / B win | both fail |
|---|---|---|---|---|
| baseline_react vs partial_s1_e4_react | 46.2% | 8.1% | 8.0% | 37.7% |
| baseline_react vs partial_s4_e1_react | 36.4% | 18.0% | 8.6% | 37.1% |
| partial_s1_e4_react vs partial_s4_e1_react | 36.1% | 18.1% | 8.9% | 36.9% |

## 8. Offline reward alignment (execution_aware_v2)

`rescore_execution_reward_v2.py` rescored saved trajectories →
`execution_reward_v2_correlation.csv` + `_summary.md`:

| run | n | Win | Pearson(full) | mean@win | mean@loss | FP(r>0.7\|win0) | FN(r<0.3\|win1) |
|---|---|---|---|---|---|---|---|
| partial_s1_e4_react | 1861 | 0.542 | 0.825 | 0.899 | 0.253 | 0.039 | 0.025 |
| partial_s4_e1_react | 1861 | 0.450 | 0.756 | 0.770 | 0.128 | 0.016 | 0.082 |
| baseline_react | 1861 | 0.543 | 0.814 | 0.885 | 0.240 | 0.037 | 0.032 |
| ALL | 5583 | 0.512 | 0.794 | 0.857 | 0.202 | 0.030 | 0.046 |

Honest findings: strong reward separation (mean@win 0.857 vs mean@loss 0.202).
`final_answer_pass` alone carries most rank-correlation; the dense terms add the
partial-credit signal GRPO needs early on. The v2 caps deliberately trade a small
amount of rank-correlation for **training safety** (parse/clipped/no-tool/terminal
trajectories forced to 0 — the exploit that drove the legacy mismatch). Residual
false positives come from `matches_gold` being more lenient than the official
scorer on formatting; they are logged per-sample for inspection.

## 9. Prompt ablation (prepared)

`prompt_ablation.py` dry-run built a fixed stratified subset
(`prompt_ablation_subset.jsonl`, n=400, +sha256) and validated all three prompt
variants (current / train_style / hardened) build and differ for every task.
`prompt_ablation_summary.csv` is seeded PENDING; the `--generate` path runs on the
pod to fill ReAct Win per (checkpoint × variant).

## 10. Logging

`grpo_train.py` now logs per-checkpoint reward-component rates (parse/clipped/
no_tool/too_few/invalid_ref/executor_error), rollout length, successful calls, KL,
mean logprob, grad-norm, lr, kl_beta (lazy `nestful_core.rewards` import; additive
and guarded). `logging_utils.REWARD_COMPONENT_FIELDS` standardises the columns.

## 11. Pilot results

**Placeholder — to be filled after `run_pilot_v2.sh` on the pod.** The pilot is
gated by `check_gates.py pilot --metrics <pilot_metrics.json>` (request §16):
validation_react_win ≥ baseline − 1pp; no_tool/too_few/parse/invalid_reference
rates not rising; executable_trajectory_rate not falling; reward-hacking flag
(reward up while Win down) stops the run.

## 12. ITAT recommendation

Frame the contribution as a **reward-design study** (trace-fidelity vs
execution-success), NOT a benchmark-improvement claim. The verified data shows
gold-trace rewards plateau at / below baseline Win and degrade with longer
curriculum; execution_aware_v2 is the targeted intervention, evaluated honestly
with consistency-gated per-sample Win and offline alignment. Do not report
macro-F1 Func as a success metric (format/tool-name compliance, ~900 classes).

---

## Success gates (encoded in `check_gates.py`)

- **preflight (PASS locally):** reward v2 + report-loadable + parser tests; gold
  replay GATE PASS; per-sample consistency OVERALL PASS.
- **pilot (pod):** validation Win ≥ baseline − 1pp; failure-mode rates not rising;
  executable rate not falling; reward-hacking guard.

## Handoff

### Files added (new)
- `experiments/nestful_core/`: `__init__.py`, `parser.py`, `executor.py`,
  `rollout.py`, `prompt.py`, `rewards.py`, `scoring.py`, `data.py`,
  `eval_loop.py`, `logging_utils.py`
- `experiments/nestful_mtgrpo_partial/`: `execution_reward_v2.py`,
  `partial_reward_v2.py`, `run_pilot_v2.sh`, `run_full_v2.sh`, `run_final_eval_v2.sh`
- `experiments/comparison/`: `recompute_per_sample_official.py`,
  `gold_replay_full.py`, `reward_v2_cases.py`, `parser_diagnostics.py`,
  `check_gates.py`, `rescore_execution_reward_v2.py`, `prompt_ablation.py`,
  `_rebuild_audit_csvs.py`, `STABILIZED_PIPELINE_V2_REPORT.md`
- `tests/`: `conftest.py`, `test_reports_loadable.py`,
  `test_execution_reward_v2.py`, `test_rewards_v2.py`
- `docs/PIPELINE_V2_STRUCTURE.md`
- `experiments/nestful_mtgrpo_minimal/data/splits/*` (synthetic train/val + manifest)

### Files modified
- `experiments/nestful_mtgrpo_partial/run.py` (recognise v2 policies)
- `experiments/nestful_mtgrpo_minimal/vllm_dp_pool.py` (resolve v2 rewards;
  `max_turns_train = gold_n + max_extra_turns_train`)
- `experiments/nestful_mtgrpo_minimal/grpo_train.py` (component + KL/logprob/
  grad-norm logging; configurable `max_turns_train`)
- `experiments/nestful_mtgrpo_minimal/run_curriculum.sh` (synthetic `VAL_JSONL`,
  per-stage mixed replay, `EXTRA_TRAIN_OVERRIDES_STR`)
- `experiments/comparison/meeting_analysis.py` (verified canonical pipeline + flags)
- re-emitted `audit_findings.csv`, `parser_executor_audit_summary.csv`

### Tests / sanity checks passed (local)
- `python experiments/comparison/check_gates.py preflight` → PREFLIGHT: PASS
  (57 unit tests pass; gold replay PASS; per-sample consistency PASS)
- `recompute_per_sample_official.py` → OVERALL PASS
- `gold_replay_full.py` → GATE PASS
- `rescore_execution_reward_v2.py` → correlation CSV + summary
- `meeting_analysis.py --assert-consistency` → verified overlap/taxonomy/brief
- `prompt_ablation.py` → subset + variant validation OK

### Remaining warnings / limitations
- No real NESTFUL in training; validation Win is a synthetic proxy (test kept clean).
- minimal_s4e2 / baseline_direct trajectories were not preserved → excluded from
  per-sample overlap; re-run their eval with `--dump-trajectories` to include them.
- execution_aware_v2 residual false positives from lenient `matches_gold`.
- No GRPO training executed locally — pilot/full/final-eval/prompt-ablation
  generation run on the pod.

### Exact commands (pod)
```bash
# 1) pilot (gated)
USE_VLLM=1 bash experiments/nestful_mtgrpo_partial/run_pilot_v2.sh
python experiments/comparison/check_gates.py pilot --metrics <pilot_metrics.json>

# 2) full run (after pilot PASS)
USE_VLLM=1 bash experiments/nestful_mtgrpo_partial/run_full_v2.sh

# 3) final eval (after full run)
USE_VLLM=1 BASELINE_DIR=<base> \
  ADAPTER_PARTIAL_S1E4=<dir> ADAPTER_MINIMAL_S4E2=<dir> \
  ADAPTER_EXEC_V2=<best_react_win_adapter> \
  bash experiments/nestful_mtgrpo_partial/run_final_eval_v2.sh
```

### Report locations
- `experiments/comparison/STABILIZED_PIPELINE_V2_REPORT.md` (this file)
- `docs/PIPELINE_V2_STRUCTURE.md`
- `experiments/comparison/`: `gold_replay_full_report.md`,
  `per_sample_consistency_report.md`, `execution_reward_v2_correlation_summary.md`,
  `MEETING_BRIEF_VERIFIED.md`, `prompt_ablation_report.md`
- final eval (pod): `experiments/nestful_mtgrpo_partial/outputs/final_eval_v2/FINAL_RESULTS_VERIFIED.md`
