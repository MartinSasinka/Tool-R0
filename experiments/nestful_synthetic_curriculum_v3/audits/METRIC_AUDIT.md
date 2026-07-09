# METRIC AUDIT — every win/accuracy/diagnostic metric in the pipeline

Date: 2026-07-09 · Read-only audit. Code references are to
`experiments/nestful_mtgrpo_minimal/` unless noted.

## 1. The three metric families and where they live

| family | source of truth | file | written to |
|---|---|---|---|
| **Official NESTFUL** (paper-grade) | official scorer code in `data/NESTFUL-main/src/scorer.py`, adapted by `nestful_official_score.py` | `nestful_official_score.py` (`score_items`, `score_items_per_sample`, `build_item`) | `metrics_official.json` (val_eval always; final_eval since the Stage-3 batch) |
| **Internal diagnostic replica** | reimplementation "definition-compatible, not byte-identical" | `metrics.py` (`compute_nestful_official_metrics`, `aggregate_final_eval`) | `metrics.json` under key `internal_metrics_diagnostic` |
| **"Our" paper-specific metrics** | strict-trace + solution-equivalence analysis | `metrics.py` (`compute_paper_metrics`, `solution_equivalent_score`) | `metrics.json` under key `our_metrics` |

Run modes (in `run.py`) and what they produce:

- `mode_rollout_eval` (per-epoch `eval/`): internal `metrics.json` only (strict_gold_trace_pass,
  final_answer_pass, zero_tool_calls, too_few_calls_rate, avg_predicted_calls…), on a
  call-count-filtered slice of `paths.eval_jsonl` (v3 runs override this to full NESTFUL,
  filtered by `data.eval_stage` → e.g. 407 3-call or 250 4-call tasks).
- `mode_val_eval` (per-epoch `val_eval/`, 200-task dev): internal `metrics.json` **and**
  official `metrics_official.json`; surfaces `react_win_rate` (:= official `win_rate`) into
  `metrics_epoch_<E>.json`, which drives checkpoint selection and stage gates. It **hard-fails
  when official win is null** (`validation.require_win_rate`, run.py:1095-1108) — good.
- `mode_final_eval` (full NESTFUL, 1,861): internal `metrics.json` always; official
  `metrics_official.json` only in the current code path — the July-7/8 final-eval batches
  predate this and have **no official numbers** (see RUN_AUDIT §2).

## 2. Metric-by-metric documentation

### Official metrics (`nestful_official_score.py`)

| metric | definition | execution? | final answer? | trace? | paper-suitable? |
|---|---|---|---|---|---|
| `win_rate` (official) | per-sample `calculate_win_score` from the official repo: re-executes the predicted call list against the IBM executable functions and compares the resulting answer to `gold_answer` (decimal-aware). Aggregated as mean of per-sample wins (`score_items` computes it via the per-sample path to survive numpy-comparison crashes; a crashing sample counts as loss, not batch abort) | **yes, real re-execution** | yes | implicitly (answer must come from executing the predicted calls) | **YES — primary** |
| `full_sequence_accuracy` | official grounded canonical-step match; 1 iff every aligned step matches | no | no | yes (exact trace) | yes, secondary |
| `partial_sequence_accuracy` | fraction of aligned canonical steps matching, after variable grounding + length alignment (`post_process_api_with_args`) | no | no | yes | yes, secondary |
| `f1_func` / `f1_param` | corpus-level set-based macro-F1 (sklearn MultiLabelBinarizer) over grounded function names / "k = v" slots | no | no | partial | diagnostics only (corpus-level; not per-sample decomposable) |
| `official_win`, `official_full_match`, `official_partial_match`, `executable`, `execution_error` (per-sample) | `score_items_per_sample` — per-item official diagnostics | yes | yes | yes | enables paired win/regression analysis |

Windows caveat: official win executes IBM functions with `signal.SIGALRM`; a threading shim
(`_patch_signal_alarm_for_win_rate`) makes it work off-pod.

### Internal diagnostic replica (`metrics.py`)

| metric | definition (file:function) | differences from official |
|---|---|---|
| `internal_metrics_diagnostic.win_rate` | `compute_nestful_official_metrics` (metrics.py:322-333): 1 iff trajectory is executable (no executor error, ≥1 tool call, not parse-fail) AND `decimal_aware_equal(trajectory.final_observation, gold_answer)` | scores the **agent's own executed trajectory** (multi-turn, with the internal executor's leniency, recovery turns allowed), whereas official re-executes the extracted call list from scratch with the IBM functions. This is the source of the systematic **+6–7 pp inflation** over official win observed in every eval cell (e.g. dev baseline .635 int vs .565 off; s3_e1 full .6131 vs .5438). Also `decimal_aware_equal` falls back to the *tolerant* `matches_gold` for non-float types. |
| `f1_func`, `f1_param` (per-sample) | count-aware multiset F1 (metrics.py:303-307) | official is corpus macro; the replica also computes `f1_*_corpus_macro` at aggregation for cross-check (present in newer metrics.json, `null` when sklearn absent) |
| `partial/full_sequence_accuracy` | grounded canonical strings + official aligner when importable, else right-padding | close replica; can drift when NESTFUL-main is absent |

### "Our" metrics (`metrics.py` `compute_paper_metrics`)

| metric | definition | execution | final answer | trace grounding |
|---|---|---|---|---|
| `strict_gold_trace_pass` | `strict_gold_trace_reward ≥ 1.0` (reward.py:51-160): binary; exact call count, tool names, argument **keys**, per-turn observation match vs precomputed gold observations, no parse fail, no clipping, AND final answer match | yes (or gold obs comparison) | yes | strict positional |
| `final_answer_pass` | `matches_gold(final_observation, gold_answer)` from strict-reward diagnostics — path-independent | trajectory execution | **only** final answer | none |
| `solution_equivalent_pass` | executable + final==gold + `answer_supported_by_observations` (every scalar in the answer appears in the model's own observations) + not noop/bruteforce (metrics.py:361-389) | yes | yes | evidence-grounded (not gold-path) |
| `alternative_valid_solution_pass` / `strict_fail_but_solution_equivalent_pass` | solution-equivalent but official full-trace ≠ 1 | yes | yes | yes |
| `correct_answer_but_unsupported_trace` | final_pass ∧ ¬solution_equivalent | — | yes | flags "guessing" |

### Behavioral diagnostics

| metric | where computed | definition |
|---|---|---|
| `too_few_calls_rate` | reward diag (`too_few_calls`) → train_log; eval: run.py aggregates `num_tool_calls < gold_n` | fraction of episodes with fewer predicted calls than gold |
| `too_many_calls` | reward diag; strict reward `too_many_turns` | calls > gold |
| `no_tool_call` / `zero_tool_calls` | `rollout.py` trajectory flag; nestful_core `has_no_tool_call` | zero parsed tool calls in the episode |
| `parse_error_rate` | parser fail_reason `parse:*` per turn; official `num_pred_parsing_errors` counts scorer-side parse failures | note: internal counts turn-level parse breaks, official counts unparseable `generated_text` — different populations |
| `avg_predicted_calls` | mean `len(predicted_calls)` | |
| `clipped_completion_rate` | trajectory `clipped_any` — completion hit max_new_tokens; clipped episodes get reward 0 and are masked from updates (`training.mask_clipped_from_update`) | |
| `executable rate` | official per-sample `executable` (calculate_ans ≠ False); internal `executable_frac` in reward | |
| paired wins/regressions | `CHECKPOINT_REEVAL_REPORT` generator over per-sample `official_win` vs baseline cell | only meaningful same-batch; Batch-3 output shows `shared=0` because the baseline cell is missing |
| `dead_group_rate` | `grpo_train.py` (corrected between-completion std == 0; old flattened definition kept as `dead_group_old_flattened`) → train_summary, stage gates | training-signal metric, not an eval metric |
| `react_win_rate` | run.py `mode_val_eval` → `metrics_epoch_<E>.json`; = official dev win | drives `checkpoint_eligibility` / best-adapter selection and `check_stage_gates.py` (`dev_win_vs_baseline`, allowed drop 0.03) |

## 3. Identified inconsistencies

1. **Internal win ≠ official win, +6–7 pp systematic** (different execution/leniency path).
   Any statement that used `internal_metrics_diagnostic.win_rate` as "win rate" overstated
   results. The code labels it correctly ("diagnostic"), but downstream reports (e.g. the SFT
   eval markdown, chat summaries) have mixed the two.
2. **Final-eval batches before 2026-07-09 have no official scores at all** — full-NESTFUL
   comparisons for Stage-2 checkpoints exist only on the internal replica.
3. **Three eval populations under one run tree** (dev-200 val_eval; call-count-filtered
   NESTFUL slices in rollout_eval; full-1861 final_eval) share metric names — e.g.
   `strict_gold_trace_pass` at stage-gate time is over a different population than in
   final_eval.
4. **Full NESTFUL includes dev**: checkpoint selection uses dev official win, then final eval
   runs on the superset (1,861 ⊃ 200).
5. Internal per-sample `f1_func/f1_param` are multiset-based and NOT comparable to official
   corpus macro-F1, though they share names inside `internal_metrics_diagnostic`.
6. `parse_error` means different things internally (turn-level) vs officially
   (item-level generated_text parse).
7. Sampled vs temp0 decoding was not pinned for the first final-eval batch; temp0 lowered all
   cells ~1 pp, so cross-batch deltas conflate decoding with checkpoints. Config default eval
   temperature is 0.7 (`generation.temperature`), overridable via `EVAL_TEMPERATURE`.
8. **Official win is platform-gated**: `run.py` sets `want_win = os.name != "nt" and
   ibm_functions_dir exists` — on Windows or without the IBM executable-functions dir the
   official win is silently skipped (`official_win=None`), which is what `mode_val_eval`'s
   `require_win_rate` hard-fail protects against. Offline re-scoring on Windows is possible
   via the standalone `nestful_official_score.py` (SIGALRM threading shim) but the pipeline
   does not use it on `nt`.
9. Numeric tolerance differs: internal `matches_gold` uses `tol=1e-3` for floats; the
   official comparison rounds the prediction to the gold value's decimal places — another
   contributor to the internal-vs-official gap.
10. The `strict_gold_trace_pass` field in `train_log.jsonl` is misnamed: it logs the group's
    mean episode *reward* (`grpo_train.py:709`), not a strict pass rate — do not read
    training-log "strict pass" as the eval metric.

See `METRIC_STANDARD_PROPOSAL.md` for the going-forward standard.
