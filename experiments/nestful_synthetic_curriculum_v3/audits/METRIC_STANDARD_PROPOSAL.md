# METRIC STANDARD PROPOSAL — one standard for all future runs

Date: 2026-07-09. Grounded in the NESTFUL paper's protocol: ReAct-style evaluation reports
**Win Rate** (executable predicted API calls whose execution reaches the gold answer);
direct function-call evaluation may additionally use partial/full sequence matching and
F1 Func/Param as corpus diagnostics.

## Primary metric (the only number allowed in headline claims)

**`official_nestful_win_rate_temp0`**

- Computed exclusively by the official scorer path (`nestful_official_score.score_items` /
  `score_items_per_sample`, wrapping the official `calculate_win_score` with real IBM-function
  re-execution).
- Decoding pinned: temperature 0.0, fixed top_p, fixed max_new_tokens, fixed seed; the
  decoding block must be embedded in the metrics file.
- Populations: report on **nestful_test (1,661)** as headline; full (1,861) may be shown but
  flagged as containing the dev tasks used for selection; dev (200) is for selection/gates
  only.
- A number without a **same-batch baseline** cell is non-reportable for comparisons.
  Confidence: report ±1.96·√(p(1−p)/n) (≈ ±2.3 pp at n=1861, ±3.5 pp per 200-task dev... use
  n-appropriate CI) and paired gained/regressed counts from per-sample `official_win`.

## Secondary diagnostics (reported in an appendix / diagnostics table, never as headline)

| metric | source | purpose |
|---|---|---|
| `internal_final_answer_win` | `internal_metrics_diagnostic.win_rate` (metrics.py) | cross-check for the official scorer; known +6–7 pp lenient |
| `final_answer_pass` | our_metrics | path-independent answer accuracy |
| `solution_equivalent` | our_metrics `solution_equivalent_pass` | evidence-grounded alternative-path solves |
| `full_trace_match` | official `full_sequence_accuracy` | strict trace reproduction |
| `partial_trace_match` | official `partial_sequence_accuracy` | graded trace overlap |
| `executable_rate` | official per-sample `executable` | how often predicted calls run at all |
| `too_few_calls_rate` | trajectory diagnostics | under-calling (the dominant failure) |
| `avg_predicted_calls` | trajectory diagnostics | call-count calibration vs gold |
| `parse_error_rate` | official `num_pred_parsing_errors / n` (state which definition) | format health |
| `no_tool_call_rate` | trajectory `zero_tool_calls` | catastrophic format collapse |
| paired `gained` / `regressed` / `net` vs baseline | per-sample official_win join on sample_id | the actual evidence for "improved" |

F1 Func / F1 Param: corpus-level diagnostics only (as in the NESTFUL paper's framing);
never per-checkpoint headline.

## Rules

1. **Never compare across eval batches.** A comparison requires: same batch directory, same
   decoding block, same dataset SHA, baseline cell present. The eval runner should enforce
   this (hard-fail without `--allow-no-baseline`).
2. **Internal replica numbers are never called "win rate" in prose.** Always
   "internal final-answer win (diagnostic)".
3. Every metrics file carries: `schema_version`, dataset name+sha256+n, decoding block,
   checkpoint path, git commit, scorer identity (`official@<NESTFUL-main commit>` vs
   `internal_replica`).
4. Checkpoint selection metric = dev `official_nestful_win_rate_temp0` (val_eval already
   uses official win; pin its temperature to 0.0 too — today val_eval inherits the training
   config's sampling temperature).
5. Stage gates may use internal diagnostics (cheap), but stage *claims* use the primary
   metric only.
6. For the paper: headline table = official Win Rate (temp0, test-1661, same-batch baseline,
   CI, paired counts); appendix = full/partial acc, F1s, diagnostics; synthetic-stage metrics
   (strict_gold_trace_pass etc.) are training-progress evidence only, never NESTFUL claims.
