# NESTFUL Evaluation Audit

Audit of our NESTFUL evaluation implementation against the official benchmark
(paper: arXiv:2409.03797; dataset: `ibm-research/nestful`; reference scorer:
`data/NESTFUL-main/src/`).

## Source-of-truth policy (read this first)

- **`official_*` = the only reportable benchmark metrics.** They come from the
  official NESTFUL scorer (`data/NESTFUL-main/src/scorer.py`) via the adapter
  `nestful_official_score.py`. All paper / benchmark tables MUST use these:
  `official_f1_func`, `official_f1_param`, `official_partial`, `official_full`,
  `official_win`.
- **`internal_*` = diagnostics and compatibility cross-check only.** They come
  from `metrics.py`, which is fixed to replicate the official semantics as
  closely as practical. They are NOT an independent source of paper numbers.
- **`mismatch` flag**: when `official_*` and `internal_*` disagree on a sample,
  trust `official_*`. The mismatch (with `mismatch_reason`) exists so we can
  debug where our internal replica diverges; it never overrides the official
  number.

## Key architectural fact

The official scorer's Win Rate path `calculate_win_score -> calculate_ans`
(`data/NESTFUL-main/src/scorer.py` lines 45-134) **re-executes the predicted
calls itself**: positional `func(*arg_val_list)` in JSON argument order, output
parameter taken from each tool's `output_parameters`, and a decimal-rounding
float comparison. Because of this, the official metrics are independent of our
runtime `executor.py`; our executor only drives the ReAct environment.

---

## A. Implemented correctly

- **Dataset structure** — `normalize_task` in `data.py` separates `input` ->
  `question`, `output` -> `gold_calls`, `tools`, and `gold_answer`, with tolerant
  field aliasing and JSON/`ast` coercion. Matches the expected NESTFUL example
  shape (`sample_id`, `input`, `output`, `tools`, `gold_answer`).
- **Direct parsing** — `parse_tool_calls_all` + `_loads_relaxed` in `parser.py`
  extract the full JSON list from a single-shot answer, strip Markdown code
  fences, and preserve `name` / `label` / `arguments`. Invalid JSON returns an
  empty list (failure), never a silent partial success.
- **Direct paradigm** — `direct_eval.py` is single-shot and scores the entire
  generated sequence through the official adapter, not just the final answer.
- **ReAct loop** — `run_episode` in `rollout.py` bounds the number of turns,
  feeds each observation back to the model, executes each call, and treats a
  parse failure as a hard stop for that episode.
- **Official adapter** — `nestful_official_score.py` builds scorer items and
  drives the real `calculate_scores`.

## B. Differences from NESTFUL (in `metrics.py`, the internal path)

These are the divergences the internal replica is being fixed to remove. They
never affect `official_*`.

- **F1 Func** — `_multiset_f1` (`metrics.py`) was per-sample and count-aware;
  official is corpus-level set-based macro-F1 via `MultiLabelBinarizer`
  (`compute_score_sklearn`, `utils.py` 34-45).
- **F1 Param** — ours used `(name, arg_key)` pairs and **ignored argument
  values**; official compares `"arg = value"` slot strings grouped per function
  (`scorer.py` 228-254). This understated F1 Param.
- **Partial / Full** — ours zipped positionally without **variable grounding**;
  official rewrites `$var_2.result$` -> `$<producing_fn>.result$` via
  `ground_seq_nested_repsonse` (`output_parsers.py` 42-92) and aligns unequal
  lengths via `post_process_api_with_args` (`utils.py` 47-92) before
  `accuracy_score`.
- **Win float compare** — our `matches_gold` uses a `1e-3` tolerance; official
  rounds the predicted float to the gold value's decimal-place count and then
  compares with exact `==` (`scorer.py` 124-127).

## C. Critical (paper-impacting) notes

- If `metrics.json` (`internal_*`) were read as the paper number, F1 Param would
  be understated and F1 Func would use a different definition. **Use `official_*`
  in all tables.** This is enforced by policy, and the report saves both with a
  `mismatch` flag.
- **Macro-F1 interpretation** (not a bug): the official F1 Func is corpus-level
  macro-F1 over a large (~900) function-name vocabulary, so it has a *different
  interpretation* than a per-sample or micro score and tends to read high
  (rare, distinctive functions are easy to match). It is the official, correct
  definition; we simply also report supplementary diagnostic metrics (micro /
  per-sample) next to it so the picture is complete. We do not claim the macro
  number is wrong.
- **Executor nested-ref leniency**: `resolve_variables` in `executor.py` ignores
  the output-parameter name and does not require a single output parameter. This
  only affects the ReAct *environment* (what the model observes), not the
  official Win Rate, which re-executes independently. Acceptable by design.

## D. Recommended fixes (priority order)

1. **Critical correctness** — make `official_*` canonical in every report; fix
   `metrics.py` to replicate official semantics (value-aware F1 Param, grounded
   Partial/Full, decimal-aware float compare, corpus macro-F1, invalid-JSON
   failure mode) as the internal cross-check.
2. **Paper compatibility** — save per-sample diagnostics with `official_*` +
   `internal_*` + `mismatch`; guarantee one bad sample never aborts the run.
3. **Nice-to-have** — 7-scenario test suite + small fixtures + interpretation
   docs.

---

## Per-metric verdict vs the requested NESTFUL semantics

- **F1 Func** — official: multilabel over function names, order-insensitive,
  argument-insensitive, corpus-level macro. Ours (official adapter): correct.
  Internal replica: fixed to compute the same corpus-level macro at aggregation.
- **F1 Param** — official: `arg = value` slot strings (values included),
  order-insensitive within a function. Official adapter: correct. Internal
  replica: fixed to include values.
- **Partial Match** — official: canonical `name(sorted("k = v"))` per step,
  positional comparison after grounding + length alignment, partial credit.
  Official adapter: correct. Internal replica: fixed to ground labels and align
  lengths (prefers the official helper where importable).
- **Full Match** — official: `Full = 1` iff the whole canonical sequence matches
  (`accuracy_combined == 1.0`). Official adapter: correct. Internal replica:
  fixed alongside Partial.
- **Win Rate** — execution-based; **taken exclusively from the official scorer**,
  which re-executes predicted calls (positional, `output_parameters`,
  decimal-rounded float compare, `signal.SIGALRM` timeout; Unix-only). Our
  `executor.py` is not the source of truth for Win Rate.

## How to read a result file

- Benchmark / paper tables: `official_f1_func`, `official_f1_param`,
  `official_partial`, `official_full`, `official_win` (corpus aggregates from
  `metrics_official.json`).
- Per-sample official diagnostics (in trajectories): `official_partial_match`,
  `official_full_match`, `official_win`, `pred_answer`, `gold_answer`,
  `parse_valid`, `executable`, `execution_error`.
- Per-sample internal diagnostics: `internal_partial`, `internal_full`,
  `internal_win`, and `internal_f1_func` / `internal_f1_param`
  (**diagnostic-only**; the real F1 is corpus-level, not per-sample).
- `mismatch` / `mismatch_reason`: investigate divergences; trust `official_*`.
- Note: Win Rate can legitimately exceed Full Match because an alternative valid
  trajectory can reach the same `gold_answer` without matching the gold steps.
