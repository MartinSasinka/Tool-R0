# SCIENTIFIC RATIONALE

## The problem this dataset attacks

The forensic audit (reports/root_cause_forensic/, 2026-07-25) showed that
GRPO training on the v3 Stage-3 synthetic curriculum improved train-side
signal without improving NESTFUL Win Rate (57.0 % → 57.4 %, n.s.). The
evidence ranked **data/transfer mismatch** as a leading root cause:

- gold-tool Jaccard between Stage-3 and NESTFUL ≈ 0.003–0.006;
- fixed 3-call structure vs NESTFUL's 2–18 calls (dev: 33 % 2-call, 22 % 6+);
- global offered catalog 163 tools vs NESTFUL's ~11 offered per task drawn
  from a much wider pool, with confusable neighbors;
- verbose renamed-arithmetic vocabulary vs NESTFUL's short math core +
  snake_case utilities;
- no controlled hard distractors;
- answer types: NESTFUL dev is 77 % float but 7 % string, 7 % list, 2 % bool,
  2 % numeric-string — Stage-3 is numeric-only.

A dataset cannot cause transfer if the capabilities it exercises are not the
capabilities the target measures. Each design element therefore maps to one
of the 11 target capabilities (table in DESIGN.md §10).

## Why these choices are scientifically defensible

- **Executor-verified oracles** remove label noise entirely; RL signal
  quality is bounded by oracle quality (APIGen's core finding, taken
  further: no LLM in the oracle path at all).
- **Target-conditioned quotas** are estimated from a held-out *dev* split
  (n=200), not the evaluation set, so distribution matching cannot be
  test-set tuning. Matching is verified with symmetric metrics (JSD),
  transport metrics (Wasserstein) and a classifier two-sample test, against
  a fixed baseline (Stage-3) — a falsifiable claim: "the new dataset is
  closer to NESTFUL structure than Stage-3 on k/n metrics, AUC x vs y."
- **Failure-driven cells** allocate the training budget where the student
  measurably fails (2-call bucket 45 % vs 62 %; too_few_calls dominant;
  distractor-poor training). This is the cheapest form of
  student-in-the-loop; the full probe cascade only refines difficulty
  metadata and never defines correctness.
- **Structural held-out** (program-family/template/tool-combination
  disjoint) separates memorization from capability before any NESTFUL
  number is consumed. Transfer claims are deferred to the pre-registered
  data-only experiment (NEXT_TRAINING_EXPERIMENT.md) with paired
  bootstrap + McNemar on the frozen diagnostic-500.

## Known threats to validity (see LIMITATIONS.md)

Template-English query style vs NESTFUL's natural phrasing; primitive
registry is arithmetic-leaning despite typed diversity; bounded minimal-path
search cannot prove absence of all shortcuts (only of cheap ones);
profile-match is necessary but not sufficient for transfer — the final
criterion is the downstream experiment, not any intrinsic metric.
