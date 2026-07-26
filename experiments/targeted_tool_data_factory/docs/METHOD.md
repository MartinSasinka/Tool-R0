# METHOD

End-to-end method of the targeted tool data factory. See DESIGN.md for
rationale, DECISIONS.md for binding choices.

## Pipeline (CLI `targeted-data`)

1. **profile** — NESTFUL dev (n=200) → aggregate `TargetProfile`
   (call-count/motif/depth distributions, per-arg reference share, argument
   & answer types, offered-tool histogram, name morphology, description
   lengths, question length) + the measured Qwen3-4B failure profile from the
   forensic audit. No raw queries/programs are stored.
2. **generate** — generation cells derived from profile × failure profile
   (2-call oversample only because 2-call is the student's weakest measured
   bucket). Per cell, deterministic candidates: typed semantic DAG →
   deterministic execution → oracle observations/answer → surface rendering
   (track-specific tool names, param styles incl. NESTFUL's `arg_0/arg_1`,
   label styles `$var1`/`$var_1`) → offered set with hard distractors →
   template query. Factory guards: no observation may equal a question
   constant (shortcut collapse), no duplicate intermediate values, answer
   not in query.
3. **validate** —
   - V1 schema/format (JSON-schema check of every rendered tool, typed gold
     args, unique names/labels, resolvable references);
   - V2 actual execution (oracle match, 2× replay, NaN/Inf);
   - V3 semantic consistency (all direct constants present in query;
     answer/intermediates do not leak, digit-boundary matching);
   - V4 bounded minimal-path search over offered tools (value-based BFS,
     depth ≤ min(N−1, 3), ≤ 20 000 evals) + single-call-shortcut detection;
   - V5 dedup (exact/normalized/program) + contamination vs NESTFUL dev+test
     (exact, normalized, rapidfuzz ratio ≥ 90, A-track gold-skeleton match,
     G-track target-name ban);
   - V6 pool distribution audit (template ≤ 5 %, cell ≤ 10 %, concentration).
4. **select** — hard gates → greedy per-cell deficit matching with novelty
   tie-breaks (program family / tool combination / template counts) →
   320 frozen tasks; decision trace JSONL; profile-match metrics (JSD,
   Wasserstein, two-sample AUC) for new-vs-target and Stage-3-vs-target.
5. **B2 expansion** (automatic in `all`) — only if cells are < 60 % covered:
   regenerate only deficient cells, cap 5000 total, revalidate, reselect.
6. **probe** — P0 structural difficulty always (heuristic, never oracle);
   P1 (1 rollout) / P2 (4) / P3 (8, borderline only) when a local
   Qwen3-4B (`Qwen/Qwen3-4B-Instruct-2507`) endpoint is reachable;
   otherwise `NOT_RUN_LOCAL` + exact command.
7. **split** — union-find over 6 group keys (program family, graph template,
   tool combination, paraphrase family, argument skeleton, value seed);
   whole components → train 160 / structural held-out 80 / reserve 80;
   leakage audit hard-fails on any collision.
8. **export** — canonical JSONL, NESTFUL-compatible (flat-dict parameters),
   GRPO train-ready (stage3_train_ready row contract), analysis CSV,
   manifests with SHA256 of every file + config/profile/registry/executor
   hashes.
9. **report** — PILOT_REPORT.md (counts, taxonomy, per-cell results, profile
   match, distributions, examples, verdict) + COST_REPORT.md.

## Determinism & hygiene

Candidate i of cell c is fully determined by `seed:cell_id:i:attempt`.
Replay is a hard gate. Any config/generator change ⇒ new `--version`,
new hashes. NESTFUL test is used exclusively as a contamination blocklist;
the diagnostic-500 is never read.
